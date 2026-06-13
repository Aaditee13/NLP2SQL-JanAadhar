from __future__ import annotations

import re
from typing import Any

from database.schema_metadata import COLUMNS, RAJASTHAN_DISTRICTS_41, RAJASTHAN_CITIES, RAJASTHAN_BLOCKS
from retrieval.schema_retriever import RetrievalResult, LOCATION_PREPOSITIONS, LOCATION_STOPWORDS


# Build fast lookup maps once at import time
_DISTRICTS_LOWER: dict[str, str] = {d.lower(): d for d in RAJASTHAN_DISTRICTS_41}
_CITIES_LOWER: dict[str, str] = {c.lower(): c for c in RAJASTHAN_CITIES}
_BLOCKS_LOWER: dict[str, str] = {b.lower(): b for b in RAJASTHAN_BLOCKS}

# Prepositions whose following word we treat as a candidate location
_LOC_PREPOSITIONS = LOCATION_PREPOSITIONS  # {"in", "from", "at"}


def _extract_location_hints(question: str) -> list[str]:
    """
    Extract candidate location tokens from the question.
    Returns a list of raw strings as the user wrote them (case-preserved).
    Tokens that are common stopwords or purely numeric are excluded.
    Supports coordinate list extraction (e.g. "Srinagar and Beejasar").
    """
    hints: list[str] = []
    seen: set[str] = set()
    for prep in _LOC_PREPOSITIONS:
        for match in re.finditer(
            rf"\b{prep}\s+([A-Za-z][A-Za-z\s-]{{1,30}}?)(?:\s+(?:and|or|where|who|that|which|with|having|are|is)\b|[,.]|$)",
            question,
            re.IGNORECASE,
        ):
            raw = match.group(1).strip()
            token = " ".join(raw.split()[:3])
            key = token.lower()
            if key in LOCATION_STOPWORDS or not token:
                continue
            if key not in seen:
                seen.add(key)
                hints.append(token)
                
            # Check for subsequent locations connected by coordinate conjunctions or commas
            end_pos = match.end(1)
            remaining = question[end_pos:]
            while True:
                conjunction_match = re.match(
                    r"^\s*(?:and|or|,)\s+([A-Za-z][A-Za-z\s-]{1,30}?)(?:\s+(?:and|or|where|who|that|which|with|having|are|is)\b|[,.]|$)",
                    remaining,
                    re.IGNORECASE
                )
                if not conjunction_match:
                    break
                next_raw = conjunction_match.group(1).strip()
                next_token = " ".join(next_raw.split()[:3])
                next_key = next_token.lower()
                if next_key not in LOCATION_STOPWORDS and next_token:
                    if next_key not in seen:
                        seen.add(next_key)
                        hints.append(next_token)
                remaining = remaining[conjunction_match.end(1):]
    return hints


def _classify_location(token: str) -> tuple[str, str | None]:
    """
    Classify token to determine if it matches a known district, city, or block.
    Returns (type, canonical_name).
    """
    lowered = token.lower().strip()
    
    # Strip off common descriptive suffixes
    suffixes = [
        " district", " zilla", " block", " city", " tehsil", 
        " village", " gaon", " gp", " gram panchayat"
    ]
    for suffix in suffixes:
        if lowered.endswith(suffix):
            lowered = lowered[:-len(suffix)].strip()
            break

    if lowered in _DISTRICTS_LOWER:
        return "district", _DISTRICTS_LOWER[lowered]
    if lowered in _CITIES_LOWER:
        return "city", _CITIES_LOWER[lowered]
    if lowered in _BLOCKS_LOWER:
        return "block", _BLOCKS_LOWER[lowered]
    return "unknown", None


class PromptBuilder:
    def build(
        self,
        result: RetrievalResult,
        previous_error: str | None = None,
        dialect: str = "sqlite",
        few_shots: list[dict[str, Any]] | None = None,
    ) -> str:
        column_lookup = {column.qualified_name: column for column in COLUMNS}
        column_lines = []
        for qualified_name in result.columns:
            column = column_lookup.get(qualified_name)
            if not column:
                continue
            indexed = "indexed" if column.indexed else "not indexed"
            sample_values = f"; valid example values: {', '.join(column.sample_values)}" if column.sample_values else ""
            column_lines.append(
                f"- {column.column} (business meaning: {column.business_name}; {column.data_type}, {indexed}): {column.description}{sample_values}"
            )

        error_block = f"\nPrevious SQL was invalid: {previous_error}\nFix it.\n" if previous_error else ""

        dialect_desc = "SQLite"
        if dialect.lower() == "postgresql":
            dialect_desc = "PostgreSQL"

        # ── Location pre-classification ───────────────────────────────────────
        location_hints = _extract_location_hints(result.question)
        location_rules: list[str] = []
        for token in location_hints:
            kind, canonical = _classify_location(token)
            display_token = " ".join(w.capitalize() for w in token.split())
            if kind == "district":
                location_rules.append(
                    f"- The location '{display_token}' IS one of the 41 Rajasthan districts. "
                    f"Filter using: citizen.district_name_eng = '{canonical}'"
                )
            elif kind == "city":
                location_rules.append(
                    f"- The location '{display_token}' IS one of the known Rajasthan cities. "
                    f"Filter using: citizen.city_name_eng = '{canonical}'"
                )
            elif kind == "block":
                location_rules.append(
                    f"- The location '{display_token}' IS one of the known Rajasthan blocks. "
                    f"Filter using: citizen.block_name_eng = '{canonical}'"
                )
            else:
                location_rules.append(
                    f"- The location '{display_token}' is not recognized as a district, city, or block. "
                    f"You MUST search across all sub-locations using: (citizen.block_name_eng LIKE '%{display_token}%' OR citizen.gp_name_eng LIKE '%{display_token}%' OR citizen.vill_name_eng LIKE '%{display_token}%'). "
                    f"Do NOT use citizen.district_name_eng for this location."
                )

        location_block = ""
        if location_rules:
            location_block = "\nLocation classification (use these exact conditions — do not override them):\n" + "\n".join(location_rules)

        # ── Dynamic column-specific rules ─────────────────────────────────────
        dynamic_rules = []

        if "citizen.education" in result.columns:
            dynamic_rules.append(
                "- education filtering:\n"
                "  * Stored education levels from lowest to highest: 'illiterate' (lowercase), 'Literate', '5 Pass', '8 Pass', '10 Pass', '12 Pass', 'Graduate', 'Post Graduate'.\n"
                "  * For hierarchical educational queries containing 'and above' or 'and below' (e.g. '10th pass and above'), you MUST use an IN clause listing all qualifying levels.\n"
                "    - Example: '10th pass and above' -> education IN ('10 Pass', '12 Pass', 'Graduate', 'Post Graduate')\n"
                "    - Example: '12th pass and below' -> education IN ('illiterate', 'Literate', '5 Pass', '8 Pass', '10 Pass', '12 Pass')\n"
                "  * Always use standard Title Case values (e.g. '10 Pass', not '10th Pass') for individual categories."
            )

        if "citizen.minority" in result.columns:
            dynamic_rules.append(
                "- minority filtering: Use minority = 'Muslim' or minority = 'Jain'. "
                "Most citizens (96%) have NULL minority — this is expected and correct. "
                "Do NOT add IS NOT NULL unless the question specifically asks for minority members."
            )

        if "citizen.caste_category" in result.columns:
            dynamic_rules.append(
                "- caste_category filtering: Use ONLY the exact stored values: 'SC', 'ST', 'OBC', 'GEN'.\n"
                "  * 'General category' or 'general' in the question means caste_category = 'GEN' (NOT 'General').\n"
                "  * 'Scheduled Caste' means caste_category = 'SC'.\n"
                "  * 'Scheduled Tribe' means caste_category = 'ST'.\n"
                "  * 'Other Backward Class/Caste' means caste_category = 'OBC'."
            )

        if "citizen.caste" in result.columns:
            dynamic_rules.append(
                "- caste column filtering:\n"
                "  * Numbers have been removed during import — search for 'Jat' not '58 Jat'.\n"
                "  * Casing is inconsistent across records — always use LIKE for caste searches.\n"
                "  * Always use a simple single-word LIKE filter for caste searches (e.g. citizen.caste LIKE '%Rajput%').\n"
                "    Do NOT use IN lists or multiple OR conditions for spellings/languages, as a post-processor\n"
                "    will automatically expand it to search in both English and Hindi.\n"
                "  * CRITICAL: If the question mentions a specific caste name (e.g., Fakir, Jat, Rajput, Brahman),\n"
                "    filter ONLY on citizen.caste using LIKE. Do NOT add a caste_category filter."
            )

        if "citizen.bank" in result.columns:
            dynamic_rules.append(
                "- bank filtering: Bank names are stored inconsistently (UPPER, Title, mixed case). "
                "Always use UPPER(bank) LIKE '%SEARCH_TERM_IN_UPPER%'. "
                "Example: UPPER(bank) LIKE '%STATE BANK%' matches 'STATE BANK OF INDIA', 'State Bank of India', etc."
            )

        # Prompt rule for unbanked / no-bank-account queries
        if "citizen.account_no" in result.columns:
            _q = result.question.lower()
            if any(w in _q for w in ["no bank", "without bank", "don't have",
                                      "do not have", "no account", "unbanked",
                                      "without account"]):
                dynamic_rules.append(
                    "- no bank account query: Use account_no IS NULL. "
                )

        if "citizen.is_rural" in result.columns:
            dynamic_rules.append(
                "- is_rural is an INTEGER column: 1 = rural family, 0 = urban family.\n"
                "  * 'rural families' or 'village families' → is_rural = 1\n"
                "  * 'urban families' or 'city families' → is_rural = 0"
            )

        # ── Family member count rule ────────────────────────────────────────────────
        if "citizen.enrollment_id" in result.columns:
            dynamic_rules.append(
                "- member counts per family: use the enrollment_id column to group family members. "
                "GROUP BY enrollment_id HAVING COUNT(*) > N. "
                "Use COUNT(*) not COUNT(member_id)."
            )

        dynamic_rules_block = "\n".join(dynamic_rules)
        if dynamic_rules_block:
            dynamic_rules_block = "\n" + dynamic_rules_block

        # Dynamic few-shots block
        few_shots_block = ""
        if few_shots:
            lines = ["### EXAMPLES"]
            for ex in few_shots:
                lines.append(f"Question: {ex['question']}")
                lines.append(f"SQL:\n{ex['sql']}\n")
            few_shots_block = "\n" + "\n".join(lines)

        return f"""You are a SQL generator for a single flat table called `citizen`.
Return SQL only. No markdown. No comments. No explanation.
Generate exactly one read-only SELECT statement.
Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, MERGE, REPLACE, VACUUM, PRAGMA, ATTACH, DETACH, GRANT, REVOKE, or any DDL/DML/admin command.
Use only the table and columns supplied below. Do not invent columns.
Every column in SELECT, WHERE, GROUP BY, and ORDER BY must exist in the schema below.
If a desired field is not listed, do not use it.
Avoid SELECT *; always select explicit identifying columns (such as name_en, age, gender, district_name_eng) unless the user explicitly asks for "all columns" or "all fields".
Never include enrollment_id or member_id in the SELECT clause unless it is used in a GROUP BY for member count aggregation or explicitly requested; they are internal keys.
Generate {dialect_desc}-compatible SQL.
CRITICAL NAME FILTERING RULE: When filtering by any person name column (name_en, father_name_en, mother_name_en, spouce_name_en), ALWAYS use LIKE with wildcards (e.g., name_en LIKE '%Vijay%'). NEVER use exact '=' for name searches.
FAMILY RELATIONS & AGGREGATES: If a query filters by family relations (e.g. "families where...", "whose sons...") or family-level aggregate properties (e.g. "family income is between X and Y", "families with more than N members"), you MUST filter using a subquery on enrollment_id with GROUP BY and HAVING. Example for family income: `enrollment_id IN (SELECT enrollment_id FROM citizen GROUP BY enrollment_id HAVING SUM(income) BETWEEN 100000 AND 300000)`. Never place aggregate functions (like SUM or COUNT) in the WHERE clause, as SQLite will raise a syntax error.
Interpret common wording precisely:
- boy or boys means gender = 'Male'.
- girl or girls means gender = 'Female'.
- man or men means gender = 'Male'.
- woman or women or ladies means gender = 'Female'.
- widow or widows or widowed means marital_status = 'Widow' AND gender = 'Female'.
- unmarried or single means marital_status = 'Unmarried'.
- family head or HOF means mem_type = 'HOF' (always female; relation_with_hof = 'Self').
- husband of the family means relation_with_hof = 'Husband'.
- son means relation_with_hof = 'Son'.
- daughter means relation_with_hof = 'Daughter'.
- above N, older than N, greater than N means age > N.
- below N, younger than N, less than N means age < N.
- between N and M means age BETWEEN N AND M.
- senior citizen, elderly, old age person means age >= 60.
- child, children, minor means age < 18.
- adult means age >= 18.
- how many, count of, number of means use COUNT(*) or COUNT(DISTINCT ...) as appropriate.
- total income means SUM(income); average age means AVG(age).
- rural families/citizens means is_rural = 1; urban families/citizens means is_rural = 0.
- Use canonical capitalization: 'Male', 'Female', 'Married', 'Unmarried', 'Widow', 'SC', 'ST', 'OBC', 'GEN'.{location_block}{dynamic_rules_block}
{few_shots_block}
Table: citizen
Columns:
{chr(10).join(column_lines)}
{error_block}
Question:
{result.question}

SQL:"""
