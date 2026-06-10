from __future__ import annotations

import re

from database.schema_metadata import COLUMNS, RAJASTHAN_DISTRICTS_41
from retrieval.schema_retriever import RetrievalResult, LOCATION_PREPOSITIONS, LOCATION_STOPWORDS


# Build a fast lookup set once at import time
_DISTRICTS_LOWER: dict[str, str] = {d.lower(): d for d in RAJASTHAN_DISTRICTS_41}

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
    Returns ('district', canonical_name) if token matches a known Rajasthan district,
    or ('unknown', None) if it doesn't match any district.
    Tries the full token first, then progressively shorter prefixes for multi-word names.
    """
    words = token.split()
    for length in range(len(words), 0, -1):
        candidate = " ".join(words[:length]).lower()
        if candidate in _DISTRICTS_LOWER:
            return "district", _DISTRICTS_LOWER[candidate]
    return "unknown", None


class PromptBuilder:
    def build(self, result: RetrievalResult, previous_error: str | None = None, dialect: str = "sqlite") -> str:
        column_lookup = {column.qualified_name: column for column in COLUMNS}
        column_lines = []
        for qualified_name in result.columns:
            column = column_lookup.get(qualified_name)
            if not column:
                continue
            indexed = "indexed" if column.indexed else "not indexed"
            sample_values = f"; valid example values: {', '.join(column.sample_values)}" if column.sample_values else ""
            column_lines.append(
                f"- {column.qualified_name} (business meaning: {column.business_name}; {column.data_type}, {indexed}): {column.description}; aliases: {', '.join(column.aliases)}{sample_values}"
            )
        relationship_lines = [
            f"- {r['from_table']}.{r['from_column']} = {r['to_table']}.{r['to_column']}"
            for r in result.relationships
        ]
        error_block = f"\nPrevious SQL was invalid: {previous_error}\nFix it.\n" if previous_error else ""

        dialect_desc = "SQLite"
        if dialect.lower() == "postgresql":
            dialect_desc = "PostgreSQL"

        # ── Location pre-classification ───────────────────────────────────────
        # Classify every location token before the LLM generates SQL so it gets
        # the exact SQL condition — no column guessing required.
        location_hints = _extract_location_hints(result.question)
        location_rules: list[str] = []
        for token in location_hints:
            kind, canonical = _classify_location(token)
            # Title-case to match how the DB stores location names
            display_token = " ".join(w.capitalize() for w in token.split())
            if kind == "district":
                location_rules.append(
                    f"- The location '{display_token}' IS one of the 41 Rajasthan districts. "
                    f"Filter using: family.district = '{canonical}'"
                )
            else:
                location_rules.append(
                    f"- The location '{display_token}' is NOT one of the 41 Rajasthan districts. "
                    f"You MUST filter using: (family.block LIKE '%{display_token}%' OR family.village LIKE '%{display_token}%'). "
                    f"Do NOT use family.district for this location."
                )

        location_block = ""
        if location_rules:
            location_block = "\nLocation classification (use these exact conditions — do not override them):\n" + "\n".join(location_rules)

        # ── Dynamic column-specific rules ─────────────────────────────────────
        dynamic_rules = []

        if "member.education" in result.columns:
            dynamic_rules.append(
                "- education filtering:\n"
                "  * 'illiterate' is stored lowercase in the DB. Use LOWER(education) = 'illiterate' or education LIKE '%illiterate%'.\n"
                "  * All other education values are Title Case: 'Literate', '5 Pass', '8 Pass', '10 Pass', '12 Pass', 'Graduate', 'Post Graduate'.\n"
                "  * For partial education matches use LIKE, e.g., education LIKE '%Pass%' to match all pass levels."
            )

        if "member.minority" in result.columns:
            dynamic_rules.append(
                "- minority filtering: Use minority = 'Muslim' or minority = 'Jain'. "
                "Most members (96%) have NULL minority — this is expected and correct. "
                "Do NOT add IS NOT NULL unless the question specifically asks for minority members."
            )

        if "member.caste_category" in result.columns:
            dynamic_rules.append(
                "- caste_category filtering: Use ONLY the exact stored values: 'SC', 'ST', 'OBC', 'GEN'.\n"
                "  * 'General category' or 'general' in the question means caste_category = 'GEN' (NOT 'General').\n"
                "  * 'Scheduled Caste' means caste_category = 'SC'.\n"
                "  * 'Scheduled Tribe' means caste_category = 'ST'.\n"
                "  * 'Other Backward Class/Caste' means caste_category = 'OBC'."
            )

        if "member.caste" in result.columns:
            dynamic_rules.append(
                "- caste column filtering:\n"
                "  * Numbers have been removed during import — search for 'Jat' not '58 Jat'.\n"
                "  * Casing is inconsistent across records — always use LIKE for caste searches.\n"
                "  * Always use a simple single-word LIKE filter for caste searches (e.g. member.caste LIKE '%Rajput%').\n"
                "    Do NOT use IN lists or multiple OR conditions for spellings/languages, as a post-processor\n"
                "    will automatically expand it to search in both English and Hindi.\n"
                "  * CRITICAL: If the question mentions a specific caste name (e.g., Fakir, Jat, Rajput, Brahman),\n"
                "    filter ONLY on member.caste using LIKE. Do NOT add a caste_category filter — you have no\n"
                "    information about which category that caste belongs to unless the question explicitly states it."
            )

        if "bank_details.bank_name" in result.columns:
            dynamic_rules.append(
                "- bank_name filtering: Bank names are stored inconsistently (UPPER, Title, mixed case). "
                "Always use UPPER(bank_name) LIKE '%SEARCH_TERM_IN_UPPER%'. "
                "Example: UPPER(bank_name) LIKE '%STATE BANK%' matches 'STATE BANK OF INDIA', 'State Bank of India', etc."
            )

        # Prompt rule for unbanked / no-bank-account queries
        if "bank_details.bank_id" in result.columns or "bank_details.bank_name" in result.columns:
            _q = result.question.lower()
            if any(w in _q for w in ["no bank", "without bank", "don't have",
                                      "do not have", "no account", "unbanked",
                                      "without account"]):
                dynamic_rules.append(
                    "- no bank account query: Use LEFT JOIN bank_details ON "
                    "bank_details.member_id = member.member_id "
                    "WHERE bank_details.bank_id IS NULL. "
                    "Never use INNER JOIN for this — INNER JOIN returns zero rows for unbanked members."
                )

        if "family.is_rural" in result.columns:
            dynamic_rules.append(
                "- is_rural is an INTEGER column: 1 = rural family, 0 = urban family.\n"
                "  * 'rural families' or 'village families' → is_rural = 1\n"
                "  * 'urban families' or 'city families' → is_rural = 0\n"
                "  * Rural families store location in block/village/gram_panchayat; urban families use city/ward."
            )

        # ── Family member count rule ────────────────────────────────────────────────
        # IMPORTANT: This block must be BEFORE dynamic_rules_block is computed.
        # If placed after, the family count rule would be silently dropped from every prompt.
        if "family.family_id" in result.columns and "member.family_id" in result.columns:
            dynamic_rules.append(
                "- member counts per family: use the member table only — no JOIN needed. "
                "Every member row already has family_id. "
                "GROUP BY member.family_id HAVING COUNT(*) > N. "
                "Use COUNT(*) not COUNT(member.member_id). "
                "Do not join the family table just to count members per family."
            )

        # ── Compute AFTER all appends so every rule is included ───────────────
        dynamic_rules_block = "\n".join(dynamic_rules)
        if dynamic_rules_block:
            dynamic_rules_block = "\n" + dynamic_rules_block

        return f"""You are a SQL generator for a Jan Aadhaar-style relational database.
Return SQL only. No markdown. No comments. No explanation.
Generate exactly one read-only SELECT statement.
Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, MERGE, REPLACE, VACUUM, PRAGMA, ATTACH, DETACH, GRANT, REVOKE, or any DDL/DML/admin command.
Use only the tables, columns, and relationships supplied below.
Do not invent tables or columns.
Every selected column, WHERE column, JOIN column, GROUP BY column, and ORDER BY column must appear in Relevant columns.
If a desired field is not listed in Relevant columns, do not use it.
If the question asks for "all" records, return identifying columns from Relevant columns, not SELECT *.
Avoid SELECT *; select explicit columns.
Never include family_id (family.family_id, member.family_id) or bank_id in the SELECT clause unless it is used in a GROUP BY for member count aggregation; they are internal surrogate keys meaningless to the user. Use them only in JOIN conditions or GROUP BY clauses.
Use joins only when required by the question.
TRANSITIVE JOIN RULE: You can only JOIN tables that have an allowed relationship listed below. Never join tables directly if there is no declared relationship between them; join them transitively through intermediate tables. For example, to join family and bank_details, you must join member in between: family JOIN member ON member.family_id = family.family_id JOIN bank_details ON bank_details.member_id = member.member_id.
MULTIPLE TABLE JOIN RULE: If columns from family, member, and bank_details are all listed in Relevant columns, you MUST join all three tables together in your query using the allowed relationships. Do not try to reference family columns (like district) or member columns (like gender) or bank_details columns (like bank_name) without joining their respective tables.
Prefer indexed columns in predicates where applicable.
MULTIPLE VALUE FILTERING RULE: If the question filters a single column by multiple different values (e.g. "from Srinagar and Beejasar", "living in Jaipur and Ajmer", "caste is Jat or Rajput"), always join those conditions using OR, not AND. A single record can never satisfy multiple different values for the same column at the same time.
CRITICAL COLUMN PREFIX RULE: Demographic columns (member_name, gender, age, income, caste, caste_category, marital_status, minority, occupation, education) ALWAYS belong to the 'member' table. NEVER prefix them with 'bank_details.' or 'family.' (e.g. use member.member_name, member.gender, member.income, NOT bank_details.member_name, bank_details.gender, family.gender). Geography columns (district, block, village, gram_panchayat, city, ward, is_rural) ALWAYS belong to the 'family' table. Bank identifiers (bank_name, bank_account, ifsc_code) ALWAYS belong to the 'bank_details' table.
Generate {dialect_desc}-compatible SQL.
CRITICAL NAME FILTERING RULE: When filtering by any person name column (member.member_name, father_name, mother_name, spouse_name, or family_head_name), ALWAYS use LIKE with wildcards (e.g., member.member_name LIKE '%Vijay%'). NEVER use exact '=' for name searches — database entries are full names with surnames and will fail exact matches.
Interpret common wording precisely:
- boy or boys means member.gender = 'Male'.
- girl or girls means member.gender = 'Female'.
- man or men means member.gender = 'Male'.
- woman or women or ladies means member.gender = 'Female'.
- widow or widows or widowed means member.marital_status = 'Widow' AND member.gender = 'Female'.
- widower or widowers means member.marital_status = 'Widow' AND member.gender = 'Male'.
- unmarried or single means member.marital_status = 'Unmarried'.
- family head or HOF means member.member_type = 'HOF' (always female; relation_with_hof = 'Self').
- husband of the family means member.relation_with_hof = 'Husband'.
- son means member.relation_with_hof = 'Son'.
- daughter means member.relation_with_hof = 'Daughter'.
- above N, older than N, greater than N means member.age > N.
- below N, younger than N, less than N means member.age < N.
- between N and M means member.age BETWEEN N AND M.
- senior citizen, elderly, old age person means member.age >= 60.
- child, children, minor means member.age < 18.
- adult means member.age >= 18.
- working age means member.age BETWEEN 18 AND 59.
- how many, count of, number of means use COUNT(*) or COUNT(DISTINCT ...) as appropriate.
- how many families means COUNT(DISTINCT family.family_id) or COUNT(DISTINCT member.family_id).
- total income means SUM(income); average age means AVG(age).
- rural families means family.is_rural = 1; urban families means family.is_rural = 0.
- Use canonical capitalization: 'Male', 'Female', 'Married', 'Unmarried', 'Widow', 'SC', 'ST', 'OBC', 'GEN'.{location_block}{dynamic_rules_block}

Available tables:
{chr(10).join(f"- {table}" for table in result.tables)}

Relevant columns:
{chr(10).join(column_lines)}

Allowed relationships:
{chr(10).join(relationship_lines) if relationship_lines else "- none"}
{error_block}
Question:
{result.question}

SQL:"""
