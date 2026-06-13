from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import settings
from database.excel_importer import import_excel_dataset
from database.query_results import execute_select_preview
from embeddings.faiss_store import FaissSchemaStore
from llm.ollama_client import OllamaModelManager, OllamaSqlGenerator
from normalization.query_normalizer import normalize_query
from optimization.query_optimizer import OptimizationReport, QueryOptimizer
from prompting.prompt_builder import PromptBuilder
from retrieval.schema_retriever import SchemaRetriever
from validation.sql_validator import SQLValidator
from retrieval.few_shot_retriever import FaissFewShotStore, FewShotRetriever
from caching.semantic_cache import FaissCacheStore, SemanticCache

from database.schema_metadata import RAJASTHAN_DISTRICTS_41

# ── Module-level lookups ─────────────────────────────────────────────────────
_DISTRICTS_LOWER    = {d.lower() for d in RAJASTHAN_DISTRICTS_41}
_DISTRICT_CANONICAL = {d.lower(): d for d in RAJASTHAN_DISTRICTS_41}

# Phrases that signal an "unbanked / no bank account" query
_NO_BANK_WORDS = [
    "no bank", "without bank", "don't have", "do not have",
    "no account", "unbanked", "without account",
]

_CASTE_GROUPS = [
    {"rajput", "rajpoot", "राजपूत"},
    {"jat", "जाट"},
    {"mina", "meena", "मीना"},
    {"brahman", "brahmin", "brahaman", "bhraman", "bharmn", "ब्राह्मण", "ब्राहम्ण"},
    {"bairwa", "berwa", "बैरवा"},
    {"gurjar", "gujar", "गुर्जर"},
    {"bazigar", "बाजीगर"},
    {"dhobi", "धोबी"},
    {"darzi", "दर्जी"},
    {"fakir", "फकीर"},
    {"valmiki", "balmiki", "वाल्मीकि"},
    {"chhipa", "chhippa", "छीपा"},
    {"daroga", "दरोगा"},
    {"jain", "जैन"},
    {"dangi", "डांगी"},
    {"deshwali", "देशवाली"},
    {"sindhi", "सिंधी"},
    {"arai", "अराई"},
    {"agrawal", "agarwal", "अग्रवाल"},
    {"mahajan", "महाजन"}
]


@dataclass
class PipelineOutput:
    question: str
    normalized_question: str
    query_corrections: dict[str, str]
    sql: str
    retrieved_tables: list[str]
    retrieved_columns: list[str]
    confidence: float
    validation_errors: list[str]
    optimization: OptimizationReport | None
    is_fuzzy: bool = False
    fuzzy_target: str | None = None
    source: str = "llm"


def _replace_outside_quotes(pattern_word: str, replacement: str, text: str) -> str:
    """
    Replace whole word pattern_word with replacement in text, but ONLY outside
    of single-quoted or double-quoted string literals.
    """
    import re
    # Match quoted string literals first, or target whole word
    regex = re.compile(rf"('[^']*'|\"[^\"]*\")|\b{pattern_word}\b", re.IGNORECASE)
    return regex.sub(lambda m: m.group(1) if m.group(1) is not None else replacement, text)


def _post_process_sql(sql: str, fuzzy_target: str | None = None) -> str:
    """
    Post-process LLM-generated SQL for the single flat citizen table.
    """
    import re

    # ── Normalize legacy multi-table qualifiers ──
    # Map family, member, bank_details table prefixes to flat attributes
    sql = re.sub(r"\bmember\.member_name\b", "name_en", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bmember\.father_name\b", "father_name_en", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bmember\.mother_name\b", "mother_name_en", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bmember\.spouse_name\b", "spouce_name_en", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bmember\.date_of_birth\b", "dob", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bmember\.mobile_number\b", "mobile_no", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bbank_details\.bank_name\b", "bank", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bbank_details\.bank_account\b", "account_no", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bbank_details\.ifsc_code\b", "ifsc_code", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bfamily\.district\b", "district_name_eng", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bfamily\.city\b", "city_name_eng", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bfamily\.block\b", "block_name_eng", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bfamily\.gram_panchayat\b", "gp_name_eng", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bfamily\.village\b", "vill_name_eng", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bfamily\.ward\b", "ward_name_eng", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bfamily\.is_rural\b", "is_rural", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bfamily\.jan_aadhaar_number\b", "enrollment_id", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bmember\.jan_aadhaar_member_id\b", "jan_aadhaar_member_id", sql, flags=re.IGNORECASE)
    
    sql = _replace_outside_quotes("member_name", "name_en", sql)
    sql = _replace_outside_quotes("father_name", "father_name_en", sql)
    sql = _replace_outside_quotes("mother_name", "mother_name_en", sql)
    sql = _replace_outside_quotes("spouse_name", "spouce_name_en", sql)
    sql = _replace_outside_quotes("spouce_name", "spouce_name_en", sql)
    sql = _replace_outside_quotes("bank_name", "bank", sql)
    sql = _replace_outside_quotes("bank_account", "account_no", sql)
    sql = _replace_outside_quotes("district", "district_name_eng", sql)
    sql = _replace_outside_quotes("city", "city_name_eng", sql)
    sql = _replace_outside_quotes("block", "block_name_eng", sql)
    sql = _replace_outside_quotes("gram_panchayat", "gp_name_eng", sql)
    sql = _replace_outside_quotes("village", "vill_name_eng", sql)
    sql = _replace_outside_quotes("ward", "ward_name_eng", sql)
    sql = _replace_outside_quotes("jan_aadhaar_number", "enrollment_id", sql)
    sql = _replace_outside_quotes("mobile_number", "mobile_no", sql)
    sql = _replace_outside_quotes("member_type", "mem_type", sql)

    # Clean up multi-table qualifiers (e.g. member.age -> age)
    sql = re.sub(r"\b(?:member|family|bank_details)\.(\w+)\b", r"\1", sql, flags=re.IGNORECASE)

    # ── Step 1: Free-text columns → LIKE '%val%' ──
    def text_replacer(match):
        col = match.group(1)
        val = match.group(2)
        return f"{col} LIKE '%{val}%'"

    _FREE_TEXT_COLS = (
        "name_en|father_name_en|mother_name_en|spouce_name_en"
        "|caste|city_name_eng|block_name_eng|gp_name_eng|vill_name_eng|occupation"
    )
    sql = re.sub(
        rf"\b({_FREE_TEXT_COLS})\s*=\s*'([^']+)'",
        text_replacer, sql, flags=re.IGNORECASE,
    )
    sql = re.sub(
        rf'\b({_FREE_TEXT_COLS})\s*=\s*"([^"]+)"',
        text_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 1.0: Fuzzy Name Broadening ──
    if fuzzy_target:
        prefix = fuzzy_target[:3]
        target_lower = fuzzy_target.lower()
        def fuzzy_repl(match):
            col_part = match.group(1)
            val = match.group(2).strip()
            from rapidfuzz.distance import JaroWinkler
            score = JaroWinkler.similarity(target_lower, val.lower())
            if score > 1.0:
                score = score / 100.0
            if score >= 0.60 or val.lower() in target_lower or target_lower in val.lower():
                return f"{col_part} LIKE '%{prefix}%'"
            return match.group(0)
        _NAME_COLS = "name_en|father_name_en|mother_name_en|spouce_name_en"
        sql = re.sub(
            rf"\b({_NAME_COLS})\s+LIKE\s+'%?([^'%]+)%?'",
            fuzzy_repl, sql, flags=re.IGNORECASE,
        )

    # ── Step 1.1: Caste IN Clause Expansion ──
    def caste_in_replacer(match):
        col = match.group(1)
        in_content = match.group(2)
        vals = [v[0] or v[1] for v in re.findall(r"'([^']+)'|\"([^\"]+)\"", in_content)]
        if not vals:
            return match.group(0)
            
        all_conditions = []
        for val in vals:
            val_l = val.strip().lower()
            matched_group = False
            for group in _CASTE_GROUPS:
                if val_l in group:
                    matched_group = True
                    for term in sorted(group, key=lambda x: len(x), reverse=True):
                        formatted = term.title() if term.isascii() else term
                        all_conditions.append(f"{col} LIKE '%{formatted}%'")
                    break
            if not matched_group:
                all_conditions.append(f"{col} LIKE '%{val}%'")
                
        return "(" + " OR ".join(all_conditions) + ")"

    sql = re.sub(
        r"\b(caste)\s+IN\s+\(([^)]+)\)",
        caste_in_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 1.2: Caste Bilingual Expansion ──
    def caste_bilingual_replacer(match):
        col = match.group(1)
        val = match.group(2).strip()
        val_l = val.lower()
        for group in _CASTE_GROUPS:
            if val_l in group:
                conditions = []
                for term in sorted(group, key=lambda x: len(x), reverse=True):
                    formatted = term.title() if term.isascii() else term
                    conditions.append(f"{col} LIKE '%{formatted}%'")
                return "(" + " OR ".join(conditions) + ")"
        return match.group(0)

    sql = re.sub(
        r"\b(caste)\s+LIKE\s+'%?([^'%]+)%?'",
        caste_bilingual_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 2: bank → UPPER(col) LIKE '%UPPER_VAL%' ──
    def bank_replace_safe(match):
        col = match.group(1)
        val = match.group(2).strip()
        return f"UPPER({col}) LIKE '%{val.upper()}%'"

    sql = re.sub(
        r"\b(bank)\s*=\s*'([^']+)'",
        bank_replace_safe, sql, flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"\b(bank)\s*LIKE\s*'%([^'%]+)%'",
        lambda m: (
            m.group(0) if m.group(0).upper().startswith("UPPER(")
            else f"UPPER({m.group(1)}) LIKE '%{m.group(2).strip().upper()}%'"
        ),
        sql, flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"\b(bank)\s*LIKE\s*'([^'%]+)'",
        lambda m: (
            m.group(0) if m.group(0).upper().startswith("UPPER(")
            else f"UPPER({m.group(1)}) LIKE '%{m.group(2).strip().upper()}%'"
        ),
        sql, flags=re.IGNORECASE,
    )

    # ── Step 2.5: bank IN (...) → UPPER(bank) IN (...) ──
    def bank_in_replacer(match):
        col = match.group(1)
        in_content = match.group(2)
        vals = [v[0] or v[1] for v in re.findall(r"'([^']+)'|\"([^\"]+)\"", in_content)]
        new_vals = [f"'{v.strip().upper()}'" for v in vals]
        return f"UPPER({col}) IN ({', '.join(new_vals)})"

    sql = re.sub(
        r"\b(bank)\s+IN\s+\(([^)]+)\)",
        bank_in_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 3: Categorical value normalization ──
    def cat_replacer(match):
        col_raw = match.group(1)
        col = col_raw.lower()
        val = match.group(2).strip()
        val_l = val.lower()

        if "gender" in col:
            if val_l in ("male", "m"):
                return f"{col_raw} = 'Male'"
            if val_l in ("female", "f"):
                return f"{col_raw} = 'Female'"

        elif "caste_category" in col:
            if val_l in ("sc", "scheduled caste", "dalit"):
                return f"{col_raw} = 'SC'"
            if val_l in ("st", "scheduled tribe", "tribal", "adivasi"):
                return f"{col_raw} = 'ST'"
            if val_l in ("obc", "other backward class", "other backward caste", "other backward", "backward class"):
                return f"{col_raw} = 'OBC'"
            if val_l in ("gen", "general", "general category", "open", "unreserved", "ur", "forward", "forward caste"):
                return f"{col_raw} = 'GEN'"
            return f"{col_raw} = '{val.upper()}'"

        elif "marital_status" in col:
            if val_l in ("married",):
                return f"{col_raw} = 'Married'"
            if val_l in ("unmarried", "single", "never married", "bachelor", "spinster"):
                return f"{col_raw} = 'Unmarried'"
            if val_l in ("widow", "widowed", "widower"):
                return f"{col_raw} = 'Widow'"

        return match.group(0)

    _CAT_COLS = r"gender|caste_category|marital_status"
    sql = re.sub(
        rf"\b({_CAT_COLS})\s*=\s*'([^']+)'",
        cat_replacer, sql, flags=re.IGNORECASE,
    )
    sql = re.sub(
        rf'\b({_CAT_COLS})\s*=\s*"([^"]+)"',
        cat_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 3.5: Categorical IN Clause Casing Normalization ──
    def cat_in_replacer(match):
        col_raw = match.group(1)
        col = col_raw.lower()
        in_content = match.group(2)
        vals = [v[0] or v[1] for v in re.findall(r"'([^']+)'|\"([^\"]+)\"", in_content)]
        
        new_vals = []
        for val in vals:
            val_l = val.strip().lower()
            if "gender" in col:
                if val_l in ("male", "m"):
                    new_vals.append("'Male'")
                elif val_l in ("female", "f"):
                    new_vals.append("'Female'")
                else:
                    new_vals.append(f"'{val}'")
            elif "caste_category" in col:
                if val_l in ("sc", "scheduled caste", "dalit"):
                    new_vals.append("'SC'")
                elif val_l in ("st", "scheduled tribe", "tribal", "adivasi"):
                    new_vals.append("'ST'")
                elif val_l in ("obc", "other backward class", "other backward caste", "other backward", "backward class"):
                    new_vals.append("'OBC'")
                elif val_l in ("gen", "general", "general category", "open", "unreserved", "ur", "forward", "forward caste"):
                    new_vals.append("'GEN'")
                else:
                    new_vals.append(f"'{val.upper()}'")
            elif "marital_status" in col:
                if val_l in ("married",):
                    new_vals.append("'Married'")
                elif val_l in ("unmarried", "single", "never married", "bachelor", "spinster"):
                    new_vals.append("'Unmarried'")
                elif val_l in ("widow", "widowed", "widower"):
                    new_vals.append("'Widow'")
                else:
                    new_vals.append(f"'{val}'")
            else:
                new_vals.append(f"'{val}'")
                
        return f"{col_raw} IN ({', '.join(new_vals)})"

    sql = re.sub(
        rf"\b({_CAT_COLS})\s+IN\s+\(([^)]+)\)",
        cat_in_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 4: Categorical LIKE → = ──
    sql = re.sub(
        r"\b(gender)\s+LIKE\s+'%?(Male|Female)%?'",
        lambda m: f"{m.group(1)} = '{m.group(2)}'",
        sql, flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"\b(caste_category)\s+LIKE\s+'%?(SC|ST|OBC|GEN)%?'",
        lambda m: f"{m.group(1)} = '{m.group(2).upper()}'",
        sql, flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"\b(marital_status)\s+LIKE\s+'%?(Married|Unmarried|Widow)%?'",
        lambda m: f"{m.group(1)} = '{m.group(2)}'",
        sql, flags=re.IGNORECASE,
    )

    # ── Step 4.5: education — 'illiterate' lowercase, others Title Case ──
    def edu_replacer(match):
        col = match.group(1)
        val = match.group(2).strip()
        if val.lower() == "illiterate":
            return f"LOWER({col}) = 'illiterate'"
        return f"{col} LIKE '%{val}%'"

    sql = re.sub(
        r"\b(education)\s*=\s*'([^']+)'",
        edu_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 5: is_rural ──
    def rural_replacer(match):
        col = match.group(1)
        val = match.group(2).strip().lower().strip("'\"")
        if val in ("true", "1", "rural", "yes"):
            return f"{col} = 1"
        if val in ("false", "0", "urban", "no"):
            return f"{col} = 0"
        return match.group(0)

    sql = re.sub(
        r"\b(is_rural)\s*(?:=|LIKE)\s*['\"]?(\w+)['\"]?",
        rural_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 6: District casing normalisation ──
    def district_exact_replacer(match: re.Match[str]) -> str:
        col = match.group(1)
        val = match.group(2).strip().lower()
        canonical = _DISTRICT_CANONICAL.get(val)
        if canonical:
            return f"{col} = '{canonical}'"
        return match.group(0)

    sql = re.sub(
        r"\b(district_name_eng)\s*=\s*'([^']+)'",
        district_exact_replacer, sql, flags=re.IGNORECASE,
    )
    sql = re.sub(
        r'\b(district_name_eng)\s*=\s*"([^"]+)"',
        district_exact_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 8: District LIKE → = ──
    def district_like_replacer(match: re.Match[str]) -> str:
        col = match.group(1)
        val = match.group(2).strip()
        canonical = _DISTRICT_CANONICAL.get(val.lower())
        if canonical:
            return f"{col} = '{canonical}'"
        return match.group(0)

    sql = re.sub(
        r"\b(district_name_eng)\s+LIKE\s+'%?([^'%]+?)%?'",
        district_like_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 8.5: Redirect district IN clauses containing non-district values ──
    def district_in_replacer(match: re.Match[str]) -> str:
        col = match.group(1)
        in_content = match.group(2)
        vals = [v[0] or v[1] for v in re.findall(r"'([^']+)'|\"([^\"]+)\"", in_content)]
        if not vals:
            return match.group(0)
            
        has_non_district = any(v.lower() not in _DISTRICTS_LOWER for v in vals)
        
        if has_non_district:
            conditions = []
            for val in vals:
                val_l = val.strip().lower()
                if val_l in _DISTRICTS_LOWER:
                    canonical = _DISTRICT_CANONICAL[val_l]
                    conditions.append(f"{col} = '{canonical}'")
                else:
                    conditions.append(f"(block_name_eng LIKE '%{val}%' OR vill_name_eng LIKE '%{val}%')")
            return "(" + " OR ".join(conditions) + ")"
        else:
            new_vals = [f"'{_DISTRICT_CANONICAL[v.lower()]}'" for v in vals]
            return f"{col} IN ({', '.join(new_vals)})"

    sql = re.sub(
        r"\b(district_name_eng)\s+IN\s+\(([^)]+)\)",
        district_in_replacer, sql, flags=re.IGNORECASE,
    )

    # ── Step 9: Redirect district → block/village for non-district locations ──
    redirect_pattern = (
        r"\b"
        r"district_name_eng\s*(?:=\s*'([^']+)'|LIKE\s*'%([^'%]+)%')"
    )

    def district_redirect_full(match: re.Match[str]) -> str:
        val = (match.group(1) or match.group(2) or "").strip()
        if not val or val.lower() in _DISTRICTS_LOWER:
            return match.group(0)
        return f"(block_name_eng LIKE '%{val}%' OR vill_name_eng LIKE '%{val}%')"

    sql = re.sub(redirect_pattern, district_redirect_full, sql, flags=re.IGNORECASE)

    # ── Steps 10-15: aggregates ──
    sql = re.sub(r"COUNT\s*\(\s*member_id\s*\)", "COUNT(*)", sql, flags=re.IGNORECASE)

    # Rename misleading family_count -> member_count
    sql = re.sub(
        r"COUNT\(\*\)\s+AS\s+family_count",
        "COUNT(*) AS member_count",
        sql, flags=re.IGNORECASE,
    )

    # ── Clean up bank terms ──
    sql = re.sub(r"\bbank_account_number\b", "account_no", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bbank_account_no\b", "account_no", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bbank_account\b", "account_no", sql, flags=re.IGNORECASE)

    # Remove trailing comments
    if ";" in sql:
        parts = sql.split(";", 1)
        sql = parts[0].strip() + ";"

    return sql


def _is_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _fix_no_bank_sql(sql: str, question: str) -> str:
    """
    Ensure "no bank account" queries filter by account_no IS NULL.
    """
    if not any(w in question.lower() for w in _NO_BANK_WORDS):
        return sql
        
    # If the SQL already checks for account_no IS NULL, we're good
    if "account_no is null" in sql.lower() or "bank is null" in sql.lower():
        sql = re.sub(r"\bbank\s+is\s+null\b", "account_no IS NULL", sql, flags=re.IGNORECASE)
        return sql

    # Replace legacy checks
    sql = re.sub(r"\bbank_details\.bank_id\s+is\s+null\b", "account_no IS NULL", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bmember\.member_id\s+is\s+null\b", "account_no IS NULL", sql, flags=re.IGNORECASE)

    if "account_no is null" not in sql.lower():
        if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
            sql = re.sub(
                r"\bWHERE\b",
                "WHERE account_no IS NULL AND",
                sql, flags=re.IGNORECASE, count=1,
            )
        else:
            injected = re.sub(
                r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b",
                r"WHERE account_no IS NULL \1",
                sql, flags=re.IGNORECASE, count=1,
            )
            if injected == sql:
                sql = sql.rstrip(";") + " WHERE account_no IS NULL;"
            else:
                sql = injected

    return sql


# Ordered education hierarchy (lowest → highest)
_EDUCATION_LEVELS = [
    "illiterate", "Literate", "5 Pass", "8 Pass",
    "10 Pass", "12 Pass", "Graduate", "Post Graduate",
]
_EDUCATION_INDEX = {lvl.lower(): i for i, lvl in enumerate(_EDUCATION_LEVELS)}

# Patterns that map NL fragments to their hierarchy level
_EDU_LEVEL_KEYWORDS = [
    (r"\bpost\s*graduate(?:s|d)?\b|\bpg\b",          "Post Graduate"),
    (r"\bgraduate(?:s|d|ion)?\b",                      "Graduate"),
    (r"\b12(?:th)?\s*(?:pass|class|std|standard)?\b|\bintermediate\b|\bhsc\b", "12 Pass"),
    (r"\b10(?:th)?\s*(?:pass|class|std|standard)?\b|\bmatric\b|\bssc\b",      "10 Pass"),
    (r"\b8(?:th)?\s*(?:pass|class|std|standard)?\b",                           "8 Pass"),
    (r"\b5(?:th)?\s*(?:pass|class|std|standard)?\b",                           "5 Pass"),
    (r"\bliterate\b|\bbasic\s+education\b",   "Literate"),
    (r"\billiterate\b|\buneducated\b",         "illiterate"),
]


def _detect_edu_level(text: str) -> str | None:
    """Return the canonical education level string matched in text, or None."""
    t = text.lower()
    for pattern, level in _EDU_LEVEL_KEYWORDS:
        if re.search(pattern, t, re.IGNORECASE):
            return level
    return None


def _fix_education_sql(sql: str, question: str) -> str:
    """
    Deterministically rewrite any incorrect/overly-broad education filter in LLM-generated SQL
    into a precise IN (...) clause drawn from the ordered education hierarchy.

    Handles all patterns the LLM may produce:
      - education LIKE '%pass%' (broad, catches ALL pass levels including wrong ones)
      - LOWER(education) LIKE '%pass%'
      - education != 'illiterate' (wrong negative exclusion)
      - combinations of the above with AND
    """
    sql_lower = sql.lower()

    # Only act if the education column is referenced
    if "education" not in sql_lower:
        return sql

    # ── Detect intended level from the question ──────────────────────────────
    edu_level = _detect_edu_level(question)
    q_lower = question.lower()
    is_above = bool(re.search(r"\band\s+above\b|\bor\s+above\b|\band\s+higher\b|\bor\s+more\b", q_lower))
    is_below = bool(re.search(r"\band\s+below\b|\bor\s+below\b|\band\s+lower\b|\bor\s+less\b", q_lower))

    # ── Only fix if question signals a hierarchy range ────────────────────────
    if not (edu_level and (is_above or is_below)):
        # Still fix the most common LLM mistake: bare education LIKE '%pass%' without context
        # Replace broad LIKE '%pass%' (catches wrong category) with all 4 pass levels + graduate
        broad_like = re.search(
            r"(?:LOWER\s*\(\s*education\s*\)|education)\s+LIKE\s+'%pass%'",
            sql, re.IGNORECASE
        )
        if broad_like:
            replacement = "education IN ('5 Pass', '8 Pass', '10 Pass', '12 Pass', 'Graduate', 'Post Graduate')"
            sql = sql[:broad_like.start()] + replacement + sql[broad_like.end():]
            # Remove any trailing AND education != 'illiterate' guard
            sql = re.sub(
                r"\s+AND\s+education\s*!=\s*'illiterate'",
                "", sql, flags=re.IGNORECASE
            )
        return sql

    # Determine qualifying levels from hierarchy
    level_idx = _EDUCATION_INDEX.get(edu_level.lower())
    if level_idx is None:
        return sql

    if is_above:
        qualifying = [_EDUCATION_LEVELS[i] for i in range(level_idx, len(_EDUCATION_LEVELS))]
    else:  # is_below
        qualifying = [_EDUCATION_LEVELS[i] for i in range(0, level_idx + 1)]

    in_clause = "education IN (" + ", ".join(f"'{lvl}'" for lvl in qualifying) + ")"

    # Replace ALL education filter patterns in the WHERE clause
    # Pattern 1: LOWER(education) LIKE '...'
    sql = re.sub(
        r"LOWER\s*\(\s*education\s*\)\s+LIKE\s+'[^']*'",
        in_clause, sql, flags=re.IGNORECASE
    )
    # Pattern 2: education LIKE '...'
    sql = re.sub(
        r"\beducation\s+LIKE\s+'[^']*'",
        in_clause, sql, flags=re.IGNORECASE
    )
    # Pattern 3: education = 'SomeLevel' (exact, wrong level)
    sql = re.sub(
        r"\beducation\s*=\s*'[^']*'",
        in_clause, sql, flags=re.IGNORECASE
    )
    # Remove any lingering AND education != 'illiterate' guard (redundant after IN clause)
    sql = re.sub(
        r"\s+AND\s+education\s*!=\s*'illiterate'",
        "", sql, flags=re.IGNORECASE
    )
    # Collapse any duplicate in_clause if patterns matched twice
    sql = re.sub(
        r"(education IN \([^)]+\))\s+AND\s+\1",
        r"\1", sql, flags=re.IGNORECASE
    )

    return sql


def generate_sql_pipeline(
    question: str,
    ask_model_pull: bool = True,
    include_optimization: bool = True,
    run_query_for_profile: bool = False,
    bypass_cache: bool = False,
) -> PipelineOutput:
    manager = OllamaModelManager()
    manager.ensure_model(settings.sql_model, ask_permission=ask_model_pull)
    manager.ensure_model(settings.embedding_model, ask_permission=ask_model_pull)

    normalized = normalize_query(question)
    
    # Fuzzy target extraction
    from normalization.fuzzy_match import is_fuzzy_intent, extract_fuzzy_target
    is_fuzzy = is_fuzzy_intent(question)
    fuzzy_target = None
    if is_fuzzy:
        target = extract_fuzzy_target(question)
        if target and len(target) >= 3:
            fuzzy_target = target
        else:
            is_fuzzy = False

    # ── Tier 0: Fast Path Check ──
    from llm.fast_path import FastPathEngine
    fast_engine = FastPathEngine()
    fast_sql = fast_engine.generate_sql_fast(question)
    if fast_sql:
        optimization = None
        if include_optimization and run_query_for_profile:
            optimization = QueryOptimizer().profile(fast_sql, run_query=run_query_for_profile)
            
        return PipelineOutput(
            question=question,
            normalized_question=normalized.normalized,
            query_corrections=normalized.corrections,
            sql=fast_sql,
            retrieved_tables=["citizen"],
            retrieved_columns=["(fast_path)"],
            confidence=1.0,
            validation_errors=[],
            optimization=optimization,
            is_fuzzy=is_fuzzy,
            fuzzy_target=fuzzy_target,
            source="fast_path"
        )

    # ── Tier 1: Semantic Cache lookup ──
    cached_sql = None
    cache_store = FaissCacheStore()
    cache = SemanticCache(cache_store)
    if not bypass_cache:
        cached_sql = cache.lookup(question)
        
        # ── Tier 1.5: AST Parameter Swapping fallback ──
        if not cached_sql and cache_store.index is not None and len(cache_store.registry) > 0:
            try:
                # Embed the query to search in FAISS
                query_vector = cache_store.embedder.embed(question).reshape(1, -1)
                scores, indexes = cache_store.index.search(query_vector, 1)
                if len(scores) > 0 and len(indexes) > 0:
                    score = float(scores[0][0])
                    idx = int(indexes[0][0])
                    if idx >= 0 and idx < len(cache_store.registry) and score >= 0.85:
                        matched_entry = cache_store.registry[idx]
                        swapped_sql = fast_engine.swap_ast_parameters(
                            matched_entry["sql"], matched_entry["question"], question
                        )
                        if swapped_sql:
                            # Verify if the swapped query is valid against Whitelisted columns
                            validator = SQLValidator()
                            validation = validator.validate(
                                swapped_sql,
                                allowed_tables=["citizen"]
                            )
                            if validation.valid:
                                optimization = None
                                if include_optimization and run_query_for_profile:
                                    optimization = QueryOptimizer().profile(swapped_sql, run_query=run_query_for_profile)
                                return PipelineOutput(
                                    question=question,
                                    normalized_question=normalized.normalized,
                                    query_corrections=normalized.corrections,
                                    sql=swapped_sql,
                                    retrieved_tables=["citizen"],
                                    retrieved_columns=["(cache_swapped)"],
                                    confidence=score,
                                    validation_errors=[],
                                    optimization=optimization,
                                    is_fuzzy=is_fuzzy,
                                    fuzzy_target=fuzzy_target,
                                    source="cache_swapped"
                                )
            except Exception:
                pass

    if cached_sql:
        # We got an exact semantic cache hit!
        optimization = None
        if include_optimization and run_query_for_profile:
            optimization = QueryOptimizer().profile(cached_sql, run_query=run_query_for_profile)
            
        return PipelineOutput(
            question=question,
            normalized_question=normalized.normalized,
            query_corrections=normalized.corrections,
            sql=cached_sql,
            retrieved_tables=["citizen"],
            retrieved_columns=[],
            confidence=1.0,
            validation_errors=[],
            optimization=optimization,
            is_fuzzy=is_fuzzy,
            fuzzy_target=fuzzy_target,
            source="cache"
        )

    # 2. Cache Miss: Retrieve Schema Context and Dynamic Few-Shots
    store = FaissSchemaStore()
    store.build()
    retrieval = SchemaRetriever(store).retrieve(normalized.normalized)
    
    few_shot_store = FaissFewShotStore()
    few_shot_store.build()
    few_shot_retriever = FewShotRetriever(few_shot_store)
    retrieved_few_shots = few_shot_retriever.retrieve(normalized.normalized, top_k=3)
    
    prompt_builder = PromptBuilder()
    generator = OllamaSqlGenerator()
    validator = SQLValidator()

    previous_error: str | None = None
    sql = ""
    validation_errors: list[str] = []
    final_sql_is_valid = False
    
    for _ in range(settings.max_retries):
        prompt = prompt_builder.build(
            retrieval,
            previous_error=previous_error,
            few_shots=retrieved_few_shots
        )
        sql = generator.generate(prompt)
        sql = _post_process_sql(sql, fuzzy_target=fuzzy_target)
        sql = _fix_no_bank_sql(sql, question)
        sql = _fix_education_sql(sql, question)
        validation = validator.validate(
            sql,
            allowed_tables=retrieval.tables,
            allowed_columns=retrieval.columns,
        )
        validation_errors = validation.errors
        if validation.valid:
            final_sql_is_valid = True
            break
        previous_error = "; ".join(validation.errors)

    optimization = None
    if include_optimization and sql and final_sql_is_valid:
        validation = validator.validate(sql, allowed_tables=retrieval.tables, allowed_columns=retrieval.columns)
        if validation.valid:
            optimization = QueryOptimizer().profile(sql, run_query=run_query_for_profile)

    # Store successful SQL to semantic cache
    if final_sql_is_valid and sql:
        cache.store(question, sql)

    return PipelineOutput(
        question=question,
        normalized_question=normalized.normalized,
        query_corrections=normalized.corrections,
        sql=sql if final_sql_is_valid else "",
        retrieved_tables=retrieval.tables,
        retrieved_columns=retrieval.columns,
        confidence=retrieval.confidence,
        validation_errors=validation_errors,
        optimization=optimization,
        is_fuzzy=is_fuzzy,
        fuzzy_target=fuzzy_target,
        source="llm"
    )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Local Jan Aadhaar-style Natural Language to SQL generator.")
    parser.add_argument("question", nargs="*", help="Natural language question to convert into SQL.")
    parser.add_argument("--build-index", action="store_true", help="Force rebuild the FAISS schema index.")
    parser.add_argument("--seed-demo-db", action="store_true", help="Create and seed the SQLite demo database.")
    parser.add_argument("--import-excel", help="Replace the local demo data with an Excel dummy dataset.")
    parser.add_argument("--show-results", action="store_true", help="Display up to 20 matching database rows after generating SQL.")
    parser.add_argument("--no-explain", action="store_true", help="Skip EXPLAIN query plan generation.")
    parser.add_argument("--run-profile-query", action="store_true", help="Execute the generated SQL while profiling.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the semantic query cache.")
    parser.add_argument("--bypass-cache", action="store_true", help="Bypass the semantic query cache.")
    args = parser.parse_args()

    if args.clear_cache:
        import os
        cache_path = settings.data_dir / "cache.faiss"
        metadata_path = settings.data_dir / "cache_metadata.json"
        if cache_path.exists():
            os.remove(cache_path)
        if metadata_path.exists():
            os.remove(metadata_path)
        print("Semantic cache cleared.")

    if args.seed_demo_db:
        import_excel_dataset("dummy_dataset/Dummy_Data_Set.xlsx")
        print(f"Demo database ready at {settings.sqlite_path} loaded from Dummy_Data_Set.xlsx")
    if args.import_excel:
        report = import_excel_dataset(args.import_excel)
        print(f"Imported {report.rows_loaded} rows from {report.source_name}.")

    manager = OllamaModelManager()
    manager.ensure_model(settings.sql_model)
    manager.ensure_model(settings.embedding_model)

    if args.build_index:
        FaissSchemaStore().build(force=True)
        print(f"FAISS schema index rebuilt at {settings.faiss_index_path}")
        FaissFewShotStore().build(force=True)
        print(f"FAISS few-shot index rebuilt at {settings.few_shot_faiss_path}")

    question = " ".join(args.question).strip()
    if not question:
        question = input("Ask a Jan Aadhaar database question: ").strip()
    output = generate_sql_pipeline(
        question,
        ask_model_pull=False,
        include_optimization=not args.no_explain,
        run_query_for_profile=args.run_profile_query,
        bypass_cache=args.bypass_cache,
    )
    print("\nGenerated SQL")
    print(output.sql)
    print("\nRetrieved tables")
    print(", ".join(output.retrieved_tables))
    print("\nRetrieved columns")
    print(", ".join(output.retrieved_columns))
    print(f"\nConfidence: {output.confidence}")
    print(f"Source: {output.source.upper()}")
    if output.query_corrections:
        print("\nQuery spelling corrections")
        print(", ".join(f"{source} -> {target}" for source, target in output.query_corrections.items()))
        print(f"Normalized question: {output.normalized_question}")
    if output.validation_errors:
        print("\nValidation errors")
        print("; ".join(output.validation_errors))
    if args.show_results and output.sql:
        preview = execute_select_preview(
            output.sql,
            max_rows=20,
            fuzzy_target=output.fuzzy_target,
            is_fuzzy=output.is_fuzzy,
        )
        if output.is_fuzzy:
            print(f"\nSimilarity matches for '{output.fuzzy_target}' (Jaro-Winkler >= 0.80)")
        else:
            print("\nMatching entries")
        print(preview.rows.to_string(index=False) if not preview.rows.empty else "No matching entries.")
        if preview.truncated:
            if output.is_fuzzy:
                print("Showing the first 20 similarity matches only.")
            else:
                print("Showing the first 20 rows only.")
    if output.optimization:
        print("\nExecution plan")
        print("\n".join(output.optimization.execution_plan))
        print(f"\nPlanning/explain time: {output.optimization.execution_time_ms} ms")
        if output.optimization.index_recommendations:
            print("\nIndex recommendations")
            print("\n".join(output.optimization.index_recommendations))


if __name__ == "__main__":
    if _is_streamlit():
        from ui.streamlit_app import render
        render()
    else:
        run_cli()
