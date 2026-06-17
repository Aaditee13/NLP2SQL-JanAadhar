from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config.settings import settings
from database.schema_metadata import COLUMNS, RAJASTHAN_DISTRICTS_41, RELATIONSHIPS, TABLES
from embeddings.faiss_store import FaissSchemaStore


# Pre-compiled regex for _terms() — defined before constants that use it
_ALPHANUM_RE = re.compile(r"[a-zA-Z0-9]+")


CASTE_DETAIL_TERMS = {"caste", "community", "jati"}
CASTE_CATEGORY_TERMS = {"category", "sc", "st", "obc", "general", "gen", "scheduled",
                        "backward", "forward", "unreserved", "open category"}
CASTE_TERMS = CASTE_DETAIL_TERMS | CASTE_CATEGORY_TERMS

WELFARE_TERMS = {"beneficiary", "beneficiaries"}
BANK_TERMS = {
    "bank", "account", "ifsc", "dbt", "payment",
    "sbi", "pnb", "bob", "hdfc", "icici", "uco", "canara", "baroda",
    "gramin", "cooperative", "union bank", "central bank", "indian bank",
    "unbanked", "no bank", "without bank", "no account", "without account",
}
IDENTITY_TERMS = {"aadhaar", "jan", "voter", "pan", "identity", "id", "mobile", "phone", "email", "photo"}
DISABILITY_TERMS = {"disabled", "disability", "divyang"}
RELIGION_TERMS = {"religion", "faith"}
MINORITY_TERMS = {"minority", "muslim", "muslims", "jain", "jains"}
EDUCATION_TERMS = {"education", "qualification", "illiterate", "literate", "graduate", "school", "pass", "matric", "intermediate"}
RURAL_TERMS = {"rural", "urban", "city dweller", "village people", "village families", "city families"}
GENDER_TERMS = {
    "gender", "sex", "male", "female", "man", "woman", "boy", "girl",
    "boys", "girls", "men", "women", "lady", "ladies", "gent", "gents", "widow", "widows"
}
AGE_TERMS = {
    "age", "year", "years", "old", "young", "adult", "minor", "senior",
    "elderly", "child", "children", "born", "dob", "birth", "date",
    "above", "below", "older", "younger", "between", "under", "over"
}

KNOWN_CASTES = {
    "jat", "arai", "fakir", "mina", "ramgadiya", "rajput", "rajpoot", "moyla", 
    "gauswami", "deshwali", "chhipa", "chhipi", "kandera", "jain", "brahman", 
    "brahmin", "gurjar", "gujar", "pathan", "valmiki", "balmiki", "berwa", 
    "bairwa", "mehtar", "bazigar", "dangi", "sidh", "dhobi", "darzi", "daroga", 
    "sindhi", "vaishya", "pinjara", "जाट", "राजपूत", "मीना", "महाजन", "बाजीगर",
    "अग्रवाल", "ब्राह्मण", "ब्राहम्ण", "sc", "st", "obc", "caste", "community"
}
RAJASTHAN_DISTRICT_TERMS = {
    term
    for district in RAJASTHAN_DISTRICTS_41
    for term in district.lower().replace("-", " ").split()
}
GEOGRAPHY_TERMS = RAJASTHAN_DISTRICT_TERMS | {
    "district",
    "zilla",
    "block",
    "village",
    "ward",
    "panchayat",
}
DISTRICT_TERMS = RAJASTHAN_DISTRICT_TERMS | {"district", "zilla", "city", "location"}
LOCATION_PREPOSITIONS = {"in", "from", "at"}
LOCATION_STOPWORDS = {
    "age",
    "years",
    "male",
    "female",
    "boys",
    "girls",
    "beneficiaries",
    "beneficiary",
    "pension",
    "scheme",
    "nfsa",
    "ekyc",
    "active",
    "pending",
}

# Pre-compiled patterns that reference the constants defined above
_LOC_PREPOSITION_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(LOCATION_PREPOSITIONS)) + r")\s+([a-zA-Z][a-zA-Z-]*)\b"
)
_LIST_TOKENS = ["show", "list", "display", "beneficiary", "all", "who", "find", "fetch", "get"]
_LIST_TOKEN_PATTERN = re.compile(r"\b(?:" + "|".join(_LIST_TOKENS) + r")\b")


@dataclass
class RetrievalResult:
    question: str
    tables: list[str]
    columns: list[str]
    relationships: list[dict[str, str]]
    documents: list[dict[str, Any]]
    confidence: float


class SchemaRetriever:
    def __init__(self, store: FaissSchemaStore | None = None):
        self.store = store or FaissSchemaStore()

    def retrieve(self, question: str, top_k: int = settings.retrieval_top_k) -> RetrievalResult:
        docs = self.store.search(question, top_k=top_k)
        tables: set[str] = {"citizen"}
        columns: set[str] = set()
        question_lower = question.lower()
        question_terms = _terms(question_lower)
        lexical_columns: set[str] = set()

        for column in COLUMNS:
            lexical_terms = [column.column.replace("_", " "), *column.aliases, *column.sample_values]
            if any(term and _matches(term, question_lower, question_terms) for term in lexical_terms):
                lexical_columns.add(column.qualified_name)

        # Add lexical columns first
        columns.update(lexical_columns)

        # Add top semantic columns that are not already present, up to a limit of 8
        semantic_added = 0
        for doc in docs:
            if doc.get("kind") != "column" or not doc.get("qualified_name"):
                continue
            qualified_name = doc["qualified_name"]
            if qualified_name in columns:
                continue
            if _column_allowed_by_domain(qualified_name, question_lower, question_terms):
                columns.add(qualified_name)
                semantic_added += 1
            if semantic_added >= 8:
                break

        # Pre-compute possible_location and non_district_loc once here;
        # pass them into _prune_columns to avoid recomputing.
        possible_location = _mentions_possible_location(question_lower)
        non_district_loc = False
        if possible_location:
            known_districts = {d.lower() for d in RAJASTHAN_DISTRICTS_41}
            if not (question_terms & known_districts):
                non_district_loc = True

        if (GEOGRAPHY_TERMS & question_terms) or possible_location:
            columns.add("citizen.district_name_eng")
            if {"block", "tehsil", "kotputli"} & question_terms or non_district_loc:
                columns.add("citizen.block_name_eng")
            if "village" in question_lower or non_district_loc:
                columns.add("citizen.vill_name_eng")

        # is_rural
        if RURAL_TERMS & question_terms or any(t in question_lower for t in ("rural", "urban", "is_rural")):
            columns.add("citizen.is_rural")

        # Force retrieve caste if query contains any known caste terms
        if (KNOWN_CASTES & question_terms) or any(caste in question_lower for caste in KNOWN_CASTES):
            columns.add("citizen.caste")

        # Display fields automatically added on list queries
        if _LIST_TOKEN_PATTERN.search(question_lower):
            columns.add("citizen.name_en")
            columns.add("citizen.enrollment_id")

        columns = _prune_columns(
            columns, question_lower, question_terms,
            possible_location=possible_location,
            non_district_loc=non_district_loc,
        )

        confidence = sum(doc["score"] for doc in docs[: min(5, len(docs))]) / max(1, min(5, len(docs)))

        return RetrievalResult(
            question=question,
            tables=["citizen"],
            columns=sorted(columns),
            relationships=[],
            documents=docs,
            confidence=round(confidence, 4),
        )


def _terms(text: str) -> set[str]:
    return set(_ALPHANUM_RE.findall(text.lower()))


def _matches(term: str, question_lower: str, question_terms: set[str]) -> bool:
    normalized = term.lower().strip()
    if not normalized:
        return False
    if " " in normalized:
        return normalized in question_lower
    return normalized in question_terms


def _column_allowed_by_domain(qualified_name: str, question_lower: str, question_terms: set[str]) -> bool:
    _, column = qualified_name.split(".")
    
    # bank mapping
    if column in ("bank", "account_no", "ifsc_code") and not (BANK_TERMS & question_terms):
        return False
    # caste_category mapping
    if column == "caste_category" and not (CASTE_CATEGORY_TERMS & question_terms):
        return False
    if column == "minority" and not (MINORITY_TERMS & question_terms):
        return False
    if column == "education" and not (EDUCATION_TERMS & question_terms):
        return False
    if column == "gender" and not (GENDER_TERMS & question_terms):
        return False
    if column in ("age", "dob") and not (AGE_TERMS & question_terms):
        return False
    return True


def _prune_columns(
    columns: set[str],
    question_lower: str,
    question_terms: set[str],
    possible_location: bool | None = None,
    non_district_loc: bool | None = None,
) -> set[str]:
    pruned: set[str] = set()
    # Reuse caller-computed values when available to avoid redundant work
    if possible_location is None:
        possible_location = _mentions_possible_location(question_lower)
    if non_district_loc is None:
        non_district_loc = False
        if possible_location:
            known_districts = {d.lower() for d in RAJASTHAN_DISTRICTS_41}
            if not (question_terms & known_districts):
                non_district_loc = True

    for qualified_name in columns:
        _, column = qualified_name.split(".")
        if not _column_allowed_by_domain(qualified_name, question_lower, question_terms):
            continue
        if column in ("bank", "account_no", "ifsc_code") and not (BANK_TERMS & question_terms or any(term in question_lower for term in BANK_TERMS)):
            continue
        if column in ("jan_aadhaar_member_id", "mobile_no") and not (IDENTITY_TERMS & question_terms or any(term in question_lower for term in IDENTITY_TERMS)):
            continue
        if column in ("enrollment_id") and not ((IDENTITY_TERMS | WELFARE_TERMS) & question_terms or any(term in question_lower for term in (IDENTITY_TERMS | WELFARE_TERMS))):
            continue
        if column == "block_name_eng" and not ({"block", "tehsil", "kotputli"} & question_terms or "block" in question_lower or "tehsil" in question_lower or "kotputli" in question_lower or non_district_loc):
            continue
        if column == "gp_name_eng" and not ({"gram", "panchayat", "gp"} & question_terms or "gram" in question_lower or "panchayat" in question_lower or "gp" in question_lower):
            continue
        if column == "vill_name_eng" and not ("village" in question_terms or "village" in question_lower or non_district_loc):
            continue
        if column == "ward_name_eng" and not ("ward" in question_terms or "ward" in question_lower):
            continue
        if column == "city_name_eng" and not ({"city", "town", "urban"} & question_terms or "city" in question_lower or "town" in question_lower or "urban" in question_lower):
            continue
        if column == "district_name_eng" and not ((DISTRICT_TERMS | GEOGRAPHY_TERMS) & question_terms or any(term in question_lower for term in (DISTRICT_TERMS | GEOGRAPHY_TERMS)) or possible_location):
            continue
        if column == "is_rural" and not (RURAL_TERMS & question_terms or any(t in question_lower for t in ("rural", "urban", "is_rural"))):
            continue
        pruned.add(qualified_name)
    return pruned


def _mentions_possible_location(question_lower: str) -> bool:
    for match in _LOC_PREPOSITION_PATTERN.finditer(question_lower):
        candidate = match.group(1).lower()
        if candidate not in LOCATION_STOPWORDS and not candidate.isdigit():
            return True
    return False
