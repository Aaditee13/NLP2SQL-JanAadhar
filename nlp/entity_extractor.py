import re
from dataclasses import dataclass, field
from typing import Any, Tuple, List, Dict

# ── Dictionaries ─────────────────────────────────────────────────────────────

_GENDER_MAP = {
    "male": "Male", "males": "Male", "boy": "Male", "boys": "Male",
    "man": "Male", "men": "Male", "gents": "Male",
    "ladka": "Male", "ladke": "Male", "purusha": "Male",
    "female": "Female", "females": "Female", "girl": "Female", "girls": "Female",
    "woman": "Female", "women": "Female", "ladies": "Female", "lady": "Female",
    "ladki": "Female", "mahila": "Female", "aurat": "Female",
}

_MARITAL_MAP = {
    re.compile(r"\bmarried\b"): "Married",
    re.compile(r"\bunmarried\b|\bsingle\b|\bbachelor\b"): "Unmarried",
    re.compile(r"\bwidow(?:s|ed|er|ers)?\b"): "Widow",
}

_CASTE_CAT_MAP = {
    "sc": "SC", "scheduled caste": "SC", "dalit": "SC",
    "st": "ST", "scheduled tribe": "ST", "tribal": "ST", "adivasi": "ST",
    "obc": "OBC", "other backward": "OBC", "backward class": "OBC",
    "gen": "GEN", "general": "GEN", "open category": "GEN",
    "unreserved": "GEN", "forward": "GEN",
}

_RURAL_MAP = {
    "rural": 1, "village": 1, "gram": 1, "gaon": 1,
    "urban": 0, "city": 0, "town": 0,
}

_OCCUPATION_MAP = {
    "farmer": "Farmer", "farmers": "Farmer",
    "labourer": "Labourer", "laborer": "Labourer", "labour": "Labourer",
    "student": "Student", "students": "Student",
    "unemployed": "Unemployed",
    "homemaker": "Home Maker", "home maker": "Home Maker",
    "businessman": "Businessman", "business": "Businessman", "businessmen": "Businessman",
}

_EDUCATION_MAP = {
    # 'illiterate' is stored lowercase in DB; all others are Title Case
    "illiterate": "illiterate", "uneducated": "illiterate",
    "literate": "Literate", "educated": "Literate",
    "graduate": "Graduate",
    "post graduate": "Post Graduate",
    "5th pass": "5 Pass", "8th pass": "8 Pass",
    "10th pass": "10 Pass", "matric": "10 Pass",
    "12th pass": "12 Pass", "intermediate": "12 Pass",
}

# Districts in Title Case — must match exactly what is stored in the database
_DISTRICTS = [
    "Ajmer", "Alwar", "Balotra", "Banswara", "Baran", "Barmer", "Beawar",
    "Bharatpur", "Bhilwara", "Bikaner", "Bundi", "Chittorgarh", "Churu",
    "Dausa", "Deeg", "Didwana-Kuchaman", "Dholpur", "Dungarpur", "Hanumangarh",
    "Jaipur", "Jaisalmer", "Jalore", "Jhalawar", "Jhunjhunu", "Jodhpur",
    "Karauli", "Khairthal-Tijara", "Kota", "Kotputli-Behror", "Nagaur", "Pali",
    "Phalodi", "Pratapgarh", "Rajsamand", "Salumbar", "Sawai Madhopur", "Sikar",
    "Sirohi", "Sri Ganganagar", "Tonk", "Udaipur",
]
# Maps lowercase user input → canonical DB-matching Title Case value
_DISTRICT_MAP = {d.lower(): d for d in _DISTRICTS}
# Short-name aliases for districts with long compound names
_DISTRICT_MAP.update({
    "ganganagar": "Sri Ganganagar",
    "kotputli": "Kotputli-Behror",
    "didwana": "Didwana-Kuchaman",
    "khairthal": "Khairthal-Tijara",
    "sawai": "Sawai Madhopur",
    "madhopur": "Sawai Madhopur",
})

_AGE_PATTERNS = [
    (r"\bsenior\s+citizen\b|\belderly\b|\bold\s+age\b", ">=", 60),
    (r"\bchild(?:ren)?\b|\bminor\b", "<", 18),
    (r"\badult\b", ">=", 18),
    (r"\bworking\s+age\b", "BETWEEN", (18, 59)),
    (r"\babove\s+(\d+)\b|\bolder\s+than\s+(\d+)\b|\bage\s*>\s*(\d+)\b", ">", None),
    (r"\bbelow\s+(\d+)\b|\byounger\s+than\s+(\d+)\b|\bage\s*<\s*(\d+)\b", "<", None),
    (r"\bage\s+between\s+(\d+)\s+and\s+(\d+)\b", "BETWEEN", None),
]

_INCOME_PATTERNS = [
    (r"\bwithout\s+(?:an\s+)?income\b|\bno\s+income\b|\bzero\s+income\b", "=", 0),
    (r"\bincome\s*(?:above|greater\s+than|more\s+than|over|>)\s*(\d[\d,]*(?:\s*(?:lakh[s]?|lac[s]?|crore[s]?))?)\b", ">", None),
    (r"\bincome\s*(?:below|less\s+than|under|<)\s*(\d[\d,]*(?:\s*(?:lakh[s]?|lac[s]?|crore[s]?))?)\b", "<", None),
    (r"\bincome\s*(?:between)\s*(\d[\d,]*(?:\s*(?:lakh[s]?|lac[s]?|crore[s]?))?)\s*(?:and|to)\s*(\d[\d,]*(?:\s*(?:lakh[s]?|lac[s]?|crore[s]?))?)\b", "BETWEEN", None),
]

_STOPWORDS = {
    "show", "data", "of", "all", "in", "the", "whose", "have", "with", "without", "an", "a",
    "and", "or", "for", "from", "where", "are", "is", "who", "whom", "that", "which",
    "give", "me", "list", "details", "find", "get", "display", "citizens", "people",
    "members", "person", "persons", "count", "number", "those", "their", "has", "been",
    "to", "on", "any", "some", "youth",
    # Bank-related words consumed partially by bank patterns
    "bank", "account", "accounts",
}

_COMPLEX_KEYWORDS = {
    "average", "avg", "sum", "total", "maximum", "minimum", "highest", "lowest",
    "how many", "ratio", "percentage", "rank", "top", "most",
    "least", "across", "compare", "except", "not in",
}
# Single compiled regex for all complex keywords — avoids re.escape + re.search
# being called in a tight loop for every keyword on every query.
_COMPLEX_KW_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _COMPLEX_KEYWORDS) + r")\b"
)

# Pre-compile age and income patterns (replace string with compiled object)
_AGE_PATTERNS_COMPILED = [
    (re.compile(p), op, val)
    for p, op, val in _AGE_PATTERNS
]
_INCOME_PATTERNS_COMPILED = [
    (re.compile(p), op, val)
    for p, op, val in _INCOME_PATTERNS
]

# ── Helper ───────────────────────────────────────────────────────────────────

def _parse_income(raw: str) -> int:
    raw = raw.replace(",", "").strip().lower()
    multiplier = 1
    if re.search(r"lakh|lac", raw):
        raw = re.sub(r"lakh[s]?|lac[s]?", "", raw).strip()
        multiplier = 100_000
    elif "crore" in raw:
        raw = re.sub(r"crore[s]?", "", raw).strip()
        multiplier = 10_000_000
    try:
        return int(float(raw) * multiplier)
    except (ValueError, TypeError):
        return 0

# ── Extractor ────────────────────────────────────────────────────────────────

@dataclass
class Condition:
    operator: str  # '=', '>', '<', '>=', '<=', 'BETWEEN', 'LIKE', 'IS NULL', 'IS NOT NULL', 'IN'
    value: Any

@dataclass
class ExtractionResult:
    entities: Dict[str, List[Condition]] = field(default_factory=dict)
    unhandled_words: List[str] = field(default_factory=list)
    has_complex_keywords: bool = False

class EntityExtractor:
    def extract(self, question: str) -> ExtractionResult:
        q = question.lower()
        res = ExtractionResult()

        # Check complex keywords — single compiled regex, O(n) scan instead of loop
        if _COMPLEX_KW_RE.search(q):
            res.has_complex_keywords = True
                
        rem = q
        entities: Dict[str, List[Condition]] = {}

        def add_cond(col: str, cond: Condition):
            if col not in entities:
                entities[col] = []
            entities[col].append(cond)

        # ── 1. Income ────────────────────────────────────────────────────────
        for compiled_pat, op, fixed_val in _INCOME_PATTERNS_COMPILED:
            m = compiled_pat.search(rem)
            if m:
                if fixed_val is not None:
                    add_cond("income", Condition(op, fixed_val))
                elif op == "BETWEEN":
                    v1 = _parse_income(m.group(1))
                    v2 = _parse_income(m.group(2))
                    add_cond("income", Condition("BETWEEN", (v1, v2)))
                else:
                    add_cond("income", Condition(op, _parse_income(m.group(1))))
                rem = compiled_pat.sub(" ", rem)
                break

        # ── 2. Age ───────────────────────────────────────────────────────────
        for compiled_pat, op, fixed_val in _AGE_PATTERNS_COMPILED:
            m = compiled_pat.search(rem)
            if m:
                if fixed_val is not None:
                    add_cond("age", Condition(op, fixed_val))
                else:
                    groups = [g for g in m.groups() if g is not None]
                    if op == "BETWEEN" and len(groups) >= 2:
                        add_cond("age", Condition("BETWEEN", (int(groups[0]), int(groups[1]))))
                    elif groups:
                        add_cond("age", Condition(op, int(groups[0])))
                rem = compiled_pat.sub(" ", rem)
                break

        # ── 3. Categorical Matches ───────────────────────────────────────────
        def match_cat(mapping: dict, col: str, is_like: bool = False):
            nonlocal rem
            matched = set()
            for kw, val in mapping.items():
                # kw is either a compiled pattern (for _MARITAL_MAP) or a plain string
                if isinstance(kw, re.Pattern):
                    compiled_kw = kw
                else:
                    pattern = rf"\b{kw}\b" if not kw.startswith(r"\b") else kw
                    compiled_kw = re.compile(pattern)
                if val not in matched and compiled_kw.search(rem):
                    matched.add(val)
                    rem = compiled_kw.sub(" ", rem)
            if matched:
                if len(matched) == 1:
                    op = "LIKE" if is_like else "="
                    add_cond(col, Condition(op, next(iter(matched))))
                else:
                    op = "LIKE_ANY" if is_like else "IN"
                    add_cond(col, Condition(op, list(matched)))

        match_cat(_GENDER_MAP, "gender")
        match_cat(_MARITAL_MAP, "marital_status")
        match_cat(_CASTE_CAT_MAP, "caste_category")
        match_cat(_RURAL_MAP, "is_rural")
        match_cat(_DISTRICT_MAP, "district_name_eng")
        match_cat(_EDUCATION_MAP, "education")
        match_cat(_OCCUPATION_MAP, "occupation", is_like=True)

        # ── 4. Minority ──────────────────────────────────────────────────────
        if re.search(r"\bmuslim\b", rem):
            add_cond("minority", Condition("=", "Muslim"))
            rem = re.sub(r"\bmuslim\b", " ", rem)
        elif re.search(r"\bjain\b", rem):
            add_cond("minority", Condition("=", "Jain"))
            rem = re.sub(r"\bjain\b", " ", rem)

        # ── 5. Bank Account ──────────────────────────────────────────────────
        p_no_bank = r"\bno\s+bank\b|\bunbanked\b|\bwithout\s+bank\b|\bno\s+account\b"
        p_has_bank = r"\bhas\s+bank\b|\bwith\s+bank\b|\bbanked\b|\bhas\s+account\b"
        if re.search(p_no_bank, rem):
            add_cond("account_no", Condition("IS NULL", None))
            rem = re.sub(p_no_bank, " ", rem)
        elif re.search(p_has_bank, rem):
            add_cond("account_no", Condition("IS NOT NULL", None))
            rem = re.sub(p_has_bank, " ", rem)

        # ── Unhandled Words ──────────────────────────────────────────────────
        leftovers = [w for w in re.findall(r"[a-z]{3,}", rem) if w not in _STOPWORDS]
        
        res.entities = entities
        res.unhandled_words = leftovers
        
        return res
