from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnMeta:
    table: str
    column: str
    description: str
    data_type: str
    aliases: list[str] = field(default_factory=list)
    indexed: bool = False
    sample_values: list[str] = field(default_factory=list)
    semantic_name: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.table}.{self.column}"

    @property
    def business_name(self) -> str:
        return self.semantic_name or self.column


@dataclass(frozen=True)
class TableMeta:
    table: str
    description: str
    aliases: list[str]


# ── Districts ─────────────────────────────────────────────────────────────────
RAJASTHAN_DISTRICTS_41 = [
    "Ajmer", "Alwar", "Balotra", "Banswara", "Baran", "Barmer", "Beawar",
    "Bharatpur", "Bhilwara", "Bikaner", "Bundi", "Chittorgarh", "Churu",
    "Dausa", "Deeg", "Didwana-Kuchaman", "Dholpur", "Dungarpur", "Hanumangarh",
    "Jaipur", "Jaisalmer", "Jalore", "Jhalawar", "Jhunjhunu", "Jodhpur",
    "Karauli", "Khairthal-Tijara", "Kota", "Kotputli-Behror", "Nagaur", "Pali",
    "Phalodi", "Pratapgarh", "Rajsamand", "Salumbar", "Sawai Madhopur", "Sikar",
    "Sirohi", "Sri Ganganagar", "Tonk", "Udaipur",
]

DISTRICT_ALIASES = [
    "district", "zilla", "city", "location", "region",
    *[d.lower() for d in RAJASTHAN_DISTRICTS_41],
    "kotputli", "ganganagar",
]

# ── Cities (towns/tehsil HQs stored in the `city_name_eng` column) ────────────
RAJASTHAN_CITIES = [
    "Ajmer", "Alwar", "Bagru", "Bansur", "Banswara", "Beawar", "Bhadra",
    "Bhilwara", "Bidasar", "Bikaner", "Borawar", "Bundi", "Chirawa", "Chomu",
    "Churu", "Dholpur", "Dudu", "Dundlod", "Gangapur City", "Hamirgarh",
    "Hindaun", "Jaipur", "Jayal", "Jodhpur", "Kapasan", "Khairthal", "Kota",
    "Kotputli", "Laxmangarh", "Malakhera", "Malpura", "Mandawa", "Mawli",
    "Nagaur", "Neem Ka Thana", "Nimbahera", "Nokha", "Pali",
    "Partapur -Garhi", "Pilibanga", "Pipar City", "Rajgarh", "Ramgarh",
    "Ratangarh", "Sadulshahar", "Sambhar", "Sanchore", "Sawai Madhopur",
    "Sikar", "Siwana", "Sri Ganganagar", "Sri Karanpur", "Sujangarh",
    "Sultanpur", "Sumerpur", "Tibbi", "Tinwari", "Tonk", "Uchchain", "Udaipur",
]

# ── Blocks (sub-district blocks stored in the `block_name_eng` column) ─────────
RAJASTHAN_BLOCKS = [
    "Aadel", "Aau", "Abu Road", "Ajeetgarh", "Alsisar", "Anupgarh", "Arain",
    "BAYTOO", "BHIM", "Badnor", "Balesar", "Bali", "Bamanwas", "Bandikui",
    "Banera", "Bansur", "Bari Sadri", "Barmer", "Barmer Rural", "Bhadesar",
    "Bhadra", "Bhawani Mandi", "Bherunda", "Bhinay", "Bhinmal", "Bhopalgarh",
    "Bhupalsagar", "Bhusawar", "Bikaner", "Buhana", "Chaksu", "Chhabra",
    "Chhipabarod", "Chhotisadri", "Chirawa", "Chohtan", "Churu",
    "Danta Ramgarh", "Deeg", "Degana", "Deoli", "Dhanaoo", "Dhawa", "Dhod",
    "Dhorimanna", "Didwana", "Dungla", "Fagliya", "Fatehpur", "Fathegarh",
    "Ganganagar", "Ghantiyali", "Gira", "Girwa", "Gogunda", "Govindgarh",
    "Gudamalani", "HADAN", "Hanumangarh", "Hindaun", "Hindoli", "Hurda",
    "Itawa", "Jahazpur", "Jaitaran", "Jalore", "Jalsoo", "Jamwa Ramgarh",
    "Jawaja", "Jayal", "Jhadol", "Jhotwara", "Jobner", "Kalyanpur", "Kaman",
    "Kapasan", "Kareda", "Kathumar", "Kekri", "Keru", "Khairabad", "Khajuwala",
    "Khandar", "Khandela", "Khanpur", "Khetri", "Khinwsar", "Kishangarh",
    "Kishangarh Renwal", "Kotputli", "Kotri", "Kuchaman City", "Kumher",
    "Ladnu", "Ladpura", "Laxmangarh", "Lohawat", "Luni", "Lunkaransar",
    "Madhorajpura", "Mahwa", "Makrana", "Malpura", "Mandal", "Mandalgarh",
    "Mandawa", "Mandor", "Manohar Thana", "Marwar Junction", "Masuda",
    "Maulasar", "Mauzamabad", "Merta", "Mundwa", "Nachana", "Nadbai", "Nadoti",
    "Nagar", "Nagaur", "Nainwa", "Nawalgarh", "Nawan", "Nechwa", "Neemkathana",
    "Niwai", "Nohar", "Nokha", "Osian", "Padampur", "Palsana", "Panchoo",
    "Paota", "Parbatsar", "Patodi", "Peeplu", "Peesangan", "Phagi", "Pilani",
    "Pindwara", "Pipar City", "Pirawa", "Poogal", "Railmagra", "Rajakhera",
    "Rajgarh", "Ramgarh Pachwara", "Ramsar", "Rani Station", "Ratangarh",
    "Rawatsar", "Reodar", "Rupwas", "Sabla", "Sadulshahar", "Sambhar",
    "Sanchore", "Sanganer", "Sangria", "Sardarshahar", "Sarwar", "Serwa",
    "Sewar", "Shahpura", "Shekhala", "Sheo", "Shergarh", "Shri Mahaveer Ji",
    "Shridungargarh", "Sindhari", "Singhana", "Siwana", "Sojat", "Srimadhopur",
    "Srinagar", "Sujangarh", "Surajgarh", "Suratgarh", "Suwana", "Taranagar",
    "Tibbi", "Tinwari", "Todaraisingh", "Tunga", "Uchain", "Udaipurwati",
    "Uniara", "Weir",
]


# ── Single table ──────────────────────────────────────────────────────────────
TABLES: list[TableMeta] = [
    TableMeta(
        "citizen",
        "Flat Jan Aadhaar citizen record — geography, demographics, and bank details in one table.",
        ["citizen", "person", "member", "resident", "people", "beneficiary",
         "family", "household", "jan aadhaar"],
    ),
]

COLUMNS: list[ColumnMeta] = [
    # Geography
    ColumnMeta("citizen", "district_name_eng", "District where the citizen lives",
               "string", DISTRICT_ALIASES, True, RAJASTHAN_DISTRICTS_41),
    ColumnMeta("citizen", "is_rural",          "1 = rural village, 0 = urban city",
               "integer", ["rural", "urban", "village", "city dwellers"], True, ["1", "0"]),
    ColumnMeta("citizen", "block_name_eng",     "Administrative block / tehsil",
               "string", ["block", "tehsil", "subdistrict"], True, RAJASTHAN_BLOCKS),
    ColumnMeta("citizen", "city_name_eng",      "City name (urban citizens only; NULL for rural)",
               "string", ["city", "town", "urban area"], True, RAJASTHAN_CITIES),
    ColumnMeta("citizen", "ward_name_eng",      "Municipal ward (urban only)",
               "string", ["ward"], True),
    ColumnMeta("citizen", "gp_name_eng",        "Gram panchayat (rural only)",
               "string", ["gram panchayat", "panchayat", "gp"], True),
    ColumnMeta("citizen", "vill_name_eng",      "Village name (rural only)",
               "string", ["village", "gaon", "gram"], True),

    # Identity
    ColumnMeta("citizen", "enrollment_id",         "Jan Aadhaar family enrollment number",
               "string", ["jan aadhaar number", "enrollment id", "family card"], True),
    ColumnMeta("citizen", "member_id",             "Member number within the family",
               "integer", ["member id"], True),
    ColumnMeta("citizen", "jan_aadhaar_member_id", "Unique Jan Aadhaar member ID (concatenated ENROLLMENT_ID and MEMBER_ID)",
               "string", ["jan aadhaar member id"], True),

    # Demographics
    ColumnMeta("citizen", "mem_type",          "HOF = Head of Family; MEM = regular member",
               "string", ["member type", "hof", "head of family", "mem"], True, ["HOF", "MEM"]),
    ColumnMeta("citizen", "relation_with_hof", "Relation to family head",
               "string", ["relation", "son", "daughter", "husband", "self"], True,
               ["Self", "Son", "Daughter", "Husband", "Daughter-in-law", "Grand Daughter", "Grand Son"]),
    ColumnMeta("citizen", "name_en",           "Citizen full name",
               "string", ["name", "person name", "citizen name", "beneficiary name"], True),
    ColumnMeta("citizen", "father_name_en",    "Father name",
               "string", ["father", "father's name"]),
    ColumnMeta("citizen", "mother_name_en",    "Mother name",
               "string", ["mother", "mother's name"]),
    ColumnMeta("citizen", "marital_status",    "Marital status: 'Married', 'Unmarried', 'Widow'",
               "string", ["marital", "married", "unmarried", "widow", "single"], True,
               ["Married", "Unmarried", "Widow"]),
    ColumnMeta("citizen", "spouce_name_en",    "Spouse name (NULL for unmarried)",
               "string", ["spouse", "wife name", "husband name"]),
    ColumnMeta("citizen", "dob",               "Date of birth",
               "date", ["dob", "birth date"]),
    ColumnMeta("citizen", "age",               "Age in years (integer, range 8–103)",
               "integer",
               ["age", "old age", "adult", "minor", "senior", "elderly", "child",
                "above", "below", "older than", "younger than", "between"], True),
    ColumnMeta("citizen", "gender",            "Gender: 'Male' or 'Female'",
               "string",
               ["gender", "sex", "male", "female", "man", "woman", "boy", "girl",
                "boys", "girls", "men", "women", "ladies", "gents"], True, ["Male", "Female"]),
    ColumnMeta("citizen", "caste_category",    "Caste category: SC, ST, OBC, GEN",
               "string", ["caste category", "category", "sc", "st", "obc", "general", "gen"],
               True, ["SC", "ST", "OBC", "GEN"]),
    ColumnMeta("citizen", "caste",             "Detailed caste name; use LIKE for searches",
               "string",
               ["caste", "community", "jat", "mina", "rajput", "brahman", "bairwa",
                "jain", "gurjar", "valmiki", "fakir", "dhobi", "darzi"], True,
               ["Jat", "Mina", "Rajput", "Brahman", "Bairwa", "Gurjar", "Jain",
                "Bazigar", "Valmiki", "Fakir", "Dhobi", "Darzi"]),
    ColumnMeta("citizen", "income",            "Annual income in rupees",
               "integer", ["income", "earning", "salary", "annual income", "rupees"], True),
    ColumnMeta("citizen", "occupation",        "Occupation; use LIKE for partial matches",
               "string",
               ["occupation", "job", "work", "farmer", "labourer", "student",
                "homemaker", "unemployed", "business"], True,
               ["Home Maker", "Farmer", "Others", "Labourer", "Unemployed",
                "State personnel", "Self-Employed", "Businessman", "Student"]),
    ColumnMeta("citizen", "minority",          "Minority community: 'Muslim', 'Jain', or NULL",
               "string", ["minority", "muslim", "muslims", "jain", "jains"], True, ["Muslim", "Jain"]),
    ColumnMeta("citizen", "education",         "Education level; 'illiterate' is lowercase",
               "string",
               ["education", "qualification", "illiterate", "literate", "graduate",
                "pass", "school", "matric", "intermediate"], True,
               ["illiterate", "Literate", "5 Pass", "8 Pass", "10 Pass",
                "12 Pass", "Graduate", "Post Graduate", "Other"]),
    ColumnMeta("citizen", "mobile_no",         "Mobile phone number",
               "string", ["mobile", "phone", "contact"]),

    # Bank
    ColumnMeta("citizen", "bank",         "Bank name (stored mostly UPPER CASE); use LIKE",
               "string",
               ["bank", "bank name", "sbi", "pnb", "bob", "hdfc", "icici",
                "gramin", "cooperative bank"], True,
               ["STATE BANK OF INDIA", "BANK OF BARODA", "PUNJAB NATIONAL BANK",
                "UCO BANK", "UNION BANK OF INDIA", "CANARA BANK",
                "Rajasthan Gramin Bank", "HDFC BANK"]),
    ColumnMeta("citizen", "ifsc_code",    "IFSC code",
               "string", ["ifsc", "ifsc code"], True),
    ColumnMeta("citizen", "account_no",   "Bank account number",
               "string", ["account", "bank account", "account number"], True),
]

# No relationships — single table
RELATIONSHIPS: list[dict] = []


def all_table_names() -> set[str]:
    return {t.table for t in TABLES}


def columns_by_table() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for col in COLUMNS:
        result.setdefault(col.table, set()).add(col.column)
    return result
