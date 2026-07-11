"""
ui/interpretation.py
Three-level result interpretation + direct answer + LLM insight.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN LABELS
# ─────────────────────────────────────────────────────────────────────────────
_COL_LABELS = {
    "name_en": "Name", "age": "Age", "gender": "Gender",
    "district_name_eng": "District", "block_name_eng": "Block",
    "city_name_eng": "City", "vill_name_eng": "Village",
    "gp_name_eng": "Gram Panchayat", "ward_name_eng": "Ward",
    "caste_category": "Caste Category", "caste": "Caste",
    "marital_status": "Marital Status", "education": "Education",
    "occupation": "Occupation", "income": "Annual Income (₹)",
    "is_rural": "Rural / Urban", "mem_type": "Member Type",
    "relation_with_hof": "Relation to HOF",
    "minority": "Minority", "bank": "Bank",
    "enrollment_id": "Enrollment ID", "member_id": "Member ID",
    "father_name_en": "Father's Name", "mother_name_en": "Mother's Name",
    "spouce_name_en": "Spouse Name", "dob": "Date of Birth",
    "mobile_no": "Mobile No.", "account_no": "Account No.",
    "ifsc_code": "IFSC Code",
}


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN INTENT → POLICY CONTEXT
# Each entry: (keywords_in_question, policy_text)
# ─────────────────────────────────────────────────────────────────────────────
_INTENT_MAP: list[tuple[list[str], str, str]] = [
    # (keywords, intent_id, policy_message)
    (["widow", "widowed", "widower"],
     "widow",
     "Widow citizens are eligible for the **Indira Gandhi National Widow Pension Scheme (IGNWPS)** "
     "— ₹500/month for widows below poverty line aged 40–79. "
     "Cross-check with Rajasthan Social Security Pension portal for enrolment status."),

    (["pension", "pensioner"],
     "pension",
     "Pension beneficiaries in this group receive welfare through state or central pension schemes. "
     "Verify payment and KYC status via the **Rajasthan Social Security Pension (SSP) portal**."),

    (["unbank", "no bank", "without bank", "no account", "without account"],
     "unbanked",
     "Citizens without bank accounts **cannot receive DBT (Direct Benefit Transfer) payments**. "
     "Enrolling them under **Pradhan Mantri Jan Dhan Yojana (PMJDY)** would immediately enable "
     "direct welfare transfers — zero-balance accounts with RuPay card and accident cover."),

    (["bpl", "below poverty", "poor"],
     "bpl",
     "Below Poverty Line citizens are priority beneficiaries for "
     "**NFSA ration cards**, **PM Awas Yojana** (housing), and "
     "**MGNREGS** employment guarantee (100 days/year)."),

    (["sc ", " sc", "scheduled caste", "dalit", "bairwa", "meghwal", "chamar"],
     "sc",
     "SC citizens are entitled to benefits under the **SC Sub-Plan** allocations, "
     "post-matric scholarships from Department of Social Justice & Empowerment, "
     "and reservation in government education and employment."),

    (["st ", " st", "scheduled tribe", "tribal", "adivasi", "meena", "bhil", "garasia"],
     "st",
     "ST citizens are covered under the **Tribal Sub-Plan**, "
     "**Van Dhan Vikas Kendra** (tribal enterprise), "
     "and **Eklavya Model Residential Schools** for children."),

    (["obc", "backward class", "backward caste"],
     "obc",
     "OBC citizens are eligible for **OBC welfare scholarships**, "
     "reservation benefits in education and employment (27% central OBC reservation), "
     "and targeted development under the **Rajasthan OBC Finance & Development Corporation**."),

    (["illiterate", "uneducated", "no education"],
     "illiterate",
     "Illiterate citizens in this group are priority targets for "
     "**Saakshar Bharat Mission** and the **Rajasthan State Literacy Mission** — "
     "free adult education programmes that improve welfare scheme awareness and self-reliance."),

    (["senior", "elderly", "aged", "old age", "60 year", "above 60", "above 65"],
     "elderly",
     "Senior citizens (60+) are eligible for **Old Age Pension** (₹1,000/month under IGNOAPS), "
     "**Senior Citizen Health Insurance** under Chiranjeevi Yojana, "
     "and priority service at all government offices under the Senior Citizens Act."),

    (["child", "minor", "below 18", "underage", "boy", "girl", "student"],
     "child",
     "Young citizens in this group may benefit from "
     "**PM Poshan** (mid-day meal scheme), "
     "**Palanhar Yojana** (Rajasthan — ₹2,500/month for orphan/destitute children), "
     "and **Beti Bachao Beti Padhao** for girls."),

    (["farmer", "kisan", "agriculture", "farming", "crop"],
     "farmer",
     "Farmer citizens may be eligible for **PM Kisan Samman Nidhi** (₹6,000/year in 3 instalments), "
     "**Rajasthan Kisan Kalyan Mission** benefits, and **PM Fasal Bima Yojana** crop insurance. "
     "Verify Jan Aadhaar linkage with land records for eligibility."),

    (["muslim", "minority", "christian", "sikh"],
     "minority",
     "Minority citizens are covered under **PM-VIKAS** (Vishwakarma Kaushal Samman), "
     "**Maulana Azad National Fellowship** (higher education), "
     "and **NMDFC** loan schemes for minority entrepreneurs."),

    (["income", "salary", "earning", "rupee", "below", "lakhs"],
     "income",
     "Low-income citizens in this result should be cross-referenced against "
     "**BPL/APL classification** and **Rajasthan Socio-Economic Census** data "
     "to verify welfare scheme eligibility."),

    (["rural", "village", "gram panchayat", "gaon"],
     "rural",
     "Rural citizens in this group are covered under **MGNREGS** employment guarantee, "
     "**PM Awas Yojana-Gramin** (rural housing), "
     "and **Jal Jeevan Mission** for household tap water connections."),

    (["urban", "city", "ward", "nagar"],
     "urban",
     "Urban citizens may access **PM Awas Yojana-Urban** (housing), "
     "**PM SVANidhi** (street vendor micro-credit), "
     "and **AMRUT 2.0** urban infrastructure scheme benefits."),

    (["bank", "account", "ifsc", "sbi", "dbt"],
     "banking",
     "Banking data in this result is critical for **DBT payment routing**. "
     "Verify IFSC codes and account numbers are current — "
     "outdated records cause DBT failures and welfare payment delays."),

    (["education", "graduate", "literate", "10 pass", "12 pass", "college"],
     "education",
     "Educated citizens in this group may qualify for "
     "**Rajasthan Yuva Vikas Prerak** placement assistance, "
     "**CM Digital Service Portal** skill certification, "
     "and state government employment schemes."),
]


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Interpretation:
    direct_answer: str          # specific answer to the exact question — shown prominently
    simple: str                 # plain English tab
    technical: str              # SQL/data-aware tab
    executive: str              # business/policy insights tab
    stats: dict = field(default_factory=dict)
    intents: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _label(col: str) -> str:
    return _COL_LABELS.get(col.lower(), col.replace("_", " ").title())


def _plural(n: int, singular: str, sfx: str = "s") -> str:
    return f"{n:,} {singular}{'' if n == 1 else sfx}"


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "0.0%"


def _top_values(series: pd.Series, n: int = 3) -> list[tuple[str, int]]:
    return [(str(k), int(v)) for k, v in series.value_counts().head(n).items()]


def _is_aggregation(sql: str) -> bool:
    return bool(re.search(r"\b(COUNT|SUM|AVG|MAX|MIN|GROUP\s+BY)\b", sql, re.IGNORECASE))


def _extract_filters(sql: str) -> list[str]:
    filters = []
    m = re.search(r"district_name_eng\s*=\s*'([^']+)'", sql, re.IGNORECASE)
    if m:
        filters.append(f"District: **{m.group(1)}**")
    m = re.search(r"gender\s*=\s*'([^']+)'", sql, re.IGNORECASE)
    if m:
        filters.append(f"Gender: **{m.group(1)}**")
    age_gt = re.search(r"age\s*>\s*(\d+)", sql, re.IGNORECASE)
    age_lt = re.search(r"age\s*<\s*(\d+)", sql, re.IGNORECASE)
    if age_gt and age_lt:
        filters.append(f"Age: **{age_gt.group(1)}–{age_lt.group(1)} years**")
    elif age_gt:
        filters.append(f"Age: **above {age_gt.group(1)} years**")
    elif age_lt:
        filters.append(f"Age: **below {age_lt.group(1)} years**")
    m = re.search(r"caste_category\s*=\s*'([^']+)'", sql, re.IGNORECASE)
    if m:
        filters.append(f"Caste Category: **{m.group(1)}**")
    m = re.search(r"marital_status\s*=\s*'([^']+)'", sql, re.IGNORECASE)
    if m:
        filters.append(f"Marital Status: **{m.group(1)}**")
    m = re.search(r"education\s*[=L][^']*'([^']+)'", sql, re.IGNORECASE)
    if m:
        filters.append(f"Education: **{m.group(1)}**")
    m = re.search(r"occupation\s+LIKE\s+'%([^%']+)%'", sql, re.IGNORECASE)
    if m:
        filters.append(f"Occupation: **{m.group(1)}**")
    m = re.search(r"is_rural\s*=\s*(\d)", sql, re.IGNORECASE)
    if m:
        filters.append(f"Area Type: **{'Rural' if m.group(1) == '1' else 'Urban'}**")
    m = re.search(r"minority\s*=\s*'([^']+)'", sql, re.IGNORECASE)
    if m:
        filters.append(f"Minority: **{m.group(1)}**")
    inc_gt = re.search(r"income\s*>\s*(\d+)", sql, re.IGNORECASE)
    inc_lt = re.search(r"income\s*<\s*(\d+)", sql, re.IGNORECASE)
    if inc_gt:
        filters.append(f"Income: **above ₹{int(inc_gt.group(1)):,}**")
    if inc_lt:
        filters.append(f"Income: **below ₹{int(inc_lt.group(1)):,}**")
    blk = re.search(r"block_name_eng\s+LIKE\s+'%([^%']+)%'", sql, re.IGNORECASE)
    if blk:
        filters.append(f"Block: **{blk.group(1)}**")
    return filters


def _detect_intents(question: str, sql: str) -> list[tuple[str, str]]:
    """Returns list of (intent_id, policy_message) for matched intents."""
    q = question.lower()
    matched = []
    seen_ids = set()
    for keywords, intent_id, policy in _INTENT_MAP:
        if intent_id not in seen_ids and any(kw in q for kw in keywords):
            matched.append((intent_id, policy))
            seen_ids.add(intent_id)
    return matched[:3]  # cap at 3 most relevant


def _compute_stats(df: pd.DataFrame, total: int) -> dict:
    s: dict = {"total": total}
    if "age" in df.columns:
        age_n = pd.to_numeric(df["age"], errors="coerce").dropna()
        if not age_n.empty:
            s.update(age_mean=round(age_n.mean(), 1),
                     age_min=int(age_n.min()),
                     age_max=int(age_n.max()))
    if "gender" in df.columns:
        s["gender_dist"] = df["gender"].value_counts().to_dict()
    if "district_name_eng" in df.columns and df["district_name_eng"].nunique() > 1:
        s["top_districts"] = _top_values(df["district_name_eng"])
    if "caste_category" in df.columns:
        s["caste_dist"] = df["caste_category"].value_counts().to_dict()
    if "marital_status" in df.columns:
        s["marital_dist"] = df["marital_status"].value_counts().to_dict()
    if "education" in df.columns:
        s["edu_dist"] = _top_values(df["education"], n=4)
    if "occupation" in df.columns:
        s["occ_dist"] = _top_values(df["occupation"], n=4)
    if "income" in df.columns:
        inc = pd.to_numeric(df["income"], errors="coerce").dropna()
        if not inc.empty:
            s.update(income_mean=int(inc.mean()),
                     income_median=int(inc.median()),
                     zero_income_pct=round((inc == 0).mean() * 100, 1))
    if "is_rural" in df.columns:
        r = (pd.to_numeric(df["is_rural"], errors="coerce") == 1).sum()
        s["rural_pct"]  = round(r / total * 100, 1) if total else 0
        s["urban_pct"]  = round(100 - s["rural_pct"], 1)
    if "bank" in df.columns:
        s["top_banks"] = _top_values(df["bank"], n=3)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# DIRECT ANSWER  ← the new core piece
# ─────────────────────────────────────────────────────────────────────────────
def _build_direct_answer(
    question: str,
    sql: str,
    df: pd.DataFrame,
    total: int,
    truncated: bool,
    filters: list[str],
    stats: dict,
    is_agg: bool,
    intents: list[tuple[str, str]],
) -> str:
    lines = []

    # ── 1.  Answer the specific question ─────────────────────────────────────
    if is_agg:
        # Single-number answer (COUNT / SUM / AVG)
        if len(df) == 1 and len(df.columns) <= 2:
            val = df.iloc[0, -1]
            try:
                val_int = int(val)
                lines.append(f"**The answer is {val_int:,}.**")
                filter_str = " and ".join(
                    f.replace("**", "").replace(":", " =") for f in filters
                ) if filters else "your search criteria"
                lines.append(
                    f"There are **{val_int:,} citizens** in the database who match {filter_str}."
                )
            except (ValueError, TypeError):
                lines.append(f"**Result: {val}**")
        else:
            # GROUP BY — summarise top row
            top = df.iloc[0].to_dict()
            top_str = " | ".join(f"**{_label(k)}:** {v}" for k, v in list(top.items())[:3])
            lines.append(f"Top result — {top_str}")
            lines.append(
                f"The query returned a breakdown across **{total:,} groups**. "
                "See the table above for the full distribution."
            )
    else:
        # Row-level SELECT — state exactly what was found
        if truncated:
            lines.append(
                f"Your query matched **at least {total:,} citizens** in the database "
                f"— currently showing the first {total:,}. "
                f"Download the CSV or increase the row limit to see all records."
            )
        else:
            lines.append(
                f"Your query found **exactly {total:,} "
                f"{'citizen' if total == 1 else 'citizens'}** in the database."
            )

    # ── 2.  Restate the specific criteria in plain English ────────────────────
    if filters:
        filter_plain = ", ".join(
            f.replace("**", "") for f in filters
        )
        lines.append(f"\n**Criteria matched:** {filter_plain}")

    # ── 3.  Key one-line stat ─────────────────────────────────────────────────
    key_parts = []
    if "age_mean" in stats:
        key_parts.append(f"avg age {stats['age_mean']} yrs")
    if "rural_pct" in stats:
        key_parts.append(f"{stats['rural_pct']}% rural")
    if "income_mean" in stats and not is_agg:
        key_parts.append(f"avg income ₹{stats['income_mean']:,}")
    if key_parts:
        lines.append("**At a glance:** " + " · ".join(key_parts))

    # ── 4.  Policy / welfare context from detected intents ────────────────────
    if intents:
        lines.append("\n---")
        lines.append("**📌 Relevant Welfare & Policy Context:**")
        for _, policy_text in intents:
            lines.append(f"\n> {policy_text}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLM INSIGHT  — calls Ollama for a specific narrative
# ─────────────────────────────────────────────────────────────────────────────
def generate_llm_insight(
    question: str,
    sql: str,
    stats: dict,
    total: int,
    truncated: bool,
    intents: list[tuple[str, str]],
) -> str:
    """
    Calls the locally running Ollama model to generate a 3-sentence
    specific interpretation of the query results.
    Returns a plain-text string or an error message.
    """
    import ollama  # noqa: PLC0415 — optional import, only needed when button pressed

    # Compact stats summary for the prompt (keeps tokens low)
    stat_parts: list[str] = [f"Records found: {total:,}"]
    if "age_mean" in stats:
        stat_parts.append(
            f"Age: mean {stats['age_mean']} yrs "
            f"(range {stats.get('age_min','?')}–{stats.get('age_max','?')})"
        )
    if "gender_dist" in stats:
        stat_parts.append(
            "Gender: " + ", ".join(
                f"{g}: {n:,}" for g, n in list(stats["gender_dist"].items())[:2]
            )
        )
    if "rural_pct" in stats:
        stat_parts.append(f"Rural: {stats['rural_pct']}%  Urban: {stats['urban_pct']}%")
    if "income_mean" in stats:
        stat_parts.append(
            f"Avg income: ₹{stats['income_mean']:,}  "
            f"({stats.get('zero_income_pct', 0)}% have zero income)"
        )
    if "caste_dist" in stats:
        top_cat = max(stats["caste_dist"], key=stats["caste_dist"].get)
        stat_parts.append(
            f"Dominant caste category: {top_cat} "
            f"({stats['caste_dist'][top_cat]:,})"
        )
    if "marital_dist" in stats:
        top_ms = max(stats["marital_dist"], key=stats["marital_dist"].get)
        stat_parts.append(f"Marital: {top_ms} most common")
    if "top_districts" in stats:
        top_d = stats["top_districts"][0]
        stat_parts.append(f"Top district: {top_d[0]} ({top_d[1]:,})")

    intent_labels = ", ".join(i[0] for i in intents) if intents else "general citizen data"

    prompt = (
        f"You are a welfare data analyst for the Government of Rajasthan, India.\n\n"
        f"A government official asked the Jan Aadhaar citizen database:\n"
        f"\"{question}\"\n\n"
        f"Database results summary:\n"
        + "\n".join(f"- {s}" for s in stat_parts)
        + (f"\n- Note: only first {total:,} records shown; more exist in database."
           if truncated else "")
        + f"\n\nDetected query domain: {intent_labels}\n\n"
        f"Write exactly 3 sentences:\n"
        f"1. A direct, specific answer to the question with exact numbers.\n"
        f"2. The single most important insight from these results.\n"
        f"3. One specific Rajasthan or central government scheme or action "
        f"relevant to this group.\n\n"
        f"Write in plain English. No bullet points. No headings. Just 3 clear sentences."
    )

    try:
        response = ollama.generate(
            model="qwen2.5-coder:3b",
            prompt=prompt,
            options={
                "temperature": 0.35,
                "num_predict": 220,
                "num_ctx": 1024,
            },
        )
        return response["response"].strip()
    except Exception as exc:
        return f"⚠️ AI insight unavailable ({exc}). Ensure Ollama is running."


# ─────────────────────────────────────────────────────────────────────────────
# TAB BUILDERS  (unchanged in logic, minor wording improvements)
# ─────────────────────────────────────────────────────────────────────────────
def _build_simple(
    question: str, sql: str, df: pd.DataFrame,
    total: int, truncated: bool, filters: list[str],
    stats: dict, is_agg: bool,
) -> str:
    lines = ["### 🔍 What your question found"]

    if is_agg:
        lines.append("Your question asked for a **summary or count** from the database.")
        if len(df) == 1 and len(df.columns) <= 2:
            val = df.iloc[0, -1]
            try:
                lines.append(f"The answer is: **{int(val):,}** records match your criteria.")
            except (ValueError, TypeError):
                lines.append(f"The result is: **{val}**")
        else:
            lines.append(f"The database returned **{_plural(total, 'group')}** of results.")
    else:
        if truncated:
            lines.append(
                f"Your question found **at least {_plural(total, 'person', 's')}** "
                f"matching your criteria — **more records exist** beyond what is displayed here. "
                f"Download the CSV or increase the row limit in the sidebar to see more."
            )
        else:
            lines.append(
                f"Your question found exactly **{_plural(total, 'person', 's')}** "
                f"matching your criteria."
            )

    if filters:
        lines.append("\n**You searched for people who are:**")
        for f in filters:
            lines.append(f"- {f}")

    lines.append("\n**What the results tell us (in simple words):**")
    if "age_mean" in stats:
        lines.append(f"- The average age is **{stats['age_mean']} years**.")
    if "gender_dist" in stats:
        for g, n in stats["gender_dist"].items():
            lines.append(f"- **{_pct(n, total)}** ({n:,}) are **{g}**.")
    if "rural_pct" in stats:
        lines.append(
            f"- **{stats['rural_pct']}%** live in **rural areas** and "
            f"**{stats['urban_pct']}%** live in **urban areas**."
        )
    if "marital_dist" in stats:
        for status, n in list(stats["marital_dist"].items())[:2]:
            lines.append(f"- **{_pct(n, total)}** are **{status}**.")
    if "income_mean" in stats:
        if stats.get("zero_income_pct", 0) > 40:
            lines.append(
                f"- More than **{stats['zero_income_pct']}%** have **zero recorded income** "
                f"— common for home makers and unpaid workers."
            )
        else:
            lines.append(f"- The average annual income is **₹{stats['income_mean']:,}**.")
    if "top_districts" in stats:
        tops = ", ".join(f"**{d}** ({n:,})" for d, n in stats["top_districts"])
        lines.append(f"- Most results are from: {tops}.")
    if "occ_dist" in stats:
        lines.append(
            f"- The most common occupation is **{stats['occ_dist'][0][0]}** "
            f"({stats['occ_dist'][0][1]:,} people)."
        )

    lines.append(
        "\n> 💡 **In short:** " +
        (f"There are **{total:,} people** in the database who match your search."
         if not is_agg else "The database has summarised the data you asked for above.")
    )
    return "\n".join(lines)


def _build_technical(
    question: str, sql: str, df: pd.DataFrame,
    total: int, filters: list[str], stats: dict, is_agg: bool,
) -> str:
    lines = ["### 🔬 Technical Breakdown"]
    lines.append(f"\n**Query type:** {'Aggregation / GROUP BY query' if is_agg else 'Row-level SELECT query'}  ")
    lines.append(f"**Rows returned (displayed):** {total:,}  ")
    lines.append(f"**Columns in result:** {', '.join(_label(c) for c in df.columns)}  ")

    lines.append("\n**Active SQL filter conditions:**")
    if filters:
        for f in filters:
            lines.append(f"- {f}")
    else:
        lines.append("- No explicit WHERE filters (full table scan or aggregation only)")

    if "age_mean" in stats:
        lines.append(
            f"\n**Age statistics:** Mean = {stats['age_mean']} yrs | "
            f"Min = {stats['age_min']} | Max = {stats['age_max']}"
        )
    if "gender_dist" in stats:
        lines.append("\n**Gender distribution:**")
        for g, n in stats["gender_dist"].items():
            lines.append(f"  - {g}: {n:,} ({_pct(n, total)})")
    if "caste_dist" in stats:
        lines.append("\n**Caste Category distribution:**")
        for c, n in stats["caste_dist"].items():
            lines.append(f"  - {c}: {n:,} ({_pct(n, total)})")
    if "edu_dist" in stats:
        lines.append("\n**Top education levels:**")
        for edu, n in stats["edu_dist"]:
            lines.append(f"  - {edu}: {n:,} ({_pct(n, total)})")
    if "income_mean" in stats:
        lines.append(
            f"\n**Income:** Mean = ₹{stats['income_mean']:,} | "
            f"Median = ₹{stats['income_median']:,} | "
            f"Zero-income = {stats['zero_income_pct']}%"
        )
    if "rural_pct" in stats:
        lines.append(f"\n**Rural / Urban:** {stats['rural_pct']}% rural | {stats['urban_pct']}% urban")
    if "top_banks" in stats:
        lines.append("\n**Bank distribution (top 3):**")
        for b, n in stats["top_banks"]:
            lines.append(f"  - {b}: {n:,} ({_pct(n, total)})")
    if "top_districts" in stats:
        lines.append("\n**Top districts in result:**")
        for d, n in stats["top_districts"]:
            lines.append(f"  - {d}: {n:,} ({_pct(n, total)})")
    if "occ_dist" in stats:
        lines.append("\n**Top occupations:**")
        for occ, n in stats["occ_dist"]:
            lines.append(f"  - {occ}: {n:,} ({_pct(n, total)})")

    lines.append("\n**Generated SQL:**")
    lines.append(f"```sql\n{sql}\n```")
    return "\n".join(lines)


def _build_executive(
    question: str, sql: str, df: pd.DataFrame,
    total: int, truncated: bool, filters: list[str],
    stats: dict, is_agg: bool,
) -> str:
    lines = ["### 📈 Executive Summary"]
    lines.append(f"\n**Query:** _{question}_")
    lines.append(
        f"\n**Total Records Identified:** {total:,}"
        + (" *(partial — more records exist)*" if truncated else "")
    )
    if filters:
        lines.append(
            f"\n**Applied Criteria:** "
            + " | ".join(f.replace("**", "") for f in filters)
        )

    lines.append("\n---\n**Key Findings:**")
    if "gender_dist" in stats:
        gd = stats["gender_dist"]
        f_n, m_n = gd.get("Female", 0), gd.get("Male", 0)
        if f_n and m_n:
            lines.append(
                f"- **Gender:** {_pct(f_n, total)} female ({f_n:,}) | "
                f"{_pct(m_n, total)} male ({m_n:,})"
            )
        elif f_n:
            lines.append(f"- **All records are female** ({f_n:,} citizens)")
        elif m_n:
            lines.append(f"- **All records are male** ({m_n:,} citizens)")

    if "age_mean" in stats:
        lines.append(
            f"- **Age Profile:** Mean {stats['age_mean']} years "
            f"(range {stats['age_min']}–{stats['age_max']} yrs)"
        )
    if "rural_pct" in stats:
        dom = ("Rural-dominant" if stats["rural_pct"] > 60
               else "Urban-dominant" if stats["urban_pct"] > 60
               else "Mixed rural-urban")
        lines.append(
            f"- **Geographic Profile:** {dom} — "
            f"{stats['rural_pct']}% rural, {stats['urban_pct']}% urban"
        )
    if "caste_dist" in stats:
        top_cat = max(stats["caste_dist"], key=stats["caste_dist"].get)
        top_n   = stats["caste_dist"][top_cat]
        lines.append(
            f"- **Dominant Category:** {top_cat} — "
            f"{_pct(top_n, total)} ({top_n:,} citizens)"
        )
    if "income_mean" in stats:
        zp = stats.get("zero_income_pct", 0)
        if zp > 50:
            lines.append(
                f"- **Income Status:** {zp}% have zero recorded income — "
                f"high welfare scheme dependency indicated."
            )
        elif stats["income_mean"] < 10000:
            lines.append(
                f"- **Income Status:** Low-income profile (avg ₹{stats['income_mean']:,}/yr) — "
                f"high potential welfare eligibility."
            )
        else:
            lines.append(
                f"- **Income:** Avg ₹{stats['income_mean']:,} | "
                f"Median ₹{stats['income_median']:,}"
            )
    if "top_districts" in stats:
        td = stats["top_districts"][0]
        lines.append(
            f"- **Geographic Concentration:** Highest in **{td[0]}** "
            f"({td[1]:,}, {_pct(td[1], total)})"
        )
    if "marital_dist" in stats:
        widow_n = stats["marital_dist"].get("Widow", 0)
        if total and widow_n / total > 0.15:
            lines.append(
                f"- **Welfare Alert:** {_pct(widow_n, total)} ({widow_n:,}) are widows — "
                f"likely eligible for IGNWPS widow pension."
            )

    lines.append("\n---\n**Strategic Implications:**")
    if total == 0:
        lines.append("- No records found. Broaden the search parameters.")
    elif total < 100:
        lines.append("- Small cohort — suitable for individual-level welfare outreach.")
    elif total < 10000:
        lines.append("- Moderate cohort — suitable for block-level programme targeting.")
    else:
        lines.append(
            f"- Large cohort of {total:,} citizens — district-level programme planning recommended."
        )
    if "zero_income_pct" in stats and stats["zero_income_pct"] > 40:
        lines.append(
            "- High zero-income proportion — strong candidate for DBT enrolment review."
        )
    if "rural_pct" in stats and stats["rural_pct"] > 70:
        lines.append(
            "- Predominantly rural — physical outreach and gram sabha engagement "
            "recommended over digital channels."
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def interpret(
    question: str,
    sql: str,
    df: pd.DataFrame,
    total_rows: int,
    truncated: bool,
    precomputed_stats: dict | None = None,
) -> Interpretation:
    is_agg  = _is_aggregation(sql)
    filters = _extract_filters(sql)
    # Use full-DB stats if available, otherwise fall back to sample stats
    stats   = precomputed_stats if precomputed_stats else _compute_stats(df, total_rows)
    # Ensure total is set
    if "total" not in stats:
        stats["total"] = total_rows
    intents = _detect_intents(question, sql)

    return Interpretation(
        direct_answer=_build_direct_answer(
            question, sql, df, total_rows, truncated,
            filters, stats, is_agg, intents,
        ),
        simple=_build_simple(
            question, sql, df, total_rows, truncated, filters, stats, is_agg,
        ),
        technical=_build_technical(
            question, sql, df, total_rows, filters, stats, is_agg,
        ),
        executive=_build_executive(
            question, sql, df, total_rows, truncated, filters, stats, is_agg,
        ),
        stats=stats,
        intents=intents,
    )
def generate_followup_answer(
    original_question: str,
    sql: str,
    stats: dict,
    total: int,
    truncated: bool,
    intents: list[tuple[str, str]],
    chat_history: list[dict],
    followup_question: str,
) -> str:
    """
    Answers a follow-up question about the current query results.
    Maintains conversation context via chat_history.
    """
    import ollama  # noqa: PLC0415

    # Build compact stats context
    stat_parts: list[str] = [f"Records in result: {total:,}"]
    if "age_mean" in stats:
        stat_parts.append(
            f"Age: mean {stats['age_mean']} yrs "
            f"(range {stats.get('age_min','?')}–{stats.get('age_max','?')})"
        )
    if "gender_dist" in stats:
        stat_parts.append(
            "Gender: " + ", ".join(
                f"{g}: {n:,}" for g, n in list(stats["gender_dist"].items())[:2]
            )
        )
    if "rural_pct" in stats:
        stat_parts.append(
            f"Rural: {stats['rural_pct']}%  Urban: {stats['urban_pct']}%"
        )
    if "income_mean" in stats:
        stat_parts.append(
            f"Avg income: ₹{stats['income_mean']:,}  "
            f"({stats.get('zero_income_pct', 0)}% zero income)"
        )
    if "caste_dist" in stats:
        top_cat = max(stats["caste_dist"], key=stats["caste_dist"].get)
        stat_parts.append(
            f"Dominant category: {top_cat} ({stats['caste_dist'][top_cat]:,})"
        )
    if "marital_dist" in stats:
        for ms, n in list(stats["marital_dist"].items())[:2]:
            stat_parts.append(f"{ms}: {n:,}")
    if "top_districts" in stats:
        td = stats["top_districts"][0]
        stat_parts.append(f"Top district: {td[0]} ({td[1]:,})")
    if "occ_dist" in stats:
        stat_parts.append(f"Top occupation: {stats['occ_dist'][0][0]}")
    if "edu_dist" in stats:
        stat_parts.append(f"Top education: {stats['edu_dist'][0][0]}")

    intent_labels = (
        ", ".join(i[0] for i in intents) if intents else "general citizen data"
    )

    # System context (always injected)
    system = (
        f"You are a welfare data analyst for the Government of Rajasthan, India.\n"
        f"You are helping a government official understand Jan Aadhaar database results.\n\n"
        f"Original query: \"{original_question}\"\n"
        f"SQL: {sql}\n"
        f"Result summary: {' | '.join(stat_parts)}\n"
        f"Domain: {intent_labels}\n"
        + (f"Note: only first {total:,} records shown — more exist.\n"
           if truncated else "")
    )

    # Build conversation history (last 8 messages for context window)
    history_block = ""
    if chat_history:
        turns = []
        for msg in chat_history[-8:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            turns.append(f"{role}: {msg['content']}")
        history_block = "\nPrevious conversation:\n" + "\n".join(turns) + "\n"

    prompt = (
        f"{system}"
        f"{history_block}\n"
        f"The user now asks: \"{followup_question}\"\n\n"
        f"Answer directly and specifically. Reference actual numbers from the result "
        f"summary where relevant. Keep the answer concise (2–4 sentences). "
        f"If the question cannot be answered from the available data, say so clearly "
        f"and suggest what query would help. Do not repeat the question."
    )

    try:
        response = ollama.generate(
            model="qwen2.5-coder:3b",
            prompt=prompt,
            options={
                "temperature": 0.4,
                "num_predict": 300,
                "num_ctx": 2048,
            },
        )
        return response["response"].strip()
    except Exception as exc:
        return f"⚠️ Could not generate answer ({exc}). Ensure Ollama is running."