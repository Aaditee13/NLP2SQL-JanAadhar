from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text

from database.connection import get_engine
from validation.sql_validator import SQLValidator


@dataclass(frozen=True)
class QueryResultPreview:
    rows: pd.DataFrame
    truncated: bool
    displayed_rows: int


def execute_select_preview(
    sql: str,
    max_rows: int = 200,
    database_url: str | None = None,
    fuzzy_target: str | None = None,
    is_fuzzy: bool = False,
    threshold: float = 0.80,
) -> QueryResultPreview:
    validation = SQLValidator().validate(sql)
    if not validation.valid:
        raise ValueError(f"Refusing to execute unsafe SQL: {'; '.join(validation.errors)}")
    safe_sql = sql.strip().rstrip(";")
    
    sql_max_rows = 1000 if is_fuzzy else max_rows
    limited_sql = f"SELECT * FROM ({safe_sql}) AS generated_result LIMIT {sql_max_rows + 1}"
    with get_engine(database_url).connect() as connection:
        result = connection.execute(text(limited_sql))
        columns = list(result.keys())
        records = result.fetchall()
        
    db_truncated = len(records) > sql_max_rows
    visible_records = records[:sql_max_rows]
    frame = pd.DataFrame(visible_records, columns=columns)
    
    if is_fuzzy and fuzzy_target:
        from normalization.fuzzy_match import fuzzy_rerank
        frame = fuzzy_rerank(frame, fuzzy_target, threshold=threshold, max_rows=max_rows)
        truncated = db_truncated
    else:
        truncated = len(records) > max_rows
        frame = frame.head(max_rows)
        
    return QueryResultPreview(rows=frame, truncated=truncated, displayed_rows=len(frame))

def compute_full_stats(sql: str) -> dict:
    """
    Runs aggregate queries against the FULL result set of the given SQL
    (not just the displayed preview rows) to get accurate stats.
    Returns a stats dict in the same format as _compute_stats() in interpretation.py
    """
    import re
    from sqlalchemy import text
    from database.connection import get_engine

    engine = get_engine()
    stats  = {}

    # Wrap the original SQL as a subquery
    base = sql.rstrip(";")
    
    # Skip aggregation queries — they already return summary data
    if re.search(r"\b(GROUP\s+BY|COUNT\s*\(|SUM\s*\(|AVG\s*\()\b", 
                 sql, re.IGNORECASE):
        return stats

    def _run(query: str):
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return result.fetchall()
        except Exception:
            return []

    # 1. Total count
    rows = _run(f"SELECT COUNT(*) FROM ({base})")
    if rows:
        stats["total"] = int(rows[0][0])

    # 2. Age stats
    rows = _run(f"""
        SELECT AVG(CAST(age AS REAL)), MIN(age), MAX(age)
        FROM ({base}) WHERE age IS NOT NULL
    """)
    if rows and rows[0][0] is not None:
        stats["age_mean"] = round(float(rows[0][0]), 1)
        stats["age_min"]  = int(rows[0][1])
        stats["age_max"]  = int(rows[0][2])

    # 3. Gender distribution
    rows = _run(f"""
        SELECT gender, COUNT(*) as n
        FROM ({base})
        WHERE gender IS NOT NULL
        GROUP BY gender ORDER BY n DESC
    """)
    if rows:
        stats["gender_dist"] = {str(r[0]): int(r[1]) for r in rows}

    # 4. Rural / Urban split
    rows = _run(f"""
        SELECT is_rural, COUNT(*) as n
        FROM ({base})
        WHERE is_rural IS NOT NULL
        GROUP BY is_rural
    """)
    if rows:
        total = stats.get("total", 1)
        rural = sum(int(r[1]) for r in rows if str(r[0]) in ("1", "1.0"))
        stats["rural_pct"] = round(rural / total * 100, 1) if total else 0
        stats["urban_pct"] = round(100 - stats["rural_pct"], 1)

    # 5. Caste category distribution
    rows = _run(f"""
        SELECT caste_category, COUNT(*) as n
        FROM ({base})
        WHERE caste_category IS NOT NULL
        GROUP BY caste_category ORDER BY n DESC
    """)
    if rows:
        stats["caste_dist"] = {str(r[0]): int(r[1]) for r in rows}

    # 6. Marital status distribution
    rows = _run(f"""
        SELECT marital_status, COUNT(*) as n
        FROM ({base})
        WHERE marital_status IS NOT NULL
        GROUP BY marital_status ORDER BY n DESC
    """)
    if rows:
        stats["marital_dist"] = {str(r[0]): int(r[1]) for r in rows}

    # 7. Income stats
    rows = _run(f"""
        SELECT AVG(CAST(income AS REAL)), 
               SUM(CASE WHEN income = 0 THEN 1 ELSE 0 END),
               COUNT(*)
        FROM ({base}) WHERE income IS NOT NULL
    """)
    if rows and rows[0][0] is not None:
        total_inc = int(rows[0][2]) or 1
        stats["income_mean"]      = int(float(rows[0][0]))
        stats["zero_income_pct"]  = round(int(rows[0][1]) / total_inc * 100, 1)

    # 8. Top districts (when multiple districts in result)
    rows = _run(f"""
        SELECT district_name_eng, COUNT(*) as n
        FROM ({base})
        WHERE district_name_eng IS NOT NULL
        GROUP BY district_name_eng ORDER BY n DESC LIMIT 3
    """)
    if rows and len(rows) > 1:
        stats["top_districts"] = [(str(r[0]), int(r[1])) for r in rows]

    # 9. Top occupations
    rows = _run(f"""
        SELECT occupation, COUNT(*) as n
        FROM ({base})
        WHERE occupation IS NOT NULL
        GROUP BY occupation ORDER BY n DESC LIMIT 4
    """)
    if rows:
        stats["occ_dist"] = [(str(r[0]), int(r[1])) for r in rows]

    # 10. Top education levels
    rows = _run(f"""
        SELECT education, COUNT(*) as n
        FROM ({base})
        WHERE education IS NOT NULL
        GROUP BY education ORDER BY n DESC LIMIT 4
    """)
    if rows:
        stats["edu_dist"] = [(str(r[0]), int(r[1])) for r in rows]

    # 11. Top banks
    rows = _run(f"""
        SELECT bank, COUNT(*) as n
        FROM ({base})
        WHERE bank IS NOT NULL
        GROUP BY bank ORDER BY n DESC LIMIT 3
    """)
    if rows:
        stats["top_banks"] = [(str(r[0]), int(r[1])) for r in rows]

    return stats