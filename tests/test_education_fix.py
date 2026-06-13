"""
Unit tests for _fix_education_sql — the deterministic education hierarchy post-processor.
Tests all patterns the LLM may generate wrongly.
"""
from app import _fix_education_sql


BASE = "SELECT name_en, age, education FROM citizen WHERE {where};"


def fix(where: str, question: str) -> str:
    return _fix_education_sql(BASE.format(where=where), question)


# ── "and above" hierarchy cases ───────────────────────────────────────────────

def test_10th_pass_and_above_broad_like():
    """LLM used broad LIKE '%pass%' — should be fixed to 10 Pass and above."""
    result = fix("LOWER(education) LIKE '%pass%' AND education != 'illiterate'",
                 "Show list of people 10th pass and above")
    assert "education IN" in result
    assert "'10 Pass'" in result
    assert "'12 Pass'" in result
    assert "'Graduate'" in result
    assert "'Post Graduate'" in result
    assert "illiterate" not in result
    assert "'5 Pass'" not in result
    assert "'8 Pass'" not in result


def test_10th_pass_and_above_exact_wrong():
    """LLM used education = '10th Pass' (wrong casing)."""
    result = fix("education = '10th Pass'", "Show 10th pass and above citizens")
    assert "'10 Pass'" in result
    assert "'Graduate'" in result
    assert "'8 Pass'" not in result


def test_12th_pass_and_above():
    result = fix("education LIKE '%12%'", "List citizens 12th pass and above")
    assert "'12 Pass'" in result
    assert "'Graduate'" in result
    assert "'Post Graduate'" in result
    assert "'10 Pass'" not in result


def test_graduate_and_above():
    result = fix("education LIKE '%graduate%'", "Show graduates and above")
    assert "'Graduate'" in result
    assert "'Post Graduate'" in result
    assert "'12 Pass'" not in result


def test_8th_pass_and_above():
    result = fix("education LIKE '%8%'", "List 8th pass and above")
    assert "'8 Pass'" in result
    assert "'10 Pass'" in result
    assert "'Graduate'" in result
    assert "'5 Pass'" not in result
    assert "illiterate" not in result


# ── "and below" hierarchy cases ───────────────────────────────────────────────

def test_10th_pass_and_below():
    result = fix("education = '10 Pass'", "List citizens 10th pass and below")
    assert "'10 Pass'" in result
    assert "'8 Pass'" in result
    assert "'5 Pass'" in result
    assert "'Literate'" in result
    assert "'illiterate'" in result
    assert "'12 Pass'" not in result
    assert "'Graduate'" not in result


def test_8th_pass_and_below():
    result = fix("education LIKE '%8%'", "Citizens 8th class and below")
    assert "'8 Pass'" in result
    assert "'5 Pass'" in result
    assert "'Literate'" in result
    assert "'illiterate'" in result
    assert "'10 Pass'" not in result


# ── Broad LIKE without hierarchy context (fallback fix) ───────────────────────

def test_broad_like_pass_no_hierarchy():
    """education LIKE '%pass%' without 'and above/below' should still be sanitised."""
    result = fix("education LIKE '%pass%'", "Show all educated citizens")
    assert "education IN" in result
    # Should include all pass levels + graduate/post graduate
    assert "'5 Pass'" in result
    assert "'10 Pass'" in result
    assert "'Graduate'" in result


def test_no_education_column_untouched():
    """SQL without education column should be returned unchanged."""
    sql = "SELECT name_en FROM citizen WHERE gender = 'Male';"
    assert _fix_education_sql(sql, "List all male citizens") == sql
