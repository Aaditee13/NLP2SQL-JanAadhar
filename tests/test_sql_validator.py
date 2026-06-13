from validation.sql_validator import SQLValidator


def test_validator_accepts_valid_citizen_query():
    sql = "SELECT name_en, age FROM citizen WHERE district_name_eng = 'Jaipur';"
    result = SQLValidator().validate(sql)
    assert result.valid, result.errors


def test_validator_rejects_hallucinated_column():
    sql = "SELECT fake_column FROM citizen;"
    result = SQLValidator().validate(sql)
    assert not result.valid
    assert any("Unknown column" in err or "disallowed column" in err for err in result.errors)


def test_validator_rejects_write_statement():
    result = SQLValidator().validate("DROP TABLE citizen;")
    assert not result.valid


def test_validator_rejects_select_plus_write_statement():
    result = SQLValidator().validate("SELECT name_en FROM citizen; DELETE FROM citizen;")
    assert not result.valid


def test_validator_rejects_select_into():
    result = SQLValidator().validate("SELECT name_en INTO backup_citizen FROM citizen;")
    assert not result.valid


def test_post_process_sql_rewrites_name_equals():
    from app import _post_process_sql
    sql = "SELECT name_en FROM citizen WHERE name_en = 'Vijay';"
    processed = _post_process_sql(sql)
    assert "name_en LIKE '%Vijay%'" in processed

    sql_multi = "SELECT name_en FROM citizen WHERE name_en = 'Vijay Kumar Laxmi';"
    processed_multi = _post_process_sql(sql_multi)
    assert "name_en LIKE '%Vijay Kumar Laxmi%'" in processed_multi

    # Categorical fields
    sql_cat = "SELECT * FROM citizen WHERE gender = 'male' AND caste_category = 'obc';"
    processed_cat = _post_process_sql(sql_cat)
    assert "gender = 'Male'" in processed_cat
    assert "caste_category = 'OBC'" in processed_cat

    # District fields
    sql_dist = "SELECT * FROM citizen WHERE district_name_eng = 'sawai madhopur';"
    processed_dist = _post_process_sql(sql_dist)
    assert "district_name_eng = 'Sawai Madhopur'" in processed_dist


def test_post_process_caste_bilingual_expansion():
    from app import _post_process_sql

    sql1 = "SELECT * FROM citizen WHERE caste LIKE '%Rajput%';"
    processed1 = _post_process_sql(sql1)
    assert "caste LIKE '%Rajput%'" in processed1
    assert "caste LIKE '%Rajpoot%'" in processed1
    assert "caste LIKE '%राजपूत%'" in processed1

    sql2 = "SELECT * FROM citizen WHERE caste = 'Rajput';"
    processed2 = _post_process_sql(sql2)
    assert "caste LIKE '%Rajput%'" in processed2
    assert "caste LIKE '%Rajpoot%'" in processed2
    assert "caste LIKE '%राजपूत%'" in processed2


def test_post_process_bank_in_clause_case_insensitivity():
    from app import _post_process_sql

    sql = "SELECT * FROM citizen WHERE bank IN ('sbi', 'HDFC', 'Icici');"
    result = _post_process_sql(sql)
    assert "UPPER(" in result
    assert "'SBI'" in result
    assert "'HDFC'" in result
    assert "'ICICI'" in result


def test_post_process_district_in_clause_non_district_redirect():
    from app import _post_process_sql

    # Srinagar is NOT a Rajasthan district -> block/village
    sql = "SELECT * FROM citizen WHERE district_name_eng IN ('Srinagar', 'Jaipur');"
    result = _post_process_sql(sql)
    assert "block_name_eng LIKE '%Srinagar%'" in result or "vill_name_eng LIKE '%Srinagar%'" in result
    assert "district_name_eng = 'Jaipur'" in result
