from llm.fast_path import FastPathEngine


def test_generate_sql_fast_simple_gender_district():
    engine = FastPathEngine()
    sql = engine.generate_sql_fast("Show all males in Jaipur")
    assert sql is not None
    # Check that canonical columns are selected
    assert "name_en" in sql
    assert "age" in sql
    assert "gender" in sql
    assert "district_name_eng" in sql
    assert "gender = 'Male'" in sql
    assert "district_name_eng = 'Jaipur'" in sql


def test_generate_sql_fast_complex_query_aborts():
    engine = FastPathEngine()
    # Average keyword is complex, should return None (fall back to LLM)
    sql = engine.generate_sql_fast("What is the average age of citizens in Jaipur?")
    assert sql is None


def test_swap_ast_parameters_gender():
    engine = FastPathEngine()
    cached_sql = "SELECT name_en, age, gender, district_name_eng FROM citizen WHERE gender = 'Female' AND district_name_eng = 'Jaipur';"
    cached_question = "Show all females in Jaipur"
    
    # Swap Female -> Male
    new_question = "Show all males in Jaipur"
    swapped_sql = engine.swap_ast_parameters(cached_sql, cached_question, new_question)
    assert swapped_sql is not None
    assert "gender = 'Male'" in swapped_sql
    assert "district_name_eng = 'Jaipur'" in swapped_sql

    # Swap Jaipur -> Jodhpur
    new_question_2 = "Show all females in Jodhpur"
    swapped_sql_2 = engine.swap_ast_parameters(cached_sql, cached_question, new_question_2)
    assert swapped_sql_2 is not None
    assert "gender = 'Female'" in swapped_sql_2
    assert "district_name_eng = 'Jodhpur'" in swapped_sql_2


def test_swap_ast_parameters_age():
    engine = FastPathEngine()
    cached_sql = "SELECT name_en, age, gender, district_name_eng FROM citizen WHERE age > 21 AND district_name_eng = 'Jaipur';"
    cached_question = "Show all citizens above 21 in Jaipur"
    
    # Swap age 21 -> 18
    new_question = "Show all citizens above 18 in Jaipur"
    swapped_sql = engine.swap_ast_parameters(cached_sql, cached_question, new_question)
    assert swapped_sql is not None
    assert "age > 18" in swapped_sql
