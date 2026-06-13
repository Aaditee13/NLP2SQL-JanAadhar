from prompting.prompt_builder import PromptBuilder
from retrieval.schema_retriever import RetrievalResult


def test_prompt_includes_dialect_rules():
    result = RetrievalResult(
        question="show all boys in Jaipur",
        tables=["citizen"],
        columns=["citizen.district_name_eng", "citizen.gender"],
        relationships=[],
        documents=[],
        confidence=0.9,
    )
    prompt = PromptBuilder().build(result, dialect="sqlite")
    assert "SQLite-compatible" in prompt
    assert "citizen" in prompt


def test_prompt_mentions_business_meaning_for_physical_columns():
    result = RetrievalResult(
        question="show all boys in Jaipur",
        tables=["citizen"],
        columns=["citizen.gender"],
        relationships=[],
        documents=[],
        confidence=0.9,
    )
    prompt = PromptBuilder().build(result)
    assert "business meaning: gender" in prompt
    assert "valid example values: Male, Female" in prompt


def test_prompt_builder_includes_dynamic_few_shots():
    result = RetrievalResult(
        question="show all boys in Jaipur",
        tables=["citizen"],
        columns=["citizen.gender"],
        relationships=[],
        documents=[],
        confidence=0.9,
    )
    few_shots = [
        {"question": "How many SC widows in Jaipur?", "sql": "SELECT COUNT(*) FROM citizen WHERE caste_category = 'SC' AND marital_status = 'Widow';"}
    ]
    prompt = PromptBuilder().build(result, few_shots=few_shots)
    assert "### EXAMPLES" in prompt
    assert "How many SC widows in Jaipur?" in prompt
    assert "SELECT COUNT(*) FROM citizen" in prompt
