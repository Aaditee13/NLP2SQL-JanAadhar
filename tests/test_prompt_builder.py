from prompting.prompt_builder import PromptBuilder
from retrieval.schema_retriever import RetrievalResult


def test_prompt_uses_only_retrieved_columns():
    result = RetrievalResult(
        question="female bank members",
        tables=["member", "bank_details"],
        columns=["member.gender", "member.member_name", "bank_details.bank_name"],
        relationships=[{"from_table": "bank_details", "from_column": "member_id", "to_table": "member", "to_column": "member_id"}],
        documents=[],
        confidence=0.9,
    )
    prompt = PromptBuilder().build(result)
    assert "member.gender" in prompt
    # Verify 'family' table is NOT listed in the Available tables section.
    # Note: '- family' appears elsewhere in the prompt instructions (e.g. '- family head or HOF'),
    # so we scope the check to the tables section only.
    tables_section = prompt[prompt.index("Available tables:"):prompt.index("Relevant columns:")]
    assert "family" not in tables_section
    assert "Do not invent tables or columns" in prompt


def test_prompt_mentions_business_meaning_for_physical_columns():
    result = RetrievalResult(
        question="boys",
        tables=["member"],
        columns=["member.gender"],
        relationships=[],
        documents=[],
        confidence=0.9,
    )
    prompt = PromptBuilder().build(result)
    assert "business meaning: gender" in prompt
    assert "valid example values: Male, Female" in prompt


def test_prompt_builder_extracts_multiple_locations():
    from prompting.prompt_builder import _extract_location_hints
    
    # 1. Test "and" conjunction
    hints = _extract_location_hints("show people from Srinagar and Beejasar")
    assert "Srinagar" in hints
    assert "Beejasar" in hints
    assert len(hints) == 2

    # 2. Test commas and "and" lists
    hints_list = _extract_location_hints("families in Jaipur, Jodhpur and Udaipur")
    assert "Jaipur" in hints_list
    assert "Jodhpur" in hints_list
    assert "Udaipur" in hints_list
    assert len(hints_list) == 3


def test_prompt_builder_builds_location_rules():
    result = RetrievalResult(
        question="show people from Jaipur and Beejasar",
        tables=["family"],
        columns=["family.district", "family.village"],
        relationships=[],
        documents=[],
        confidence=0.9,
    )
    prompt = PromptBuilder().build(result)
    
    # Jaipur is a district
    assert "The location 'Jaipur' IS one of the 41 Rajasthan districts." in prompt
    # Beejasar is not a district
    assert "The location 'Beejasar' is NOT one of the 41 Rajasthan districts." in prompt
    assert "MULTIPLE VALUE FILTERING RULE:" in prompt

