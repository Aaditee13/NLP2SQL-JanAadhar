from database.schema_metadata import COLUMNS, ColumnMeta, RAJASTHAN_DISTRICTS_41, RELATIONSHIPS


def test_basic_metadata_exists():
    names = {column.qualified_name for column in COLUMNS}
    assert "citizen.bank" in names
    assert "citizen.district_name_eng" in names
    assert "citizen.gender" in names


def test_relationships_are_empty():
    assert len(RELATIONSHIPS) == 0


def test_rajasthan_districts_match_current_wikipedia_count():
    assert len(RAJASTHAN_DISTRICTS_41) == 41
    assert "Bikaner" in RAJASTHAN_DISTRICTS_41
    assert "Kotputli-Behror" in RAJASTHAN_DISTRICTS_41
    assert "Anupgarh" not in RAJASTHAN_DISTRICTS_41
    assert "Jaipur Rural" not in RAJASTHAN_DISTRICTS_41
    assert "Sanchore" not in RAJASTHAN_DISTRICTS_41


def test_column_metadata_supports_legacy_misspelled_physical_names():
    column = ColumnMeta(
        table="citizen",
        column="gendr",
        description="Citizen gender",
        data_type="string",
        aliases=["gender", "male", "female"],
        semantic_name="gender",
    )
    assert column.qualified_name == "citizen.gendr"
    assert column.business_name == "gender"
