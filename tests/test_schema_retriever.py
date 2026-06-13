from retrieval.schema_retriever import SchemaRetriever


class EmptyStore:
    def search(self, question, top_k):
        return []


class NoisyStore:
    def search(self, question, top_k):
        noisy_columns = [
            "citizen.block_name_eng",
            "citizen.district_name_eng",
            "citizen.gp_name_eng",
            "citizen.enrollment_id",
            "citizen.mobile_no",
            "citizen.age",
            "citizen.caste_category",
            "citizen.dob",
            "citizen.gender",
            "citizen.jan_aadhaar_member_id",
        ]
        return [
            {
                "kind": "column",
                "table": qualified_name.split(".")[0],
                "qualified_name": qualified_name,
                "score": 0.9,
            }
            for qualified_name in noisy_columns
        ]


def test_retriever_adds_explicit_geography_columns():
    result = SchemaRetriever(EmptyStore()).retrieve("all female citizens in Jaipur district")
    assert "citizen" in result.tables
    assert "citizen.district_name_eng" in result.columns
    assert "citizen.gender" in result.columns
    assert "citizen.name_en" in result.columns


def test_retriever_adds_boys_age_and_jodhpur_columns():
    result = SchemaRetriever(EmptyStore()).retrieve("show me all boys above 18 in jodhpur")
    assert "citizen.district_name_eng" in result.columns
    assert "citizen.gender" in result.columns
    assert "citizen.age" in result.columns
    assert "citizen.caste_category" not in result.columns
    assert len(result.columns) <= 7


def test_retriever_handles_bikaner_as_district():
    result = SchemaRetriever(EmptyStore()).retrieve("all boys above 21 in bikaner")
    assert "citizen.district_name_eng" in result.columns
    assert "citizen.gender" in result.columns
    assert "citizen.age" in result.columns
    assert "citizen.name_en" in result.columns


def test_retriever_uses_generic_location_fallback():
    result = SchemaRetriever(EmptyStore()).retrieve("all girls above 21 in phulera")
    assert "citizen.district_name_eng" in result.columns
    assert "citizen.gender" in result.columns
    assert "citizen.age" in result.columns


def test_retriever_prunes_noisy_unrequested_domains():
    result = SchemaRetriever(NoisyStore()).retrieve("All boys above 21 in jaipur")
    assert "citizen" in result.tables
    expected = {
        "citizen.district_name_eng",
        "citizen.age",
        "citizen.gender",
        "citizen.name_en",
    }
    assert expected.issubset(set(result.columns))


def test_retriever_uses_education_and_minority_for_illiterate_muslims():
    result = SchemaRetriever(NoisyStore()).retrieve("Show all illiterate muslims in Jaipur")
    assert "citizen.education" in result.columns
    assert "citizen.minority" in result.columns
    assert "citizen.district_name_eng" in result.columns
    assert "citizen.caste_category" not in result.columns
    assert "citizen.gender" not in result.columns
