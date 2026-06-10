from __future__ import annotations

import pandas as pd
import pytest

from normalization.fuzzy_match import is_fuzzy_intent, extract_fuzzy_target, fuzzy_rerank


def test_is_fuzzy_intent():
    assert is_fuzzy_intent("show names similar to vijay") is True
    assert is_fuzzy_intent("find members whose name is like suresh") is True
    assert is_fuzzy_intent("show names that sound like ramesh") is True
    assert is_fuzzy_intent("members spelled like preeya") is True
    assert is_fuzzy_intent("do a fuzzy search for dinesh") is True
    assert is_fuzzy_intent("find approximate matches for kamlesh") is True
    assert is_fuzzy_intent("show members resembling sunita") is True
    assert is_fuzzy_intent("show all members in jaipur") is False
    assert is_fuzzy_intent("how many families have income > 50000") is False


def test_extract_fuzzy_target():
    assert extract_fuzzy_target("show names similar to vijay in jaipur") == "Vijay"
    assert extract_fuzzy_target("find members whose name is like suresh from ajmer") == "Suresh"
    assert extract_fuzzy_target("show members resembling sunita") == "Sunita"
    assert extract_fuzzy_target("members spelled like preeya who are female") == "Preeya"
    assert extract_fuzzy_target("similar to vijay laxmi bhambhoo") == "Vijay Laxmi Bhambhoo"
    assert extract_fuzzy_target("do a fuzzy search for ram") == "Ram"


def test_fuzzy_rerank():
    # Setup mock DataFrame
    data = {
        "member_name": ["Vijay Laxmi", "Vijai", "Vijendra", "Ramesh Kumar", "Sunita Kanwar"],
        "age": [32, 25, 45, 50, 28]
    }
    df = pd.DataFrame(data)

    # Search for "Vijay"
    result_df = fuzzy_rerank(df, "Vijay", threshold=0.75, max_rows=3)
    
    assert len(result_df) <= 3
    # Check that similarity scores were added and sorted descending
    assert "similarity_score" in result_df.columns
    scores = result_df["similarity_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    
    # "Vijay Laxmi" and "Vijai" should match and have high Jaro-Winkler scores
    matched_names = result_df["member_name"].tolist()
    assert "Vijay Laxmi" in matched_names
    assert "Vijai" in matched_names
    assert "Ramesh Kumar" not in matched_names  # below threshold

    # Search for "Ram" — all Ram-prefix names score >= 0.86 via full-string JaroWinkler
    result_ram_df = fuzzy_rerank(df, "Ram", threshold=0.75)
    matched_ram_names = result_ram_df["member_name"].tolist()
    assert "Vijai" not in matched_ram_names
    ram_data = {
        "member_name": ["Ramu", "Ramesh", "Rama", "Ramkesh"],
        "age": [10, 20, 30, 40]
    }
    ram_df = pd.DataFrame(ram_data)
    result_ram_df = fuzzy_rerank(ram_df, "Ram", threshold=0.75)
    matched_ram_names = result_ram_df["member_name"].tolist()
    # All Ram-prefix names match (full-string JaroWinkler ≥ 0.86 for all)
    assert "Ramu" in matched_ram_names
    assert "Rama" in matched_ram_names
    assert "Ramesh" in matched_ram_names
    assert "Ramkesh" in matched_ram_names
    # Search for "Ramesh" - Rameshwari should be allowed (starts with Ramesh, target len >= 5)
    ramesh_data = {
        "member_name": ["Rameshwari", "Rameshwar"],
        "age": [30, 40]
    }
    ramesh_df = pd.DataFrame(ramesh_data)
    result_ramesh_df = fuzzy_rerank(ramesh_df, "Ramesh", threshold=0.75)
    matched_ramesh_names = result_ramesh_df["member_name"].tolist()
    assert "Rameshwari" in matched_ramesh_names
    assert "Rameshwar" in matched_ramesh_names
    # Multi-word exact target: "Palo Devi" must score 1.0 and appear first
    # Regression test — old bug scored it 0 because individual words ("palo", "devi")
    # were compared against the full target "palo devi" and failed the length guard.
    palo_data = {
        "member_name": ["Palo Devi", "Pallavi Jain", "Pallo Singh", "Ram Kumar"],
        "age": [30, 25, 40, 50]
    }
    palo_df = pd.DataFrame(palo_data)
    result_palo = fuzzy_rerank(palo_df, "Palo Devi", threshold=0.80)
    palo_names = result_palo["member_name"].tolist()
    assert "Palo Devi" in palo_names, "Exact multi-word name must be returned"
    assert palo_names[0] == "Palo Devi", "Exact match must rank first (score 1.0)"
    assert result_palo.iloc[0]["similarity_score"] == 1.0

    # Edge cases
    empty_df = pd.DataFrame(columns=["member_name"])
    assert fuzzy_rerank(empty_df, "Vijay").empty is True

    no_name_col_df = pd.DataFrame({"age": [32, 25]})
    assert "similarity_score" not in fuzzy_rerank(no_name_col_df, "Vijay").columns
