from __future__ import annotations

import pandas as pd

from normalization.fuzzy_match import is_fuzzy_intent, extract_fuzzy_target, fuzzy_rerank, phonetic_key


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(names: list[str]) -> pd.DataFrame:
    """Build a minimal member_name DataFrame for rerank tests."""
    return pd.DataFrame({"member_name": names, "age": range(len(names))})


# ── Intent detection ──────────────────────────────────────────────────────────

def test_is_fuzzy_intent_detects_all_trigger_phrases():
    triggers = [
        "show names similar to X",
        "find members whose name is like X",
        "show names that sound like X",
        "members spelled like X",
        "do a fuzzy search for X",
        "find approximate matches for X",
        "show members resembling X",
    ]
    for phrase in triggers:
        assert is_fuzzy_intent(phrase) is True, f"Should detect fuzzy intent: {phrase!r}"


def test_is_fuzzy_intent_rejects_non_fuzzy_queries():
    non_fuzzy = [
        "show all members in jaipur",
        "how many families have income > 50000",
        "list female members from ajmer",
        "count members with bank account",
    ]
    for phrase in non_fuzzy:
        assert is_fuzzy_intent(phrase) is False, f"Should NOT detect fuzzy intent: {phrase!r}"


# ── Target extraction ─────────────────────────────────────────────────────────

def test_extract_fuzzy_target_stops_at_location_prepositions():
    # "in", "from" are stop words — target should end before them
    assert extract_fuzzy_target("similar to Abc in jaipur") == "Abc"
    assert extract_fuzzy_target("name like Xyz from ajmer") == "Xyz"


def test_extract_fuzzy_target_captures_multi_word_names():
    # Multi-word targets should be fully captured until a stop word
    result = extract_fuzzy_target("similar to Foo Bar Baz")
    assert result == "Foo Bar Baz"


def test_extract_fuzzy_target_returns_title_case():
    # Output must always be Title Case regardless of input casing
    result = extract_fuzzy_target("similar to foo bar")
    assert result == result.title()


def test_extract_fuzzy_target_returns_none_for_non_fuzzy():
    assert extract_fuzzy_target("show all members in jaipur") is None


# ── Fuzzy reranking ───────────────────────────────────────────────────────────

def test_fuzzy_rerank_adds_similarity_score_column():
    df = _make_df(["Alpha", "Beta", "Gamma"])
    result = fuzzy_rerank(df, "Alpha", threshold=0.0)
    assert "similarity_score" in result.columns


def test_fuzzy_rerank_results_sorted_descending():
    df = _make_df(["Alpha", "Aleph", "Zebra", "Zeta"])
    result = fuzzy_rerank(df, "Alpha", threshold=0.0)
    scores = result["similarity_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_fuzzy_rerank_respects_max_rows():
    df = _make_df(["Aaaa", "Aaab", "Aaac", "Aaad", "Aaae"])
    result = fuzzy_rerank(df, "Aaaa", threshold=0.0, max_rows=3)
    assert len(result) <= 3


def test_fuzzy_rerank_respects_threshold():
    # Names that are completely unrelated to the target should be filtered out
    df = _make_df(["Aaaa", "Zzzz", "Qqqq"])
    result = fuzzy_rerank(df, "Aaaa", threshold=0.95)
    # Only the exact match should survive at 0.95 threshold
    assert all(s >= 0.95 for s in result["similarity_score"].tolist())


def test_fuzzy_rerank_exact_single_word_match_scores_one():
    # An exact match must always return similarity_score == 1.0
    df = _make_df(["Exact", "Something Else"])
    result = fuzzy_rerank(df, "Exact", threshold=0.0)
    exact_row = result[result["member_name"] == "Exact"]
    assert not exact_row.empty
    assert exact_row.iloc[0]["similarity_score"] == 1.0


def test_fuzzy_rerank_exact_multi_word_match_scores_one():
    # Regression: two-word exact target must score 1.0 and rank first.
    # Previously, individual words were compared against the full target string
    # and failed the length-difference guard, causing a score of 0.
    target = "Foo Bar"
    exact_name = "Foo Bar"
    similar_name = "Foobar Xyz"
    unrelated_name = "Zzz Qqq"
    df = _make_df([exact_name, similar_name, unrelated_name])

    result = fuzzy_rerank(df, target, threshold=0.80)
    names = result["member_name"].tolist()

    assert exact_name in names, "Exact multi-word name must be returned"
    assert names[0] == exact_name, "Exact match must rank first"
    assert result.iloc[0]["similarity_score"] == 1.0


def test_fuzzy_rerank_prefix_names_match_shorter_target():
    # A short target should match names that start with that prefix
    # e.g. target "Ram" should match "Ramu", "Rama" (prefix variants)
    prefix = "Ram"
    prefix_variants = ["Ramu", "Rama"]
    unrelated = "Zzz"
    df = _make_df(prefix_variants + [unrelated])

    result = fuzzy_rerank(df, prefix, threshold=0.75)
    names = result["member_name"].tolist()

    for name in prefix_variants:
        assert name in names, f"{name!r} should match prefix target {prefix!r}"
    assert unrelated not in names


def test_fuzzy_rerank_returns_empty_for_empty_dataframe():
    empty_df = pd.DataFrame(columns=["member_name"])
    assert fuzzy_rerank(empty_df, "Anything").empty is True


def test_fuzzy_rerank_graceful_when_no_name_column():
    # DataFrame with no name column — should return unchanged (no crash, no score col)
    no_name_df = pd.DataFrame({"age": [30, 25]})
    result = fuzzy_rerank(no_name_df, "Anything")
    assert "similarity_score" not in result.columns


# ── phonetic_key equivalences ─────────────────────────────────────────────────

def test_phonetic_key_vowel_doubling_oo():
    # oo → u: Poonam and Punam are the same phoneme
    assert phonetic_key("Poonam") == phonetic_key("Punam")


def test_phonetic_key_vowel_doubling_ee():
    # ee → i: Geeta and Gita are the same phoneme
    assert phonetic_key("Geeta") == phonetic_key("Gita")
    assert phonetic_key("Suneeta") == phonetic_key("Sunita")
    assert phonetic_key("Preeti") == phonetic_key("Priti")


def test_phonetic_key_consonant_sh():
    # sh → s: Shweta and Sweta are the same phoneme cluster
    assert phonetic_key("Shweta") == phonetic_key("Sweta")


def test_phonetic_key_aspiration_th():
    # th → t: Thakur and Takur are the same phoneme
    assert phonetic_key("Thakur") == phonetic_key("Takur")


def test_phonetic_key_aspiration_bh_gh_kh():
    assert phonetic_key("Bhanu") == phonetic_key("Banu")
    assert phonetic_key("Ghosh") == phonetic_key("Gosh")
    assert phonetic_key("Phool") == phonetic_key("Pool")


def test_phonetic_key_v_b_interchange():
    # v/b are phonetically interchangeable in North India
    assert phonetic_key("Vijay") == phonetic_key("Bijay")
    assert phonetic_key("Vimal") == phonetic_key("Bimal")


def test_phonetic_key_gemination():
    # Double consonants reduce to single
    assert phonetic_key("Rammesh") == phonetic_key("Ramesh")


def test_phonetic_key_aa_normalization():
    assert phonetic_key("Raadha") == phonetic_key("Radha")


def test_phonetic_key_distinct_names_differ():
    # Unrelated names must NOT collapse to the same key
    assert phonetic_key("Sunita") != phonetic_key("Savita")
    assert phonetic_key("Geeta") != phonetic_key("Rita")
    assert phonetic_key("Poonam") != phonetic_key("Meena")


def test_phonetic_key_empty_string():
    assert phonetic_key("") == ""


# ── fuzzy_rerank with phonetic strategy ──────────────────────────────────────

def test_fuzzy_rerank_phonetic_catches_punam_when_searching_poonam():
    # JW("poonam", "punam") ≈ 0.84 — below threshold 0.88.
    # Phonetic keys match → score floored at 0.90 → should pass.
    df = _make_df(["Poonam", "Punam", "Zzz"])
    result = fuzzy_rerank(df, "Poonam", threshold=0.88)
    names = result["member_name"].tolist()
    assert "Punam" in names, "Phonetic match should include Punam when searching Poonam"


def test_fuzzy_rerank_phonetic_catches_sweta_when_searching_shweta():
    # JW("shweta", "sweta") ≈ 0.95 — above 0.88 on its own.
    # Confirm it's still returned (either via JW or phonetic).
    df = _make_df(["Shweta", "Sweta", "Zzz"])
    result = fuzzy_rerank(df, "Shweta", threshold=0.88)
    names = result["member_name"].tolist()
    assert "Sweta" in names


def test_fuzzy_rerank_phonetic_works_for_multi_word_db_names():
    # Single-word target "Punam" should match "Punam Devi" and "Poonam Devi"
    # via the per-word phonetic check.
    df = _make_df(["Poonam Devi", "Punam Devi", "Zzz Abc"])
    result = fuzzy_rerank(df, "Punam", threshold=0.88)
    names = result["member_name"].tolist()
    assert "Punam Devi" in names
    assert "Poonam Devi" in names


def test_fuzzy_rerank_phonetic_does_not_inflate_unrelated_names():
    # Names with completely different phonetic keys should not get the 0.90 floor.
    df = _make_df(["Sunita", "Rohit", "Zzz"])
    result = fuzzy_rerank(df, "Poonam", threshold=0.88)
    names = result["member_name"].tolist()
    assert "Rohit" not in names
    assert "Zzz" not in names
