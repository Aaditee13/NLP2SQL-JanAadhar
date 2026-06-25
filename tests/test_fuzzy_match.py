from __future__ import annotations

import pandas as pd

from normalization.fuzzy_match import (
    is_fuzzy_intent, extract_fuzzy_target, fuzzy_rerank, phonetic_key,
    classify_query_name, score_name_pair, _score_token_pair,
)


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


def test_is_fuzzy_intent_detects_members_like_pattern():
    triggers = [
        "show members like Kumar Ashok",
        "find members like Sunita Devi",
        "show people like Ram Prasad",
        "members are like Geeta",
        "find persons like Vijay Singh",
        "citizens like Rekha Devi",
        "beneficiaries like Ashok Kumar",
    ]
    for phrase in triggers:
        assert is_fuzzy_intent(phrase) is True, f"Should detect fuzzy intent: {phrase!r}"

    # Extraction should work too
    assert extract_fuzzy_target("show members like Kumar Ashok") == "Kumar Ashok"
    assert extract_fuzzy_target("find members like Sunita Devi") == "Sunita Devi"


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


# ── classify_query_name ───────────────────────────────────────────────────────

def test_classify_query_name_single_word():
    tokens = classify_query_name("Sunita")
    assert tokens == [("sunita", 1.0)]


def test_classify_query_name_two_words_weights():
    tokens = classify_query_name("Ramesh Sharma")
    assert len(tokens) == 2
    first_w = tokens[0][1]
    second_w = tokens[1][1]
    assert first_w > second_w, "First name must have higher weight than second token"
    assert first_w == 1.0
    # Position 1 is treated as a middle-name slot (weight 0.55); the last-name
    # slot (0.40) only activates at position 2 in a 3-word name.
    assert second_w == 0.55


def test_classify_query_name_three_words_weights_decreasing():
    tokens = classify_query_name("Ram Kumar Sharma")
    weights = [w for _, w in tokens]
    assert weights == sorted(weights, reverse=True), "Weights must be non-increasing"


def test_classify_query_name_strips_honorifics():
    # "S/O" and "Shri" are honorifics and must be removed
    tokens = classify_query_name("S/O Ramesh")
    assert len(tokens) == 1
    assert tokens[0][0] == "ramesh"

    tokens = classify_query_name("Shri Mohan Lal")
    assert tokens[0][0] == "mohan"


def test_classify_query_name_filler_token_capped():
    # "Devi" at position 1 must have weight <= 0.45
    tokens = classify_query_name("Sunita Devi")
    devi_weight = dict(tokens)["devi"]
    assert devi_weight <= 0.45


def test_classify_query_name_returns_lowercase():
    tokens = classify_query_name("GEETA DEVI")
    assert all(t == t.lower() for t, _ in tokens)


# ── _score_token_pair ─────────────────────────────────────────────────────────

def test_score_token_pair_exact_match():
    assert _score_token_pair("ramesh", "ramesh") == 1.0


def test_score_token_pair_phonetic_match():
    # Poonam and Punam share the same phonetic key
    assert _score_token_pair("poonam", "punam") == 0.92


def test_score_token_pair_initial_match():
    # Single char "R" should match "Ramesh" at 0.88
    assert _score_token_pair("r", "ramesh") == 0.88


def test_score_token_pair_initial_no_match():
    # "R" should not give 0.88 against a name not starting with R
    assert _score_token_pair("r", "sunita") < 0.88


def test_score_token_pair_jw_fallback():
    score = _score_token_pair("ramesh", "suresh")
    assert 0.0 < score < 1.0


# ── score_name_pair ───────────────────────────────────────────────────────────

def test_score_name_pair_exact_two_word_match():
    tokens = classify_query_name("Ramesh Sharma")
    assert score_name_pair(tokens, "Ramesh Sharma") == 1.0


def test_score_name_pair_first_name_match_ranks_above_surname_match():
    # "Geeta Sharma": compare a record with an exact first-name match against a
    # record with a completely different first name but exact surname.
    # "Bimla" has low JW to "Geeta" (≈0.46), so the exact first-name match
    # on "Geeta Gupta" should rank above the unrelated-first + exact-surname "Bimla Sharma".
    tokens = classify_query_name("Geeta Sharma")
    score_first_name_match = score_name_pair(tokens, "Geeta Gupta")
    score_surname_match = score_name_pair(tokens, "Bimla Sharma")
    assert score_first_name_match > score_surname_match, (
        f"Exact first-name match ({score_first_name_match:.3f}) must rank above "
        f"unrelated-first + exact-surname match ({score_surname_match:.3f})"
    )


def test_score_name_pair_missing_middle_name_scores_high():
    # Query "Ramesh Sharma" vs DB "Ramesh Kumar Sharma" — middle name gap must not penalize heavily
    tokens = classify_query_name("Ramesh Sharma")
    score_with_middle = score_name_pair(tokens, "Ramesh Kumar Sharma")
    assert score_with_middle >= 0.90, (
        f"Missing middle name should not reduce score below 0.90, got {score_with_middle:.3f}"
    )


def test_score_name_pair_phonetic_variant_in_first_position():
    # "Poonam Devi" vs query "Punam Devi" — phonetic variant in first position
    tokens = classify_query_name("Punam Devi")
    score = score_name_pair(tokens, "Poonam Devi")
    assert score >= 0.88, f"Phonetic first-name variant should score >= 0.88, got {score:.3f}"


def test_score_name_pair_phonetic_variant_preserved_with_weights():
    # Phonetic first-name variant must still rank above a different-first-name record
    tokens = classify_query_name("Punam Sharma")
    score_phonetic_first = score_name_pair(tokens, "Poonam Sharma")   # phonetic first match
    score_different_first = score_name_pair(tokens, "Sunita Sharma")  # different first name
    assert score_phonetic_first > score_different_first


def test_score_name_pair_reordered_tokens_score_lower():
    # "Sharma Ramesh" should score meaningfully below "Ramesh Sharma".
    # The per-token backward-shift penalty (0.90) should produce a visible gap,
    # not just a tiny alignment-bonus difference.
    tokens = classify_query_name("Ramesh Sharma")
    score_aligned = score_name_pair(tokens, "Ramesh Sharma")
    score_reordered = score_name_pair(tokens, "Sharma Ramesh")
    assert score_aligned > score_reordered, "Aligned must score higher than reordered"
    assert score_aligned - score_reordered >= 0.05, (
        f"Gap must be >= 0.05 to meaningfully separate reordered results "
        f"(aligned={score_aligned:.3f}, reordered={score_reordered:.3f})"
    )


def test_score_name_pair_single_token_query():
    # A single token classify gives weight 1.0; score_name_pair should handle it
    tokens = classify_query_name("Sunita")
    score_exact = score_name_pair(tokens, "Sunita")
    score_with_suffix = score_name_pair(tokens, "Sunita Devi")
    assert score_exact == 1.0
    assert score_with_suffix >= 0.90


def test_score_name_pair_empty_inputs():
    assert score_name_pair([], "Ramesh Sharma") == 0.0
    assert score_name_pair(classify_query_name("Ramesh"), "") == 0.0


def test_score_name_pair_initial_in_query():
    # "R. K. Sharma" → tokens ["r", "k", "sharma"]
    tokens = classify_query_name("R K Sharma")
    score = score_name_pair(tokens, "Ramesh Kumar Sharma")
    assert score >= 0.80, f"Initial match should score >= 0.80, got {score:.3f}"


# ── fuzzy_rerank with multi-word targets (position-aware path) ────────────────

def test_fuzzy_rerank_multiword_first_name_match_ranks_above_surname_match():
    df = _make_df(["Ramesh Devi", "Suresh Sharma", "Ramesh Kumar Sharma", "Zzz Qqq"])
    result = fuzzy_rerank(df, "Ramesh Sharma", threshold=0.75)
    names = result["member_name"].tolist()

    assert "Ramesh Kumar Sharma" in names, "Full-name record must be returned"
    assert "Ramesh Devi" in names, "First-name-only match must be returned"
    assert "Suresh Sharma" in names, "Surname-only match must be returned"

    ramesh_kumar_idx = names.index("Ramesh Kumar Sharma")
    suresh_sharma_idx = names.index("Suresh Sharma")
    assert ramesh_kumar_idx < suresh_sharma_idx, (
        "Ramesh Kumar Sharma (first+last match) must rank above Suresh Sharma (surname-only)"
    )


def test_fuzzy_rerank_multiword_phonetic_first_name_variant():
    # "Shweta Singh" query must match "Sweta Singh" (phonetic first-name variant)
    df = _make_df(["Sweta Singh", "Shweta Singh", "Geeta Singh", "Zzz Abc"])
    result = fuzzy_rerank(df, "Shweta Singh", threshold=0.80)
    names = result["member_name"].tolist()
    assert "Sweta Singh" in names, "Phonetic first-name variant Sweta must match Shweta"
    assert "Shweta Singh" in names


def test_fuzzy_rerank_multiword_missing_middle_name():
    df = _make_df(["Ramesh Kumar Sharma", "Ramesh Sharma", "Suresh Kumar Sharma"])
    result = fuzzy_rerank(df, "Ramesh Sharma", threshold=0.80)
    names = result["member_name"].tolist()
    assert "Ramesh Kumar Sharma" in names, "Middle-name gap must not filter out the record"
    assert "Ramesh Sharma" in names


def test_fuzzy_rerank_multiword_unrelated_filtered_out():
    df = _make_df(["Ramesh Sharma", "Geeta Devi", "Palo Devi", "Xyz Abc"])
    result = fuzzy_rerank(df, "Ramesh Sharma", threshold=0.80)
    names = result["member_name"].tolist()
    assert "Xyz Abc" not in names
    assert "Palo Devi" not in names
