from __future__ import annotations

import re
import pandas as pd
from rapidfuzz.distance import JaroWinkler


def phonetic_key(name: str) -> str:
    """
    Reduce an Indian name to a canonical phonetic key.

    Maps common romanization variants to a single form so that spelling
    differences that represent the same sound collapse to the same key:
      - Vowel doubling:        oo/ou→u, aa→a, ee/ii→i   (Poonam↔Punam, Geeta↔Gita)
      - Consonant aspiration:  sh→s, bh→b, kh→k, dh→d, gh→g, th→t, ph→p, ch→c, jh→j
      - v/b interchange:       v→b                        (Vijay↔Bijay)
      - Gemination:            double consonant → single  (Suneeta↔Sunita via ee→i + ll→l)

    Known limitation: Arabic-origin name pairs such as Mohammad/Muhammed share
    a mid-vowel o/u substitution that these rules do not cover; JW similarity
    handles those near-threshold cases.
    """
    if not name:
        return ""
    s = re.sub(r'[^a-z\s]', '', name.lower().strip())
    # Vowel length normalization (romanization of long vowels)
    s = re.sub(r'oo|ou', 'u', s)           # Poonam→punam, Gourav→gurav
    s = re.sub(r'aa', 'a', s)               # Raadha→radha
    s = re.sub(r'ee|ii', 'i', s)            # Geeta→gita, Preeti→priti
    # Consonant cluster simplification
    s = re.sub(r'sh', 's', s)               # Shweta→sweta, Shyam→syam
    s = re.sub(r'([bdfgkpt])h', r'\1', s)   # bh→b, ph→p, kh→k, dh→d, gh→g, th→t
    s = re.sub(r'chh?', 'c', s)             # chh→c, ch→c
    s = re.sub(r'jh', 'j', s)               # Jha→ja
    # North Indian v/b interchange
    s = s.replace('v', 'b')                 # Vijay→bijay, Vimal→bimal
    # Gemination: double consonants → single
    s = re.sub(r'(.)\1+', r'\1', s)         # tt→t, nn→n, ll→l, mm→m
    return s


# ── Position-aware name scoring ───────────────────────────────────────────────

# Weights for first, middle, last, and beyond-third name positions.
_POSITIONAL_WEIGHTS: list[float] = [1.0, 0.55, 0.40, 0.30]

# Relational and honorific prefixes that appear before the actual name.
# Stripped from the query before positional weight assignment.
_HONORIFICS: frozenset[str] = frozenset({
    "s/o", "d/o", "w/o", "c/o", "shri", "smt", "sh", "dr", "mr", "mrs", "late",
})

# Tokens that are almost exclusively used as gender/rank suffixes in Indian female
# names (never standalone first names). When one of these appears at position ≥ 1,
# its weight is capped to avoid over-rewarding a suffix match.
_FILLER_TOKENS: frozenset[str] = frozenset({"devi", "bai", "kumari"})


def classify_query_name(target: str) -> list[tuple[str, float]]:
    """
    Return [(lowercase_token, positional_weight), ...] for a name query string.

    Strips leading honorifics, then assigns decreasing positional weights:
    first name = 1.0, middle = 0.55, last = 0.40, beyond = 0.30.
    Common suffix-only tokens (Devi, Bai, Kumari) at non-first position are
    capped at 0.45 so a suffix match cannot dominate the score.
    """
    raw = [w.strip().lower() for w in target.split() if w.strip()]
    tokens = [t for t in raw if t not in _HONORIFICS]
    result: list[tuple[str, float]] = []
    for i, tok in enumerate(tokens):
        w = _POSITIONAL_WEIGHTS[min(i, len(_POSITIONAL_WEIGHTS) - 1)]
        if i > 0 and tok in _FILLER_TOKENS:
            w = min(w, 0.45)
        result.append((tok, w))
    return result


def _score_token_pair(q_tok: str, d_tok: str) -> float:
    """
    Score similarity between a single query token and a single DB name token.

    Priority order:
      1. Exact case-insensitive match → 1.0
      2. Phonetic key match           → 0.92  (Poonam↔Punam, Shweta↔Sweta)
      3. Initial match (1-char query) → 0.88  ("R" matches "Ramesh")
      4. Jaro-Winkler similarity      → raw JW score
    """
    if q_tok == d_tok:
        return 1.0
    pk_q = phonetic_key(q_tok)
    pk_d = phonetic_key(d_tok)
    if pk_q and pk_d and pk_q == pk_d:
        return 0.92
    if len(q_tok) == 1 and d_tok.startswith(q_tok):
        return 0.88
    jw = JaroWinkler.similarity(q_tok, d_tok)
    if jw > 1.0:
        jw /= 100.0
    return jw


def score_name_pair(
    query_tokens: list[tuple[str, float]],
    db_name: str,
) -> float:
    """
    Score a DB name string against weighted query tokens (position-aware).

    Algorithm:
      1. For each query token q_i, find the best-matching DB token
         (exact > phonetic > initial > JW).
      2. Weighted sum: Σ(w_i * best_i) / Σ(w_i).
      3. Alignment bonus (+0.04 * ratio): reward queries where token order
         in the DB name matches the query token order.
      4. Length penalty: mild discount when the DB name has many more tokens
         than the query (loose match against a very long name).
      5. Full-string JW as a fallback floor (* 0.90) for near-exact matches.

    Returns a score in [0.0, 1.0].
    """
    if not query_tokens or not db_name:
        return 0.0

    db_lower = db_name.lower().strip()
    db_tokens = [t for t in db_lower.split() if t]
    if not db_tokens:
        return 0.0

    total_w = sum(w for _, w in query_tokens)
    if total_w == 0.0:
        return 0.0

    weighted_sum = 0.0
    aligned_count = 0

    for i, (q_tok, w) in enumerate(query_tokens):
        best_score = 0.0
        best_j = -1
        for j, d_tok in enumerate(db_tokens):
            s = _score_token_pair(q_tok, d_tok)
            if s > best_score:
                best_score = s
                best_j = j
        if best_j == i:
            aligned_count += 1
        weighted_sum += w * best_score

    primary = weighted_sum / total_w

    alignment_bonus = 0.04 * (aligned_count / len(query_tokens))

    len_diff = len(db_tokens) - len(query_tokens)
    length_penalty = max(0.0, min(0.06, 0.02 * (len_diff - 2))) if len_diff > 2 else 0.0

    full_q = " ".join(t for t, _ in query_tokens)
    full_jw = JaroWinkler.similarity(full_q, db_lower)
    if full_jw > 1.0:
        full_jw /= 100.0

    primary_adjusted = primary + alignment_bonus - length_penalty
    return min(1.0, max(primary_adjusted, full_jw * 0.90))


# ── Fuzzy intent detection patterns ──────────────────────────────────────────

_FUZZY_PATTERNS = [
    re.compile(r"\bsimilar\s+to\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bname(?:s)?\s+(?:is\s+)?like\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bsound(?:s)?\s+like\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bspell(?:ed)?\s+like\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bfuzzy\s+(?:search\s+)?(?:for\s+)?([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bapproximate\s+(?:matches\s+)?(?:for\s+)?([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bresembl(?:e|es|ing)\s+([a-zA-Z\s]+)", re.IGNORECASE),
]

_STOP_WORDS = {
    "in", "from", "at", "who", "where", "with", "and", "or",
    "whose", "of", "having", "is", "are", "limit", "show", "find"
}


def is_fuzzy_intent(question: str) -> bool:
    """Detects whether the question indicates a request for similar or fuzzy name matching."""
    for pattern in _FUZZY_PATTERNS:
        if pattern.search(question):
            return True
    return False


def extract_fuzzy_target(question: str) -> str | None:
    """
    Extracts the name to search for from a fuzzy query.
    Stops extracting if it encounters a stop word (e.g. location prepositions).
    """
    for pattern in _FUZZY_PATTERNS:
        match = pattern.search(question)
        if match:
            raw_target = match.group(1).strip()
            words = raw_target.split()
            name_words = []
            for word in words:
                if word.lower() in _STOP_WORDS:
                    break
                name_words.append(word)
            if name_words:
                return " ".join(name_words).strip().title()
    return None


def fuzzy_rerank(
    df: pd.DataFrame,
    target_name: str,
    threshold: float = 0.80,
    max_rows: int = 30
) -> pd.DataFrame:
    """
    Re-scores rows in df by how well a name column matches target_name, then
    filters by threshold and returns the top max_rows sorted by score descending.

    Scoring strategy depends on the number of effective query tokens:

    Multi-word targets (≥ 2 tokens after stripping honorifics):
      Uses position-aware scoring via score_name_pair().
      First name carries the highest weight (1.0), middle names less (0.55),
      surname/last the least (0.40+). Each query token is matched against its
      best DB token using exact → phonetic → JW priority. An alignment bonus
      rewards in-order matches. A mild length penalty discounts DB names with
      many more tokens than the query.

    Single-word targets:
      Strategy 1: Full-string Jaro-Winkler against the DB name.
      Strategy 2: Per-DB-token JW with a length-difference guard (catches
                  "Geeta" inside "Geeta Devi").
      Strategy 3: Phonetic key floor at 0.90 (catches Poonam/Punam,
                  Shweta/Sweta, Vijay/Bijay).
    """
    if df.empty or not target_name:
        return df

    # Detect name column
    name_cols = ["member_name", "father_name", "mother_name", "spouse_name", "family_head_name"]
    df_cols_lower = {col.lower(): col for col in df.columns}

    match_col = None
    for col_key in name_cols:
        if col_key in df_cols_lower:
            match_col = df_cols_lower[col_key]
            break

    if not match_col:
        for col in df.columns:
            if "name" in col.lower():
                match_col = col
                break

    if not match_col:
        return df

    # Classify the query into weighted tokens
    query_tokens = classify_query_name(target_name)
    is_multi_word = len(query_tokens) > 1

    # Pre-compute values needed for the single-word path
    target_lower = target_name.lower()
    target_words = [w.strip() for w in target_lower.split() if w.strip()]
    max_len_diff = 2 if len(target_name) <= 5 else 3
    target_phonetic = phonetic_key(target_lower)
    target_phonetic_words = [w for w in target_phonetic.split() if w]

    scores: list[float] = []
    for val in df[match_col]:
        if pd.isna(val) or not isinstance(val, str):
            scores.append(0.0)
            continue

        val_clean = val.strip()

        if is_multi_word:
            scores.append(score_name_pair(query_tokens, val_clean))
        else:
            val_lower = val_clean.lower()
            val_words = [w.strip() for w in val_lower.split() if w.strip()]

            # Strategy 1: full-string JW
            full_score = JaroWinkler.similarity(target_lower, val_lower)
            if full_score > 1.0:
                full_score /= 100.0

            # Strategy 2: per-word JW with positional discount.
            # A match at position 0 (first name) is kept at full score; matches
            # further right are discounted so first-name matches rank above
            # middle- or surname matches for the same query word.
            _POS_DISCOUNT = [1.0, 0.92, 0.85, 0.80]
            best_word_score = 0.0
            t_word = query_tokens[0][0] if query_tokens else target_lower
            if len(t_word) == 1:
                # Initial match: check if any DB token starts with this char
                for pos, v_word in enumerate(val_words):
                    if v_word.startswith(t_word):
                        s = _score_token_pair(t_word, v_word)
                        s *= _POS_DISCOUNT[min(pos, len(_POS_DISCOUNT) - 1)]
                        if s > best_word_score:
                            best_word_score = s
            else:
                for pos, v_word in enumerate(val_words):
                    len_diff = abs(len(v_word) - len(t_word))
                    is_prefix_match = len(t_word) >= 4 and v_word.startswith(t_word)
                    if len_diff <= max_len_diff or is_prefix_match:
                        s = JaroWinkler.similarity(t_word, v_word)
                        if s > 1.0:
                            s /= 100.0
                        s *= _POS_DISCOUNT[min(pos, len(_POS_DISCOUNT) - 1)]
                        if s > best_word_score:
                            best_word_score = s

            # Strategy 3: phonetic key → floor at 0.90, with positional discount.
            # Full-string phonetic match (same word count) keeps 0.90.
            # Single-token phonetic match inside a multi-token DB name is
            # discounted by position so first-name phonetic matches rank higher.
            phonetic_score = 0.0
            if target_phonetic:
                val_phonetic = phonetic_key(val_lower)
                val_phonetic_words = [w for w in val_phonetic.split() if w]
                if val_phonetic == target_phonetic:
                    phonetic_score = 0.90
                elif len(target_phonetic_words) == 1:
                    pk = target_phonetic_words[0]
                    for pos, vp_word in enumerate(val_phonetic_words):
                        if pk == vp_word:
                            s = 0.90 * _POS_DISCOUNT[min(pos, len(_POS_DISCOUNT) - 1)]
                            if s > phonetic_score:
                                phonetic_score = s

            scores.append(max(full_score, best_word_score, phonetic_score))

    df_copy = df.copy()
    df_copy["similarity_score"] = scores
    df_copy = df_copy[df_copy["similarity_score"] >= threshold]
    df_copy = df_copy.sort_values(by="similarity_score", ascending=False)
    df_copy["similarity_score"] = df_copy["similarity_score"].round(2)
    return df_copy.head(max_rows)
