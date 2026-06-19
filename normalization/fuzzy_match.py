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


# Compile regex patterns for fuzzy name queries
_FUZZY_PATTERNS = [
    re.compile(r"\bsimilar\s+to\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bname(?:s)?\s+(?:is\s+)?like\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bsound(?:s)?\s+like\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bspell(?:ed)?\s+like\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bfuzzy\s+(?:search\s+)?(?:for\s+)?([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bapproximate\s+(?:matches\s+)?(?:for\s+)?([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bresembl(?:e|es|ing)\s+([a-zA-Z\s]+)", re.IGNORECASE),
]

# Words that indicate a stop in the extracted target name
_STOP_WORDS = {
    "in", "from", "at", "who", "where", "with", "and", "or",
    "whose", "of", "having", "is", "are", "limit", "show", "find"
}


def is_fuzzy_intent(question: str) -> bool:
    """
    Detects whether the question indicates a request for similar or fuzzy name matching.
    """
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
    Calculates similarity scores between target_name and values in the
    first detected name column of the DataFrame. Filters by threshold, sorts descending,
    and returns up to max_rows.

    Three scoring strategies are combined (best wins):
      1. Full-string JW: compare entire DB name against entire target.
      2. Per-word JW: for single-word targets, compare against each word in multi-word names.
      3. Phonetic key: if phonetic_key(target_word) matches any phonetic_key(db_name_word),
         floor score at 0.90. Catches romanization variants JW misses (Poonam/Punam,
         Shweta/Sweta, Vijay/Bijay, Geeta/Gita).
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
        # Fallback to first column containing 'name'
        for col in df.columns:
            if "name" in col.lower():
                match_col = col
                break

    if not match_col:
        return df

    target_lower = target_name.lower()
    target_words = [w.strip() for w in target_lower.split() if w.strip()]
    max_len_diff = 2 if len(target_name) <= 5 else 3

    # Pre-compute phonetic info for the target once
    target_phonetic = phonetic_key(target_lower)
    target_phonetic_words = [w for w in target_phonetic.split() if w]

    scores = []
    for val in df[match_col]:
        if pd.isna(val) or not isinstance(val, str):
            scores.append(0.0)
        else:
            val_clean = val.strip()
            val_lower = val_clean.lower()
            val_words = [w.strip() for w in val_lower.split() if w.strip()]

            # Strategy 1: full-string JW score
            # "Geeta Devi" vs "Geeta Devi" → 1.0 (exact)
            # "Geeta Devi" vs "Geeta"        → ~0.77 (different — surname missing)
            # "Geeta Devi" vs "Geeta Choudhary" → ~0.82 (different surname)
            full_score = JaroWinkler.similarity(target_lower, val_lower)
            if full_score > 1.0:
                full_score = full_score / 100.0

            # Strategy 2: per-word JW score — ONLY for single-word targets.
            # When the user types one word (e.g. "Geeta"), match it against each
            # word in a multi-word DB name so "Geeta Devi" is still found.
            # NOT used for multi-word targets: otherwise "Geeta" (DB) would score
            # 1.0 against target "Geeta Devi" by matching just the first word.
            best_word_score = 0.0
            if len(target_words) == 1:
                t_word = target_words[0]
                for v_word in val_words:
                    len_diff = abs(len(v_word) - len(t_word))
                    is_prefix_match = len(t_word) >= 4 and v_word.startswith(t_word)
                    if len_diff <= max_len_diff or is_prefix_match:
                        score = JaroWinkler.similarity(t_word, v_word)
                        if score > 1.0:
                            score = score / 100.0
                        if score > best_word_score:
                            best_word_score = score

            # Strategy 3: phonetic key match — floors score at 0.90.
            # Catches systematic romanization variants where JW alone scores just
            # below the threshold (e.g. Poonam/Punam JW ≈ 0.84 at threshold 0.88).
            phonetic_score = 0.0
            if target_phonetic:
                val_phonetic = phonetic_key(val_lower)
                val_phonetic_words = [w for w in val_phonetic.split() if w]
                if val_phonetic == target_phonetic:
                    # Full-name phonetic match (same number of words)
                    phonetic_score = 0.90
                elif len(target_phonetic_words) == 1 and target_phonetic_words[0] in val_phonetic_words:
                    # Single-word target whose phonetic key matches a word in a multi-word DB name
                    phonetic_score = 0.90

            scores.append(max(full_score, best_word_score, phonetic_score))

    df_copy = df.copy()
    df_copy["similarity_score"] = scores
    df_copy = df_copy[df_copy["similarity_score"] >= threshold]
    df_copy = df_copy.sort_values(by="similarity_score", ascending=False)
    df_copy["similarity_score"] = df_copy["similarity_score"].round(2)
    return df_copy.head(max_rows)
