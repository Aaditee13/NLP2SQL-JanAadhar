from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config.settings import settings
from database.schema_metadata import RAJASTHAN_DISTRICTS_41, RAJASTHAN_CITIES, RAJASTHAN_BLOCKS
from embeddings.ollama_embeddings import OllamaEmbedder


# Lowercase lookup sets
_DISTRICTS_SET = {d.lower() for d in RAJASTHAN_DISTRICTS_41}
_CITIES_SET = {c.lower() for c in RAJASTHAN_CITIES}
_BLOCKS_SET = {b.lower() for b in RAJASTHAN_BLOCKS}
_LOCATIONS_SET = _DISTRICTS_SET | _CITIES_SET | _BLOCKS_SET


def _extract_numbers_and_operators(text: str) -> str:
    """Extract numbers, math symbols, and comparison keywords to distinguish filters."""
    digits = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    words = [
        "above", "below", "greater", "less", "more", "fewer", "between", "under", "over",
        "at least", "at most", "exactly", "older", "younger"
    ]
    pattern = r'\b(?:' + '|'.join(words) + r')\b'
    found_words = re.findall(pattern, text.lower())
    return ",".join(sorted(digits + found_words))


def _extract_intent(text: str) -> str:
    """Determine if query is 'count' or 'list' to prevent cache collisions."""
    lowered = text.lower()
    if any(w in lowered for w in ["how many", "count", "total", "number of"]):
        return "count"
    if any(w in lowered for w in ["list", "show", "who", "find", "get", "details"]):
        return "list"
    return "unknown"


def _extract_gender(text: str) -> str:
    """Identify explicit gender filters."""
    import re
    lowered = text.lower()
    words = set(re.findall(r"\b[a-zA-Z]+\b", lowered))
    male_terms = {"male", "boy", "man", "men", "gent", "gentleman"}
    female_terms = {"female", "girl", "woman", "women", "lady", "widow"}
    has_male = bool(words & male_terms)
    has_female = bool(words & female_terms)
    if has_male and has_female:
        return "both"
    if has_male:
        return "male"
    if has_female:
        return "female"
    return "none"


def _extract_locations(text: str) -> str:
    """Extract Rajasthan geographical proper nouns mentioned in the question."""
    words = set(re.findall(r"\b[a-zA-Z-]+\b", text.lower()))
    found = words & _LOCATIONS_SET
    return ",".join(sorted(list(found)))


def _prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()[:32]


class FaissCacheStore:
    def __init__(
        self,
        index_path: Path = settings.data_dir / "cache.faiss",
        metadata_path: Path = settings.data_dir / "cache_metadata.json",
        embedder: OllamaEmbedder | None = None,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedder = embedder or OllamaEmbedder()
        self.index: faiss.Index | None = None
        self.registry: list[dict[str, Any]] = []

    def build_empty(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        # nomic-embed-text has 768 dimensions
        self.index = faiss.IndexFlatIP(768)
        faiss.write_index(self.index, str(self.index_path))
        self.registry = []
        self.metadata_path.write_text("[]", encoding="utf-8")

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            self.build_empty()
            return
        self.index = faiss.read_index(str(self.index_path))
        self.registry = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def add(self, question: str, sql: str, metadata: dict[str, str]) -> None:
        if self.index is None:
            self.load()
        assert self.index is not None
        
        # Prevent duplicate entries by hash ID
        q_id = _prompt_id(question)
        for entry in self.registry:
            if entry["id"] == q_id:
                return
                
        vector = self.embedder.embed(question).reshape(1, -1)
        self.index.add(vector)
        faiss.write_index(self.index, str(self.index_path))
        
        self.registry.append({
            "id": q_id,
            "question": question,
            "sql": sql,
            "metadata": metadata
        })
        self.metadata_path.write_text(json.dumps(self.registry, indent=2), encoding="utf-8")


class SemanticCache:
    def __init__(self, store: FaissCacheStore | None = None):
        self.cache_store = store or FaissCacheStore()
        
    def lookup(self, question: str) -> str | None:
        try:
            if self.cache_store.index is None:
                self.cache_store.load()
            
            if not self.cache_store.registry:
                return None

            # 1. Exact string match first (O(1))
            q_id = _prompt_id(question)
            for entry in self.cache_store.registry:
                if entry["id"] == q_id:
                    return entry["sql"]

            # 2. Semantic Search Match
            query_vector = self.cache_store.embedder.embed(question).reshape(1, -1)
            scores, indexes = self.cache_store.index.search(query_vector, 1)
            
            if len(scores) == 0 or len(indexes) == 0:
                return None
                
            score = float(scores[0][0])
            idx = int(indexes[0][0])
            
            # cosine similarity threshold limit
            if idx < 0 or idx >= len(self.cache_store.registry) or score < 0.95:
                return None
                
            matched_entry = self.cache_store.registry[idx]
            
            # 3. Guardrail validation checks
            incoming_numbers = _extract_numbers_and_operators(question)
            cached_numbers = matched_entry["metadata"].get("numbers", "")
            if incoming_numbers != cached_numbers:
                return None

            incoming_intent = _extract_intent(question)
            cached_intent = matched_entry["metadata"].get("intent", "unknown")
            if incoming_intent != "unknown" and cached_intent != "unknown" and incoming_intent != cached_intent:
                return None

            incoming_gender = _extract_gender(question)
            cached_gender = matched_entry["metadata"].get("gender", "none")
            if incoming_gender != "none" and cached_gender != "none" and incoming_gender != cached_gender:
                return None

            incoming_locs = _extract_locations(question)
            cached_locs = matched_entry["metadata"].get("locations", "")
            if incoming_locs != cached_locs:
                return None
                
            return matched_entry["sql"]
        except Exception:
            return None

    def store(self, question: str, sql: str) -> None:
        try:
            if self.cache_store.index is None:
                self.cache_store.load()
            
            metadata = {
                "numbers": _extract_numbers_and_operators(question),
                "intent": _extract_intent(question),
                "gender": _extract_gender(question),
                "locations": _extract_locations(question),
            }
            self.cache_store.add(question, sql, metadata)
        except Exception:
            pass
