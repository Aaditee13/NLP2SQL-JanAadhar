from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config.settings import settings
from embeddings.ollama_embeddings import OllamaEmbedder


class FaissFewShotStore:
    def __init__(
        self,
        json_path: Path = settings.few_shot_json_path,
        index_path: Path = settings.few_shot_faiss_path,
        metadata_path: Path = settings.few_shot_metadata_path,
        embedder: OllamaEmbedder | None = None,
    ):
        self.json_path = json_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedder = embedder or OllamaEmbedder()
        self.index: faiss.Index | None = None
        self.examples: list[dict[str, Any]] = []

    def build(self, force: bool = False) -> None:
        if self.index_path.exists() and self.metadata_path.exists() and not force:
            self.load()
            return
        
        if not self.json_path.exists():
            raise FileNotFoundError(f"Few-shots pool JSON not found at {self.json_path}")
            
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, encoding="utf-8") as f:
            self.examples = json.load(f)
            
        if not self.examples:
            return

        vectors = self.embedder.embed_many([ex["question"] for ex in self.examples])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.examples, indent=2), encoding="utf-8")

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self.examples = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self.index is None:
            try:
                self.build()
            except Exception:
                # If building fails (e.g. Ollama offline), return default static examples
                if self.json_path.exists():
                    with open(self.json_path, encoding="utf-8") as f:
                        return json.load(f)[:top_k]
                return []
                
        assert self.index is not None
        query_vector = self.embedder.embed(query).reshape(1, -1)
        scores, indexes = self.index.search(query_vector, top_k)
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indexes[0]):
            if idx < 0 or idx >= len(self.examples):
                continue
            ex = dict(self.examples[idx])
            ex["score"] = float(score)
            results.append(ex)
        return results


class FewShotRetriever:
    def __init__(self, store: FaissFewShotStore | None = None):
        self.store = store or FaissFewShotStore()

    def retrieve(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        try:
            return self.store.search(question, top_k=top_k)
        except Exception:
            # Fallback to first K elements from JSON
            try:
                if self.store.json_path.exists():
                    with open(self.store.json_path, encoding="utf-8") as f:
                        return json.load(f)[:top_k]
            except Exception:
                pass
            return []
