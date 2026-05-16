"""
JSONL-based vector store with configurable similarity metrics.
Lightweight, inspectable, no external DB dependency.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

VECTOR_STORE_PATH = Path("data/vectors.jsonl")
VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return -float(np.linalg.norm(a - b))  # negative so higher = more similar


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


METRICS = {
    "cosine": cosine_similarity,
    "l2": l2_distance,
    "dot": dot_product,
}


class VectorStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or VECTOR_STORE_PATH
        self.entries: list[dict] = []
        self._load()

    def _load(self):
        """Load all vectors from JSONL file."""
        self.entries = []
        if self.path.exists():
            with open(self.path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))

    def add(self, text: str, embedding: list[float], source: str, chunk_index: int, metadata: dict = {}):
        """Add a single chunk + embedding to the store."""
        entry = {
            "text": text,
            "embedding": embedding,
            "source": source,
            "chunk_index": chunk_index,
            "metadata": metadata,
        }
        self.entries.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metric: str = "cosine",
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for the most similar chunks.
        Returns list of {text, score, source, chunk_index}.
        """
        if metric not in METRICS:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(METRICS.keys())}")

        sim_fn = METRICS[metric]
        q_vec = np.array(query_embedding, dtype=np.float32)

        scored = []
        for entry in self.entries:
            if source_filter and entry.get("source") != source_filter:
                continue
            e_vec = np.array(entry["embedding"], dtype=np.float32)
            score = sim_fn(q_vec, e_vec)
            scored.append({
                "text": entry["text"],
                "score": round(score, 4),
                "source": entry["source"],
                "chunk_index": entry["chunk_index"],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def clear(self):
        """Wipe the store."""
        self.entries = []
        if self.path.exists():
            self.path.unlink()

    def stats(self) -> dict:
        sources = {}
        for e in self.entries:
            src = e.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {
            "total_chunks": len(self.entries),
            "sources": sources,
            "embedding_dim": len(self.entries[0]["embedding"]) if self.entries else 0,
        }


# Module-level singleton
_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
