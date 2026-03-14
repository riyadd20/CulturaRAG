import os
import pickle
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from loguru import logger
from app.core.config import get_settings

settings = get_settings()

def _simple_embed(text: str) -> List[float]:
    words = text.lower().split()
    vec = np.zeros(384)
    for i, word in enumerate(words):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % 384
        vec[idx] += 1.0 / (i + 1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

class VectorStoreService:
    def __init__(self):
        self._chunks: List[Dict[str, Any]] = []
        self._embeddings: List[List[float]] = []
        self._store_path = Path(settings.faiss_index_path)
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        p = self._store_path / "store.pkl"
        if p.exists():
            with open(p, "rb") as f:
                data = pickle.load(f)
                self._chunks = data.get("chunks", [])
                self._embeddings = data.get("embeddings", [])
            logger.info(f"Loaded {len(self._chunks)} chunks from store.")

    def _persist(self):
        with open(self._store_path / "store.pkl", "wb") as f:
            pickle.dump({"chunks": self._chunks, "embeddings": self._embeddings}, f)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        for chunk in chunks:
            emb = _simple_embed(chunk["content"])
            self._chunks.append(chunk)
            self._embeddings.append(emb)
        self._persist()
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self._chunks)}")
        return len(chunks)

    def search(self, query: str, top_k: int = 5, culture_filter=None, language_filter=None):
        if not self._chunks:
            return []
        q_emb = _simple_embed(query)
        scored = []
        for chunk, emb in zip(self._chunks, self._embeddings):
            if culture_filter and chunk.get("culture","").lower() != culture_filter.lower():
                continue
            score = _cosine_similarity(q_emb, emb)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_stats(self):
        cultures = list({c.get("culture","unknown") for c in self._chunks if c.get("culture")})
        languages = list({c.get("language","en") for c in self._chunks if c.get("language")})
        return {
            "total_chunks": len(self._chunks),
            "total_documents": len({c.get("source") for c in self._chunks}),
            "cultures_indexed": sorted(cultures),
            "languages_indexed": sorted(languages),
            "index_size_mb": 0.0,
            "last_updated": None,
        }

_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store
