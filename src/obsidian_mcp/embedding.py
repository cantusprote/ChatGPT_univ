from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol


class Embedder(Protocol):
    model_name: str

    def documents(self, texts: list[str]) -> list[list[float]]: ...

    def query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, cache_folder: str | None = None):
        # Transformers 5 may perform a model-info lookup even when all files are
        # cached. Force the first attempt to be genuinely offline.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        try:
            self._model = SentenceTransformer(
                model_name, cache_folder=cache_folder, local_files_only=True
            )
        except OSError:
            # A new independent installation has no model cache yet. Download it
            # once into this project's data directory; later starts stay local.
            self._model = SentenceTransformer(model_name, cache_folder=cache_folder)

    def documents(self, texts: list[str]) -> list[list[float]]:
        values = [f"passage: {text}" for text in texts]
        return self._model.encode(values, normalize_embeddings=True).tolist()

    def query(self, text: str) -> list[float]:
        return self._model.encode(
            [f"query: {text}"], normalize_embeddings=True
        )[0].tolist()


class HashEmbedder:
    """Small deterministic embedder used only by tests."""

    model_name = "test-hash-embedder"

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:2], "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def query(self, text: str) -> list[float]:
        return self._one(text)
