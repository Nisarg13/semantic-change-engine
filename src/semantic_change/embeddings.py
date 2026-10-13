"""Stage 2 - dense vector mapping: one sentence-transformer vector for each unit.

The vectors are L2-normalized, so the cosine similarity of two units is the dot product
of their vectors. Stage 3 uses the similarity to find the units that correspond. The
similarity does not show if the meaning changed: "Users may share their credentials."
and "Users may not share their credentials." have a similarity of 0.88. Stage 4 judges
the meaning.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


class SentenceEmbedder:
    """Encodes texts with a sentence-transformers model. Threads can share one instance."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = model
        self._encoder: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the model on the first call. The first load downloads it from Hugging Face."""
        with self._lock:
            if self._encoder is None:
                # Imported here because the import loads PyTorch, which takes seconds.
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.model)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return one normalized vector for each text, as a (len(texts), dim) array."""
        self.load()
        with self._lock:
            vectors = self._encoder.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            )
        return np.asarray(vectors, dtype=np.float32)
