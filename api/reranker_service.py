#!/usr/bin/env python3
"""
Optional Reranker Service for OSRS RAG
- Uses BAAI/bge-reranker-large via FlagEmbedding if installed
- Falls back to disabled (no-op) if dependency or model is unavailable

Enable via env:
  OSRS_USE_RERANKER=1            # default 1
  OSRS_RERANKER_MODEL=BAAI/bge-reranker-large

Install dependency (requires permission):
  pip install FlagEmbedding
"""
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    # FlagEmbedding provides a light-weight cross-encoder reranker
    from FlagEmbedding import FlagReranker  # type: ignore
except Exception:
    FlagReranker = None  # Dependency not present


class RerankerService:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", use_fp16: bool = True):
        self.model_name = model_name
        self.available = False
        self._reranker = None

        if FlagReranker is None:
            logger.warning("Reranker disabled: FlagEmbedding not installed")
            return

        try:
            # Loads model weights on first use; large model (~1.3GB)
            self._reranker = FlagReranker(model_name, use_fp16=use_fp16)
            self.available = True
            logger.info(f"Initialized reranker: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize reranker '{model_name}': {e}")
            self.available = False
            self._reranker = None

    def score(self, query: str, docs: List[str], normalize: bool = True) -> List[float]:
        """Return cross-encoder relevance scores for (query, doc) pairs.
        If unavailable, returns zeros to allow graceful fallback.
        """
        if not self.available or self._reranker is None:
            return [0.0] * len(docs)

        try:
            pairs = [[query, d] for d in docs]
            scores = self._reranker.compute_score(pairs, normalize=normalize)
            return [float(s) for s in scores]
        except Exception as e:
            logger.error(f"Reranker scoring failed: {e}")
            return [0.0] * len(docs)

