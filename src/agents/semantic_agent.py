"""
semantic_agent.py  —  Agent 2: Semantic Consistency Agent (SCA)

Measures whether the response addresses the query at all.
Combines two signals:
  - Cosine similarity  (fast, geometric)
  - NLI entailment     (slower, linguistically grounded)

NLI model loads on first use and is cached globally (~180 MB download).
Set use_nli=False for fast development runs.
"""
from __future__ import annotations

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from src.core.embeddings import embed_single, embed_texts
from src.config import config

log = logging.getLogger(__name__)

# ── NLI model singleton ──────────────────────────────────────────────────────
_nli_model = None

# Label index mapping for cross-encoder/nli-deberta-v3-small
# id2label: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
_ENTAILMENT_IDX = 1


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"[SCA] Loading NLI model '{config.NLI_MODEL}' "
                  "(first time ~180 MB download) ...")
            _nli_model = CrossEncoder(config.NLI_MODEL, num_labels=3)
            print("[SCA] NLI model ready.")
        except Exception as e:
            log.warning("SCA: Could not load NLI model (%s). "
                        "Falling back to cosine-only mode.", e)
            _nli_model = None
    return _nli_model


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class SemanticConsistencyAgent:

    def __init__(self, use_nli: bool = True) -> None:
        """
        Parameters
        ----------
        use_nli : bool
            If False, only cosine similarity is used (fast, no model download).
        """
        self.use_nli = use_nli

    # ── Sub-scores ───────────────────────────────────────────────────────────

    def _cosine(self, query: str, response: str) -> float:
        q = embed_single(query).reshape(1, -1)
        r = embed_single(response).reshape(1, -1)
        return float(round(float(sk_cosine(q, r)[0][0]), 4))

    def _nli_entailment(self, query: str, response: str) -> float:
        model = _get_nli_model()
        if model is None:
            return self._cosine(query, response)
        try:
            raw = model.predict([(query, response)])   # shape (1, 3)
            probs = _softmax(raw[0])
            return float(round(float(probs[_ENTAILMENT_IDX]), 4))
        except Exception as e:
            log.warning("SCA NLI predict failed (%s). Using cosine.", e)
            return self._cosine(query, response)

    # ── Public API ───────────────────────────────────────────────────────────

    def check(self, query: str, response: str) -> float:
        """
        Returns semantic consistency score ∈ [0, 1].
        1.0 → response is fully consistent with / entailed by the query.
        0.0 → response is off-topic or contradictory.
        """
        cos = self._cosine(query, response)

        if not self.use_nli:
            return cos

        nli = self._nli_entailment(query, response)
        combined = 0.40 * cos + 0.60 * nli
        return round(float(np.clip(combined, 0.0, 1.0)), 4)