"""
pipeline.py  —  Master MASA-SVD pipeline.

Connects embeddings → SVD engine → three agents → final verdict.
This is the single entry point for all analysis.
"""
from __future__ import annotations

import re
import logging
from typing import Any

from src.config import config
from src.core.svd_engine        import SVDSubspaceEngine
from src.agents.factual_agent   import FactualVerificationAgent
from src.agents.semantic_agent  import SemanticConsistencyAgent
from src.agents.calibration_agent import ConfidenceCalibrationAgent

log = logging.getLogger(__name__)


def _split_sentences(text: str) -> list[str]:
    """
    Split text into a list of non-empty sentences.
    Handles '.', '!', '?' as sentence boundaries.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    result = [p.strip() for p in parts if len(p.strip()) > 3]
    return result if result else [text.strip()]  # fallback: whole text as one


class MASASVDPipeline:
    """
    The complete MASA-SVD pipeline.

    Instantiate once, then call .analyze() for each query-response pair.
    All models are loaded on first instantiation; subsequent calls are fast.

    Parameters
    ----------
    svd_k     : Number of SVD principal components (default from config).
    use_nli   : Whether to use the NLI model in SCA (default True).
    alpha/beta/gamma : Agent weights (default from config).
    threshold : Hallucination decision threshold (default from config).
    """

    def __init__(
        self,
        svd_k:     int   = config.SVD_K,
        use_nli:   bool  = True,
        alpha:     float = config.ALPHA,
        beta:      float = config.BETA,
        gamma:     float = config.GAMMA,
        threshold: float = config.HALLUCINATION_THRESHOLD,
    ) -> None:
        print("[Pipeline] Initialising MASA-SVD …")
        self._svd = SVDSubspaceEngine(k=svd_k)
        self._fva = FactualVerificationAgent()
        self._sca = SemanticConsistencyAgent(use_nli=use_nli)
        self._cca = ConfidenceCalibrationAgent(
            alpha=alpha, beta=beta, gamma=gamma, threshold=threshold
        )
        print("[Pipeline] Ready.\n")

    # ── Public API ───────────────────────────────────────────────────────────

    def analyze(self, query: str, llm_response: str,
                use_nli: bool | None = None) -> dict[str, Any]:
        """
        Run the full MASA-SVD analysis on one query-response pair.

        Parameters
        ----------
        query        : The user's original question.
        llm_response : The LLM-generated answer to evaluate.
        use_nli      : Override instance-level use_nli for this call only.

        Returns
        -------
        dict with keys:
            verdict, final_risk_score, action, breakdown, svd_analysis,
            query, llm_response
        """
        # ── 1. Sentence splitting ────────────────────────────────────────────
        ctx_sentences = _split_sentences(query)
        out_sentences = _split_sentences(llm_response)

        # ── 2. SVD subspace divergence ───────────────────────────────────────
        sds          = self._svd.compute_sds(ctx_sentences, out_sentences)
        svd_analysis = self._svd.full_analysis(ctx_sentences, out_sentences)

        # ── 3. Factual verification (API call) ───────────────────────────────
        fva_score = self._fva.verify(query, llm_response)

        # ── 4. Semantic consistency ──────────────────────────────────────────
        _use_nli  = use_nli if use_nli is not None else self._sca.use_nli
        old_nli   = self._sca.use_nli
        self._sca.use_nli = _use_nli
        sca_score = self._sca.check(query, llm_response)
        self._sca.use_nli = old_nli

        # ── 5. Aggregation ───────────────────────────────────────────────────
        result = self._cca.aggregate(sds, fva_score, sca_score)

        return {
            **result.to_dict(),
            "svd_analysis": svd_analysis,
            "query":        query,
            "llm_response": llm_response,
        }