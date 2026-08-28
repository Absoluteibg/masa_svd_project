"""
svd_engine.py  —  SVD-based Subspace Divergence Score (SDS).

ALGORITHM
---------
1. Embed context sentences   → matrix C  (n × 384)
2. Embed output  sentences   → matrix G  (m × 384)
3. SVD of C: C = U_C Σ_C Vᵀ   →  P_C = U_C[:, :k]  (principal subspace)
4. SVD of G: G = U_G Σ_G Vᵀ   →  P_G = U_G[:, :k]
5. Cross-product              M = P_Cᵀ P_G            (k × k)
6. SVD of M                   →  singular values σ
7. SDS = 1 − σ_min     (worst-case misalignment ∈ [0, 1])
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd as scipy_svd
from typing import List

from src.config import config
from src.core.embeddings import embed_texts


class SVDSubspaceEngine:

    def __init__(self, k: int | None = None):
        self.k: int = k if k is not None else config.SVD_K

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _pad(sentences: List[str]) -> List[str]:
        """SVD needs ≥ 2 rows. Duplicate single-sentence inputs."""
        return sentences * 2 if len(sentences) == 1 else sentences

    @staticmethod
    def _principal_subspace(matrix: np.ndarray, k: int) -> np.ndarray:
        """
        Compute top-k left singular vectors of matrix.

        Parameters
        ----------
        matrix : np.ndarray  shape (n, d)
        k      : int          number of components to keep

        Returns
        -------
        np.ndarray  shape (n, min(k, n, d))
        """
        try:
            U, _, _ = scipy_svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback to numpy's SVD with a different LAPACK driver
            U, _, _ = np.linalg.svd(matrix, full_matrices=False)
        actual_k = min(k, U.shape[1])
        return U[:, :actual_k]

    # ── Public API ───────────────────────────────────────────────────────────

    def compute_sds(self,
                    context_sentences: List[str],
                    output_sentences:  List[str]) -> float:
        """
        Compute the Subspace Divergence Score between context and output.

        Returns
        -------
        float ∈ [0, 1]
            0 → perfectly aligned (likely correct)
            1 → maximally diverged (likely hallucinated)
        """
        ctx = self._pad(context_sentences)
        out = self._pad(output_sentences)

        C = embed_texts(ctx)
        G = embed_texts(out)

        k = min(self.k, C.shape[0], G.shape[0], C.shape[1])

        P_C = self._principal_subspace(C, k)
        P_G = self._principal_subspace(G, k)

        M = P_C.T @ P_G  # (k × k) cross-product
        try:
            _, sigma, _ = scipy_svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            _, sigma, _ = np.linalg.svd(M, full_matrices=False)

        sds = float(1.0 - float(np.min(sigma)))
        return round(float(np.clip(sds, 0.0, 1.0)), 4)

    def full_analysis(self,
                      context_sentences: List[str],
                      output_sentences:  List[str]) -> dict:
        """
        Extended SDS report — includes all singular values and matrix shapes.
        Useful for debugging and ablation studies.
        """
        ctx = self._pad(context_sentences)
        out = self._pad(output_sentences)

        C = embed_texts(ctx)
        G = embed_texts(out)
        k = min(self.k, C.shape[0], G.shape[0], C.shape[1])

        P_C = self._principal_subspace(C, k)
        P_G = self._principal_subspace(G, k)
        M   = P_C.T @ P_G

        try:
            _, sigma, _ = scipy_svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            _, sigma, _ = np.linalg.svd(M, full_matrices=False)

        sds = float(np.clip(1.0 - float(np.min(sigma)), 0.0, 1.0))
        return {
            "sds":          round(sds, 4),
            "sigma_values": [round(float(s), 4) for s in sigma],
            "sigma_min":    round(float(np.min(sigma)), 4),
            "sigma_max":    round(float(np.max(sigma)), 4),
            "sigma_mean":   round(float(np.mean(sigma)), 4),
            "k_used":       k,
            "context_shape": list(C.shape),
            "output_shape":  list(G.shape),
        }