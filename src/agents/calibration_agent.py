"""
calibration_agent.py  —  Agent 3: Confidence Calibration Agent (CCA)

Aggregates SDS, FVA score, and SCA score into a single risk score
and produces a human-readable verdict.

Formula
-------
risk = α · SDS_risk + β · FVA_risk + γ · SCA_risk

Where:
  SDS_risk  = SDS                  (already risk-oriented)
  FVA_risk  = 1 − FVA_score       (low factual confidence → high risk)
  SCA_risk  = 1 − SCA_score       (low semantic consistency → high risk)
  α = 0.40,  β = 0.35,  γ = 0.25
"""
from __future__ import annotations

from dataclasses import dataclass
from src.config import config


@dataclass
class VerificationResult:
    """Structured output from CCA."""
    verdict:          str    # "VERIFIED" | "UNCERTAIN" | "HALLUCINATED"
    final_risk_score: float
    action:           str
    sds:              float
    fva_score:        float
    sca_score:        float
    sds_risk:         float
    fva_risk:         float
    sca_risk:         float

    def to_dict(self) -> dict:
        return {
            "verdict":          self.verdict,
            "final_risk_score": self.final_risk_score,
            "action":           self.action,
            "breakdown": {
                "sds":      self.sds,
                "fva_score": self.fva_score,
                "sca_score": self.sca_score,
                "sds_risk": self.sds_risk,
                "fva_risk": self.fva_risk,
                "sca_risk": self.sca_risk,
            },
        }


class ConfidenceCalibrationAgent:

    def __init__(
        self,
        alpha:     float = config.ALPHA,
        beta:      float = config.BETA,
        gamma:     float = config.GAMMA,
        threshold: float = config.HALLUCINATION_THRESHOLD,
        uncertain: float = config.UNCERTAIN_THRESHOLD,
    ) -> None:
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            f"Weights must sum to 1.0 (got {alpha+beta+gamma:.4f})"
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma
        self.threshold = threshold
        self.uncertain = uncertain

    def aggregate(self, sds: float, fva_score: float,
                  sca_score: float) -> VerificationResult:
        """
        Parameters
        ----------
        sds       : Subspace Divergence Score (risk-oriented, ↑ = more risky)
        fva_score : Factual confidence (quality-oriented, ↑ = more confident)
        sca_score : Semantic consistency (quality-oriented, ↑ = more consistent)

        Returns
        -------
        VerificationResult with verdict, score, and full breakdown.
        """
        sds_risk = float(sds)
        fva_risk = float(1.0 - fva_score)
        sca_risk = float(1.0 - sca_score)

        risk = (self.alpha * sds_risk +
                self.beta  * fva_risk  +
                self.gamma * sca_risk)
        risk = round(float(min(max(risk, 0.0), 1.0)), 4)

        if risk > self.threshold:
            verdict = "HALLUCINATED"
            action  = "Reject or regenerate. Do NOT show to user as-is."
        elif risk > self.uncertain:
            verdict = "UNCERTAIN"
            action  = "Show with low-confidence warning. Consider regenerating."
        else:
            verdict = "VERIFIED"
            action  = "Safe to present to user."

        return VerificationResult(
            verdict          = verdict,
            final_risk_score = risk,
            action           = action,
            sds              = round(sds,       4),
            fva_score        = round(fva_score, 4),
            sca_score        = round(sca_score, 4),
            sds_risk         = round(sds_risk,  4),
            fva_risk         = round(fva_risk,  4),
            sca_risk         = round(sca_risk,  4),
        )