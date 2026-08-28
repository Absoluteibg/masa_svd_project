"""
test_pipeline.py  —  Integration tests for the full MASA-SVD pipeline.

All external calls (OpenAI, NLI model) are mocked.
Run with:  pytest tests/test_pipeline.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock

from src.pipeline import MASASVDPipeline, _split_sentences


# ── Sentence splitter ─────────────────────────────────────────────────────────

class TestSplitSentences:

    def test_basic_split(self):
        r = _split_sentences("Hello world. This is a test. Final sentence.")
        assert len(r) == 3

    def test_single_sentence(self):
        r = _split_sentences("Just one sentence here.")
        assert len(r) == 1

    def test_empty_fallback(self):
        r = _split_sentences("Hi")
        assert r == ["Hi"]

    def test_question(self):
        r = _split_sentences("Who won? Argentina did. By penalties!")
        assert len(r) == 3


# ── Pipeline integration ──────────────────────────────────────────────────────

def _make_pipeline_with_mocks(fva_score: float = 0.8,
                              sca_score: float = 0.75) -> MASASVDPipeline:
    """Create a pipeline with FVA and SCA NLI mocked."""
    pipeline = MASASVDPipeline(use_nli=False)   # skip NLI model load
    # Patch FVA to return fixed score
    pipeline._fva.verify = lambda q, r: fva_score
    # Patch SCA to return fixed score  
    pipeline._sca.check  = lambda q, r: sca_score
    return pipeline


class TestPipeline:

    def test_result_has_required_keys(self):
        p = _make_pipeline_with_mocks()
        r = p.analyze("What is AI?", "AI is artificial intelligence.")
        required = {"verdict", "final_risk_score", "action",
                    "breakdown", "svd_analysis", "query", "llm_response"}
        assert required.issubset(r.keys())

    def test_verdict_is_valid_enum(self):
        p = _make_pipeline_with_mocks()
        r = p.analyze("Test question.", "Test answer.")
        assert r["verdict"] in ("VERIFIED", "UNCERTAIN", "HALLUCINATED")

    def test_risk_score_in_range(self):
        p = _make_pipeline_with_mocks()
        r = p.analyze("Capital of France?", "Paris is the capital.")
        assert 0.0 <= r["final_risk_score"] <= 1.0

    def test_verified_scenario(self):
        """High FVA + high SCA + identical text → should be VERIFIED."""
        p = _make_pipeline_with_mocks(fva_score=0.97, sca_score=0.95)
        text = "Paris is the capital of France."
        r = p.analyze(text, text)
        # SDS will be near 0 (identical text); FVA and SCA are high
        assert r["verdict"] in ("VERIFIED", "UNCERTAIN")

    def test_hallucinated_scenario(self):
        """Low FVA + low SCA → should be HALLUCINATED or UNCERTAIN."""
        p = _make_pipeline_with_mocks(fva_score=0.05, sca_score=0.10)
        r = p.analyze(
            "Who won the 2022 FIFA World Cup?",
            "Brazil won the 2022 FIFA World Cup defeating France 3-0."
        )
        assert r["verdict"] in ("HALLUCINATED", "UNCERTAIN")

    def test_breakdown_keys(self):
        p = _make_pipeline_with_mocks()
        r = p.analyze("Q", "A")
        bd = r["breakdown"]
        for key in ("sds", "fva_score", "sca_score",
                    "sds_risk", "fva_risk", "sca_risk"):
            assert key in bd, f"Missing key: {key}"

    def test_svd_analysis_keys(self):
        p = _make_pipeline_with_mocks()
        r = p.analyze("What is gravity?", "Gravity is a force.")
        svd = r["svd_analysis"]
        for key in ("sds", "sigma_values", "k_used"):
            assert key in svd, f"Missing SVD key: {key}"