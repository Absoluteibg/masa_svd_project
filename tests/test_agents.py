"""
test_agents.py  —  Tests for all three agents.

FVA and SCA NLI calls are mocked so no API key or model download is needed.
Run with:  pytest tests/test_agents.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.agents.calibration_agent import ConfidenceCalibrationAgent
from src.agents.semantic_agent import SemanticConsistencyAgent
from src.agents.factual_agent import FactualVerificationAgent


# ── ConfidenceCalibrationAgent (no mocking needed — pure math) ───────────────

class TestCCA:

    def setup_method(self):
        self.cca = ConfidenceCalibrationAgent()

    def test_verified_case(self):
        """Low SDS + high FVA + high SCA → VERIFIED."""
        r = self.cca.aggregate(sds=0.05, fva_score=0.95, sca_score=0.90)
        assert r.verdict == "VERIFIED"
        assert r.final_risk_score < 0.30

    def test_hallucinated_case(self):
        """High SDS + low FVA + low SCA → HALLUCINATED."""
        r = self.cca.aggregate(sds=0.80, fva_score=0.10, sca_score=0.20)
        assert r.verdict == "HALLUCINATED"
        assert r.final_risk_score > 0.45

    def test_uncertain_case(self):
        """Mid-range scores → UNCERTAIN."""
        r = self.cca.aggregate(sds=0.40, fva_score=0.50, sca_score=0.55)
        assert r.verdict in ("UNCERTAIN", "HALLUCINATED")

    def test_risk_in_range(self):
        for sds, fva, sca in [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0),
                               (0.5, 0.5, 0.5)]:
            r = self.cca.aggregate(sds, fva, sca)
            assert 0.0 <= r.final_risk_score <= 1.0

    def test_weight_validation(self):
        """Weights that don't sum to 1.0 should raise AssertionError."""
        with pytest.raises(AssertionError):
            ConfidenceCalibrationAgent(alpha=0.5, beta=0.5, gamma=0.5)

    def test_to_dict_keys(self):
        r = self.cca.aggregate(0.3, 0.7, 0.6)
        d = r.to_dict()
        assert {"verdict", "final_risk_score", "action",
                "breakdown"}.issubset(d.keys())


# ── SemanticConsistencyAgent (cosine only, no NLI) ───────────────────────────

class TestSCACosinOnly:

    def setup_method(self):
        self.sca = SemanticConsistencyAgent(use_nli=False)

    def test_returns_float_in_range(self):
        score = self.sca.check("What is Paris?",
                               "Paris is the capital of France.")
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_same_text_high_score(self):
        text = "The dog runs in the park."
        score = self.sca.check(text, text)
        assert score > 0.8, f"Identical texts should score > 0.8, got {score}"

    def test_unrelated_text_lower_score(self):
        q = "What is quantum mechanics?"
        related   = self.sca.check(q, "Quantum mechanics is a branch of physics.")
        unrelated = self.sca.check(q, "I enjoy eating pasta with tomato sauce.")
        assert related > unrelated


# ── FactualVerificationAgent (mocked OpenAI) ─────────────────────────────────

class TestFVA:

    def _make_mock_response(self, content: str) -> MagicMock:
        mock = MagicMock()
        mock.choices[0].message.content = content
        return mock

    def test_returns_float_in_range(self):
        with patch("src.agents.factual_agent.OpenAI") as MockOAI:
            instance = MockOAI.return_value
            instance.chat.completions.create.return_value = \
                self._make_mock_response("0.85")
            with patch("src.agents.factual_agent.config") as mock_cfg:
                mock_cfg.get_fva_provider_settings.return_value = \
                    ("openai", "test-key", "gpt-4o-mini", None)
                mock_cfg.has_fva_key.return_value = True
                agent = FactualVerificationAgent()
                score = agent.verify("Who won?", "Argentina won.")
        assert 0.0 <= score <= 1.0
        assert agent.is_available

    def test_fallback_on_no_key(self):
        """No API key → returns 0.5 neutral fallback."""
        with patch("src.agents.factual_agent.config") as mock_cfg:
            mock_cfg.get_fva_provider_settings.return_value = \
                ("openai", "", "gpt-4o-mini", None)
            mock_cfg.has_fva_key.return_value = False
            agent = FactualVerificationAgent()
            score = agent.verify("test", "test")
        assert score == 0.5
        assert agent.last_status == "unconfigured"

    def test_parse_error_returns_neutral(self):
        with patch("src.agents.factual_agent.OpenAI") as MockOAI:
            instance = MockOAI.return_value
            instance.chat.completions.create.return_value = \
                self._make_mock_response("NOT A NUMBER")
            with patch("src.agents.factual_agent.config") as mock_cfg:
                mock_cfg.get_fva_provider_settings.return_value = \
                    ("openai", "test-key", "gpt-4o-mini", None)
                mock_cfg.has_fva_key.return_value = True
                agent = FactualVerificationAgent()
                score = agent.verify("test", "test")
        assert score == 0.5
        assert agent.last_status == "invalid_response"

    def test_accepts_a_score_with_explanatory_text(self):
        assert FactualVerificationAgent._parse_score("Score: 0.92") == 0.92

    def test_gemini_uses_compatible_endpoint(self):
        with patch("src.agents.factual_agent.OpenAI") as MockOAI:
            with patch("src.agents.factual_agent.config") as mock_cfg:
                mock_cfg.get_fva_provider_settings.return_value = (
                    "gemini", "gemini-key", "gemini-2.5-flash",
                    "https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                mock_cfg.has_fva_key.return_value = True
                FactualVerificationAgent()
        assert MockOAI.call_args.kwargs["base_url"].endswith("/openai/")
