"""
factual_agent.py  —  Agent 1: Factual Verification Agent (FVA)

Asks a configured OpenAI or Gemini model to rate factual accuracy.
Returns a confidence float ∈ [0, 1].
Falls back to 0.5 (neutral) if API is unavailable.
"""
from __future__ import annotations

import logging
import re
from openai import OpenAI, OpenAIError
from src.config import config

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a strict factual verification expert. Given a QUESTION and an \
AI-generated RESPONSE, assess how factually accurate the response is.

Steps:
1. List every specific factual claim (names, dates, numbers, citations).
2. For each claim, judge: TRUE / FALSE / UNVERIFIABLE.
3. Output a single float between 0.0 and 1.0.
   0.0 = completely false  |  0.5 = mixed  |  1.0 = completely true

Rules:
- Return ONLY the float. No words, no explanation, no punctuation.
- If the response is empty or nonsensical, return 0.0.
"""


class FactualVerificationAgent:

    def __init__(self) -> None:
        self.provider, api_key, self.model, base_url = \
            config.get_fva_provider_settings()
        self._enabled = config.has_fva_key()
        self.last_status = "unconfigured"
        if self._enabled:
            client_options = {"api_key": api_key}
            if base_url:
                client_options["base_url"] = base_url
            self._client = OpenAI(**client_options)
        else:
            log.warning("FVA: No supported API key found. Using an unavailable "
                        "factual-verification result.")

    @property
    def is_available(self) -> bool:
        """Whether the latest score was produced by the configured provider."""
        return self.last_status == "success"

    @staticmethod
    def _parse_score(raw: object) -> float:
        """Accept a bare score and harmless prose such as 'Score: 0.92'."""
        text = str(raw or "").strip()
        match = re.search(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", text)
        if match is None:
            raise ValueError(f"No 0-1 score found in {text!r}")
        return float(match.group(1))

    def verify(self, query: str, response: str) -> float:
        """
        Parameters
        ----------
        query    : The user's original question.
        response : The LLM-generated answer to verify.

        Returns
        -------
        float ∈ [0, 1]   factual confidence (higher = more accurate)
        """
        if not self._enabled:
            self.last_status = "unconfigured"
            return 0.5   # Neutral fallback when no API key

        user_msg = (
            f"QUESTION: {query}\n\n"
            f"AI RESPONSE: {response}\n\n"
            "Factual accuracy score:"
        )

        raw = ""
        try:
            completion = self._client.chat.completions.create(
                model       = self.model,
                messages    = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = 0,
                max_tokens  = 10,
            )
            raw = completion.choices[0].message.content
            score = self._parse_score(raw)
            self.last_status = "success"
            return round(float(min(max(score, 0.0), 1.0)), 4)

        except (ValueError, TypeError, AttributeError, IndexError) as e:
            self.last_status = "invalid_response"
            log.warning("FVA parse error (%s). Raw='%s'. Using 0.5.", e, raw)
            return 0.5
        except OpenAIError as e:
            self.last_status = "provider_error"
            log.warning("FVA OpenAI error: %s. Using 0.5.", e)
            return 0.5
