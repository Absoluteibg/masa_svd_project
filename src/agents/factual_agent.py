"""
factual_agent.py  —  Agent 1: Factual Verification Agent (FVA)

Asks GPT-4o-mini to rate how factually accurate the LLM response is.
Returns a confidence float ∈ [0, 1].
Falls back to 0.5 (neutral) if API is unavailable.
"""
from __future__ import annotations

import os
import logging
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
        self._enabled = config.has_openai_key()
        if self._enabled:
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            log.warning("FVA: No OpenAI API key found. Using fallback score 0.5.")

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
            return 0.5   # Neutral fallback when no API key

        user_msg = (
            f"QUESTION: {query}\n\n"
            f"AI RESPONSE: {response}\n\n"
            "Factual accuracy score:"
        )

        try:
            completion = self._client.chat.completions.create(
                model       = config.FVA_LLM_MODEL,
                messages    = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = 0,
                max_tokens  = 10,
            )
            raw = completion.choices[0].message.content.strip()
            score = float(raw)
            return round(float(min(max(score, 0.0), 1.0)), 4)

        except (ValueError, TypeError) as e:
            log.warning("FVA parse error (%s). Raw='%s'. Using 0.5.", e, raw)
            return 0.5
        except OpenAIError as e:
            log.warning("FVA OpenAI error: %s. Using 0.5.", e)
            return 0.5