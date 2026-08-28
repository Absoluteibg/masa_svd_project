"""
config.py  —  Single source of truth for all project settings.
Every module imports from here. Change a value here and it
propagates everywhere automatically.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent   # masa_svd_project/
load_dotenv(PROJECT_ROOT / ".env")            # Load .env file silently


class Config:
    # ── Paths ─────────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = PROJECT_ROOT
    RESULTS_DIR:  Path = PROJECT_ROOT / "results"
    DATA_DIR:     Path = PROJECT_ROOT / "data"

    # ── API Keys ──────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # ── Model Names ───────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"       # 80 MB, fast
    NLI_MODEL:       str = "cross-encoder/nli-deberta-v3-small"  # 180 MB
    FVA_LLM_MODEL:   str = "gpt-4o-mini"            # Cheapest capable model

    # ── SVD Hyperparameters ───────────────────────────────────────────────
    SVD_K: int = 5           # Principal components to keep

    # ── Agent Weights (must sum to 1.0) ──────────────────────────────────
    ALPHA: float = 0.40      # SDS  weight (SVD geometry)
    BETA:  float = 0.35      # FVA  weight (factual accuracy)
    GAMMA: float = 0.25      # SCA  weight (semantic consistency)

    # ── Verdict Thresholds ────────────────────────────────────────────────
    HALLUCINATION_THRESHOLD: float = 0.45
    UNCERTAIN_THRESHOLD:     float = 0.30

    # ── FastAPI Server ────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int  = 8000

    # ── Evaluation ────────────────────────────────────────────────────────
    DEFAULT_EVAL_N: int = 50  # Default TruthfulQA samples to evaluate

    def has_openai_key(self) -> bool:
        return bool(self.OPENAI_API_KEY and
                    self.OPENAI_API_KEY != "sk-your-openai-key-here")

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Singleton ────────────────────────────────────────────────────────────────
config = Config()