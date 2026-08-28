"""
embeddings.py  —  Converts text into numerical vectors.

The model (all-MiniLM-L6-v2) is loaded once globally and reused.
Every call to embed_texts() reuses the same model instance.
"""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from src.config import config

# ── Singleton model (loaded once, shared across all calls) ───────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the sentence embedding model. Downloads on first call (~80 MB)."""
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model '{config.EMBEDDING_MODEL}' ...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("[Embeddings] Model ready.")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of sentences into a 2-D numpy array.

    Parameters
    ----------
    texts : List[str]
        One or more sentences / passages.

    Returns
    -------
    np.ndarray  shape (len(texts), 384)
        Row i is the 384-dimensional embedding of texts[i].

    Raises
    ------
    ValueError  if texts is empty.
    """
    if not texts:
        raise ValueError("embed_texts() received an empty list.")

    cleaned = [str(t).strip() or " " for t in texts]   # Guard empty strings
    model   = _get_model()
    return model.encode(cleaned, convert_to_numpy=True,
                        show_progress_bar=False, normalize_embeddings=True)


def embed_single(text: str) -> np.ndarray:
    """Embed a single string. Returns shape (384,)."""
    return embed_texts([text])[0]