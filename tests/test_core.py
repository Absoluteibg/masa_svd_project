"""
test_core.py  —  Tests for embeddings and SVD engine.

No mocking needed — pure numpy/scipy operations.
Run with:  pytest tests/test_core.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import patch

from src.core.embeddings import embed_texts, embed_single
from src.core.svd_engine import SVDSubspaceEngine


class TestEmbeddings:

    def test_shape_single(self):
        emb = embed_texts(["Hello world"])
        assert emb.shape == (1, 384), f"Expected (1,384) got {emb.shape}"

    def test_shape_multiple(self):
        emb = embed_texts(["First sentence.", "Second sentence.", "Third."])
        assert emb.shape == (3, 384)

    def test_returns_numpy(self):
        emb = embed_texts(["Test"])
        assert isinstance(emb, np.ndarray)

    def test_embed_single_shape(self):
        v = embed_single("A single sentence.")
        assert v.shape == (384,)

    def test_normalized(self):
        """Since we set normalize_embeddings=True, vectors should be unit length."""
        v = embed_single("Normalized sentence.")
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-4, f"Expected unit vector, got norm={norm}"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            embed_texts([])

    def test_similar_sentences_close(self):
        a = embed_single("The dog ran quickly.")
        b = embed_single("The puppy sprinted fast.")
        c = embed_single("Quantum mechanics describes subatomic particles.")
        cos_ab = float(np.dot(a, b))   # unit vectors: dot = cosine
        cos_ac = float(np.dot(a, c))
        assert cos_ab > cos_ac, "Similar sentences should be closer than unrelated ones."


class TestSVDEngine:

    def setup_method(self):
        self.engine = SVDSubspaceEngine(k=3)

    def test_sds_in_range(self):
        sds = self.engine.compute_sds(
            ["What is the capital of France?"],
            ["Paris is the capital of France."]
        )
        assert 0.0 <= sds <= 1.0, f"SDS out of range: {sds}"

    def test_identical_texts_low_sds(self):
        text = ["The quick brown fox jumps over the lazy dog."]
        sds = self.engine.compute_sds(text, text)
        assert sds < 0.15, f"Identical texts should have very low SDS, got {sds}"

    def test_unrelated_texts_higher_sds(self):
        science = ["Quantum entanglement is a physical phenomenon."]
        cooking = ["To make pasta, boil water and add salt."]
        sds_related = self.engine.compute_sds(
            ["What is quantum entanglement?"], science
        )
        sds_unrelated = self.engine.compute_sds(
            ["What is quantum entanglement?"], cooking
        )
        assert sds_unrelated > sds_related, \
            f"Unrelated text ({sds_unrelated}) should have higher SDS than related ({sds_related})"

    def test_single_sentence_no_crash(self):
        """Single-sentence inputs should be represented in embedding space."""
        sds = self.engine.compute_sds(
            ["Who invented the telephone?"],
            ["Alexander Graham Bell."]
        )
        assert 0.0 <= sds <= 1.0

    def test_unequal_sentence_counts_use_shared_embedding_space(self):
        """A one-sentence query and multi-sentence answer must not crash."""
        embeddings = {
            "q": [1.0, 0.0, 0.0], "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0],
        }
        with patch("src.core.svd_engine.embed_texts",
                   side_effect=lambda texts: np.array([embeddings[t] for t in texts])):
            sds = self.engine.compute_sds(["q"], ["a", "b", "c"])
        assert 0.0 <= sds <= 1.0

    def test_full_analysis_keys(self):
        analysis = self.engine.full_analysis(
            ["What is gravity?"],
            ["Gravity is a fundamental force."]
        )
        required = {"sds", "sigma_values", "sigma_min", "sigma_max",
                    "sigma_mean", "k_used", "context_shape", "output_shape"}
        assert required.issubset(analysis.keys())

    def test_different_k_values(self):
        """Changing k should give valid SDS values."""
        ctx = ["The model predicts the next word in a sequence."]
        out = ["Language models use transformers to generate text."]
        for k in [1, 3, 5, 7]:
            engine = SVDSubspaceEngine(k=k)
            sds = engine.compute_sds(ctx, out)
            assert 0.0 <= sds <= 1.0, f"Invalid SDS for k={k}: {sds}"
