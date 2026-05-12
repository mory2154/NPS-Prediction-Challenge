"""
Phase 8 tests — Hybrid (tabular + verbatim embeddings).

Run with: pytest tests/test_phase8.py -v

Most tests are "skip if not run yet" — they validate output artifacts once the
phase is complete, so they pass cleanly during partial runs of the pipeline.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR
from src.features.embeddings import (
    DEFAULT_EMBEDDING_DIM,
    EMBEDDINGS_CACHE,
    EMBEDDINGS_META,
)


HYBRID_DIR = MODELS_DIR / "hybrid"


# ============================================================
# Embeddings artifacts
# ============================================================
class TestEmbeddings:
    @pytest.fixture(scope="class")
    def emb_df(self):
        if not EMBEDDINGS_CACHE.exists():
            pytest.skip(
                f"{EMBEDDINGS_CACHE.name} missing — run `make compute-embeddings`"
            )
        return pd.read_parquet(EMBEDDINGS_CACHE)

    @pytest.fixture(scope="class")
    def df(self):
        path = DATA_PROCESSED / "dataset_with_verbatims.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make load-verbatims`")
        return pd.read_parquet(path)

    def test_embeddings_have_correct_shape(self, emb_df, df):
        assert emb_df.shape[0] == df.shape[0], (
            f"row count mismatch: emb={emb_df.shape[0]} vs df={df.shape[0]}"
        )
        assert emb_df.shape[1] == DEFAULT_EMBEDDING_DIM, (
            f"expected {DEFAULT_EMBEDDING_DIM} dims, got {emb_df.shape[1]}"
        )

    def test_no_nan(self, emb_df):
        assert not emb_df.isna().any().any(), "NaN in embeddings"

    def test_no_inf(self, emb_df):
        arr = emb_df.to_numpy()
        assert np.isfinite(arr).all(), "Inf or NaN in embeddings"

    def test_index_aligned_with_dataset(self, emb_df, df):
        assert emb_df.index.equals(df.index), (
            "Embedding index ≠ dataset index — Customer IDs misaligned"
        )

    def test_embeddings_are_not_constant(self, emb_df):
        """Sanity: at least 90% of dims should have non-zero variance."""
        std = emb_df.std()
        n_active = (std > 1e-6).sum()
        assert n_active >= DEFAULT_EMBEDDING_DIM * 0.9, (
            f"Only {n_active}/{DEFAULT_EMBEDDING_DIM} dims have variance — "
            f"suspicious"
        )

    def test_meta_file_exists_and_consistent(self, emb_df):
        if not EMBEDDINGS_META.exists():
            pytest.skip("meta.json not produced (older code path)")
        meta = json.loads(EMBEDDINGS_META.read_text())
        assert "model_name" in meta
        assert meta["embedding_dim"] == emb_df.shape[1]
        assert meta["n_customers"] == emb_df.shape[0]


# ============================================================
# Hybrid models persisted
# ============================================================
class TestHybridModels:
    EXPECTED_FILES = [
        "lightgbm_concat.joblib",
        "lightgbm_pca32.joblib",
        "logistic_concat.joblib",
        "logistic_pca32.joblib",
    ]

    def test_at_least_one_hybrid_model_exists(self):
        existing = [
            f for f in self.EXPECTED_FILES if (HYBRID_DIR / f).exists()
        ]
        if len(existing) == 0:
            pytest.skip("No hybrid models — run `make tune-hybrid`")
        assert len(existing) >= 1

    def test_loaded_models_can_predict(self):
        """Each hybrid model that exists should be loadable and have .predict."""
        for fname in self.EXPECTED_FILES:
            path = HYBRID_DIR / fname
            if not path.exists():
                continue
            model = joblib.load(path)
            assert hasattr(model, "predict"), f"{fname} has no .predict"


# ============================================================
# Hybrid tuning trials
# ============================================================
class TestHybridTrials:
    @pytest.fixture(scope="class")
    def trials_df(self):
        path = RESULTS_DIR / "hybrid_tuning_trials.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make tune-hybrid`")
        return pd.read_parquet(path)

    def test_has_at_least_one_row(self, trials_df):
        assert len(trials_df) > 0

    def test_required_columns(self, trials_df):
        for col in ("model", "feature_space", "number", "value"):
            assert col in trials_df.columns, f"missing column {col}"

    def test_qwk_values_in_valid_range(self, trials_df):
        completed = trials_df[trials_df.get("state", "COMPLETE") == "COMPLETE"]
        if len(completed) == 0:
            pytest.skip("No completed trials")
        vals = completed["value"].dropna()
        assert (vals >= -1).all() and (vals <= 1).all()

    def test_each_combination_was_tuned(self, trials_df):
        """At least 2 of 4 (model × feature_space) combinations should have run."""
        combos = trials_df.groupby(["model", "feature_space"]).size()
        assert len(combos) >= 2


# ============================================================
# Hybrid eval results
# ============================================================
class TestHybridResults:
    @pytest.fixture(scope="class")
    def hybrid_df(self):
        path = RESULTS_DIR / "hybrid_results.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make tune-hybrid`")
        return pd.read_parquet(path)

    def test_required_columns(self, hybrid_df):
        for col in ("model", "feature_space", "split", "qwk", "phase"):
            assert col in hybrid_df.columns, f"missing column {col}"

    def test_phase_label(self, hybrid_df):
        assert (hybrid_df["phase"] == "hybrid").all()

    def test_qwk_in_valid_range(self, hybrid_df):
        assert hybrid_df["qwk"].min() >= -1
        assert hybrid_df["qwk"].max() <= 1

    def test_evaluated_on_all_4_splits(self, hybrid_df):
        expected = {"train", "val", "respondent_test", "silent_test"}
        actual = set(hybrid_df["split"].unique())
        assert expected.issubset(actual), f"missing splits: {expected - actual}"


# ============================================================
# Honesty bar — hybrid should not be DRAMATICALLY worse than tabular tuned
# ============================================================
class TestHybridSanityVsPhase7:
    @pytest.fixture(scope="class")
    def comparison(self):
        tuned_path = RESULTS_DIR / "tuned_results.parquet"
        hybrid_path = RESULTS_DIR / "hybrid_results.parquet"
        if not tuned_path.exists() or not hybrid_path.exists():
            pytest.skip("Need both tuned_results and hybrid_results")
        return pd.read_parquet(tuned_path), pd.read_parquet(hybrid_path)

    def test_hybrid_lightgbm_val_not_dramatically_worse(self, comparison):
        """At least ONE of the 2 LGBM hybrid variants should be ≥ tuned LGBM
        val QWK − 0.05. Otherwise something is broken (alignment, leakage,
        bad PCA, etc.)."""
        tuned, hybrid = comparison
        if "target" not in hybrid.columns:
            pytest.skip("'target' missing in hybrid_results")
        target = hybrid["target"].iloc[0]

        tuned_val = tuned[
            (tuned["model"] == "lightgbm")
            & (tuned["split"] == "val")
            & (tuned["target"] == target)
        ]
        hybrid_val = hybrid[
            (hybrid["model"] == "lightgbm")
            & (hybrid["split"] == "val")
            & (hybrid["target"] == target)
        ]
        if tuned_val.empty or hybrid_val.empty:
            pytest.skip("Cannot compare (missing rows)")
        best_hybrid = hybrid_val["qwk"].max()
        baseline = tuned_val["qwk"].iloc[0]
        assert best_hybrid >= baseline - 0.05, (
            f"All hybrid LGBM val QWK ({best_hybrid:.3f}) much worse than "
            f"tuned tabular ({baseline:.3f}) — likely a bug "
            f"(alignment, leakage, PCA fit on test, etc.)"
        )
