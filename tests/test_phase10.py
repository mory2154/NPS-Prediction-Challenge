"""
Phase 10 tests — Interpretability invariants.

Run with: pytest tests/test_phase10.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, RESULTS_DIR
from src.interpretation.shap_utils import (
    C1_PATH,
    _stratified_sample,
    _hybrid_feature_names,
)


HYBRID_MODEL = MODELS_DIR / "hybrid" / "lightgbm_pca32.joblib"
TUNED_MODEL = MODELS_DIR / "tuned" / "logistic_tuned.joblib"


# ============================================================
# Unit tests — sampling
# ============================================================
class TestStratifiedSample:
    def test_returns_correct_total_size(self):
        y = np.array([0] * 100 + [1] * 60 + [2] * 40)
        idx = _stratified_sample(y, n_target=50, seed=42)
        assert 45 <= len(idx) <= 55  # rounding tolerance

    def test_preserves_class_proportions(self):
        y = np.array([0] * 580 + [1] * 250 + [2] * 170)  # 58/25/17
        idx = _stratified_sample(y, n_target=200, seed=42)
        counts = np.bincount(y[idx], minlength=3)
        props = counts / counts.sum()
        # Should be close to 0.58 / 0.25 / 0.17 ± 0.03
        assert abs(props[0] - 0.58) < 0.05
        assert abs(props[1] - 0.25) < 0.05
        assert abs(props[2] - 0.17) < 0.05

    def test_returns_all_when_total_less_than_target(self):
        y = np.array([0] * 10 + [1] * 5 + [2] * 5)
        idx = _stratified_sample(y, n_target=100, seed=42)
        assert len(idx) == 20

    def test_seed_is_reproducible(self):
        y = np.array([0] * 50 + [1] * 30 + [2] * 20)
        idx_a = _stratified_sample(y, n_target=40, seed=42)
        idx_b = _stratified_sample(y, n_target=40, seed=42)
        np.testing.assert_array_equal(idx_a, idx_b)


# ============================================================
# Unit tests — feature naming
# ============================================================
class TestFeatureNames:
    def test_pc_labels_are_zero_padded(self):
        import joblib
        if not (MODELS_DIR / "preprocessing_pipeline.joblib").exists():
            pytest.skip("preprocessing_pipeline.joblib missing")
        pipe = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
        names = _hybrid_feature_names(pipe, n_pcs=32)
        pc_names = [n for n in names if n.startswith("PC")]
        assert len(pc_names) == 32
        assert pc_names[0] == "PC00"
        assert pc_names[-1] == "PC31"

    def test_total_count_matches_hybrid_dim(self):
        import joblib
        if not HYBRID_MODEL.exists():
            pytest.skip("C1 missing")
        pipe = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
        names = _hybrid_feature_names(pipe, n_pcs=32)
        c1 = joblib.load(HYBRID_MODEL)
        assert len(names) == c1.n_features_in_, (
            f"feature names ({len(names)}) ≠ model expected ({c1.n_features_in_})"
        )


# ============================================================
# Integration — SHAP artifacts
# ============================================================
class TestShapArtifacts:
    @pytest.fixture(scope="class")
    def global_df(self):
        path = RESULTS_DIR / "shap_global_C1.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make interpret`")
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def local_df(self):
        path = RESULTS_DIR / "shap_local_C1.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make interpret`")
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def segment_df(self):
        path = RESULTS_DIR / "shap_segment_C1.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing")
        return pd.read_parquet(path)

    def test_global_has_three_classes(self, global_df):
        classes = set(global_df["class"].unique())
        assert classes == {"Detractor", "Passive", "Promoter"}

    def test_global_required_columns(self, global_df):
        for col in ("class", "feature", "mean_abs_shap", "rank_in_class"):
            assert col in global_df.columns

    def test_global_mean_abs_shap_is_nonneg(self, global_df):
        assert (global_df["mean_abs_shap"] >= 0).all()

    def test_global_rank_starts_at_1(self, global_df):
        for cls in global_df["class"].unique():
            sub = global_df[global_df["class"] == cls]
            assert sub["rank_in_class"].min() == 1
            assert sub["rank_in_class"].max() == sub.shape[0]

    def test_local_has_3_archetypes(self, local_df):
        n_customers = local_df["customer_id"].nunique()
        assert n_customers == 3, f"expected 3 archetypes, got {n_customers}"

    def test_local_each_customer_has_all_features(self, local_df):
        n_feat = local_df.groupby("customer_id")["feature"].nunique()
        # Every archetype should explain ALL features (same N for each)
        assert n_feat.nunique() == 1

    def test_segment_has_yes_no_groups(self, segment_df):
        if segment_df.empty:
            pytest.skip("segment_df empty — segments not in dataset")
        for seg in segment_df["segment"].unique():
            groups = set(segment_df[segment_df["segment"] == seg]["group"].unique())
            # Should have at least 2 groups per segment
            assert len(groups) >= 2, f"segment {seg} has < 2 groups"


# ============================================================
# Integration — linear coef artifacts
# ============================================================
class TestLinearCoefArtifacts:
    @pytest.fixture(scope="class")
    def coef_df(self):
        path = RESULTS_DIR / "linear_coef_C2.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make interpret`")
        return pd.read_parquet(path)

    def test_has_three_classes(self, coef_df):
        assert set(coef_df["class"].unique()) == {"Detractor", "Passive", "Promoter"}

    def test_required_columns(self, coef_df):
        for col in ("class", "feature", "coef", "std", "coef_times_std"):
            assert col in coef_df.columns

    def test_sign_consistent_with_coef(self, coef_df):
        """If coef > 0 the sign must be '+', if < 0 it must be '−'."""
        if "sign" not in coef_df.columns:
            pytest.skip()
        pos = coef_df[coef_df["coef"] > 0]
        neg = coef_df[coef_df["coef"] < 0]
        assert (pos["sign"] == "+").all()
        assert (neg["sign"] == "−").all()

    def test_std_is_nonneg(self, coef_df):
        assert (coef_df["std"] >= 0).all()


# ============================================================
# Integration — PCA loadings
# ============================================================
class TestPcaLoadings:
    @pytest.fixture(scope="class")
    def loadings(self):
        path = RESULTS_DIR / "pca_loadings.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make interpret`")
        return pd.read_parquet(path)

    def test_32_pcs(self, loadings):
        assert loadings["pc"].nunique() == 32

    def test_each_pc_has_topk_dims(self, loadings):
        for pc, sub in loadings.groupby("pc"):
            assert len(sub) >= 1
            assert sub["rank"].min() == 1

    def test_explained_variance_sums_below_one(self, loadings):
        total_var = loadings.groupby("pc")["explained_variance"].first().sum()
        # PCA on 384 dims with only 32 components → < 1.0
        assert 0 < total_var <= 1.0

    def test_loadings_are_orthonormal_signed(self, loadings):
        """|loading| ≤ 1 (PCA components are unit vectors in dim space)."""
        assert loadings["abs_loading"].max() <= 1.0


# ============================================================
# Sanity: C1 vs C2 agreement on the top "tabular" features
# ============================================================
class TestC1vsC2Agreement:
    @pytest.fixture(scope="class")
    def both(self):
        gp = RESULTS_DIR / "shap_global_C1.parquet"
        cp = RESULTS_DIR / "linear_coef_C2.parquet"
        if not gp.exists() or not cp.exists():
            pytest.skip("Both interpretability artifacts required")
        return pd.read_parquet(gp), pd.read_parquet(cp)

    def test_top_tab_features_overlap_for_detractor(self, both):
        """At least 1 of the top-5 tab features should appear in both rankings
        for the Detractor class. Lower bar = synthetic data, real data should
        be higher."""
        g_shap, g_coef = both
        # Restrict SHAP to non-PC features (tabular only)
        top_shap = (g_shap[(g_shap["class"] == "Detractor") & (~g_shap["is_pc"])]
                    .head(5)["feature"].tolist())
        top_coef = (g_coef[g_coef["class"] == "Detractor"]
                    .head(5)["feature"].tolist())
        common = set(top_shap) & set(top_coef)
        assert len(common) >= 1, (
            f"No overlap between C1 and C2 top-5 Detractor features.\n"
            f"  C1 top: {top_shap}\n  C2 top: {top_coef}"
        )
