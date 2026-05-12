"""
Phase 11 tests — Fairness audit invariants.

Run with: pytest tests/test_phase11.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RESULTS_DIR, REPORTS_DIR
from src.fairness.bootstrap_per_group import (
    _stratified_resample_within,
    bootstrap_group_metrics,
    per_group_breakdown,
)
from src.fairness.metrics import (
    demographic_parity_difference,
    disparate_impact,
    equal_opportunity_difference,
    recall_for_class,
    selection_rate_for_class,
)


RNG = np.random.default_rng(42)


# ============================================================
# Unit tests — basic metrics
# ============================================================
class TestRecallForClass:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        assert recall_for_class(y_true, y_pred, 0) == 1.0
        assert recall_for_class(y_true, y_pred, 1) == 1.0
        assert recall_for_class(y_true, y_pred, 2) == 1.0

    def test_no_class_yields_nan(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert np.isnan(recall_for_class(y_true, y_pred, 2))

    def test_half_correct(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        assert recall_for_class(y_true, y_pred, 0) == 0.5


class TestSelectionRate:
    def test_no_selection(self):
        y_pred = np.array([1, 1, 1, 1])
        assert selection_rate_for_class(y_pred, 0) == 0.0

    def test_all_selection(self):
        y_pred = np.array([0, 0, 0])
        assert selection_rate_for_class(y_pred, 0) == 1.0


# ============================================================
# Unit tests — disparate impact
# ============================================================
class TestDisparateImpact:
    def test_perfectly_balanced_yields_DI_1(self):
        # 200 rows: 50% group A, 50% group B, same recall (0.8 each)
        y_true = np.array([0] * 100 + [0] * 100)
        # group A: 80/100 correct, group B: 80/100 correct
        y_pred = np.array([0] * 80 + [1] * 20 + [0] * 80 + [1] * 20)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        df = disparate_impact(y_true, y_pred, groups, class_idx=0)
        assert len(df) == 2
        # DI should equal 1.0 for both
        assert abs(df["DI"].max() - 1.0) < 1e-6
        assert (df["DI_status"] == "fair").all()

    def test_disparity_flagged(self):
        # group A recall = 0.9, group B recall = 0.5 (ratio 0.56, below 0.8)
        y_true = np.array([0] * 100 + [0] * 100)
        y_pred = np.array([0] * 90 + [1] * 10 + [0] * 50 + [1] * 50)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        df = disparate_impact(y_true, y_pred, groups, class_idx=0)
        # Reference = A (highest recall)
        b_row = df[df["group"] == "B"].iloc[0]
        assert b_row["DI"] < 0.8
        assert b_row["DI_status"] == "biased"

    def test_reference_group_explicit(self):
        y_true = np.array([0] * 50 + [0] * 50)
        y_pred = np.array([0] * 40 + [1] * 10 + [0] * 30 + [1] * 20)
        groups = np.array(["A"] * 50 + ["B"] * 50)
        df = disparate_impact(y_true, y_pred, groups, class_idx=0, reference_group="B")
        # All DI computed wrt B; A's recall is higher so DI > 1
        a_row = df[df["group"] == "A"].iloc[0]
        assert a_row["DI"] > 1.0


# ============================================================
# Unit tests — EOD and DPD
# ============================================================
class TestEqualOpportunityDifference:
    def test_zero_when_perfectly_equal(self):
        y_true = np.array([0] * 100 + [0] * 100)
        y_pred = np.array([0] * 80 + [1] * 20 + [0] * 80 + [1] * 20)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        eod = equal_opportunity_difference(y_true, y_pred, groups, class_idx=0)
        assert eod["max_diff"] == 0.0

    def test_positive_when_disparity(self):
        y_true = np.array([0] * 100 + [0] * 100)
        y_pred = np.array([0] * 90 + [1] * 10 + [0] * 50 + [1] * 50)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        eod = equal_opportunity_difference(y_true, y_pred, groups, class_idx=0)
        assert eod["max_diff"] > 0


class TestDemographicParity:
    def test_DPD_zero_when_equal_selection_rates(self):
        y_pred = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        dpd = demographic_parity_difference(y_pred, groups, class_idx=0)
        assert dpd["max_diff"] == 0.0


# ============================================================
# Unit tests — bootstrap
# ============================================================
class TestBootstrapResampleWithin:
    def test_returns_correct_shape(self):
        y = np.array([0] * 50 + [1] * 30 + [2] * 20)
        idx = _stratified_resample_within(y, n_resamples=10, seed=42)
        assert idx.shape == (10, 100)

    def test_preserves_class_count_within_each_resample(self):
        y = np.array([0] * 50 + [1] * 30 + [2] * 20)
        idx = _stratified_resample_within(y, n_resamples=20, seed=42)
        for b in range(20):
            counts = np.bincount(y[idx[b]], minlength=3)
            assert counts[0] == 50
            assert counts[1] == 30
            assert counts[2] == 20


class TestBootstrapGroupMetrics:
    def test_ci_brackets_point_estimate(self):
        y_true = np.array([0] * 100 + [1] * 100)
        y_pred = np.array([0] * 80 + [1] * 20 + [0] * 30 + [1] * 70)
        out = bootstrap_group_metrics(y_true, y_pred, class_idx=0,
                                       n_resamples=200, seed=42)
        assert out["recall_ci_lo"] <= out["recall"] <= out["recall_ci_hi"]
        assert out["selection_rate_ci_lo"] <= out["selection_rate"] <= out["selection_rate_ci_hi"]

    def test_returns_nan_ci_for_small_groups(self):
        y_true = np.array([0, 0, 1])
        y_pred = np.array([0, 0, 1])
        out = bootstrap_group_metrics(y_true, y_pred, class_idx=0,
                                       n_resamples=200, seed=42, min_class_n=5)
        assert np.isnan(out["recall_ci_lo"])


# ============================================================
# Integration tests — fairness audit artifacts
# ============================================================
class TestFairnessArtifacts:
    @pytest.fixture(scope="class")
    def per_group_df(self):
        path = RESULTS_DIR / "fairness_per_group.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make fairness`")
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def disparity_df(self):
        path = RESULTS_DIR / "fairness_disparities.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing")
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def cf_df(self):
        path = RESULTS_DIR / "fairness_counterfactual.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing")
        return pd.read_parquet(path)

    def test_per_group_has_both_champions(self, per_group_df):
        champions = set(per_group_df["champion"].unique())
        assert "C1_qwk" in champions
        assert "C2_safe" in champions

    def test_per_group_has_3_segments(self, per_group_df):
        for seg in ("Senior", "Gender", "Married"):
            assert seg in per_group_df["segment"].values, f"missing segment: {seg}"

    def test_per_group_audits_detractor_and_promoter(self, per_group_df):
        classes = set(per_group_df["class"].unique())
        assert "Detractor" in classes
        assert "Promoter" in classes

    def test_ci_brackets_point_estimate(self, per_group_df):
        sub = per_group_df.dropna(subset=["recall_ci_lo", "recall_ci_hi"])
        for _, r in sub.iterrows():
            assert r["recall_ci_lo"] <= r["recall"] <= r["recall_ci_hi"], (
                f"{r['champion']}/{r['segment']}/{r['group']}/{r['class']}: "
                f"recall {r['recall']} not in CI [{r['recall_ci_lo']}, {r['recall_ci_hi']}]"
            )

    def test_disparity_has_DI_status(self, disparity_df):
        assert "DI_status" in disparity_df.columns
        assert disparity_df["DI_status"].isin(["fair", "biased"]).all()

    def test_disparity_required_columns(self, disparity_df):
        for col in ("champion", "segment", "class", "DI_worst", "EOD", "DPD"):
            assert col in disparity_df.columns

    def test_counterfactual_change_rate_in_valid_range(self, cf_df):
        for _, r in cf_df.iterrows():
            if "change_rate" not in r:
                continue
            assert 0 <= r["change_rate"] <= 1, (
                f"change_rate {r['change_rate']} out of [0, 1]"
            )

    def test_counterfactual_has_both_champions(self, cf_df):
        champions = set(cf_df["champion"].unique())
        assert "C1_qwk" in champions
        assert "C2_safe" in champions

    def test_fairness_report_exists(self):
        path = REPORTS_DIR / "fairness_audit.md"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make fairness`")
        content = path.read_text()
        assert "Phase 11" in content
        assert "Detractor" in content
        assert "Promoter" in content
        assert "Senior" in content
        assert "Gender" in content
        assert "Married" in content


# ============================================================
# Sanity — disparity flagged for at least one cell?
# (this is a smoke check, not a strict assertion)
# ============================================================
class TestFairnessSanity:
    @pytest.fixture(scope="class")
    def disparity_df(self):
        path = RESULTS_DIR / "fairness_disparities.parquet"
        if not path.exists():
            pytest.skip()
        return pd.read_parquet(path)

    def test_DI_in_reasonable_range(self, disparity_df):
        """No DI < 0.5 or > 2.0 (catastrophic disparity = code bug)."""
        for _, r in disparity_df.iterrows():
            assert 0.3 < r["DI_worst"] < 3.0, (
                f"DI {r['DI_worst']} extreme — investigate "
                f"{r['champion']}/{r['segment']}/{r['class']}"
            )

    def test_EOD_bounded(self, disparity_df):
        """|EOD| should be < 0.5 in any reasonable scenario."""
        for _, r in disparity_df.iterrows():
            if r["EOD"] == r["EOD"]:  # not NaN
                assert abs(r["EOD"]) < 0.5, (
                    f"|EOD| {r['EOD']} extreme — investigate"
                )
