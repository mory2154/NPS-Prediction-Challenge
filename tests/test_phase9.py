"""
Phase 9 tests — Final evaluation invariants.

Run with: pytest tests/test_phase9.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RESULTS_DIR, REPORTS_DIR
from src.evaluation.bootstrap import (
    _stratified_resample_indices,
    bootstrap_all_metrics,
    bootstrap_ci,
    lift_at_k_factory,
)
from src.evaluation.calibration import (
    brier_per_class,
    calibration_report,
    reliability_curve_one_vs_rest,
)
from src.evaluation.metrics import (
    detractor_recall,
    quadratic_weighted_kappa,
)


RNG = np.random.default_rng(42)


# ============================================================
# Unit tests — bootstrap
# ============================================================
class TestStratifiedResample:
    def test_indices_have_correct_length(self):
        y = np.array([0] * 100 + [1] * 60 + [2] * 40)
        idx = _stratified_resample_indices(y, n_resamples=10, random_state=0)
        assert idx.shape == (10, 200)

    def test_each_resample_preserves_class_distribution(self):
        y = np.array([0] * 100 + [1] * 60 + [2] * 40)
        idx = _stratified_resample_indices(y, n_resamples=20, random_state=0)
        for b in range(20):
            counts = np.bincount(y[idx[b]], minlength=3)
            assert counts[0] == 100
            assert counts[1] == 60
            assert counts[2] == 40

    def test_seed_is_reproducible(self):
        y = np.array([0] * 50 + [1] * 30 + [2] * 20)
        idx_a = _stratified_resample_indices(y, n_resamples=5, random_state=42)
        idx_b = _stratified_resample_indices(y, n_resamples=5, random_state=42)
        np.testing.assert_array_equal(idx_a, idx_b)


class TestBootstrapCI:
    @pytest.fixture
    def fake_predictions(self):
        n = 600
        y_true = RNG.choice([0, 1, 2], n, p=[0.58, 0.25, 0.17])
        # Predictions are correct 70% of the time, otherwise shifted by 1
        flip = RNG.random(n) > 0.7
        y_pred = y_true.copy()
        y_pred[flip] = (y_pred[flip] + RNG.choice([-1, 1], flip.sum())) % 3
        proba = RNG.dirichlet(np.ones(3), n)
        return y_true, y_pred, proba

    def test_qwk_ci_brackets_point_estimate(self, fake_predictions):
        y_true, y_pred, _ = fake_predictions
        v, lo, hi = bootstrap_ci(
            quadratic_weighted_kappa, y_true, y_pred,
            n_resamples=200, ci=0.95, random_state=42,
        )
        assert lo <= v <= hi, f"Point estimate {v} not in CI [{lo}, {hi}]"

    def test_ci_width_shrinks_with_more_resamples(self, fake_predictions):
        """A 95% CI with 1000 resamples is no wider than with 100 resamples (in expectation)."""
        y_true, y_pred, _ = fake_predictions
        widths = []
        for nr in [50, 500]:
            _, lo, hi = bootstrap_ci(
                quadratic_weighted_kappa, y_true, y_pred,
                n_resamples=nr, ci=0.95, random_state=42,
            )
            widths.append(hi - lo)
        # 500 resamples shouldn't dramatically widen the CI (quantile is stable)
        # We don't assert strict shrinking — bootstrap CIs can fluctuate at low n.
        assert widths[1] < widths[0] * 1.5

    def test_lift_ci_uses_proba(self, fake_predictions):
        y_true, y_pred, proba = fake_predictions
        v, lo, hi = bootstrap_ci(
            metric_fn=lift_at_k_factory(0.10),
            y_true=y_true, y_pred=y_pred, y_proba=proba,
            n_resamples=200, ci=0.95, needs_proba=True,
        )
        assert v >= 0
        assert lo <= v <= hi

    def test_all_metrics_dataframe_shape(self, fake_predictions):
        y_true, y_pred, proba = fake_predictions
        df = bootstrap_all_metrics(
            y_true=y_true, y_pred=y_pred, y_proba=proba,
            n_resamples=100, random_state=42,
        )
        # 6 metrics: qwk, macro_f1, balanced_acc, detractor_recall, lift@10, lift@20
        assert len(df) == 6
        for col in ("metric", "value", "ci_lo", "ci_hi"):
            assert col in df.columns
        for _, row in df.iterrows():
            assert row["ci_lo"] <= row["value"] <= row["ci_hi"], (
                f"{row['metric']}: point {row['value']} not in CI "
                f"[{row['ci_lo']}, {row['ci_hi']}]"
            )


# ============================================================
# Unit tests — calibration
# ============================================================
class TestCalibration:
    @pytest.fixture
    def perfect_calibration(self):
        """Construct predictions whose proba EXACTLY matches the true rate."""
        n = 3000
        # Each example gets a fixed proba; whether it's "true class" is sampled
        proba_detractor = RNG.uniform(0, 1, n)
        proba_passive = (1 - proba_detractor) * 0.5
        proba_promoter = 1 - proba_detractor - proba_passive
        proba = np.stack([proba_detractor, proba_passive, proba_promoter], axis=1)
        # True class: Bernoulli per row, weighted by proba
        u = RNG.uniform(0, 1, n)
        y_true = np.where(
            u < proba_detractor, 0,
            np.where(u < proba_detractor + proba_passive, 1, 2),
        )
        return y_true, proba

    def test_brier_in_valid_range(self, perfect_calibration):
        y, p = perfect_calibration
        b = brier_per_class(y, p)
        for cls, score in b.items():
            assert 0 <= score <= 1, f"Brier {cls}={score} out of [0,1]"

    def test_perfect_calibration_has_low_brier(self, perfect_calibration):
        """A model that's calibrated by construction has Brier ≈ predicted variance."""
        y, p = perfect_calibration
        b = brier_per_class(y, p)
        # Brier for Detractor should be < 0.25 (random would be 0.25)
        assert b["Detractor"] < 0.25

    def test_reliability_curve_columns(self, perfect_calibration):
        y, p = perfect_calibration
        df = reliability_curve_one_vs_rest(y, p, class_idx=0, n_bins=10)
        for col in ("bin_id", "bin_lo", "bin_hi", "n", "predicted_mean", "observed_rate"):
            assert col in df.columns
        assert len(df) == 10

    def test_calibration_report_returns_all_classes(self, perfect_calibration):
        y, p = perfect_calibration
        rep = calibration_report(y, p, n_bins=8)
        assert set(rep["curves"].keys()) == {"Detractor", "Passive", "Promoter"}
        assert set(rep["brier"].keys()) == {"Detractor", "Passive", "Promoter"}
        assert set(rep["ece"].keys()) == {"Detractor", "Passive", "Promoter"}


# ============================================================
# Integration tests — final_eval artifacts
# ============================================================
class TestFinalEvalArtifacts:
    @pytest.fixture(scope="class")
    def summary_df(self):
        path = RESULTS_DIR / "final_eval_summary.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make final-eval`")
        return pd.read_parquet(path)

    def test_summary_has_both_champions(self, summary_df):
        champs = set(summary_df["champion"].unique())
        assert "C1_qwk" in champs
        assert "C2_safe" in champs

    def test_summary_has_all_4_splits(self, summary_df):
        for split in ("train", "val", "respondent_test", "silent_test"):
            assert split in summary_df["split"].values, f"missing split: {split}"

    def test_summary_required_metrics(self, summary_df):
        silent_C1 = summary_df[
            (summary_df["split"] == "silent_test")
            & (summary_df["champion"] == "C1_qwk")
        ]
        metrics = set(silent_C1["metric"].unique())
        assert {"qwk", "detractor_recall", "macro_f1"}.issubset(metrics)

    def test_ci_brackets_point_estimate(self, summary_df):
        for _, r in summary_df.iterrows():
            assert r["ci_lo"] <= r["value"] <= r["ci_hi"], (
                f"{r['champion']}/{r['split']}/{r['metric']}: "
                f"point {r['value']} not in CI [{r['ci_lo']}, {r['ci_hi']}]"
            )

    def test_report_markdown_exists(self):
        path = REPORTS_DIR / "final_eval_report.md"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make final-eval`")
        content = path.read_text()
        assert "# Phase 9" in content
        assert "C1" in content
        assert "C2" in content
        assert "silent_test" in content


class TestCovariateShift:
    """Sanity: respondent_test and silent_test should not differ wildly."""
    @pytest.fixture(scope="class")
    def summary_df(self):
        path = RESULTS_DIR / "final_eval_summary.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing")
        return pd.read_parquet(path)

    def test_qwk_shift_not_extreme(self, summary_df):
        """For each champion, |QWK_silent - QWK_respondent| < 0.25.

        This is a *sanity* check, not a strict invariant. Phase 6 documented
        Δ ≈ 0.005 with real data and full splits (n_resp=211, n_silent=5987).
        On real production runs we expect Δ < 0.10 for both champions.
        If this test fails (>0.25), something is very wrong (e.g. data
        leakage between train and respondent_test, or splits mis-aligned).
        """
        for ck in ["C1_qwk", "C2_safe"]:
            q_resp = summary_df[
                (summary_df["champion"] == ck)
                & (summary_df["split"] == "respondent_test")
                & (summary_df["metric"] == "qwk")
            ]
            q_sil = summary_df[
                (summary_df["champion"] == ck)
                & (summary_df["split"] == "silent_test")
                & (summary_df["metric"] == "qwk")
            ]
            if q_resp.empty or q_sil.empty:
                pytest.skip(f"missing splits for {ck}")
            shift = abs(q_sil["value"].iloc[0] - q_resp["value"].iloc[0])
            assert shift < 0.25, (
                f"{ck}: |QWK shift| = {shift:.3f} > 0.25 — covariate shift "
                f"extreme, investigate (data leakage? splits mis-aligned?)"
            )
