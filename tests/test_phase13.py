"""
Phase 13 tests — Monitoring lightweight invariants.

Run with: pytest tests/test_phase13.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, RESULTS_DIR
from src.monitoring.alerts import (
    _normalize_champion_key,
    escalate_alerts,
    evaluate_alert,
    lookup_reference_ci,
)


C2_CALIBRATED_PATH = MODELS_DIR / "tuned" / "logistic_C2_calibrated.joblib"


# ============================================================
# Unit tests — alerts
# ============================================================
class TestEvaluateAlert:
    def test_in_band(self):
        assert evaluate_alert(0.85, 0.80, 0.90) == "in_band"

    def test_drift_low(self):
        assert evaluate_alert(0.70, 0.80, 0.90) == "drift_low"

    def test_drift_high(self):
        assert evaluate_alert(0.95, 0.80, 0.90) == "drift_high"

    def test_no_reference_with_nan(self):
        assert evaluate_alert(0.85, float("nan"), 0.90) == "no_reference"
        assert evaluate_alert(float("nan"), 0.80, 0.90) == "no_reference"


class TestEscalation:
    def test_single_drift_warns(self):
        s = pd.Series(["in_band", "drift_high", "in_band"])
        result = escalate_alerts(s).tolist()
        assert result == ["ok", "warn", "ok"]

    def test_two_consecutive_same_direction_alert(self):
        s = pd.Series(["in_band", "drift_high", "drift_high", "in_band"])
        result = escalate_alerts(s).tolist()
        assert result == ["ok", "warn", "alert", "ok"]

    def test_alternating_drifts_only_warn(self):
        s = pd.Series(["drift_high", "drift_low", "drift_high"])
        result = escalate_alerts(s).tolist()
        # Direction changes each time, so each is "warn" not "alert"
        assert result == ["warn", "warn", "warn"]

    def test_three_consecutive_same_direction_all_alert(self):
        s = pd.Series(["drift_low", "drift_low", "drift_low"])
        result = escalate_alerts(s).tolist()
        # First is warn (no prior), then alert, alert
        assert result == ["warn", "alert", "alert"]

    def test_no_reference_treated_as_ok(self):
        s = pd.Series(["no_reference", "drift_high"])
        result = escalate_alerts(s).tolist()
        assert result == ["ok", "warn"]


class TestNormalizeChampionKey:
    def test_C1_variants(self):
        assert _normalize_champion_key("C1") == "C1_qwk"
        assert _normalize_champion_key("C1_qwk") == "C1_qwk"

    def test_C2_variants(self):
        assert _normalize_champion_key("C2") == "C2_safe"
        assert _normalize_champion_key("C2_safe") == "C2_safe"


class TestLookupReferenceCI:
    @pytest.fixture
    def fake_phase11(self):
        return pd.DataFrame({
            "champion":        ["C2_safe", "C2_safe"],
            "segment":         ["Senior", "Senior"],
            "class":           ["Detractor", "Detractor"],
            "group":           ["Yes", "No"],
            "recall":          [0.91, 0.82],
            "recall_ci_lo":    [0.88, 0.79],
            "recall_ci_hi":    [0.94, 0.84],
        })

    @pytest.fixture
    def fake_phase9(self):
        return pd.DataFrame({
            "champion":   ["C2_safe", "C2_safe"],
            "split":      ["silent_test", "silent_test"],
            "metric":     ["qwk", "detractor_recall"],
            "value":      [0.355, 0.840],
            "ci_lo":      [0.334, 0.820],
            "ci_hi":      [0.375, 0.859],
        })

    def test_phase11_lookup_succeeds(self, fake_phase11, fake_phase9):
        lo, hi, src = lookup_reference_ci(
            fake_phase11, fake_phase9,
            champion="C2", metric="detractor_recall",
            segment="Senior", group="Yes",
        )
        assert lo == 0.88
        assert hi == 0.94
        assert src == "phase11_per_segment"

    def test_phase9_fallback_for_global(self, fake_phase11, fake_phase9):
        lo, hi, src = lookup_reference_ci(
            fake_phase11, fake_phase9,
            champion="C2", metric="qwk",
        )
        assert lo == 0.334
        assert hi == 0.375
        assert src == "phase9_headline"

    def test_no_reference_when_metric_unknown(self, fake_phase11, fake_phase9):
        lo, hi, src = lookup_reference_ci(
            fake_phase11, fake_phase9,
            champion="C2", metric="unknown_metric",
        )
        assert np.isnan(lo)
        assert np.isnan(hi)
        assert src == "no_reference"


# ============================================================
# Integration tests — artifacts
# ============================================================
class TestRecalibrationArtifacts:
    def test_calibrated_model_exists(self):
        if not C2_CALIBRATED_PATH.exists():
            pytest.skip(f"{C2_CALIBRATED_PATH.name} missing — run `make recalibrate`")
        import joblib
        model = joblib.load(C2_CALIBRATED_PATH)
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "predict")

    @pytest.fixture
    def cal_audit(self):
        path = RESULTS_DIR / "calibration_before_after.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing")
        return pd.read_parquet(path)

    def test_audit_has_all_three_classes(self, cal_audit):
        classes = set(cal_audit["class"].unique())
        assert classes == {"Detractor", "Passive", "Promoter"}

    def test_audit_columns_present(self, cal_audit):
        for col in ("brier_before", "brier_after", "brier_delta",
                    "ece_before", "ece_after", "ece_delta"):
            assert col in cal_audit.columns

    def test_brier_in_valid_range(self, cal_audit):
        for col in ("brier_before", "brier_after"):
            assert (cal_audit[col] >= 0).all()
            assert (cal_audit[col] <= 1).all()

    def test_ece_in_valid_range(self, cal_audit):
        for col in ("ece_before", "ece_after"):
            assert (cal_audit[col] >= 0).all()
            assert (cal_audit[col] <= 1).all()

    def test_recalibration_helps_or_neutral(self, cal_audit):
        """ECE after should be at most ECE before + small tolerance.

        Isotonic on a small calibration set can in rare cases make ECE marginally
        worse (overfit on val), so we allow a small slack."""
        for _, r in cal_audit.iterrows():
            assert r["ece_after"] <= r["ece_before"] + 0.05, (
                f"Class {r['class']}: ECE got worse "
                f"({r['ece_before']:.3f} → {r['ece_after']:.3f})"
            )


class TestDriftSimulationArtifacts:
    @pytest.fixture
    def drift(self):
        path = RESULTS_DIR / "drift_simulation.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing — run `make simulate-drift`")
        return pd.read_parquet(path)

    def test_required_columns(self, drift):
        for col in ("champion", "batch_id", "batch_size", "metric", "value"):
            assert col in drift.columns

    def test_both_champions(self, drift):
        champs = set(drift["champion"].unique())
        assert "C1" in champs
        assert "C2" in champs

    def test_metrics_in_valid_range(self, drift):
        for m in ("qwk", "detractor_recall", "macro_f1"):
            vals = drift[drift["metric"] == m]["value"].dropna()
            if vals.empty:
                continue
            assert vals.min() >= -1
            assert vals.max() <= 1

    def test_batch_sizes_reasonable(self, drift):
        sizes = drift["batch_size"].unique()
        assert all(s > 0 for s in sizes)


class TestAlertArtifacts:
    @pytest.fixture
    def alerts(self):
        path = RESULTS_DIR / "monitoring_alerts.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing")
        return pd.read_parquet(path)

    def test_severity_values_valid(self, alerts):
        valid = {"ok", "warn", "alert"}
        assert set(alerts["severity"].unique()).issubset(valid)

    def test_status_values_valid(self, alerts):
        valid = {"in_band", "drift_low", "drift_high", "no_reference"}
        assert set(alerts["status"].unique()).issubset(valid)

    def test_observed_within_ci_implies_in_band(self, alerts):
        sub = alerts.dropna(subset=["ref_lo", "ref_hi"])
        in_band = sub[(sub["observed"] >= sub["ref_lo"])
                      & (sub["observed"] <= sub["ref_hi"])]
        # Allow tiny floating-point slack
        statuses = set(in_band["status"].unique())
        assert statuses.issubset({"in_band"})
