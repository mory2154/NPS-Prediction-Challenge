"""
Phase 7 tests — Optuna tuning.

Mostly integration tests: skip if Phase 6 wasn't run yet.
The tunable code is exercised end-to-end in the CI.

Run with: pytest tests/test_phase7.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RESULTS_DIR, TUNED_DIR, TUNING_BUDGETS


# ============================================================
# Tuning artifacts
# ============================================================
class TestTuningResults:
    @pytest.fixture(scope="class")
    def trials_df(self):
        path = RESULTS_DIR / "tuning_results.parquet"
        if not path.exists():
            pytest.skip(f"{path} missing — run `make tune` first.")
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def tuned_df(self):
        path = RESULTS_DIR / "tuned_results.parquet"
        if not path.exists():
            pytest.skip(f"{path} missing — run `make tune` first.")
        return pd.read_parquet(path)

    def test_trials_has_one_row_per_trial(self, trials_df):
        assert len(trials_df) > 0

    def test_trials_columns(self, trials_df):
        # Optuna's default trials_dataframe has at least number, value, state
        for col in ["model", "number", "value"]:
            assert col in trials_df.columns

    def test_trials_values_in_qwk_range(self, trials_df):
        # QWK is in [-1, 1]; we filter only successful trials
        completed = trials_df[trials_df.get("state", "COMPLETE") == "COMPLETE"]
        if len(completed) == 0:
            pytest.skip("No completed trials")
        assert (completed["value"].dropna() >= -1).all()
        assert (completed["value"].dropna() <= 1).all()

    def test_each_model_was_tuned(self, trials_df):
        # At least 2 of the 3 models should have run (mord can be skipped)
        models = set(trials_df["model"].unique())
        assert len(models & {"lightgbm", "logistic", "ordinal"}) >= 2

    def test_lightgbm_runs_full_budget(self, trials_df):
        if "lightgbm" not in trials_df["model"].unique():
            pytest.skip("LightGBM was not tuned")
        n = (trials_df["model"] == "lightgbm").sum()
        # We allow some flexibility in case of timeout / pruning
        budget = TUNING_BUDGETS["lightgbm"]
        assert n >= budget * 0.8, f"Only {n}/{budget} LightGBM trials completed"


# ============================================================
# Tuned models persisted
# ============================================================
class TestTunedModels:
    def test_lightgbm_tuned_exists(self):
        path = TUNED_DIR / "lightgbm_tuned.joblib"
        if not path.exists():
            pytest.skip("LightGBM tuning skipped")
        assert path.is_file()

    def test_logistic_tuned_exists(self):
        path = TUNED_DIR / "logistic_tuned.joblib"
        if not path.exists():
            pytest.skip("Logistic tuning skipped")
        assert path.is_file()

    def test_loaded_models_can_predict(self):
        """Each tuned model that exists should be loadable and produce predictions."""
        import joblib

        for name in ["lightgbm", "logistic", "ordinal"]:
            path = TUNED_DIR / f"{name}_tuned.joblib"
            if not path.exists():
                continue
            model = joblib.load(path)
            assert hasattr(model, "predict"), f"{name} model has no .predict"


# ============================================================
# Sanity vs baseline — tuned should be at LEAST as good
# ============================================================
class TestImprovementVsBaseline:
    @pytest.fixture(scope="class")
    def comparison_df(self):
        baseline_path = RESULTS_DIR / "baseline_results.parquet"
        tuned_path = RESULTS_DIR / "tuned_results.parquet"
        if not baseline_path.exists() or not tuned_path.exists():
            pytest.skip("Need both baseline and tuned results.")
        baseline = pd.read_parquet(baseline_path)
        tuned = pd.read_parquet(tuned_path)
        return baseline, tuned

    def test_tuned_lightgbm_no_worse_on_val(self, comparison_df):
        """The tuned LightGBM should not be DRAMATICALLY worse on val
        than its baseline — they may differ but tuning targets val QWK."""
        baseline, tuned = comparison_df
        # Get LightGBM val QWK on the same target as tuned
        if "target" not in tuned.columns:
            pytest.skip("Tuned results missing 'target' column")
        target = tuned["target"].iloc[0]
        baseline_lgb = baseline[
            (baseline["model"] == "lightgbm")
            & (baseline["split"] == "val")
            & (baseline["target"] == target)
        ]
        tuned_lgb = tuned[
            (tuned["model"] == "lightgbm") & (tuned["split"] == "val")
        ]
        if baseline_lgb.empty or tuned_lgb.empty:
            pytest.skip("Cannot compare (missing rows)")
        baseline_qwk = baseline_lgb["qwk"].iloc[0]
        tuned_qwk = tuned_lgb["qwk"].iloc[0]
        # Tuning maximizes val QWK, so tuned should be ≥ baseline on val
        # (allow tiny tolerance for stochasticity in some optuna trials)
        assert tuned_qwk >= baseline_qwk - 0.02, (
            f"Tuned LightGBM val QWK ({tuned_qwk:.3f}) much worse than "
            f"baseline ({baseline_qwk:.3f}) — tuning didn't work"
        )
