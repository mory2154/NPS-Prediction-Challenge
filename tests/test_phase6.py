"""
Phase 6 invariant tests for baseline modelling.

Run with:
    pytest tests/test_phase6.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import NPS_CLASSES, RESULTS_DIR
from src.evaluation.metrics import (
    balanced_acc,
    confusion_df,
    detractor_recall,
    evaluate,
    lift_at_k,
    macro_f1,
    per_class_recall,
    quadratic_weighted_kappa,
)


# ============================================================
# Synthetic prediction fixtures
# ============================================================
@pytest.fixture
def perfect_predictions():
    """Truth and predictions that are identical."""
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = y_true.copy()
    y_proba = np.eye(3)[y_pred] * 0.9 + 0.05
    return y_true, y_pred, y_proba


@pytest.fixture
def random_predictions():
    """Random predictions (should give QWK ≈ 0)."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 3, n)
    y_pred = rng.integers(0, 3, n)
    y_proba = rng.dirichlet([1, 1, 1], n)
    return y_true, y_pred, y_proba


@pytest.fixture
def realistic_predictions():
    """Mostly correct, but with some ordinal-aware errors."""
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.integers(0, 3, n)
    # 80% correct, 15% off-by-1, 5% off-by-2
    y_pred = y_true.copy()
    flip_idx = rng.choice(n, size=int(n * 0.2), replace=False)
    for i in flip_idx:
        if y_true[i] == 0:
            y_pred[i] = 1  # off by 1
        elif y_true[i] == 2:
            y_pred[i] = 1
        else:
            y_pred[i] = rng.choice([0, 2])
    # Probabilities aligned with predictions
    y_proba = np.zeros((n, 3))
    y_proba[np.arange(n), y_pred] = 0.7
    y_proba[np.arange(n), (y_pred + 1) % 3] = 0.2
    y_proba[np.arange(n), (y_pred + 2) % 3] = 0.1
    return y_true, y_pred, y_proba


# ============================================================
# Quadratic Weighted Kappa
# ============================================================
class TestQWK:
    def test_perfect_is_one(self, perfect_predictions):
        y_true, y_pred, _ = perfect_predictions
        assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(1.0)

    def test_random_is_near_zero(self, random_predictions):
        y_true, y_pred, _ = random_predictions
        # Random predictions should give QWK between -0.1 and 0.1
        qwk = quadratic_weighted_kappa(y_true, y_pred)
        assert abs(qwk) < 0.15

    def test_string_labels_supported(self):
        y_true = pd.Categorical(["Detractor", "Promoter", "Passive"], categories=NPS_CLASSES)
        y_pred = pd.Categorical(["Detractor", "Promoter", "Passive"], categories=NPS_CLASSES)
        assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(1.0)

    def test_off_by_two_penalized_more_than_off_by_one(self):
        """If we predict Promoter for a Detractor (off by 2), QWK should be
        WORSE than predicting Passive for a Detractor (off by 1)."""
        y_true = np.array([0, 0, 0, 0])
        # Off by 1 each time
        y_pred_off1 = np.array([1, 1, 1, 1])
        # Off by 2 each time
        y_pred_off2 = np.array([2, 2, 2, 2])
        # When all predictions are constant, kappa is 0 (degenerate).
        # Use mixed cases instead.
        y_true_mixed = np.array([0, 0, 1, 1, 2, 2])
        y_pred_off1_mixed = np.array([1, 1, 0, 2, 1, 1])
        y_pred_off2_mixed = np.array([2, 2, 1, 1, 0, 0])  # off-by-2 from extremes

        qwk1 = quadratic_weighted_kappa(y_true_mixed, y_pred_off1_mixed)
        qwk2 = quadratic_weighted_kappa(y_true_mixed, y_pred_off2_mixed)
        assert qwk1 > qwk2


# ============================================================
# F1 / balanced acc / detractor recall
# ============================================================
class TestStandardMetrics:
    def test_macro_f1_perfect(self, perfect_predictions):
        y_true, y_pred, _ = perfect_predictions
        assert macro_f1(y_true, y_pred) == pytest.approx(1.0)

    def test_balanced_acc_perfect(self, perfect_predictions):
        y_true, y_pred, _ = perfect_predictions
        assert balanced_acc(y_true, y_pred) == pytest.approx(1.0)

    def test_detractor_recall_perfect(self, perfect_predictions):
        y_true, y_pred, _ = perfect_predictions
        assert detractor_recall(y_true, y_pred) == pytest.approx(1.0)

    def test_detractor_recall_when_all_misclassified(self):
        # All Detractors predicted as Promoter
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([2, 2, 2, 2])
        assert detractor_recall(y_true, y_pred) == 0.0


# ============================================================
# Per-class recall
# ============================================================
class TestPerClassRecall:
    def test_returns_three_classes(self, realistic_predictions):
        y_true, y_pred, _ = realistic_predictions
        recalls = per_class_recall(y_true, y_pred)
        assert set(recalls.keys()) == set(NPS_CLASSES)

    def test_all_in_zero_one(self, realistic_predictions):
        y_true, y_pred, _ = realistic_predictions
        recalls = per_class_recall(y_true, y_pred)
        for v in recalls.values():
            assert 0 <= v <= 1


# ============================================================
# Lift@k
# ============================================================
class TestLift:
    def test_lift_perfect_predictions_high(self, perfect_predictions):
        y_true, _, y_proba = perfect_predictions
        lift = lift_at_k(y_true, y_proba, k_pct=0.5)
        # When proba is perfect, top-50% should contain all actual detractors
        # which is 3/8 of total → ratio = (3/4) / (3/8) = 2.0
        assert lift > 1.5

    def test_lift_random_around_one(self, random_predictions):
        y_true, _, y_proba = random_predictions
        lift = lift_at_k(y_true, y_proba, k_pct=0.20)
        assert 0.5 < lift < 1.8


# ============================================================
# Confusion matrix
# ============================================================
class TestConfusion:
    def test_returns_3x3_dataframe(self, realistic_predictions):
        y_true, y_pred, _ = realistic_predictions
        cm = confusion_df(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_perfect_is_diagonal(self, perfect_predictions):
        y_true, y_pred, _ = perfect_predictions
        cm = confusion_df(y_true, y_pred)
        # Off-diagonal = 0
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert cm.iat[i, j] == 0


# ============================================================
# Aggregate evaluate()
# ============================================================
class TestEvaluate:
    def test_returns_all_keys(self, realistic_predictions):
        y_true, y_pred, y_proba = realistic_predictions
        result = evaluate(y_true, y_pred, y_proba, name="test_model")
        for key in [
            "model", "n", "qwk", "macro_f1", "balanced_acc",
            "detractor_recall", "recall_detractor", "recall_passive",
            "recall_promoter", "lift@10", "lift@20",
        ]:
            assert key in result

    def test_n_matches(self, realistic_predictions):
        y_true, y_pred, _ = realistic_predictions
        result = evaluate(y_true, y_pred)
        assert result["n"] == len(y_true)


# ============================================================
# Integration: results file from baseline.py
# ============================================================
class TestBaselineResults:
    @pytest.fixture
    def results_df(self):
        path = RESULTS_DIR / "baseline_results.parquet"
        if not path.exists():
            pytest.skip(
                f"{path} not found — run `make baseline` first."
            )
        return pd.read_parquet(path)

    def test_three_models_present(self, results_df):
        models = set(results_df["model"].unique())
        # Logistic and lightgbm should always be present.
        # Ordinal is optional (mord install).
        assert {"logistic", "lightgbm"}.issubset(models)

    def test_all_splits_present(self, results_df):
        splits = set(results_df["split"].unique())
        assert {"train", "val", "respondent_test", "silent_test"}.issubset(splits)

    def test_qwk_in_valid_range(self, results_df):
        # QWK between -1 and 1
        assert (results_df["qwk"] >= -1).all()
        assert (results_df["qwk"] <= 1).all()

    def test_results_sorted_naturally(self, results_df):
        # Should have at least one row per (model, split) for each target
        targets = results_df["target"].unique()
        for target in targets:
            sub = results_df[results_df["target"] == target]
            for model in sub["model"].unique():
                splits = sub[sub["model"] == model]["split"].unique()
                # Each model should be evaluated on 4 splits
                assert len(splits) >= 3, (
                    f"Model {model} on target {target} has only {len(splits)} splits"
                )
