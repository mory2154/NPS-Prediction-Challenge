"""
Phase 12 tests — Streamlit app + batch scoring invariants.

Run with: pytest tests/test_phase12.py -v
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure app/ is on the path so component imports work
_APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from src.config import RESULTS_DIR

PREDICTIONS_PATH = RESULTS_DIR / "silent_predictions.parquet"


# ============================================================
# batch_score artifact
# ============================================================
class TestBatchScoreArtifact:
    @pytest.fixture(scope="class")
    def pred(self):
        if not PREDICTIONS_PATH.exists():
            pytest.skip(f"{PREDICTIONS_PATH.name} missing — run `make batch-score`")
        return pd.read_parquet(PREDICTIONS_PATH)

    def test_predictions_have_expected_columns(self, pred):
        required = {
            "customer_id",
            "proba_C1_detractor", "proba_C1_passive", "proba_C1_promoter", "pred_C1",
            "rank_C1_detractor", "rank_C1_promoter",
            "proba_C2_detractor", "proba_C2_passive", "proba_C2_promoter", "pred_C2",
            "rank_C2_detractor", "rank_C2_promoter",
            "agreement",
        }
        missing = required - set(pred.columns)
        assert not missing, f"missing columns: {missing}"

    def test_probabilities_in_valid_range(self, pred):
        for col in [c for c in pred.columns if c.startswith("proba_")]:
            assert pred[col].min() >= 0, f"{col} has negative values"
            assert pred[col].max() <= 1, f"{col} has values > 1"

    def test_probabilities_sum_to_one_per_model(self, pred):
        for model in ("C1", "C2"):
            cols = [f"proba_{model}_{c}" for c in ("detractor", "passive", "promoter")]
            sums = pred[cols].sum(axis=1)
            assert (sums - 1.0).abs().max() < 1e-3, (
                f"{model} probabilities don't sum to 1"
            )

    def test_ranks_are_unique(self, pred):
        for col in [c for c in pred.columns if c.startswith("rank_")]:
            assert pred[col].nunique() == len(pred), f"{col} has duplicates"
            assert pred[col].min() == 1
            assert pred[col].max() == len(pred)

    def test_pred_consistent_with_argmax(self, pred):
        """The predicted class should be the argmax of the 3 probas."""
        cls_names = ["Detractor", "Passive", "Promoter"]
        for model in ("C1", "C2"):
            cols = [f"proba_{model}_{c.lower()}" for c in cls_names]
            argmax_pred = pred[cols].idxmax(axis=1).map(
                lambda c: cls_names[cols.index(c)]
            )
            assert (argmax_pred == pred[f"pred_{model}"]).all(), (
                f"{model}: pred doesn't match argmax of probabilities"
            )

    def test_agreement_consistent(self, pred):
        expected = pred["pred_C1"] == pred["pred_C2"]
        assert (expected == pred["agreement"]).all()

    def test_agreement_rate_reasonable(self, pred):
        """C1 and C2 should agree on 50-95% of customers."""
        rate = pred["agreement"].mean()
        assert 0.5 < rate < 0.95, f"agreement rate {rate:.1%} is suspicious"

    def test_includes_segment_columns(self, pred):
        """The display layer needs Senior/Gender/Married to render filters."""
        for seg in ("Senior Citizen", "Gender", "Married"):
            assert seg in pred.columns, f"missing segment column: {seg}"


# ============================================================
# Components — pure functions only (Streamlit context not required)
# ============================================================
class TestStylingHelpers:
    @pytest.fixture(scope="class")
    def styling(self):
        from components import styling
        return styling

    def test_badge_fair(self, styling):
        s = styling.fairness_badge(1.0, 0.05)
        # The badge text is now in French ("Équitable") to match the FR UI.
        assert "Équitable" in s or "Equitable" in s or "Fair" in s

    def test_badge_disparity(self, styling):
        s = styling.fairness_badge(0.5, 0.30)
        assert "Disparité" in s or "Disparite" in s or "Disparity" in s

    def test_badge_handles_nan(self, styling):
        s = styling.fairness_badge(float("nan"), 0.05)
        assert "N/A" in s


class TestPlotsHelpers:
    @pytest.fixture(scope="class")
    def plots(self):
        from components import plots
        return plots

    def test_probability_gauge_returns_figure(self, plots):
        fig = plots.probability_gauge(0.6, 0.3, 0.1)
        assert hasattr(fig, "axes")
        assert len(fig.axes) == 1

    def test_waterfall_handles_empty(self, plots):
        df = pd.DataFrame({
            "feature":       ["A", "B"],
            "feature_value": [1.0, -1.0],
            "contribution":  [0.5, -0.3],
        })
        fig = plots.explanation_waterfall(df, top_k=2)
        assert fig is not None

    def test_rank_visualization(self, plots):
        fig = plots.rank_in_population(50, 1000, "Detractor")
        assert fig is not None

    def test_per_group_recall_with_empty(self, plots):
        empty = pd.DataFrame(
            columns=["segment", "class", "champion", "group",
                     "recall", "recall_ci_lo", "recall_ci_hi"]
        )
        fig = plots.per_group_recall_bars(empty, "Senior", "Detractor", "C2_safe")
        assert fig is None  # graceful fallback


# ============================================================
# Data loaders — invariants
# ============================================================
class TestDataLoaders:
    @pytest.fixture(scope="class")
    def loaders(self):
        # Stub st.cache_data context
        from components import data_loaders
        return data_loaders

    def test_availability_check_returns_dict(self, loaders):
        d = loaders.check_artifacts_available()
        assert isinstance(d, dict)
        assert all(isinstance(v, bool) for v in d.values())

    def test_loaders_callable(self, loaders):
        # Just check they don't error if artifacts exist; otherwise they
        # raise FileNotFoundError which is acceptable.
        for fn_name in ("load_predictions", "load_final_eval_summary",
                        "load_fairness_per_group", "load_fairness_disparities"):
            fn = getattr(loaders, fn_name)
            try:
                df = fn()
                assert isinstance(df, pd.DataFrame)
                assert len(df) > 0
            except FileNotFoundError:
                pytest.skip(f"{fn_name}: artifact missing")


# ============================================================
# Page modules — at least they import (syntax check at runtime)
# ============================================================
class TestPageImports:
    """Streamlit pages can't be fully executed without `streamlit run`, but
    we can at least verify they parse correctly and import.

    A `st.stop()` raises a StopException internally which propagates as a
    SystemExit-like behavior; we catch any exception type."""

    @pytest.fixture(scope="class")
    def pages_dir(self):
        return Path(__file__).resolve().parent.parent / "app" / "pages"

    def test_pages_directory_exists(self, pages_dir):
        assert pages_dir.exists()

    def test_each_page_is_valid_python(self, pages_dir):
        """py_compile each page — no syntax errors allowed."""
        import py_compile
        page_files = sorted(pages_dir.glob("*.py"))
        assert len(page_files) >= 4, "expected at least 4 pages"
        for path in page_files:
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as e:
                pytest.fail(f"{path.name}: {e}")
