"""
Smoke tests to verify the project setup is correct.

Run with:
    pytest tests/ -v
"""

import importlib

import pytest


def test_config_imports():
    """src.config imports and exposes expected constants."""
    from src import config

    assert hasattr(config, "RANDOM_SEED")
    assert config.RANDOM_SEED == 42
    assert hasattr(config, "NPS_CLASSES")
    assert config.NPS_CLASSES == ["Detractor", "Passive", "Promoter"]


def test_leakage_list_includes_critical_features():
    """Sanity check: critical leaky features are in the drop list."""
    from src.config import LEAKY_FEATURES

    must_drop = [
        "Satisfaction Score",
        "Churn Score",
        "Churn Value",
        "Churn Reason",
    ]
    for feat in must_drop:
        assert feat in LEAKY_FEATURES, f"{feat} missing from LEAKY_FEATURES"


def test_directories_created():
    """All project directories exist after importing config."""
    from src.config import (
        DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_EXTERNAL,
        MODELS_DIR, FIGURES_DIR,
    )
    for d in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_EXTERNAL,
              MODELS_DIR, FIGURES_DIR]:
        assert d.exists(), f"Directory {d} does not exist"


@pytest.mark.parametrize("pkg", [
    "pandas", "numpy", "sklearn", "lightgbm", "xgboost",
    "shap", "fairlearn", "streamlit", "joblib",
])
def test_core_dependencies_importable(pkg):
    """Required dependencies are installed."""
    importlib.import_module(pkg)


def test_satisfaction_mapping_covers_all_scores():
    """Baseline mapping covers Satisfaction Score 1-5 fully."""
    from src.config import SATISFACTION_TO_NPS_BASELINE, NPS_CLASSES

    assert set(SATISFACTION_TO_NPS_BASELINE.keys()) == {1, 2, 3, 4, 5}
    assert all(v in NPS_CLASSES for v in SATISFACTION_TO_NPS_BASELINE.values())
