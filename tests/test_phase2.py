"""
Invariant tests for the Phase 2 processed dataset.

Run with:
    pytest tests/test_phase2.py -v

These tests verify the **outputs** of the build_dataset pipeline. They do not
re-implement the pipeline — they read what's been saved and assert structural
properties that the modelling phase relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    DATA_PROCESSED,
    INDEX_COLUMN,
    LEAKY_FEATURES,
    NPS_CLASSES,
    NPS_MAPPINGS,
)


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """Load the built dataset; skip if Phase 2 hasn't been run yet."""
    path = DATA_PROCESSED / "dataset.parquet"
    if not path.exists():
        pytest.skip(
            f"{path} does not exist — run `make build-dataset` first."
        )
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def metadata() -> dict:
    """Load the metadata file alongside the dataset."""
    path = DATA_PROCESSED / "dataset_metadata.json"
    if not path.exists():
        pytest.skip(f"{path} does not exist.")
    with open(path) as f:
        return json.load(f)


# ============================================================
# No leakage
# ============================================================
class TestNoLeakage:
    def test_no_leaky_feature_in_columns(self, dataset):
        leakers = [c for c in LEAKY_FEATURES if c in dataset.columns]
        assert not leakers, f"Leakers leaked: {leakers}"

    def test_no_satisfaction_score(self, dataset):
        assert "Satisfaction Score" not in dataset.columns

    def test_no_churn_columns(self, dataset):
        churn_cols = [c for c in dataset.columns if c.startswith("Churn")]
        assert not churn_cols, f"Churn columns must be dropped: {churn_cols}"

    def test_no_customer_status(self, dataset):
        assert "Customer Status" not in dataset.columns

    def test_no_cltv(self, dataset):
        assert "CLTV" not in dataset.columns


# ============================================================
# Targets are present and valid
# ============================================================
class TestTargets:
    @pytest.mark.parametrize("mapping_name", list(NPS_MAPPINGS))
    def test_target_column_exists(self, dataset, mapping_name):
        col = f"NPS_{mapping_name}"
        assert col in dataset.columns

    @pytest.mark.parametrize("mapping_name", list(NPS_MAPPINGS))
    def test_target_no_nan(self, dataset, mapping_name):
        col = f"NPS_{mapping_name}"
        assert dataset[col].isna().sum() == 0

    @pytest.mark.parametrize("mapping_name", list(NPS_MAPPINGS))
    def test_target_three_classes(self, dataset, mapping_name):
        col = f"NPS_{mapping_name}"
        unique = set(dataset[col].dropna().unique())
        assert unique == set(NPS_CLASSES), (
            f"Expected {set(NPS_CLASSES)}, got {unique}"
        )

    def test_targets_differ_for_score_3(self, dataset):
        """Sat=3 → Detractor in baseline, Passive in alternative.
        So the two targets must disagree on at least *some* rows.
        (We can't test on Sat=3 directly — Sat Score has been dropped.)"""
        baseline = dataset["NPS_baseline"].astype(str)
        alternative = dataset["NPS_alternative"].astype(str)
        assert (baseline != alternative).sum() > 0


# ============================================================
# Imputation worked
# ============================================================
class TestImputation:
    def test_no_nan_in_total_charges(self, dataset):
        if "Total Charges" in dataset.columns:
            assert dataset["Total Charges"].isna().sum() == 0

    def test_no_nan_in_internet_type(self, dataset):
        if "Internet Type" in dataset.columns:
            assert dataset["Internet Type"].isna().sum() == 0

    def test_internet_type_has_none_category(self, dataset):
        if "Internet Type" in dataset.columns:
            values = set(dataset["Internet Type"].astype(str).unique())
            # If imputation kicked in, "None" should appear
            assert "None" in values or len(values) <= 4


# ============================================================
# Shape and structure
# ============================================================
class TestShape:
    def test_row_count(self, dataset):
        assert 7000 <= len(dataset) <= 7100

    def test_minimum_features(self, dataset):
        # ~40 features + 2 targets minimum
        assert dataset.shape[1] >= 35

    def test_index_is_customer_id(self, dataset):
        assert dataset.index.name == INDEX_COLUMN

    def test_index_unique(self, dataset):
        assert dataset.index.is_unique

    def test_no_constant_columns(self, dataset):
        for col in dataset.columns:
            n_unique = dataset[col].nunique(dropna=False)
            assert n_unique > 1, f"Column '{col}' is constant"


# ============================================================
# Metadata sanity
# ============================================================
class TestMetadata:
    def test_metadata_has_version(self, metadata):
        assert "version" in metadata

    def test_metadata_has_random_seed(self, metadata):
        assert metadata.get("random_seed") == 42

    def test_metadata_lists_targets(self, metadata):
        targets = metadata.get("target_columns", [])
        assert "NPS_baseline" in targets
        assert "NPS_alternative" in targets

    def test_metadata_records_imputations(self, metadata):
        assert "imputations" in metadata
        assert "total_charges" in metadata["imputations"]
        assert "internet_type" in metadata["imputations"]

    def test_metadata_records_drops(self, metadata):
        dropped = metadata.get("dropped_columns", {}).get("dropped", [])
        assert "Satisfaction Score" in dropped
        assert "Churn Value" in dropped
        assert "Customer Status" in dropped
