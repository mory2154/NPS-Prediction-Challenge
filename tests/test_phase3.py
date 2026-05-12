"""
Invariant tests for the Phase 3 splits.

Run with:
    pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DATA_PROCESSED, DEFAULT_TARGET
from src.data.split import (
    SPLITS_DIR,
    compute_response_propensity,
    get_split,
    load_splits,
    make_naive_splits,
    make_response_biased_splits,
)


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    path = DATA_PROCESSED / "dataset.parquet"
    if not path.exists():
        pytest.skip(f"{path} missing — run `make build-dataset` first.")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def naive_splits(dataset) -> pd.Series:
    return make_naive_splits(dataset)


@pytest.fixture(scope="module")
def biased_splits(dataset) -> pd.Series:
    splits, _ = make_response_biased_splits(dataset)
    return splits


# ============================================================
# Naive split — basic invariants
# ============================================================
class TestNaiveSplit:
    def test_covers_all_rows(self, dataset, naive_splits):
        assert len(naive_splits) == len(dataset)
        assert naive_splits.index.equals(dataset.index)

    def test_three_classes_only(self, naive_splits):
        assert set(naive_splits.unique()) == {"train", "val", "test"}

    def test_no_empty_split(self, naive_splits):
        for s in ["train", "val", "test"]:
            assert (naive_splits == s).sum() > 0

    def test_approximate_ratios(self, naive_splits):
        n = len(naive_splits)
        n_train = (naive_splits == "train").sum() / n
        n_val = (naive_splits == "val").sum() / n
        n_test = (naive_splits == "test").sum() / n
        assert 0.55 < n_train < 0.65
        assert 0.15 < n_val < 0.25
        assert 0.15 < n_test < 0.25

    def test_target_distribution_preserved(self, dataset, naive_splits):
        """Stratified → all splits share the global class ratio (±2 pts)."""
        global_dist = dataset[DEFAULT_TARGET].value_counts(normalize=True)
        for s in ["train", "val", "test"]:
            mask = naive_splits == s
            local = dataset.loc[mask, DEFAULT_TARGET].value_counts(normalize=True)
            for cls in global_dist.index:
                assert abs(local.get(cls, 0) - global_dist[cls]) < 0.02, (
                    f"Class {cls} drift in '{s}' split exceeds 2 pts"
                )


# ============================================================
# Response-biased split
# ============================================================
class TestResponseBiasedSplit:
    def test_covers_all_rows(self, dataset, biased_splits):
        assert len(biased_splits) == len(dataset)

    def test_four_classes(self, biased_splits):
        expected = {"train", "val", "respondent_test", "silent_test"}
        assert set(biased_splits.unique()) == expected

    def test_silent_is_about_85_percent(self, biased_splits):
        n = len(biased_splits)
        silent_share = (biased_splits == "silent_test").sum() / n
        assert 0.83 < silent_share < 0.87, (
            f"silent_test share is {silent_share:.3f}, expected ~0.85"
        )

    def test_respondent_pool_split(self, biased_splits):
        """Within respondents, splits are 60/20/20 of respondents."""
        respondent_mask = biased_splits.isin(["train", "val", "respondent_test"])
        n_resp = respondent_mask.sum()
        train_share = (biased_splits == "train").sum() / n_resp
        val_share = (biased_splits == "val").sum() / n_resp
        rt_share = (biased_splits == "respondent_test").sum() / n_resp
        assert 0.55 < train_share < 0.65
        assert 0.15 < val_share < 0.25
        assert 0.15 < rt_share < 0.25

    def test_respondents_have_higher_tenure(self, dataset, biased_splits):
        """Sanity: response mechanism prefers longer-tenured customers."""
        if "Tenure Months" not in dataset.columns:
            pytest.skip("Tenure Months not in dataset")
        is_resp = biased_splits.isin(["train", "val", "respondent_test"])
        mean_resp = dataset.loc[is_resp, "Tenure Months"].mean()
        mean_silent = dataset.loc[~is_resp, "Tenure Months"].mean()
        assert mean_resp > mean_silent, (
            f"Respondents tenure {mean_resp:.1f} should exceed silent {mean_silent:.1f}"
        )


# ============================================================
# Reproducibility
# ============================================================
class TestReproducibility:
    def test_naive_split_same_seed_same_result(self, dataset):
        s1 = make_naive_splits(dataset, seed=42)
        s2 = make_naive_splits(dataset, seed=42)
        assert s1.equals(s2)

    def test_naive_split_different_seed_different_result(self, dataset):
        s1 = make_naive_splits(dataset, seed=42)
        s2 = make_naive_splits(dataset, seed=99)
        assert not s1.equals(s2)

    def test_biased_split_same_seed_same_result(self, dataset):
        s1, _ = make_response_biased_splits(dataset, seed=42)
        s2, _ = make_response_biased_splits(dataset, seed=42)
        assert s1.equals(s2)

    def test_propensity_same_seed_same_result(self, dataset):
        p1 = compute_response_propensity(dataset, seed=42)
        p2 = compute_response_propensity(dataset, seed=42)
        pd.testing.assert_series_equal(p1, p2)


# ============================================================
# Persistence
# ============================================================
class TestPersistence:
    def test_save_and_load_naive(self, dataset, naive_splits, tmp_path, monkeypatch):
        # Use the public save/load API via real module path
        from src.data import split as split_mod

        monkeypatch.setattr(split_mod, "SPLITS_DIR", tmp_path)
        split_mod.save_splits(naive_splits, "naive_test")
        loaded = split_mod.load_splits("naive_test")
        # Compare values not dtypes (parquet round-trip may convert object → str)
        assert loaded.tolist() == naive_splits.tolist()
        assert loaded.index.equals(naive_splits.index)


# ============================================================
# get_split helper
# ============================================================
class TestGetSplit:
    def test_get_split_returns_features_and_target(self, dataset, naive_splits):
        X, y = get_split(dataset, naive_splits, "train")
        assert len(X) == (naive_splits == "train").sum()
        assert len(y) == len(X)
        # Targets should not be in feature matrix
        assert "NPS_baseline" not in X.columns
        assert "NPS_alternative" not in X.columns

    def test_get_split_unknown_kind_raises(self, dataset, naive_splits):
        with pytest.raises(ValueError):
            get_split(dataset, naive_splits, "nonexistent")
