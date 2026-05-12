"""
Invariant tests for the Phase 4 feature engineering layer.

Run with:
    pytest tests/test_phase4.py -v

These tests are **deterministic**: they use a small synthetic DataFrame with
columns matching the IBM Telco v11.1.3 schema, so they don't depend on
having run `make build-dataset` first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import LEAKY_FEATURES, NPS_CLASSES
from src.features.derive import (
    add_all_derived_features,
    add_financial_ratios,
    add_friction_signals,
    add_loyalty_signals,
    add_population_bucket,
    add_service_bundle,
    add_tenure_bucket,
    list_derived_features,
)
from src.features.pipeline import (
    DROP_FROM_FEATURES,
    ONEHOT_MAX_CARDINALITY,
    build_preprocessing_pipeline,
    describe_pipeline,
    get_feature_names_after_transform,
    split_X_y,
)


# ============================================================
# Synthetic dataset matching the v11.1.3 schema
# ============================================================
@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """A small DataFrame mimicking the v11.1.3 dataset schema."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "Customer ID": [f"C{i:04d}" for i in range(n)],
        "Tenure Months": rng.integers(0, 72, n),
        "Monthly Charges": rng.uniform(20, 120, n).round(2),
        "Total Charges": rng.uniform(0, 8000, n).round(2),
        "Phone Service": rng.choice(["Yes", "No"], n),
        "Multiple Lines": rng.choice(["Yes", "No"], n),
        "Internet Service": rng.choice(["Yes", "No"], n),
        "Internet Type": rng.choice(["Fiber", "DSL", "Cable", "None"], n),
        "Online Security": rng.choice(["Yes", "No"], n),
        "Online Backup": rng.choice(["Yes", "No"], n),
        "Device Protection": rng.choice(["Yes", "No"], n),
        "Tech Support": rng.choice(["Yes", "No"], n),
        "Streaming TV": rng.choice(["Yes", "No"], n),
        "Streaming Movies": rng.choice(["Yes", "No"], n),
        "Streaming Music": rng.choice(["Yes", "No"], n),
        "Unlimited Data": rng.choice(["Yes", "No"], n),
        "Paperless Billing": rng.choice(["Yes", "No"], n),
        "Payment Method": rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], n,
        ),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "Number of Referrals": rng.integers(0, 6, n),
        "Total Refunds": rng.choice([0.0] * 9 + [25.0], n),
        "Total Extra Data Charges": rng.choice([0.0] * 9 + [10.0], n),
        "Offer": rng.choice([None] * 4 + ["A", "B"], n),
        "Population": rng.integers(500, 100000, n),
        "Zip Code": rng.integers(90001, 96162, n).astype(str),
        "City": rng.choice([f"City_{i}" for i in range(50)], n),
        "Latitude": rng.uniform(32, 42, n).round(2),
        "Longitude": rng.uniform(-124, -114, n).round(2),
        "Gender": rng.choice(["Male", "Female"], n),
        "Senior Citizen": rng.choice(["Yes", "No"], n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Married": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "Number of Dependents": rng.integers(0, 5, n),
        "Age": rng.integers(18, 80, n),
        "Under 30": rng.choice(["Yes", "No"], n),
        # Target
        "NPS_baseline": pd.Categorical(
            rng.choice(NPS_CLASSES, n), categories=NPS_CLASSES, ordered=True,
        ),
        "NPS_alternative": pd.Categorical(
            rng.choice(NPS_CLASSES, n), categories=NPS_CLASSES, ordered=True,
        ),
    })
    return df.set_index("Customer ID")


# ============================================================
# Individual feature derivations
# ============================================================
class TestServiceBundle:
    def test_n_services_in_range(self, synthetic_df):
        out = add_service_bundle(synthetic_df)
        assert "n_services" in out.columns
        assert out["n_services"].min() >= 0
        # 11 service columns max in v11.1.3
        assert out["n_services"].max() <= 11

    def test_n_addons_in_range(self, synthetic_df):
        out = add_service_bundle(synthetic_df)
        assert "n_addons" in out.columns
        assert 0 <= out["n_addons"].min()
        assert out["n_addons"].max() <= 4

    def test_security_bundle_flag(self, synthetic_df):
        out = add_service_bundle(synthetic_df)
        assert "has_security_bundle" in out.columns
        assert set(out["has_security_bundle"].unique()) <= {0, 1}


class TestFinancialRatios:
    def test_avg_monthly_charge_finite(self, synthetic_df):
        out = add_service_bundle(synthetic_df)
        out = add_financial_ratios(out)
        assert "avg_monthly_charge" in out.columns
        assert out["avg_monthly_charge"].isna().sum() == 0
        assert np.isfinite(out["avg_monthly_charge"]).all()

    def test_charges_per_service_no_div_by_zero(self, synthetic_df):
        out = add_service_bundle(synthetic_df)
        out = add_financial_ratios(out)
        assert np.isfinite(out["charges_per_service"]).all()


class TestTenureBucket:
    def test_buckets_categorical(self, synthetic_df):
        out = add_tenure_bucket(synthetic_df)
        assert "tenure_bucket" in out.columns
        expected = {"0-6m", "7-24m", "25-48m", "49m+"}
        assert set(out["tenure_bucket"].unique()) <= expected

    def test_no_nan_buckets(self, synthetic_df):
        out = add_tenure_bucket(synthetic_df)
        assert out["tenure_bucket"].isna().sum() == 0


class TestPopulationBucket:
    def test_quartiles(self, synthetic_df):
        out = add_population_bucket(synthetic_df)
        assert "pop_density_bucket" in out.columns
        expected = {"Q1_rural", "Q2", "Q3", "Q4_urban", "unknown"}
        assert set(out["pop_density_bucket"].unique()) <= expected


class TestFriction:
    def test_extra_charges_flag(self, synthetic_df):
        out = add_friction_signals(synthetic_df)
        assert "has_extra_charges" in out.columns
        assert set(out["has_extra_charges"].unique()) <= {0, 1}

    def test_refund_flag(self, synthetic_df):
        out = add_friction_signals(synthetic_df)
        assert "has_refund" in out.columns
        assert set(out["has_refund"].unique()) <= {0, 1}


class TestLoyalty:
    def test_referrer_flag(self, synthetic_df):
        out = add_loyalty_signals(synthetic_df)
        assert "is_referrer" in out.columns
        assert set(out["is_referrer"].unique()) <= {0, 1}


# ============================================================
# Combined pipeline
# ============================================================
class TestAddAll:
    def test_all_listed_features_present(self, synthetic_df):
        out = add_all_derived_features(synthetic_df)
        for feat in list_derived_features():
            assert feat in out.columns, f"Derived feature missing: {feat}"

    def test_no_input_columns_lost(self, synthetic_df):
        out = add_all_derived_features(synthetic_df)
        for col in synthetic_df.columns:
            assert col in out.columns, f"Original column lost: {col}"

    def test_idempotent(self, synthetic_df):
        once = add_all_derived_features(synthetic_df)
        twice = add_all_derived_features(once)
        # Same shape, same column set
        assert once.shape == twice.shape
        assert set(once.columns) == set(twice.columns)


# ============================================================
# Sklearn pipeline
# ============================================================
class TestPipeline:
    def test_pipeline_builds(self, synthetic_df):
        df_with = add_all_derived_features(synthetic_df)
        X, _ = split_X_y(df_with)
        pipe = build_preprocessing_pipeline(X)
        assert pipe is not None

    def test_pipeline_fits_and_transforms(self, synthetic_df):
        df_with = add_all_derived_features(synthetic_df)
        X, _ = split_X_y(df_with)
        pipe = build_preprocessing_pipeline(X)
        Xt = pipe.fit_transform(X)
        assert Xt.shape[0] == len(X)
        assert Xt.shape[1] >= X.shape[1]  # one-hot expands columns
        assert np.isfinite(Xt).all()

    def test_geographic_columns_dropped(self, synthetic_df):
        df_with = add_all_derived_features(synthetic_df)
        X, _ = split_X_y(df_with)
        pipe = build_preprocessing_pipeline(X)
        feature_names = get_feature_names_after_transform(pipe, X)
        for geo in DROP_FROM_FEATURES:
            assert geo not in feature_names, (
                f"{geo} should be dropped before encoding"
            )

    def test_target_columns_not_in_X(self, synthetic_df):
        df_with = add_all_derived_features(synthetic_df)
        X, y = split_X_y(df_with)
        for col in X.columns:
            assert not col.startswith("NPS_"), (
                f"Target column {col} leaked into X"
            )
        assert y.name == "NPS_baseline"

    def test_no_leakers_in_pipeline_output(self, synthetic_df):
        """The preprocessing pipeline must not output any LEAKY_FEATURES,
        even if they somehow got through."""
        df_with = add_all_derived_features(synthetic_df)
        X, _ = split_X_y(df_with)
        pipe = build_preprocessing_pipeline(X)
        feature_names = get_feature_names_after_transform(pipe, X)
        for leaker in LEAKY_FEATURES:
            assert leaker not in feature_names, (
                f"Leaker '{leaker}' present in pipeline output"
            )

    def test_describe_pipeline_returns_one_row_per_input(self, synthetic_df):
        df_with = add_all_derived_features(synthetic_df)
        X, _ = split_X_y(df_with)
        desc = describe_pipeline(X)
        assert len(desc) == X.shape[1]


# ============================================================
# Cardinality & encoding policy
# ============================================================
class TestEncodingPolicy:
    def test_low_cardinality_goes_to_onehot(self, synthetic_df):
        df_with = add_all_derived_features(synthetic_df)
        X, _ = split_X_y(df_with)
        desc = describe_pipeline(X)
        for _, row in desc.iterrows():
            if row["policy"] == "categorical" and row["n_unique"] <= ONEHOT_MAX_CARDINALITY:
                assert row["encoding"] == "OneHot"

    def test_high_cardinality_goes_to_ordinal(self, synthetic_df):
        df_with = add_all_derived_features(synthetic_df)
        X, _ = split_X_y(df_with)
        desc = describe_pipeline(X)
        for _, row in desc.iterrows():
            if row["policy"] == "categorical" and row["n_unique"] > ONEHOT_MAX_CARDINALITY:
                assert row["encoding"] == "Ordinal"
