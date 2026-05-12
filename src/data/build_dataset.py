"""
Build the modelling-ready dataset: raw → processed.

Pipeline:
    1. Load raw IBM Telco (5 customer-keyed files + 1 zip-keyed)
    2. Add both NPS targets (baseline + alternative)
    3. Impute Total Charges (= Tenure × Monthly Charges for the 11 missing)
    4. Impute Internet Type (NaN → "None" — clients without internet)
    5. Drop leaky features and constants
    6. Set Customer ID as index
    7. Validate invariants (no leak, no NaN target, etc.)
    8. Save to data/processed/dataset.parquet
    9. Save schema and decisions to data/processed/dataset_metadata.json

CLI:
    python -m src.data.build_dataset
    python -m src.data.build_dataset --no-save     # dry run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DEFAULT_TARGET,
    ID_AND_CONSTANT_FEATURES,
    INDEX_COLUMN,
    LEAKY_FEATURES,
    NPS_MAPPINGS,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.data.load import load_raw_telco
from src.data.target import TARGET_PREFIX, add_all_targets


# ============================================================
# Imputations
# ============================================================
def impute_total_charges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing Total Charges as Tenure Months × Monthly Charges.

    For the 11 brand-new customers (tenure≈0), this gives 0 or one month of
    charges, which is more semantically defensible than the median.
    """
    out = df.copy()
    info: dict = {"strategy": "tenure_x_monthly_charges", "n_imputed": 0}

    if "Total Charges" not in out.columns:
        return out, info

    n_missing = int(out["Total Charges"].isna().sum())
    info["n_imputed"] = n_missing
    if n_missing == 0:
        return out, info

    fill = (
        out["Tenure Months"].astype("float").fillna(0)
        * out["Monthly Charges"].astype("float").fillna(0)
    )
    out["Total Charges"] = out["Total Charges"].fillna(fill)
    info["sample_imputed_values"] = out.loc[
        df["Total Charges"].isna(), "Total Charges"
    ].head(5).round(2).tolist()
    return out, info


def impute_internet_type(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Replace NaN in Internet Type with the explicit "None" category.

    These NaN correspond to customers without Internet Service — a real
    semantic value, not missing data. Encoding it as "None" makes the
    feature usable downstream (LightGBM can use it, SHAP can interpret it).
    """
    out = df.copy()
    info: dict = {"strategy": "fill_with_none_category", "n_imputed": 0}

    if "Internet Type" not in out.columns:
        return out, info

    n_missing = int(out["Internet Type"].isna().sum())
    info["n_imputed"] = n_missing
    if n_missing == 0:
        return out, info

    out["Internet Type"] = out["Internet Type"].fillna("None").astype("string")
    return out, info


# ============================================================
# Drop leakers and constants
# ============================================================
def drop_leakers_and_constants(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Drop LEAKY_FEATURES and constant/identifier columns."""
    out = df.copy()
    actually_dropped: list[str] = []
    not_in_data: list[str] = []

    candidates = LEAKY_FEATURES + ID_AND_CONSTANT_FEATURES
    for col in candidates:
        if col in out.columns:
            out = out.drop(columns=col)
            actually_dropped.append(col)
        else:
            not_in_data.append(col)

    info = {
        "dropped": actually_dropped,
        "not_in_data": not_in_data,
        "n_dropped": len(actually_dropped),
    }
    return out, info


# ============================================================
# Validation
# ============================================================
def validate_dataset(df: pd.DataFrame) -> None:
    """Hard assertions that must hold on the processed dataset."""
    # No leaker survived the drop
    surviving_leakers = [c for c in LEAKY_FEATURES if c in df.columns]
    assert not surviving_leakers, f"Leakers survived: {surviving_leakers}"

    # No constants survived
    surviving_constants = [c for c in ID_AND_CONSTANT_FEATURES if c in df.columns]
    assert not surviving_constants, f"Constants survived: {surviving_constants}"

    # Target columns present and well-formed
    target_cols = [f"{TARGET_PREFIX}{name}" for name in NPS_MAPPINGS]
    for col in target_cols:
        assert col in df.columns, f"Target column missing: {col}"
        assert df[col].isna().sum() == 0, f"NaN values in {col}"
        assert df[col].nunique() == 3, (
            f"{col} should have 3 levels, has {df[col].nunique()}"
        )

    # Total Charges no longer has NaN
    if "Total Charges" in df.columns:
        assert df["Total Charges"].isna().sum() == 0, (
            "Total Charges still has NaN values"
        )

    # Internet Type imputed if present
    if "Internet Type" in df.columns:
        assert df["Internet Type"].isna().sum() == 0, (
            "Internet Type still has NaN values"
        )

    # Index is unique
    if df.index.name == INDEX_COLUMN:
        assert df.index.is_unique, f"Index {INDEX_COLUMN} is not unique"

    # Reasonable shape (not lower than expected)
    assert len(df) >= 7000, f"Suspiciously few rows: {len(df)}"
    assert df.shape[1] >= 30, f"Suspiciously few columns: {df.shape[1]}"


# ============================================================
# Metadata
# ============================================================
def build_metadata(
    df_final: pd.DataFrame,
    df_raw_shape: tuple[int, int],
    impute_charges_info: dict,
    impute_internet_info: dict,
    drop_info: dict,
) -> dict:
    """Capture the choices we made so the dataset is fully self-describing."""
    return {
        "version": "phase2-v1",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": RANDOM_SEED,
        "raw_shape": list(df_raw_shape),
        "processed_shape": list(df_final.shape),
        "default_target": DEFAULT_TARGET,
        "nps_mappings": {
            name: {str(k): v for k, v in m.items()}
            for name, m in NPS_MAPPINGS.items()
        },
        "imputations": {
            "total_charges": impute_charges_info,
            "internet_type": impute_internet_info,
        },
        "dropped_columns": drop_info,
        "feature_columns": [
            c for c in df_final.columns
            if not c.startswith(TARGET_PREFIX)
        ],
        "target_columns": [
            c for c in df_final.columns if c.startswith(TARGET_PREFIX)
        ],
        "dtypes": {c: str(df_final[c].dtype) for c in df_final.columns},
    }


# ============================================================
# Main pipeline
# ============================================================
def build_dataset(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """End-to-end raw → processed pipeline. Returns (df, metadata)."""
    if verbose:
        print("=" * 60)
        print("BUILDING PROCESSED DATASET")
        print("=" * 60)

    # 1. Load raw
    if verbose:
        print("\n[1/6] Loading raw...")
    df = load_raw_telco(verbose=verbose)
    raw_shape = df.shape

    # 2. Add both NPS targets
    if verbose:
        print("\n[2/6] Adding NPS targets (baseline + alternative)...")
    df = add_all_targets(df)
    for name in NPS_MAPPINGS:
        col = f"{TARGET_PREFIX}{name}"
        dist = df[col].value_counts(normalize=True).round(3) * 100
        if verbose:
            print(f"  {col}: " + " / ".join(
                f"{cls}={dist[cls]:.1f}%" for cls in dist.index
            ))

    # 3. Impute Total Charges
    if verbose:
        print("\n[3/6] Imputing Total Charges (Tenure × Monthly Charges)...")
    df, charges_info = impute_total_charges(df)
    if verbose:
        print(f"  Imputed {charges_info['n_imputed']} rows.")

    # 4. Impute Internet Type
    if verbose:
        print('\n[4/6] Imputing Internet Type (NaN → "None")...')
    df, internet_info = impute_internet_type(df)
    if verbose:
        print(f"  Imputed {internet_info['n_imputed']} rows.")

    # 5. Drop leakers and constants
    if verbose:
        print("\n[5/6] Dropping leakers and constants...")
    df, drop_info = drop_leakers_and_constants(df)
    if verbose:
        print(f"  Dropped {drop_info['n_dropped']} columns: {drop_info['dropped']}")

    # 6. Set index
    if INDEX_COLUMN in df.columns:
        df = df.set_index(INDEX_COLUMN)

    # Validate
    if verbose:
        print("\n[6/6] Validating invariants...")
    validate_dataset(df)
    if verbose:
        print("  ✓ All invariants hold")

    metadata = build_metadata(df, raw_shape, charges_info, internet_info, drop_info)

    if verbose:
        print(f"\n✓ Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  ({df.shape[1] - len(metadata['target_columns'])} features "
              f"+ {len(metadata['target_columns'])} targets)")

    return df, metadata


def save(df: pd.DataFrame, metadata: dict) -> None:
    """Persist dataset + metadata side-by-side in data/processed/."""
    out_data = DATA_PROCESSED / "dataset.parquet"
    out_meta = DATA_PROCESSED / "dataset_metadata.json"

    df.to_parquet(out_data)
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved {out_data.relative_to(PROJECT_ROOT)}")
    print(f"✓ Saved {out_meta.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-save", action="store_true", help="Build but do not write to disk"
    )
    args = parser.parse_args()

    try:
        df, metadata = build_dataset(verbose=True)
        if not args.no_save:
            save(df, metadata)
        else:
            print("\n[--no-save] Not writing to disk.")
    except (FileNotFoundError, ValueError, AssertionError) as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
