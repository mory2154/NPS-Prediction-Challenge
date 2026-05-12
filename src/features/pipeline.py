"""
scikit-learn pipeline for the NPS modelling phase.

Exposes a single function `build_preprocessing_pipeline(df)` that returns a
ready-to-use `ColumnTransformer`. Compatible with logistic regression,
ordinal regression (mord), LightGBM, XGBoost — every model in Phase 6+.

Encoding policy:
    * **Numeric features** → `StandardScaler` (helps logistic / mord;
      neutral for tree models).
    * **Low-cardinality categorical** (≤ 8 levels) → `OneHotEncoder`
      with `handle_unknown="ignore"`.
    * **High-cardinality categorical** (≥ 9 levels) → `OrdinalEncoder`
      with `handle_unknown="use_encoded_value"`.
    * **ZIP / City / Latitude / Longitude** → DROPPED (Population kept
      instead — Phase 11 fairness compliance).
    * **Customer ID** → not in features (it's the index).

Usage:
    from src.features.pipeline import build_preprocessing_pipeline
    from src.features.derive import add_all_derived_features

    df_with = add_all_derived_features(df)
    X = df_with.drop(columns=["NPS_baseline", "NPS_alternative"])
    pipeline = build_preprocessing_pipeline(X)
    X_encoded = pipeline.fit_transform(X)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ============================================================
# Configuration
# ============================================================
ONEHOT_MAX_CARDINALITY = 8

# Geographic features that are too granular or too socio-economic-proxying.
# Population is kept (we use pop_density_bucket from derive.py).
DROP_FROM_FEATURES = [
    "Zip Code",
    "City",
    "Latitude",
    "Longitude",
]


# ============================================================
# Column type detection
# ============================================================
def _classify_columns(X: pd.DataFrame) -> dict[str, list[str]]:
    """
    Inspect dtypes and cardinality, return columns split by encoding policy.

    Returns
    -------
    {
        "numeric":      [...],
        "onehot":       [...],   # categorical, ≤ ONEHOT_MAX_CARDINALITY
        "ordinal":      [...],   # categorical, > ONEHOT_MAX_CARDINALITY
        "dropped":      [...],   # explicitly dropped (geo, etc.)
    }
    """
    numeric: list[str] = []
    onehot: list[str] = []
    ordinal: list[str] = []
    dropped: list[str] = []

    for col in X.columns:
        if col in DROP_FROM_FEATURES:
            dropped.append(col)
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric.append(col)
        else:
            n_unique = X[col].nunique(dropna=True)
            if n_unique <= ONEHOT_MAX_CARDINALITY:
                onehot.append(col)
            else:
                ordinal.append(col)

    return {
        "numeric": numeric,
        "onehot": onehot,
        "ordinal": ordinal,
        "dropped": dropped,
    }


# ============================================================
# Pipeline construction
# ============================================================
def build_preprocessing_pipeline(
    X: pd.DataFrame,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build a preprocessing ColumnTransformer matched to the feature matrix X.

    Parameters
    ----------
    X : the feature matrix (no target columns) — used for column inspection
    scale_numeric : whether to z-score numeric features. Set False for
        tree-only pipelines if you prefer to skip the (cheap) scaling.

    Returns
    -------
    Unfitted ColumnTransformer.
    """
    cols = _classify_columns(X)

    transformers: list[tuple] = []

    # Numeric
    if cols["numeric"]:
        numeric_transformer = (
            StandardScaler() if scale_numeric else "passthrough"
        )
        transformers.append(("num", numeric_transformer, cols["numeric"]))

    # One-hot for low cardinality
    if cols["onehot"]:
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop=None,  # keep all levels — interpretability matters more than colinearity
        )
        transformers.append(("ohe", ohe, cols["onehot"]))

    # Ordinal for high cardinality
    if cols["ordinal"]:
        ordinal = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        transformers.append(("ord", ordinal, cols["ordinal"]))

    # Anything else is dropped silently — including DROP_FROM_FEATURES
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0,
    )


# ============================================================
# Diagnostics
# ============================================================
def describe_pipeline(X: pd.DataFrame) -> pd.DataFrame:
    """Show how each input column will be processed."""
    cols = _classify_columns(X)
    rows = []
    for col in X.columns:
        if col in cols["numeric"]:
            policy = "scale" if True else "passthrough"
            n = X[col].nunique(dropna=True)
            rows.append({"column": col, "policy": "numeric", "encoding": "StandardScaler", "n_unique": n})
        elif col in cols["onehot"]:
            n = X[col].nunique(dropna=True)
            rows.append({"column": col, "policy": "categorical", "encoding": "OneHot", "n_unique": n})
        elif col in cols["ordinal"]:
            n = X[col].nunique(dropna=True)
            rows.append({"column": col, "policy": "categorical", "encoding": "Ordinal", "n_unique": n})
        elif col in cols["dropped"]:
            rows.append({"column": col, "policy": "dropped", "encoding": "—", "n_unique": X[col].nunique(dropna=True)})
        else:
            rows.append({"column": col, "policy": "??", "encoding": "??", "n_unique": -1})
    return pd.DataFrame(rows)


def get_feature_names_after_transform(
    pipeline: ColumnTransformer, X: pd.DataFrame,
) -> list[str]:
    """
    Return the output column names after `pipeline.fit_transform(X)`.

    Useful for SHAP plots, feature importance tables, etc.
    """
    if not hasattr(pipeline, "transformers_"):
        # Pipeline not fitted yet → fit on a sample
        pipeline.fit(X)
    return pipeline.get_feature_names_out().tolist()


# ============================================================
# Targets / features split
# ============================================================
TARGET_PREFIX = "NPS_"


def split_X_y(
    df: pd.DataFrame, target_col: str = "NPS_baseline",
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into (X, y), excluding all NPS_* columns from X."""
    feature_cols = [c for c in df.columns if not c.startswith(TARGET_PREFIX)]
    if target_col not in df.columns:
        raise ValueError(
            f"Target '{target_col}' not in DataFrame. "
            f"Available: {[c for c in df.columns if c.startswith(TARGET_PREFIX)]}"
        )
    X = df[feature_cols]
    y = df[target_col]
    return X, y
