"""
Train and evaluate the three baseline models for NPS prediction.

Models:
    1. Logistic Regression (multinomial)
    2. Ordinal Logistic Regression (mord.LogisticAT)
    3. LightGBM (default params)

CLI:
    python -m src.models.baseline
    python -m src.models.baseline --mapping baseline    # only NPS_baseline
    python -m src.models.baseline --quick               # skip ordinal
"""

from __future__ import annotations

import argparse
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.config import (
    BASELINES_DIR,
    DATA_PROCESSED,
    LIGHTGBM_DEFAULT_PARAMS,
    MODELS_DIR,
    NPS_CLASS_TO_INT,
    NPS_MAPPINGS,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.data.split import get_split, load_splits
from src.evaluation.metrics import evaluate_on_splits
# Import OrdinalWrapper from wrappers module to ensure stable pickle paths
from src.models.wrappers import OrdinalWrapper

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Model factories
# ============================================================
def make_logistic() -> LogisticRegression:
    """Multinomial logistic regression — simplest baseline.

    sklearn ≥ 1.5 uses multinomial automatically when there are >2 classes.
    """
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def make_ordinal():
    """Ordinal logistic (LogisticAT from `mord`)."""
    try:
        from mord import LogisticAT
    except ImportError as e:
        raise ImportError(
            "mord is required for ordinal regression. Install via:\n"
            "  pip install mord"
        ) from e
    return LogisticAT(alpha=1.0)


def make_lightgbm():
    """LightGBM with default params + class_weight='balanced'."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        raise ImportError(
            "lightgbm is required. Install via:\n  pip install lightgbm"
        ) from e
    params = LIGHTGBM_DEFAULT_PARAMS.copy()
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_data_in_leaf"],
        feature_fraction=params["feature_fraction"],
        bagging_fraction=params["bagging_fraction"],
        bagging_freq=params["bagging_freq"],
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )


MODEL_FACTORIES = {
    "logistic": make_logistic,
    "ordinal": make_ordinal,
    "lightgbm": make_lightgbm,
}


# ============================================================
# Train one model
# ============================================================
def train_one_model(
    name: str,
    df: pd.DataFrame,
    splits: pd.Series,
    target_col: str,
    pipeline,
    feature_cols: list[str],
):
    """Train one baseline model on the train split, evaluate on all splits."""
    X_train, y_train = get_split(df, splits, "train", target_col=target_col)
    X_train = X_train[feature_cols]

    print(f"  Training {name} on n={len(X_train)} (target={target_col})...")
    model = MODEL_FACTORIES[name]()

    X_train_enc = pipeline.transform(X_train)

    if name == "ordinal":
        wrapped = OrdinalWrapper(model)
        wrapped.fit(X_train_enc, y_train)
        fitted = wrapped
    else:
        if hasattr(y_train, "to_numpy"):
            y_train_arr = y_train.to_numpy()
        else:
            y_train_arr = y_train
        if y_train_arr.dtype.kind in {"U", "O"} or hasattr(y_train_arr, "categories"):
            y_train_int = np.array([NPS_CLASS_TO_INT[str(v)] for v in y_train_arr], dtype=int)
        else:
            y_train_int = np.asarray(y_train_arr, dtype=int)
        model.fit(X_train_enc, y_train_int)
        fitted = model

    results = evaluate_on_splits(
        model=fitted, df=df, splits=splits,
        target_col=target_col, pipeline=pipeline,
        feature_cols=feature_cols,
    )
    results["model"] = name
    results["target"] = target_col
    return fitted, results


def run_baselines(
    mappings: list[str],
    skip_ordinal: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print("=" * 70)
        print("TRAINING BASELINES")
        print("=" * 70)

    df = pd.read_parquet(DATA_PROCESSED / "dataset_with_features.parquet")
    splits = load_splits("response_biased")
    pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
    feature_cols = [c for c in df.columns if not c.startswith("NPS_")]

    if verbose:
        print(f"\nDataset: {df.shape[0]:,} × {df.shape[1]} ({len(feature_cols)} features)")
        print(f"Splits: {dict(splits.value_counts())}")
        print()

    all_results: list[pd.DataFrame] = []
    model_names = ["logistic", "ordinal", "lightgbm"]
    if skip_ordinal:
        model_names.remove("ordinal")

    for target_col in mappings:
        target_full = f"NPS_{target_col}" if not target_col.startswith("NPS_") else target_col
        if target_full not in df.columns:
            print(f"⚠ Target {target_full} not in dataset — skipping")
            continue

        if verbose:
            print(f"--- Target: {target_full} ---")

        for name in model_names:
            try:
                fitted, results = train_one_model(
                    name=name, df=df, splits=splits,
                    target_col=target_full, pipeline=pipeline,
                    feature_cols=feature_cols,
                )
                out_path = BASELINES_DIR / f"{name}_{target_col}.joblib"
                joblib.dump(fitted, out_path)
                if verbose:
                    print(f"    ✓ Saved {out_path.name}")
                all_results.append(results)
            except ImportError as e:
                print(f"    ⚠ Skipping {name}: {e}")
                continue
        if verbose:
            print()

    if not all_results:
        raise RuntimeError("No models trained successfully — check imports.")

    return pd.concat(all_results, ignore_index=True)


def save_results(results_df: pd.DataFrame, name: str = "baseline_results") -> None:
    out_data = RESULTS_DIR / f"{name}.parquet"
    results_df.to_parquet(out_data, index=False)
    print(f"✓ Saved results: {out_data}")

    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 70)
    cols = ["target", "model", "split", "n", "qwk", "macro_f1", "balanced_acc",
            "detractor_recall", "lift@10"]
    cols = [c for c in cols if c in results_df.columns]
    summary = results_df[cols].sort_values(["target", "model", "split"])
    print(summary.to_string(index=False))


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        choices=["baseline", "alternative", "both"],
        default="both",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.mapping == "both":
        mappings = list(NPS_MAPPINGS.keys())
    else:
        mappings = [args.mapping]

    try:
        results = run_baselines(
            mappings=mappings, skip_ordinal=args.quick, verbose=True,
        )
        save_results(results)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n✗ {e}", file=sys.stderr)
        print(
            "\nMake sure you've run:\n"
            "  make build-dataset\n"
            "  make build-features\n"
            "  make build-splits\n",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
