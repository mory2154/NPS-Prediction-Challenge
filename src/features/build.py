"""
Build derived features and fit the preprocessing pipeline.

CLI:
    python -m src.features.build
"""

from __future__ import annotations

import sys

import joblib
import pandas as pd

from src.config import DATA_PROCESSED, DEFAULT_TARGET, MODELS_DIR
from src.features.derive import add_all_derived_features, list_derived_features
from src.features.pipeline import (
    build_preprocessing_pipeline, describe_pipeline, split_X_y,
)


def main() -> None:
    print("=" * 60)
    print("BUILDING DERIVED FEATURES + FITTING PIPELINE")
    print("=" * 60)

    in_path = DATA_PROCESSED / "dataset.parquet"
    if not in_path.exists():
        print(f"\n✗ {in_path} missing — run `make build-dataset` first.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(in_path)
    print(f"\n[1/4] Loaded {df.shape[0]:,} × {df.shape[1]} from {in_path.name}")

    # Add derived features
    df_with = add_all_derived_features(df)
    new_cols = [c for c in df_with.columns if c not in df.columns]
    print(f"\n[2/4] Added {len(new_cols)} derived features:")
    for c in new_cols:
        print(f"  + {c:<25} dtype={df_with[c].dtype}")

    # Save enriched dataset
    out_data = DATA_PROCESSED / "dataset_with_features.parquet"
    df_with.to_parquet(out_data)
    print(f"\n[3/4] Saved {out_data.name}")

    # Build & fit pipeline
    X, y = split_X_y(df_with, target_col=DEFAULT_TARGET)
    pipe = build_preprocessing_pipeline(X)
    pipe.fit(X)

    # Describe what was fit
    desc = describe_pipeline(X)
    print(f"\n[4/4] Pipeline encoding plan:")
    print(desc.to_string(index=False))

    out_pipe = MODELS_DIR / "preprocessing_pipeline.joblib"
    joblib.dump(pipe, out_pipe)
    print(f"\n✓ Saved fitted pipeline: {out_pipe.name}")
    print(f"  Output dimensions: {pipe.transform(X.head(5)).shape[1]} encoded features")


if __name__ == "__main__":
    main()
