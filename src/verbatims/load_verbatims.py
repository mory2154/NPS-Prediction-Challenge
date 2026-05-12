"""
Load Colab-generated verbatims and integrate them into the local dataset.

CLI:
    python -m src.verbatims.load_verbatims

Expected input:
    data/external/verbatims.parquet
    Columns: Customer ID (index), verbatim, expected_class, counter_intuitive,
             generation_metadata (optional)

Outputs:
    data/processed/dataset_with_verbatims.parquet
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import DATA_EXTERNAL, DATA_PROCESSED


def main() -> None:
    df_path = DATA_PROCESSED / "dataset_with_features.parquet"
    verbatims_path = DATA_EXTERNAL / "verbatims.parquet"

    if not df_path.exists():
        print(f"\n✗ {df_path} missing — run `make build-features` first.",
              file=sys.stderr)
        sys.exit(1)
    if not verbatims_path.exists():
        print(f"\n✗ {verbatims_path} missing — run Colab notebook first.",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("INTEGRATING VERBATIMS")
    print("=" * 60)

    df = pd.read_parquet(df_path)
    verbatims = pd.read_parquet(verbatims_path)
    print(f"\n[1/3] Loaded {len(df):,} customers and {len(verbatims):,} verbatims")

    # Coverage check
    missing = df.index.difference(verbatims.index)
    extra = verbatims.index.difference(df.index)
    if len(missing) > 0:
        print(f"⚠ {len(missing)} customers missing a verbatim")
    if len(extra) > 0:
        print(f"⚠ {len(extra)} verbatims for unknown customers — will be ignored")

    # Quality stats
    if "verbatim" in verbatims.columns:
        lens = verbatims["verbatim"].str.len()
        print(f"\n[2/3] Verbatim length stats:")
        print(f"  Mean: {lens.mean():.0f} chars  /  Median: {lens.median():.0f}")
        print(f"  Min:  {lens.min()} chars      /  Max: {lens.max()}")
        n_short = (lens < 30).sum()
        if n_short > 0:
            print(f"  ⚠ {n_short} verbatims are < 30 chars (suspicious)")

    # Merge
    keep_cols = [c for c in ["verbatim", "counter_intuitive"] if c in verbatims.columns]
    enriched = df.join(verbatims[keep_cols], how="left")
    out_path = DATA_PROCESSED / "dataset_with_verbatims.parquet"
    enriched.to_parquet(out_path)
    print(f"\n[3/3] ✓ Saved {out_path.relative_to(out_path.parents[2])}")
    print(f"  Final shape: {enriched.shape}")


if __name__ == "__main__":
    main()
