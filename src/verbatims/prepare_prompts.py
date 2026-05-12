"""
Build the prompts file from the local enriched dataset.

CLI:
    python -m src.verbatims.prepare_prompts

Output:
    data/external/verbatim_prompts.parquet  ← upload this to Google Colab
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import DATA_EXTERNAL, DATA_PROCESSED, DEFAULT_TARGET
from src.verbatims.prompts import build_prompts, summarize_prompts


def main() -> None:
    in_path = DATA_PROCESSED / "dataset_with_features.parquet"
    if not in_path.exists():
        print(
            f"\n✗ {in_path} missing — run `make build-features` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("BUILDING VERBATIM PROMPTS")
    print("=" * 60)

    df = pd.read_parquet(in_path)
    print(f"\n[1/3] Loaded {len(df):,} rows from {in_path.name}")

    prompts_df = build_prompts(df, target_col=DEFAULT_TARGET)
    print(f"[2/3] Built {len(prompts_df):,} prompts")

    summary = summarize_prompts(prompts_df)
    print("\nPrompts summary by expected class:")
    print(summary.to_string())

    out_path = DATA_EXTERNAL / "verbatim_prompts.parquet"
    prompts_df.to_parquet(out_path)
    print(f"\n[3/3] ✓ Saved {out_path.relative_to(out_path.parents[2])}")
    print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")
    print("\nNext step: upload this file to Google Colab and run colab_generate_verbatims.ipynb")


if __name__ == "__main__":
    main()
