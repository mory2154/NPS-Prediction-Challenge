"""
Build the NPS classification target from `Satisfaction Score`.

This module is a small, importable library. The orchestration (drop leakers,
impute, save) lives in `src.data.build_dataset`.

Example usage:
    from src.data.target import add_all_targets, build_target

    df_with_targets = add_all_targets(df)            # adds NPS_baseline + NPS_alternative
    df_one, col = build_target(df, mapping="baseline")

CLI usage (only prints a summary):
    python -m src.data.target
"""

from __future__ import annotations

import pandas as pd

from src.config import NPS_CLASSES, NPS_MAPPINGS

SOURCE_COLUMN = "Satisfaction Score"
TARGET_PREFIX = "NPS_"


def build_target(
    df: pd.DataFrame, mapping: str = "baseline",
) -> tuple[pd.DataFrame, str]:
    """
    Add a single NPS target column to `df`.

    Parameters
    ----------
    df : DataFrame containing 'Satisfaction Score'.
    mapping : key into NPS_MAPPINGS — 'baseline' or 'alternative'.

    Returns
    -------
    (df, target_column_name)
        df with a new ordered Categorical column 'NPS_<mapping>'.
    """
    if SOURCE_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{SOURCE_COLUMN}' not found. Has the data been loaded "
            f"via src.data.load.load_raw_telco?"
        )
    if mapping not in NPS_MAPPINGS:
        raise ValueError(
            f"Unknown mapping '{mapping}'. Available: {list(NPS_MAPPINGS)}"
        )

    target_col = f"{TARGET_PREFIX}{mapping}"
    out = df.copy()

    out[target_col] = out[SOURCE_COLUMN].map(NPS_MAPPINGS[mapping])

    # Sanity: every Satisfaction Score should have a mapping
    n_missing = out[target_col].isna().sum() - df[SOURCE_COLUMN].isna().sum()
    if n_missing > 0:
        unmapped = sorted(
            df.loc[out[target_col].isna() & df[SOURCE_COLUMN].notna(), SOURCE_COLUMN]
            .unique()
        )
        raise ValueError(
            f"{n_missing} rows could not be mapped. Unmapped scores: {unmapped}"
        )

    out[target_col] = pd.Categorical(
        out[target_col], categories=NPS_CLASSES, ordered=True,
    )
    return out, target_col


def add_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add ALL NPS target columns (one per mapping in NPS_MAPPINGS)."""
    out = df.copy()
    for name in NPS_MAPPINGS:
        out, _ = build_target(out, mapping=name)
    return out


def summarise_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each mapping, return the resulting class counts and percentages.

    Returns a DataFrame indexed by NPS class with one column per mapping.
    """
    rows: dict[str, pd.Series] = {}
    for name in NPS_MAPPINGS:
        col = f"{TARGET_PREFIX}{name}"
        if col not in df.columns:
            df_with, col = build_target(df, mapping=name)
        else:
            df_with = df
        counts = df_with[col].value_counts().reindex(NPS_CLASSES, fill_value=0)
        pct = (counts / counts.sum() * 100).round(1)
        rows[f"{name}_n"] = counts
        rows[f"{name}_pct"] = pct
    return pd.DataFrame(rows).reindex(NPS_CLASSES)


# ============================================================
# CLI
# ============================================================
def main() -> None:
    from src.data.load import load_raw_telco

    df = load_raw_telco(verbose=False)
    print(f"Loaded {len(df):,} rows.\n")

    print("Mappings registered in src.config.NPS_MAPPINGS:")
    for name, m in NPS_MAPPINGS.items():
        print(f"\n  {name}:")
        for score in sorted(m):
            print(f"    Sat={score} → {m[score]}")

    print("\n" + "=" * 60)
    print("Class distribution per mapping")
    print("=" * 60)
    print(summarise_mappings(df))


if __name__ == "__main__":
    main()
