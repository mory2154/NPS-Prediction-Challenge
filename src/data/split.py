"""
Splitting strategies for the NPS prediction problem.

Two strategies are implemented:

  A. **Naive stratified split** (60/20/20)
     Baseline reference. Stratified by the target class. Used to measure
     the effect of selection bias by comparison.

  B. **Response-biased split** (recommended)
     Models the survey response mechanism explicitly. ~15 % of customers
     are sampled into a "respondent pool" with probability proportional
     to engagement signals (tenure, paperless billing, autopay, etc.).
     The respondent pool is then split 60/20/20 train/val/respondent_test;
     the remaining 85 % become `silent_test` — the honest evaluation
     of how the model would perform on the deployment population.

The split outputs are saved as parquet files indexed by Customer ID,
with a single `split` column taking values:
    'train' / 'val' / 'test'           (naive)
    'train' / 'val' / 'respondent_test' / 'silent_test'  (response-biased)

CLI:
    python -m src.data.split
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DEFAULT_TARGET,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
)


# ============================================================
# Response propensity model
# ============================================================
DEFAULT_RESPONSE_WEIGHTS = {
    # Strong positive — engaged customers respond more
    "tenure": 1.5,           # primary signal (per user choice)
    "paperless_billing": 0.7,
    "autopay": 0.6,
    # Mild positive — older customers often more responsive
    "age": 0.3,
    # Negative — fast-moving customers respond less
    "month_to_month": -0.5,
}


def _zscore(x: pd.Series) -> pd.Series:
    """Z-score with safe handling of constant series."""
    std = x.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean()) / std


def _build_features_for_propensity(df: pd.DataFrame) -> pd.DataFrame:
    """Extract & normalize the features used to compute response propensity."""
    feats = pd.DataFrame(index=df.index)

    # Tenure (numeric, z-scored)
    if "Tenure Months" in df.columns:
        feats["tenure"] = _zscore(df["Tenure Months"].astype(float))
    else:
        feats["tenure"] = 0.0

    # Paperless billing (binary)
    if "Paperless Billing" in df.columns:
        feats["paperless_billing"] = (
            df["Paperless Billing"].astype(str).str.lower() == "yes"
        ).astype(float)
    else:
        feats["paperless_billing"] = 0.5

    # Autopay (string-detected from Payment Method)
    if "Payment Method" in df.columns:
        feats["autopay"] = (
            df["Payment Method"]
            .astype(str)
            .str.lower()
            .str.contains("automatic", na=False)
        ).astype(float)
    else:
        feats["autopay"] = 0.5

    # Age (z-scored if present)
    if "Age" in df.columns:
        feats["age"] = _zscore(df["Age"].astype(float))
    else:
        feats["age"] = 0.0

    # Month-to-month flag
    if "Contract" in df.columns:
        feats["month_to_month"] = (
            df["Contract"].astype(str).str.lower() == "month-to-month"
        ).astype(float)
    else:
        feats["month_to_month"] = 0.5

    return feats


def compute_response_propensity(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    noise_scale: float = 0.5,
    seed: int = RANDOM_SEED,
) -> pd.Series:
    """
    Score each customer with a response propensity.

    Higher propensity = more likely to answer the NPS survey.

    Parameters
    ----------
    df : the dataset (Customer ID as index expected)
    weights : optional override of DEFAULT_RESPONSE_WEIGHTS
    noise_scale : standard deviation of Gaussian noise added to the logit.
                  This is what makes the sampling "weighted random" rather
                  than purely deterministic.
    seed : reproducibility seed

    Returns
    -------
    Series indexed like df, values in (0, 1)
    """
    weights = weights or DEFAULT_RESPONSE_WEIGHTS
    feats = _build_features_for_propensity(df)

    logit = pd.Series(0.0, index=df.index)
    for feat_name, beta in weights.items():
        if feat_name in feats.columns:
            logit += beta * feats[feat_name]

    rng = np.random.default_rng(seed)
    logit += rng.normal(0, noise_scale, size=len(logit))

    # Sigmoid to (0, 1)
    propensity = 1.0 / (1.0 + np.exp(-logit))
    return propensity.rename("response_propensity")


# ============================================================
# Strategy A — naive stratified split
# ============================================================
def make_naive_splits(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    train_size: float = 1 - TEST_SIZE - VAL_SIZE,  # 0.6
    val_size: float = VAL_SIZE,                     # 0.2
    seed: int = RANDOM_SEED,
) -> pd.Series:
    """
    Stratified train/val/test split (naive baseline).

    Returns
    -------
    Series indexed by df.index, values: 'train' / 'val' / 'test'
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in df.")

    rng = np.random.default_rng(seed)
    splits = pd.Series("", index=df.index, dtype=object)
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError(
            f"train+val ({train_size + val_size}) >= 1.0; no room for test"
        )

    # Stratified: same proportion of each class in each split
    for cls in df[target_col].unique():
        idx = df.index[df[target_col] == cls].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_size))
        n_val = int(round(n * val_size))
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        splits.loc[train_idx] = "train"
        splits.loc[val_idx] = "val"
        splits.loc[test_idx] = "test"

    splits.name = "split"
    return splits


# ============================================================
# Strategy B — response-biased split
# ============================================================
def make_response_biased_splits(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    response_rate: float = 0.15,
    train_share: float = 0.60,
    val_share: float = 0.20,
    seed: int = RANDOM_SEED,
) -> tuple[pd.Series, pd.Series]:
    """
    Split simulating the 15 % respondents / 85 % silent gap.

    1. Compute response propensity for each customer.
    2. Sample `response_rate` × n customers WITHOUT replacement, weighted by
       propensity. These are the "respondents".
    3. Within respondents, split (stratified on target) into:
         train (60%) / val (20%) / respondent_test (20%)
    4. The 85 % NOT sampled become `silent_test`.

    Returns
    -------
    splits : Series indexed by df.index. Values:
        'train' / 'val' / 'respondent_test' / 'silent_test'
    propensity : Series of response propensity scores (for diagnostics)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in df.")
    test_share_within = 1.0 - train_share - val_share
    if test_share_within < 0:
        raise ValueError(
            f"train+val={train_share + val_share}, leaves no room for "
            f"respondent_test"
        )

    propensity = compute_response_propensity(df, seed=seed)
    rng = np.random.default_rng(seed + 1)  # different stream for sampling

    n = len(df)
    n_respondents = int(round(response_rate * n))

    # Weighted sampling without replacement, normalized propensity as weights
    weights = propensity.to_numpy()
    weights = weights / weights.sum()

    respondent_pos = rng.choice(
        np.arange(n), size=n_respondents, replace=False, p=weights
    )
    respondent_idx = df.index.to_numpy()[respondent_pos]
    silent_idx = df.index.difference(respondent_idx)

    splits = pd.Series("silent_test", index=df.index, dtype=object)

    # Stratified split within respondents
    rng2 = np.random.default_rng(seed + 2)
    df_resp = df.loc[respondent_idx]
    for cls in df_resp[target_col].unique():
        cls_idx = df_resp.index[df_resp[target_col] == cls].to_numpy()
        rng2.shuffle(cls_idx)
        n_cls = len(cls_idx)
        n_train = int(round(n_cls * train_share))
        n_val = int(round(n_cls * val_share))
        splits.loc[cls_idx[:n_train]] = "train"
        splits.loc[cls_idx[n_train:n_train + n_val]] = "val"
        splits.loc[cls_idx[n_train + n_val:]] = "respondent_test"

    splits.name = "split"
    return splits, propensity


# ============================================================
# Diagnostics
# ============================================================
def summarize_split(
    splits: pd.Series,
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
) -> pd.DataFrame:
    """Per-split: count, target distribution, mean tenure (sanity check)."""
    rows: list[dict] = []
    for s in splits.unique():
        mask = splits == s
        sub = df[mask]
        row = {"split": s, "n": int(mask.sum())}

        # Target distribution
        target_dist = sub[target_col].value_counts(normalize=True) * 100
        for cls, pct in target_dist.items():
            row[f"%_{cls}"] = round(pct, 1)

        # Tenure mean (engagement check)
        if "Tenure Months" in df.columns:
            row["mean_tenure"] = round(sub["Tenure Months"].mean(), 1)

        # Age mean
        if "Age" in df.columns:
            row["mean_age"] = round(sub["Age"].mean(), 1)

        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["pct"] = (summary["n"] / summary["n"].sum() * 100).round(1)
    cols = ["split", "n", "pct"] + [c for c in summary.columns if c not in ("split", "n", "pct")]
    return summary[cols]


# ============================================================
# Public load/save helpers
# ============================================================
SPLITS_DIR = DATA_PROCESSED / "splits"


def save_splits(splits: pd.Series, name: str) -> Path:
    """Persist a splits Series to data/processed/splits/<name>.parquet."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out = SPLITS_DIR / f"splits_{name}.parquet"
    splits.to_frame().to_parquet(out)
    return out


def load_splits(name: str) -> pd.Series:
    """Load a previously saved splits file."""
    path = SPLITS_DIR / f"splits_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `make build-splits` first."
        )
    return pd.read_parquet(path)["split"]


def get_split(
    df: pd.DataFrame, splits: pd.Series, kind: str,
    target_col: str = DEFAULT_TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convenience: extract X, y for a given split kind.

    Example:
        X_train, y_train = get_split(df, splits, "train")
    """
    mask = splits == kind
    if not mask.any():
        raise ValueError(f"No rows with split == '{kind}'")
    feature_cols = [c for c in df.columns if not c.startswith("NPS_")]
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, target_col]
    return X, y


# ============================================================
# CLI
# ============================================================
def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "dataset.parquet")
    print(f"Loaded {len(df):,} rows from data/processed/dataset.parquet\n")

    print("=" * 60)
    print("STRATEGY A — Naive stratified split")
    print("=" * 60)
    splits_naive = make_naive_splits(df)
    print(summarize_split(splits_naive, df).to_string(index=False))
    out_a = save_splits(splits_naive, "naive")
    print(f"\n✓ Saved {out_a.name}")

    print("\n" + "=" * 60)
    print("STRATEGY B — Response-biased split (15% respondents)")
    print("=" * 60)
    splits_rb, propensity = make_response_biased_splits(df)
    print(summarize_split(splits_rb, df).to_string(index=False))
    out_b = save_splits(splits_rb, "response_biased")
    print(f"\n✓ Saved {out_b.name}")

    # Compare: how shifted is the respondent pool from silent?
    print("\n" + "=" * 60)
    print("SHIFT CHECK — respondents vs silent")
    print("=" * 60)
    is_resp = splits_rb.isin(["train", "val", "respondent_test"])
    if "Tenure Months" in df.columns:
        print(f"Mean tenure — respondents: {df.loc[is_resp, 'Tenure Months'].mean():.1f}")
        print(f"Mean tenure — silent:      {df.loc[~is_resp, 'Tenure Months'].mean():.1f}")
    if "Age" in df.columns:
        print(f"Mean age    — respondents: {df.loc[is_resp, 'Age'].mean():.1f}")
        print(f"Mean age    — silent:      {df.loc[~is_resp, 'Age'].mean():.1f}")
    print(
        f"\nA visible difference confirms the response mechanism is working. "
        f"This shift is what makes silent_test a more honest evaluation than naive test."
    )


if __name__ == "__main__":
    main()
