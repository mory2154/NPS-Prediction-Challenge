"""
Quantitative leakage audit.

For every feature, measures statistical association with 'Satisfaction Score'
(the source of the NPS target). Outputs a verdict per feature:

  STRONG LEAK  → assoc ≥ 0.5, must be dropped
  WEAK LEAK    → 0.3 ≤ assoc < 0.5, dropped by default but defensible to keep
  CLEAN        → assoc < 0.3, safe to use as a feature
  CONFIGURED   → already in LEAKY_FEATURES (skipped from features regardless)

Confirms (or challenges) the LEAKY_FEATURES list configured in src/config.py.

Run as CLI:
    python -m src.data.audit_leaks
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.config import LEAKY_FEATURES, REPORTS_DIR
from src.data.load import load_raw_telco

STRONG_LEAK_THRESHOLD = 0.5
WEAK_LEAK_THRESHOLD = 0.3


def _verdict(score: float, in_drop_list: bool) -> str:
    """Categorize a feature based on its association score."""
    if in_drop_list:
        return "CONFIGURED"
    if score >= STRONG_LEAK_THRESHOLD:
        return "STRONG LEAK"
    if score >= WEAK_LEAK_THRESHOLD:
        return "WEAK LEAK"
    return "CLEAN"


def _numeric_correlation(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Pearson + Spearman correlation between numeric features and target."""
    rows: list[dict] = []
    numeric_cols = (
        df.select_dtypes(include=[np.number])
        .columns.drop(target, errors="ignore")
        .tolist()
    )
    for col in numeric_cols:
        try:
            pearson = df[col].corr(df[target])
            spearman = df[col].corr(df[target], method="spearman")
            score = abs(pearson) if pd.notna(pearson) else 0.0
            rows.append({
                "feature": col,
                "type": "numeric",
                "pearson_r": round(pearson, 3) if pd.notna(pearson) else np.nan,
                "spearman_r": round(spearman, 3) if pd.notna(spearman) else np.nan,
                "score": round(score, 3),
            })
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(rows)


def _categorical_association(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Eta-squared (R² of one-way ANOVA): 0 = no association, 1 = perfect.
    """
    rows: list[dict] = []
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    grand_mean = df[target].mean()
    ss_total = ((df[target] - grand_mean) ** 2).sum()

    for col in cat_cols:
        try:
            grp = df.groupby(col, observed=True)[target]
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in grp)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
            rows.append({
                "feature": col,
                "type": "categorical",
                "n_categories": int(df[col].nunique(dropna=True)),
                "eta_squared": round(eta_sq, 3),
                "score": round(eta_sq, 3),
            })
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(rows)


def audit(df: pd.DataFrame, target: str = "Satisfaction Score") -> pd.DataFrame:
    """Combined numeric + categorical association audit."""
    num = _numeric_correlation(df, target)
    cat = _categorical_association(df, target)
    audit_df = pd.concat([num, cat], ignore_index=True, sort=False)
    audit_df = audit_df.sort_values("score", ascending=False).reset_index(drop=True)
    audit_df["in_drop_list"] = audit_df["feature"].isin(LEAKY_FEATURES)
    audit_df["verdict"] = audit_df.apply(
        lambda r: _verdict(r["score"], r["in_drop_list"]), axis=1
    )
    return audit_df


def _print_section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    df = load_raw_telco()
    _print_section("LEAKAGE AUDIT — vs 'Satisfaction Score'")

    audit_df = audit(df)

    # Top 25 by score
    cols = [
        "feature", "type", "verdict", "score",
        "pearson_r", "spearman_r", "eta_squared", "in_drop_list",
    ]
    cols = [c for c in cols if c in audit_df.columns]
    print("\nTop 25 features by association with Satisfaction Score:\n")
    print(audit_df.head(25)[cols].to_string(index=False))

    # Save
    out = REPORTS_DIR / "leakage_audit.csv"
    audit_df.to_csv(out, index=False)
    print(f"\n✓ Full audit saved to {out}")

    # Summary by verdict
    _print_section("VERDICT SUMMARY")
    counts = audit_df["verdict"].value_counts().reindex(
        ["STRONG LEAK", "WEAK LEAK", "CONFIGURED", "CLEAN"], fill_value=0
    )
    for v, n in counts.items():
        print(f"  {v:<14} : {n:>3} features")

    # Action items
    _print_section("ACTION ITEMS")

    # Strong leaks not yet in drop list
    new_strong = audit_df[
        (audit_df["verdict"] == "STRONG LEAK") & (~audit_df["in_drop_list"])
    ]
    if not new_strong.empty:
        print("\n⚠ STRONG leaks NOT yet in LEAKY_FEATURES — ADD them to src/config.py:")
        print(new_strong[["feature", "type", "score"]].to_string(index=False))
    else:
        print("\n✓ All strong leaks are already in LEAKY_FEATURES.")

    # Weak leaks (judgment call)
    weak = audit_df[audit_df["verdict"] == "WEAK LEAK"]
    if not weak.empty:
        print("\nℹ WEAK leaks — judgment call (drop = safer; keep = more signal):")
        print(weak[["feature", "type", "score"]].to_string(index=False))

    # Configured features that look mild
    mild_configured = audit_df[
        audit_df["in_drop_list"] & (audit_df["score"] < WEAK_LEAK_THRESHOLD)
    ]
    if not mild_configured.empty:
        print("\nℹ Features in LEAKY_FEATURES with low measured association:")
        print(mild_configured[["feature", "score"]].to_string(index=False))
        print("  → These could be reconsidered (e.g. CLTV may still leak temporally).")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)
