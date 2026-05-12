"""
Inspect generated verbatims for quality:
  - length distributions
  - vocabulary leakage (any forbidden words like "Detractor", "NPS")
  - sentiment-class correlation (sanity check)
  - duplicates / templated outputs

CLI:
    python -m src.verbatims.inspect
"""

from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from src.config import DATA_PROCESSED, NPS_CLASSES

# Words the model should never output (rule encoded in the system prompt)
FORBIDDEN_WORDS = ["detractor", "passive", "promoter", "nps", "satisfaction score"]


def quality_audit(df: pd.DataFrame) -> dict:
    """Compute a quality report on a verbatims DataFrame."""
    if "verbatim" not in df.columns:
        raise ValueError("'verbatim' column missing")

    text = df["verbatim"].astype(str)
    lens = text.str.len()
    word_counts = text.str.split().apply(len)

    # Forbidden vocabulary check
    forbidden_hits: dict[str, int] = {}
    text_lower = text.str.lower()
    for w in FORBIDDEN_WORDS:
        hits = text_lower.str.contains(w, regex=False).sum()
        if hits > 0:
            forbidden_hits[w] = int(hits)

    # Duplicates (sign of model collapse)
    n_duplicates = int(text.duplicated().sum())

    # Length stats per expected class (if column present)
    per_class: dict[str, dict] = {}
    if "expected_class" in df.columns:
        for cls in NPS_CLASSES:
            sub = df[df["expected_class"] == cls]["verbatim"]
            if len(sub) == 0:
                continue
            per_class[cls] = {
                "n": len(sub),
                "mean_chars": int(sub.str.len().mean()),
                "median_chars": int(sub.str.len().median()),
            }

    return {
        "total": len(df),
        "char_length": {
            "mean": int(lens.mean()),
            "median": int(lens.median()),
            "min": int(lens.min()),
            "max": int(lens.max()),
            "below_30": int((lens < 30).sum()),
            "above_500": int((lens > 500).sum()),
        },
        "word_length": {
            "mean": int(word_counts.mean()),
            "median": int(word_counts.median()),
        },
        "forbidden_words": forbidden_hits,
        "duplicates": n_duplicates,
        "per_class": per_class,
    }


def sample_verbatims(df: pd.DataFrame, n_per_class: int = 3, seed: int = 42) -> pd.DataFrame:
    """Return a random sample of verbatims per expected class for manual inspection."""
    if "expected_class" not in df.columns:
        raise ValueError("'expected_class' column missing")
    samples = []
    for cls in NPS_CLASSES:
        sub = df[df["expected_class"] == cls]
        if len(sub) > 0:
            samples.append(sub.sample(min(n_per_class, len(sub)), random_state=seed))
    return pd.concat(samples) if samples else pd.DataFrame()


def main() -> None:
    in_path = DATA_PROCESSED / "dataset_with_verbatims.parquet"
    if not in_path.exists():
        print(f"\n✗ {in_path} missing — run `make load-verbatims` first.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(in_path)
    if "verbatim" not in df.columns:
        print("\n✗ 'verbatim' column not in the dataset.", file=sys.stderr)
        sys.exit(1)

    # We need expected_class for per-class stats; rebuild from baseline target if absent
    if "expected_class" not in df.columns and "NPS_baseline" in df.columns:
        df["expected_class"] = df["NPS_baseline"].astype(str)

    print("=" * 60)
    print("VERBATIM QUALITY AUDIT")
    print("=" * 60)

    report = quality_audit(df)

    print(f"\nTotal verbatims:    {report['total']:,}")
    cl = report["char_length"]
    print(f"\nCharacter length:")
    print(f"  mean   = {cl['mean']}")
    print(f"  median = {cl['median']}")
    print(f"  min    = {cl['min']}    (below 30 chars: {cl['below_30']})")
    print(f"  max    = {cl['max']}    (above 500 chars: {cl['above_500']})")
    print(f"\nWord length:")
    wl = report["word_length"]
    print(f"  mean   = {wl['mean']}")
    print(f"  median = {wl['median']}")

    print(f"\nDuplicates: {report['duplicates']}")
    if report["forbidden_words"]:
        print(f"\n⚠ Forbidden words detected:")
        for w, n in report["forbidden_words"].items():
            print(f"  '{w}' appears in {n} verbatims")
    else:
        print("\n✓ No forbidden words detected")

    if report["per_class"]:
        print("\nLength per expected class:")
        for cls, stats in report["per_class"].items():
            print(f"  {cls:<10}: n={stats['n']:>5}, mean={stats['mean_chars']:>3} chars")

    # Sample for manual review
    if "expected_class" in df.columns:
        print("\n" + "=" * 60)
        print("SAMPLE (3 per class) — manual review")
        print("=" * 60)
        samples = sample_verbatims(df, n_per_class=3)
        for cls in NPS_CLASSES:
            print(f"\n--- {cls} ---")
            sub = samples[samples["expected_class"] == cls]
            for _, row in sub.iterrows():
                ci_marker = " [CI]" if row.get("counter_intuitive", False) else ""
                print(f"\n  {row.name}{ci_marker}:")
                print(f"  \"{row['verbatim']}\"")


if __name__ == "__main__":
    main()
