"""
Phase 10 — Linear coefficients as native interpretability for C2.

For C2 (`tuned/logistic/tabular`), a Logistic Regression with `class_weight=
'balanced'` and `C=0.027` (Phase 7), the coefficients are *already* a SHAP-
equivalent attribution: for a standardized feature `x_std`, the contribution
of that feature to the log-odds of class k is exactly `coef[k] * x_std`.

Three outputs match the SHAP module's structure:

    1. **Global**  — |coef × std| per feature, per class
    2. **Local**   — coef × x_std on individual archetypes
    3. **Segment** — same but restricted to a subgroup

Numeric features are already z-scored by the pipeline (StandardScaler). For
one-hot and ordinal features, we use their *empirical* std on the train split
so the magnitudes are comparable.

CLI
---
    python -m src.interpretation.linear_coef
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DEFAULT_TARGET,
    MODELS_DIR,
    NPS_CLASSES,
    NPS_CLASS_TO_INT,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.data.split import load_splits
from src.models.tuning_hybrid import VERBATIM_AUX_COLS


C2_PATH = MODELS_DIR / "tuned" / "logistic_tuned.joblib"

# Same segment columns as SHAP — for parallel storytelling
SEGMENT_COLS = {
    "Senior":  "Senior Citizen",
    "Gender":  "Gender",
    "Married": "Married",
}


# ============================================================
# Build the X matrix for C2 (tab-only) + feature names
# ============================================================
def _load_C2_data(target_col: str):
    """Return (model, X_train, X_silent, y_silent, df_silent, feature_names)."""
    if not target_col.startswith("NPS_"):
        target_col = f"NPS_{target_col}"
    if not C2_PATH.exists():
        raise FileNotFoundError(
            f"{C2_PATH} missing — run `make tune` first (Phase 7)."
        )

    model = joblib.load(C2_PATH)
    df = pd.read_parquet(DATA_PROCESSED / "dataset_with_verbatims.parquet")
    splits = load_splits("response_biased")
    pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")

    feature_cols = [
        c for c in df.columns
        if not c.startswith("NPS_") and c not in VERBATIM_AUX_COLS
    ]
    feature_names = list(pipeline.get_feature_names_out())

    # Build X for train (for the std reference) and silent_test (for groupings)
    X_train = pipeline.transform(df.loc[splits == "train", feature_cols])
    X_silent = pipeline.transform(df.loc[splits == "silent_test", feature_cols])
    y_silent = np.array(
        [NPS_CLASS_TO_INT[str(v)]
         for v in df.loc[splits == "silent_test", target_col].to_numpy()],
        dtype=int,
    )
    df_silent = df.loc[splits == "silent_test"]
    return model, X_train, X_silent, y_silent, df_silent, feature_names


# ============================================================
# Global: |coef × std| per feature, per class
# ============================================================
def global_coef_importance(target_col: str = DEFAULT_TARGET) -> pd.DataFrame:
    """One row per (class, feature) with coef, std, coef×std, |coef×std|."""
    model, X_train, X_silent, y_silent, df_silent, feature_names = _load_C2_data(target_col)
    coef = model.coef_  # (n_classes, n_features)
    std = X_train.std(axis=0)

    rows = []
    for k, cls in enumerate(NPS_CLASSES):
        for j, name in enumerate(feature_names):
            std_j = float(std[j])
            cv = float(coef[k, j])
            cs = cv * std_j
            rows.append({
                "class":            cls,
                "feature":          name,
                "coef":             cv,
                "std":              std_j,
                "coef_times_std":   cs,
                "abs_coef_times_std": abs(cs),
                "sign":             "+" if cv > 0 else "−" if cv < 0 else "0",
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["class", "abs_coef_times_std"], ascending=[True, False],
    )
    df["rank_in_class"] = df.groupby("class")["abs_coef_times_std"].rank(
        ascending=False, method="first",
    ).astype(int)
    return df.reset_index(drop=True)


# ============================================================
# Local: coef × x_std per archetype customer
# ============================================================
def _pick_archetypes(model, X, y, n_per_class: int = 1) -> dict[str, list[int]]:
    """One correctly-classified row per class, highest predicted proba."""
    proba = model.predict_proba(X)
    y_pred = model.predict(X)
    correct = y_pred == y
    out: dict[str, list[int]] = {}
    for k, cls in enumerate(NPS_CLASSES):
        cand = np.where((y == k) & correct)[0]
        if len(cand) == 0:
            cand = np.where(y == k)[0]
        if len(cand) == 0:
            out[cls] = []
            continue
        order = np.argsort(-proba[cand, k])
        out[cls] = cand[order[:n_per_class]].tolist()
    return out


def local_coef_records(target_col: str = DEFAULT_TARGET) -> pd.DataFrame:
    model, X_train, X_silent, y_silent, df_silent, feature_names = _load_C2_data(target_col)
    coef = model.coef_

    archetypes = _pick_archetypes(model, X_silent, y_silent)
    rows = []
    for true_class, idxs in archetypes.items():
        for s_idx in idxs:
            customer_id = df_silent.index[s_idx]
            y_pred_int = int(model.predict(X_silent[s_idx:s_idx + 1])[0])
            y_pred_cls = NPS_CLASSES[y_pred_int]
            x = X_silent[s_idx]
            for j, fname in enumerate(feature_names):
                contrib = float(coef[y_pred_int, j] * x[j])
                rows.append({
                    "customer_id":     str(customer_id),
                    "archetype":       true_class,
                    "y_true":          true_class,
                    "y_pred":          y_pred_cls,
                    "explained_class": y_pred_cls,
                    "feature":         fname,
                    "feature_value":   float(x[j]),
                    "contribution":    contrib,
                    "abs_contribution": abs(contrib),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["customer_id", "abs_contribution"], ascending=[True, False],
        ).reset_index(drop=True)
    return df


# ============================================================
# Segment: |coef × std| within subgroups
# ============================================================
def _normalize_segment_value(v):
    if isinstance(v, (int, np.integer)):
        return "Yes" if v == 1 else "No"
    return str(v)


def segment_coef_importance(
    target_col: str = DEFAULT_TARGET,
    segments: dict[str, str] | None = None,
    top_k: int = 15,
) -> pd.DataFrame:
    if segments is None:
        segments = SEGMENT_COLS

    model, X_train, X_silent, y_silent, df_silent, feature_names = _load_C2_data(target_col)
    coef = model.coef_
    rows = []

    for seg_label, col_name in segments.items():
        if col_name not in df_silent.columns:
            continue
        vals = df_silent[col_name].apply(_normalize_segment_value)
        for group_value in sorted(vals.unique()):
            mask = (vals == group_value).to_numpy()
            n_in_group = int(mask.sum())
            if n_in_group < 20:
                continue
            X_g = X_silent[mask]
            std_g = X_g.std(axis=0)
            for k_cls, cls in enumerate(NPS_CLASSES):
                tmp = pd.DataFrame({
                    "segment":       seg_label,
                    "group":         group_value,
                    "class":         cls,
                    "feature":       feature_names,
                    "coef":          coef[k_cls],
                    "std_in_group":  std_g,
                    "abs_contrib":   np.abs(coef[k_cls] * std_g),
                    "n_in_group":    n_in_group,
                })
                tmp = tmp.nlargest(top_k, "abs_contrib")
                tmp["rank"] = np.arange(1, len(tmp) + 1)
                rows.append(tmp)

    if not rows:
        return pd.DataFrame(
            columns=["segment", "group", "class", "feature",
                     "coef", "std_in_group", "abs_contrib", "n_in_group", "rank"],
        )
    return pd.concat(rows, ignore_index=True)


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 10 — LINEAR COEFFICIENTS FOR C2 (Logistic tabular)")
    print("=" * 70)

    try:
        gdf = global_coef_importance(args.target)
        out = RESULTS_DIR / "linear_coef_C2.parquet"
        gdf.to_parquet(out, index=False)
        print(f"\n✓ {out.name}  ({len(gdf)} rows)")
        for cls in NPS_CLASSES:
            top = gdf[gdf["class"] == cls].head(5)
            print(f"\n  Top-5 |coef × std| for {cls}:")
            for _, r in top.iterrows():
                print(
                    f"    {r['rank_in_class']:>2}. {r['feature']:<35} "
                    f"{r['sign']} {abs(r['coef_times_std']):.4f}"
                )

        ldf = local_coef_records(args.target)
        out = RESULTS_DIR / "linear_coef_local_C2.parquet"
        ldf.to_parquet(out, index=False)
        print(f"\n✓ {out.name}  ({len(ldf)} rows, "
              f"{ldf['customer_id'].nunique() if not ldf.empty else 0} customers)")

        sdf = segment_coef_importance(args.target)
        out = RESULTS_DIR / "linear_coef_segment_C2.parquet"
        sdf.to_parquet(out, index=False)
        print(f"✓ {out.name}  ({len(sdf)} rows)")

    except FileNotFoundError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
