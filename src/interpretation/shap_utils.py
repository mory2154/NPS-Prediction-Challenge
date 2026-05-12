"""
Phase 10 — TreeSHAP for C1 (hybrid LightGBM on tab + PCA(32) embeddings).

Provides three views of feature importance:

    1. **Global** — mean |SHAP| per feature, per class.
       Answers: "across the silent population, which features push predictions?"

    2. **Local** — SHAP values for individual customers.
       Answers: "for *this* customer, why did the model predict Detractor?"

    3. **By segment** — mean |SHAP| within {Senior, Gender, Married} subgroups.
       Answers: "do the top drivers differ for seniors vs non-seniors?"
       Sets up Phase 11 (fairness audit).

Note on hybrid features
-----------------------
C1 was trained on  [tab_encoded (~94 cols) | PCA(32) on embeddings].
The 32 PCA components are NOT directly interpretable. We surface them as
"PC00"..."PC31" here and rely on `pca_loadings.py` (separate module) to map
each PC to its dominant embedding dimensions.

CLI
---
    python -m src.interpretation.shap_utils
    python -m src.interpretation.shap_utils --n-sample 1000
    python -m src.interpretation.shap_utils --no-segments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from src.features.embeddings import load_or_compute_embeddings
from src.models.tuning_hybrid import VERBATIM_AUX_COLS, _build_split_matrices


# ============================================================
# Champion C1 — frozen from Phase 8
# ============================================================
C1_PATH = MODELS_DIR / "hybrid" / "lightgbm_pca32.joblib"
DEFAULT_SAMPLE_SIZE = 1000

# Segments for the "by segment" view (matches Phase 11 fairness audit)
SEGMENT_COLS = {
    "Senior":  "Senior Citizen",   # 0/1 or Yes/No depending on dataset cleaning
    "Gender":  "Gender",            # Male/Female
    "Married": "Married",           # Yes/No
}

# Local explanations: pick 3 archetypal customers (one per class)
N_LOCAL_PER_CLASS = 1


# ============================================================
# Feature names — assemble tab_names + PC labels
# ============================================================
def _hybrid_feature_names(pipeline, n_pcs: int = 32) -> list[str]:
    """[tab_feature_names..., PC00, PC01, ..., PCnn]"""
    try:
        tab_names = list(pipeline.get_feature_names_out())
    except Exception:
        # Fallback if a transformer doesn't implement it
        n_tab = pipeline.transform(
            pd.DataFrame([[None] * len(pipeline.feature_names_in_)],
                         columns=pipeline.feature_names_in_)
        ).shape[1]
        tab_names = [f"tab_{i:03d}" for i in range(n_tab)]
    pc_names = [f"PC{i:02d}" for i in range(n_pcs)]
    return tab_names + pc_names


# ============================================================
# Stratified sampling — preserve class marginals
# ============================================================
def _stratified_sample(
    y: np.ndarray,
    n_target: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Return indices for a stratified sample of size ~n_target."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    counts = {c: int((y == c).sum()) for c in classes}
    total = sum(counts.values())
    if total <= n_target:
        return np.arange(total)

    quotas = {c: max(1, int(round(n_target * counts[c] / total))) for c in classes}
    out = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        q = min(quotas[c], len(idx_c))
        out.append(rng.choice(idx_c, size=q, replace=False))
    return np.concatenate(out)


# ============================================================
# Compute TreeSHAP values for C1
# ============================================================
def compute_shap_values_C1(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    split: str = "silent_test",
    target_col: str = DEFAULT_TARGET,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict:
    """
    Run TreeSHAP on C1 on a stratified sample of `split`.

    Returns
    -------
    {
        "model":            <LGBMClassifier>,
        "explainer":        <shap.TreeExplainer>,
        "shap_values":      list of 3 ndarrays, one per class, shape (n_sample, n_feat)
        "expected_values":  ndarray (3,)
        "X_sample":         pd.DataFrame (n_sample, n_feat) — features fed to model
        "y_sample":         np.ndarray (n_sample,) — int labels
        "feature_names":    list[str]
        "df_sample":        pd.DataFrame — original rows for segment grouping
        "split":            str
    }
    """
    import shap

    if not target_col.startswith("NPS_"):
        target_col = f"NPS_{target_col}"
    if not C1_PATH.exists():
        raise FileNotFoundError(
            f"{C1_PATH} missing — run `make tune-hybrid` first (Phase 8)."
        )

    # Load champion + data
    if verbose:
        print(f"Loading C1 from {C1_PATH.name}")
    model = joblib.load(C1_PATH)

    df = pd.read_parquet(DATA_PROCESSED / "dataset_with_verbatims.parquet")
    splits = load_splits("response_biased")
    pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
    embeddings_df = load_or_compute_embeddings(df, verbose=False)

    # Align (defensive)
    common = df.index.intersection(embeddings_df.index)
    df = df.loc[common]
    splits = splits.loc[common]
    embeddings_df = embeddings_df.loc[common]

    feature_cols = [
        c for c in df.columns
        if not c.startswith("NPS_") and c not in VERBATIM_AUX_COLS
    ]

    # Build the full hybrid matrices via the same code path used during tuning
    X_per, y_per, info = _build_split_matrices(
        df=df, splits=splits, pipeline=pipeline,
        embeddings_df=embeddings_df,
        target_col=target_col, feature_cols=feature_cols,
        feature_space="pca32", verbose=False,
    )

    if split not in X_per:
        raise ValueError(f"Split '{split}' not found in {list(X_per.keys())}")

    X_full = X_per[split]
    y_full = y_per[split]
    df_split = df.loc[splits == split]

    # Stratified sub-sample
    idx_sample = _stratified_sample(y_full, n_target=sample_size, seed=seed)
    X_sample = X_full[idx_sample]
    y_sample = y_full[idx_sample]
    df_sample = df_split.iloc[idx_sample].copy()

    if verbose:
        print(
            f"Sampling: {len(idx_sample)}/{len(X_full)} rows from {split} "
            f"(class distribution: {dict(zip(*np.unique(y_sample, return_counts=True)))})"
        )

    feature_names = _hybrid_feature_names(pipeline, n_pcs=info["pca_components"])
    if len(feature_names) != X_sample.shape[1]:
        # Defensive — fallback to generic names
        feature_names = [f"feat_{i:03d}" for i in range(X_sample.shape[1])]

    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

    # ------------------------------------------------------------
    # TreeSHAP
    # ------------------------------------------------------------
    if verbose:
        print("Computing TreeSHAP values...")
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample_df)

    # Normalize the shape: shap returns either a list of 3 arrays (older API)
    # or one (n, p, 3) array (newer API). Normalize to list-of-3-(n,p).
    if isinstance(raw, list):
        shap_values = raw
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        shap_values = [raw[:, :, k] for k in range(raw.shape[2])]
    else:
        raise TypeError(f"Unexpected SHAP output: {type(raw)} {getattr(raw, 'shape', '')}")

    if len(shap_values) != 3:
        raise ValueError(f"Expected 3 classes, got {len(shap_values)}")

    expected = np.atleast_1d(explainer.expected_value)
    if verbose:
        print(f"  shap_values[0].shape = {shap_values[0].shape}")
        print(f"  expected_values = {expected}")

    return {
        "model": model,
        "explainer": explainer,
        "shap_values": shap_values,
        "expected_values": expected,
        "X_sample": X_sample_df,
        "y_sample": y_sample,
        "feature_names": feature_names,
        "df_sample": df_sample,
        "split": split,
        "sample_indices": idx_sample,
    }


# ============================================================
# Global feature importance per class
# ============================================================
def global_importance(shap_bundle: dict) -> pd.DataFrame:
    """
    Mean |SHAP| per feature, per class. One row per (class, feature).
    """
    rows = []
    for k, cls in enumerate(NPS_CLASSES):
        sv = shap_bundle["shap_values"][k]
        mean_abs = np.abs(sv).mean(axis=0)
        for j, name in enumerate(shap_bundle["feature_names"]):
            rows.append({
                "class":          cls,
                "feature":        name,
                "mean_abs_shap":  float(mean_abs[j]),
                "is_pc":          name.startswith("PC"),
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["class", "mean_abs_shap"], ascending=[True, False])
    df["rank_in_class"] = df.groupby("class")["mean_abs_shap"].rank(
        ascending=False, method="first",
    ).astype(int)
    return df.reset_index(drop=True)


# ============================================================
# Local SHAP — 1 archetypal customer per class
# ============================================================
def pick_archetypes(
    shap_bundle: dict, n_per_class: int = N_LOCAL_PER_CLASS,
) -> dict[str, list[int]]:
    """
    For each class, pick `n_per_class` rows where the model is CORRECT *and*
    the predicted probability for that class is among the highest in the sample.

    Returns a dict mapping class name → list of row indices into X_sample.
    """
    model = shap_bundle["model"]
    X = shap_bundle["X_sample"]
    y = shap_bundle["y_sample"]

    proba = model.predict_proba(X.values)
    y_pred = model.predict(X.values)
    correct = y_pred == y

    out: dict[str, list[int]] = {}
    for k, cls in enumerate(NPS_CLASSES):
        # candidates: correctly classified AND true class == k
        cand_idx = np.where((y == k) & correct)[0]
        if len(cand_idx) == 0:
            cand_idx = np.where(y == k)[0]  # fallback
        if len(cand_idx) == 0:
            out[cls] = []
            continue
        # rank by predicted proba for that class
        order = np.argsort(-proba[cand_idx, k])
        out[cls] = cand_idx[order[:n_per_class]].tolist()
    return out


def local_shap_records(shap_bundle: dict) -> pd.DataFrame:
    """
    One row per (customer, feature) with SHAP contribution + feature value.

    The customer set = archetypes picked by `pick_archetypes`. For each customer
    we record SHAP values for the *predicted* class (not all three) — the
    explanation of "why the model predicted Detractor for THIS customer".
    """
    archetypes = pick_archetypes(shap_bundle)
    model = shap_bundle["model"]
    X = shap_bundle["X_sample"]
    y = shap_bundle["y_sample"]
    df_sample = shap_bundle["df_sample"]
    shap_vals = shap_bundle["shap_values"]
    expected = shap_bundle["expected_values"]
    feature_names = shap_bundle["feature_names"]

    rows = []
    for true_class, sample_idxs in archetypes.items():
        for s_idx in sample_idxs:
            customer_id = df_sample.index[s_idx]
            y_true_int = int(y[s_idx])
            y_true = NPS_CLASSES[y_true_int]
            y_pred_int = int(model.predict(X.values[s_idx:s_idx + 1])[0])
            y_pred_cls = NPS_CLASSES[y_pred_int]
            sv_for_pred = shap_vals[y_pred_int][s_idx]
            for j, fname in enumerate(feature_names):
                rows.append({
                    "customer_id":    str(customer_id),
                    "archetype":      true_class,
                    "y_true":         y_true,
                    "y_pred":         y_pred_cls,
                    "explained_class": y_pred_cls,
                    "feature":        fname,
                    "feature_value":  float(X.values[s_idx, j]),
                    "shap_value":     float(sv_for_pred[j]),
                    "expected_value": float(expected[y_pred_int]),
                })
    df = pd.DataFrame(rows)
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values(
        ["customer_id", "abs_shap"], ascending=[True, False],
    ).reset_index(drop=True)
    return df


# ============================================================
# Segment-level importance
# ============================================================
def _normalize_segment_value(v):
    """Coerce yes/no flags to a uniform string representation."""
    if isinstance(v, (int, np.integer)):
        return "Yes" if v == 1 else "No"
    return str(v)


def segment_importance(
    shap_bundle: dict,
    segments: dict[str, str] | None = None,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Mean |SHAP| per feature, computed WITHIN each subgroup.

    For each segment column (Senior / Gender / Married) and each value
    (e.g. Senior=Yes, Senior=No), produce a DataFrame of top-`k` features.

    Returns one combined DataFrame with columns:
        segment, group, class, feature, mean_abs_shap, n_in_group, rank
    """
    if segments is None:
        segments = SEGMENT_COLS

    df_sample = shap_bundle["df_sample"]
    feature_names = shap_bundle["feature_names"]
    shap_vals = shap_bundle["shap_values"]

    rows = []
    for seg_label, col_name in segments.items():
        if col_name not in df_sample.columns:
            # Skip silently — segment may be absent in this dataset
            continue
        vals_series = df_sample[col_name].apply(_normalize_segment_value)
        for group_value in sorted(vals_series.unique()):
            mask = (vals_series == group_value).to_numpy()
            n_in_group = int(mask.sum())
            if n_in_group < 20:
                # too small to be meaningful — skip
                continue
            for k_cls, cls in enumerate(NPS_CLASSES):
                sv = shap_vals[k_cls][mask]
                mean_abs = np.abs(sv).mean(axis=0)
                tmp = pd.DataFrame({
                    "segment":       seg_label,
                    "group":         group_value,
                    "class":         cls,
                    "feature":       feature_names,
                    "mean_abs_shap": mean_abs,
                    "n_in_group":    n_in_group,
                })
                tmp = tmp.nlargest(top_k, "mean_abs_shap")
                tmp["rank"] = np.arange(1, len(tmp) + 1)
                rows.append(tmp)

    if not rows:
        return pd.DataFrame(
            columns=["segment", "group", "class", "feature",
                     "mean_abs_shap", "n_in_group", "rank"],
        )
    return pd.concat(rows, ignore_index=True)


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sample", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--split", default="silent_test")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--no-segments", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 10 — SHAP ANALYSIS FOR C1 (hybrid LGBM)")
    print("=" * 70)

    try:
        bundle = compute_shap_values_C1(
            sample_size=args.n_sample,
            split=args.split,
            target_col=args.target,
            seed=RANDOM_SEED,
            verbose=True,
        )
    except (FileNotFoundError, ImportError) as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[1/3] Global importance")
    gdf = global_importance(bundle)
    out = RESULTS_DIR / "shap_global_C1.parquet"
    gdf.to_parquet(out, index=False)
    print(f"  ✓ {out.name} ({len(gdf)} rows)")
    for cls in NPS_CLASSES:
        top = gdf[gdf["class"] == cls].head(5)
        print(f"  Top-5 for {cls}:")
        for _, r in top.iterrows():
            print(f"    {r['rank_in_class']:>2}. {r['feature']:<35} {r['mean_abs_shap']:.4f}")

    print("\n[2/3] Local (archetypes)")
    ldf = local_shap_records(bundle)
    out = RESULTS_DIR / "shap_local_C1.parquet"
    ldf.to_parquet(out, index=False)
    print(f"  ✓ {out.name} ({len(ldf)} rows, {ldf['customer_id'].nunique()} customers)")

    if not args.no_segments:
        print("\n[3/3] Segment importance")
        sdf = segment_importance(bundle)
        out = RESULTS_DIR / "shap_segment_C1.parquet"
        sdf.to_parquet(out, index=False)
        print(f"  ✓ {out.name} ({len(sdf)} rows)")
    else:
        print("\n[3/3] Segments skipped (--no-segments)")


if __name__ == "__main__":
    main()
