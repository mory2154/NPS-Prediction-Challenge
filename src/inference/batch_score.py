"""
Phase 12 — Pre-compute predictions on silent_test for the Streamlit app.

The app loads a single parquet rather than re-running model inference on every
page load: this keeps the UI responsive (no LightGBM/PCA cold-start per query)
and means the artifact is reproducible (re-run when champions change).

Output schema (one row per silent_test customer):

    customer_id, segment_columns..., display_columns...,
    nps_true,
    proba_C1_detractor, proba_C1_passive, proba_C1_promoter,
    pred_C1, rank_C1_detractor, rank_C1_promoter,
    proba_C2_detractor, proba_C2_passive, proba_C2_promoter,
    pred_C2, rank_C2_detractor, rank_C2_promoter,
    agreement   (C1 and C2 predict same class? bool)

Re-run after every Phase 7/8 retune:
    make batch-score
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
from src.features.embeddings import load_or_compute_embeddings
from src.models.tuning_hybrid import VERBATIM_AUX_COLS, _build_split_matrices


# ============================================================
# Champion registry — frozen from Phase 8-9
# ============================================================
CHAMPIONS = {
    "C1": {
        "label":         "C1 — QWK champion",
        "phase":         "hybrid",
        "model":         "lightgbm",
        "feature_space": "pca32",
        "path":          MODELS_DIR / "hybrid" / "lightgbm_pca32.joblib",
        "uses_text":     True,
    },
    "C2": {
        "label":         "C2 — Production-safe",
        "phase":         "tuned",
        "model":         "logistic",
        "feature_space": "tabular",
        "path":          MODELS_DIR / "tuned" / "logistic_tuned.joblib",
        "uses_text":     False,
    },
}

# Columns surfaced in the Streamlit UI (Customer Lookup page)
DISPLAY_COLUMNS = [
    "Tenure Months", "Monthly Charges", "Total Charges",
    "Contract", "Internet Type", "Payment Method", "Paperless Billing",
    "Number of Referrals", "Number of Dependents",
    "Age", "Gender", "Senior Citizen", "Married", "Partner",
    "City", "Zip Code",
    "Online Security", "Online Backup", "Device Protection Plan",
    "Premium Tech Support", "Streaming TV", "Streaming Movies",
]

SEGMENT_COLUMNS = ["Senior Citizen", "Gender", "Married"]


# ============================================================
# Score a single champion → 3 probability columns + pred + ranks
# ============================================================
def _score_champion(
    champion_key: str,
    df: pd.DataFrame,
    splits: pd.Series,
    pipeline,
    embeddings_df,
    target_col: str,
    split_name: str = "silent_test",
    verbose: bool = True,
) -> pd.DataFrame:
    champ = CHAMPIONS[champion_key]
    if verbose:
        print(f"  loading {champion_key} from {champ['path'].name}")
    model = joblib.load(champ["path"])

    feature_cols = [
        c for c in df.columns
        if not c.startswith("NPS_") and c not in VERBATIM_AUX_COLS
    ]

    if champ["uses_text"]:
        X_per, y_per, _ = _build_split_matrices(
            df=df, splits=splits, pipeline=pipeline,
            embeddings_df=embeddings_df,
            target_col=target_col, feature_cols=feature_cols,
            feature_space=champ["feature_space"], verbose=False,
        )
        X = X_per[split_name]
    else:
        mask = splits == split_name
        X = pipeline.transform(df.loc[mask, feature_cols])

    proba = model.predict_proba(X)
    pred = model.predict(X)

    idx = df.index[splits == split_name]
    out = pd.DataFrame(index=idx)
    for k, cls in enumerate(NPS_CLASSES):
        out[f"proba_{champion_key}_{cls.lower()}"] = proba[:, k]
    out[f"pred_{champion_key}"] = [NPS_CLASSES[i] for i in pred]

    # Ranks (lower rank = more likely detractor / promoter)
    # rank 1 = highest predicted probability for that class
    out[f"rank_{champion_key}_detractor"] = (
        (-out[f"proba_{champion_key}_detractor"]).rank(method="min").astype(int)
    )
    out[f"rank_{champion_key}_promoter"] = (
        (-out[f"proba_{champion_key}_promoter"]).rank(method="min").astype(int)
    )

    if verbose:
        cls_dist = pd.Series(out[f"pred_{champion_key}"]).value_counts().to_dict()
        print(f"    {champion_key} predictions: {cls_dist}")

    return out


# ============================================================
# Main orchestrator
# ============================================================
def run_batch_score(
    target_col: str = DEFAULT_TARGET,
    split_name: str = "silent_test",
    verbose: bool = True,
) -> pd.DataFrame:
    if not target_col.startswith("NPS_"):
        target_col = f"NPS_{target_col}"

    if verbose:
        print("=" * 60)
        print("PHASE 12 — BATCH SCORING")
        print("=" * 60)
        print(f"Target : {target_col}")
        print(f"Split  : {split_name}")
        print(f"Models : {list(CHAMPIONS)}")

    # Load all assets
    df_path = DATA_PROCESSED / "dataset_with_verbatims.parquet"
    if not df_path.exists():
        raise FileNotFoundError(f"{df_path} missing — run `make load-verbatims`")
    df = pd.read_parquet(df_path)
    splits = load_splits("response_biased")
    pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
    embeddings_df = load_or_compute_embeddings(df, verbose=False)

    # Align
    common = df.index.intersection(embeddings_df.index)
    df = df.loc[common]; splits = splits.loc[common]
    embeddings_df = embeddings_df.loc[common]

    # Verify champions
    missing = [k for k, c in CHAMPIONS.items() if not c["path"].exists()]
    if missing:
        raise FileNotFoundError(
            f"Champion(s) missing: {missing}. Run Phases 7 and 8 first."
        )

    # Score each champion
    silent_idx = df.index[splits == split_name]
    if verbose:
        print(f"\nScoring {len(silent_idx)} customers on {split_name}")

    score_dfs = []
    for ck in CHAMPIONS:
        score_dfs.append(_score_champion(
            ck, df, splits, pipeline, embeddings_df,
            target_col=target_col, split_name=split_name, verbose=verbose,
        ))

    # Combine scores
    scores = pd.concat(score_dfs, axis=1)

    # Add true label (silent_test has it since IBM Telco dataset is fully labelled)
    silent_df = df.loc[silent_idx]
    scores["nps_true"] = silent_df[target_col].astype(str)

    # Display columns (only those present in the dataset)
    display_cols_present = [c for c in DISPLAY_COLUMNS if c in silent_df.columns]
    segment_cols_present = [c for c in SEGMENT_COLUMNS if c in silent_df.columns]

    for c in display_cols_present + segment_cols_present:
        if c not in scores.columns:
            scores[c] = silent_df[c]

    # Agreement column
    scores["agreement"] = scores["pred_C1"] == scores["pred_C2"]

    # Reset index → bring Customer ID back as a column (the index name varies
    # by dataset: "Customer ID", "customerID", or None). Normalise to
    # "customer_id" for downstream code.
    original_index_name = scores.index.name
    scores = scores.reset_index()
    if original_index_name and original_index_name in scores.columns:
        scores = scores.rename(columns={original_index_name: "customer_id"})
    elif "index" in scores.columns:
        scores = scores.rename(columns={"index": "customer_id"})

    # Reorder columns for readability
    score_cols = [c for c in scores.columns
                  if c.startswith(("proba_", "pred_", "rank_"))]
    front = (["customer_id", "nps_true"]
             + segment_cols_present
             + score_cols + ["agreement"])
    back = [c for c in scores.columns if c not in front]
    scores = scores[front + back]

    # Persist
    out_path = RESULTS_DIR / "silent_predictions.parquet"
    scores.to_parquet(out_path, index=False)
    if verbose:
        print(f"\n✓ {out_path.name}  ({scores.shape[0]} rows × {scores.shape[1]} cols)")
        print(f"  C1 vs C2 agreement: {scores['agreement'].mean():.1%}")
        print(f"  Memory: {scores.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return scores


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--split", default="silent_test")
    args = parser.parse_args()

    try:
        run_batch_score(
            target_col=args.target,
            split_name=args.split,
            verbose=True,
        )
    except FileNotFoundError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
