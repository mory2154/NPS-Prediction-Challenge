"""
Phase 10 — PCA loadings: map opaque PC dimensions back to embedding dims.

C1 uses PCA(32) on the 384-dim sentence-transformers embeddings. SHAP tells us
"PC07 was the most influential" but PC07 is itself a linear combination of all
384 embedding dimensions, which are themselves opaque. We can't fully
"interpret" the text channel — that would require feature-attribution on the
encoder model — but we can at least surface, for each PC, the embedding dims
that dominate its loading.

Useful for the report write-up: "PC07 is essentially driven by emb_142,
emb_201, emb_318 — these dims may correspond to sentiment-related directions
of the encoder, though without ground-truth labelling of embedding dims, we
cannot claim semantic meaning."

CLI
---
    python -m src.interpretation.pca_loadings
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.config import (
    DATA_PROCESSED,
    DEFAULT_TARGET,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.data.split import load_splits
from src.features.embeddings import load_or_compute_embeddings


PCA_N_COMPONENTS = 32


def fit_pca_on_train(seed: int = RANDOM_SEED) -> tuple[PCA, list]:
    """Fit PCA(32) on TRAIN embeddings — same as Phase 8 hybrid pipeline."""
    df = pd.read_parquet(DATA_PROCESSED / "dataset_with_verbatims.parquet")
    splits = load_splits("response_biased")
    embeddings_df = load_or_compute_embeddings(df, verbose=False)
    common = df.index.intersection(embeddings_df.index)
    df = df.loc[common]; splits = splits.loc[common]
    embeddings_df = embeddings_df.loc[common]
    train_idx = df.index[splits == "train"]
    emb_train = embeddings_df.loc[train_idx].to_numpy()
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=seed)
    pca.fit(emb_train)
    return pca, list(embeddings_df.columns)


def loadings_table(top_k_dims_per_pc: int = 5) -> pd.DataFrame:
    """
    For each PC, return the top-k embedding dims with largest |loading|.

    Returns one row per (PC, dim) with columns:
        pc, pc_id, dim_name, loading, abs_loading, rank, explained_variance
    """
    pca, dim_names = fit_pca_on_train()
    rows = []
    for k in range(PCA_N_COMPONENTS):
        comp = pca.components_[k]  # shape (384,)
        order = np.argsort(-np.abs(comp))[:top_k_dims_per_pc]
        for rank, j in enumerate(order, start=1):
            rows.append({
                "pc":                 f"PC{k:02d}",
                "pc_id":              k,
                "dim_name":           dim_names[j],
                "loading":            float(comp[j]),
                "abs_loading":        float(abs(comp[j])),
                "rank":               rank,
                "explained_variance": float(pca.explained_variance_ratio_[k]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 60)
    print("PHASE 10 — PCA LOADINGS (PC → embedding dim mapping)")
    print("=" * 60)
    try:
        df = loadings_table(top_k_dims_per_pc=5)
    except FileNotFoundError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)

    out = RESULTS_DIR / "pca_loadings.parquet"
    df.to_parquet(out, index=False)
    print(f"\n✓ {out.name}  ({len(df)} rows)")
    print(
        f"\n  Total variance explained by 32 PCs: "
        f"{df.groupby('pc')['explained_variance'].first().sum():.1%}"
    )
    print("\n  Top loading for first 5 PCs:")
    for pc, sub in df.groupby("pc"):
        if int(pc[2:]) >= 5:
            continue
        top = sub.iloc[0]
        print(
            f"    {pc} (var={top['explained_variance']:.1%}) "
            f"→ {top['dim_name']} (loading={top['loading']:+.3f})"
        )


if __name__ == "__main__":
    main()
