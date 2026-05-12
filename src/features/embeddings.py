"""
Phase 8 — SentenceTransformer embeddings for customer verbatims.

Embeddings are cached to data/processed/verbatim_embeddings.parquet (indexed
by Customer ID) along with a meta.json file. The cache invalidates when:

    * the content hash of (sorted_index + concatenated_verbatim_text + model_name)
      changes (i.e. anything affecting the input)
    * `force=True` is passed

CLI:
    python -m src.features.embeddings
    python -m src.features.embeddings --model sentence-transformers/all-MiniLM-L6-v2
    python -m src.features.embeddings --force

Default model: `sentence-transformers/all-MiniLM-L6-v2`
    - English-only, 384 dims, ~80 MB on disk
    - First call downloads the model from HuggingFace; subsequent calls use cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED

# ============================================================
# Configuration
# ============================================================
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 64

EMBEDDINGS_CACHE = DATA_PROCESSED / "verbatim_embeddings.parquet"
EMBEDDINGS_META = DATA_PROCESSED / "verbatim_embeddings.meta.json"

VERBATIM_COL = "verbatim"


# ============================================================
# Cache invalidation
# ============================================================
def _content_hash(verbatims: pd.Series, model_name: str) -> str:
    """Hash of (sorted index + concatenated text + model name).

    Detects any change in the input that would require recomputing.
    """
    sig = (
        f"{model_name}\n"
        f"{','.join(sorted(verbatims.index.astype(str)))}\n"
        f"{verbatims.fillna('').str.cat(sep='|')}"
    )
    return hashlib.md5(sig.encode("utf-8")).hexdigest()


# ============================================================
# Compute embeddings
# ============================================================
def compute_embeddings(
    verbatims: pd.Series,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Run SentenceTransformer over the verbatims and return a (n, dim) array.

    Empty / missing verbatims get a zero vector (defensive — should be 0 in
    practice since Phase 5 generates one verbatim per customer).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers not installed. Run:\n"
            "  pip install sentence-transformers\n"
            "or uncomment the line in requirements.txt then `pip install -r ...`"
        ) from e

    model = SentenceTransformer(model_name)
    texts = verbatims.fillna("").astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    # Defensive: zero out empty rows
    empty_mask = (verbatims.fillna("").str.strip() == "").to_numpy()
    if empty_mask.any():
        embeddings[empty_mask] = 0.0
    return embeddings.astype(np.float32)


# ============================================================
# Load-or-compute with caching
# ============================================================
def load_or_compute_embeddings(
    df: pd.DataFrame,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_path: Path | None = None,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Return a (n_customers, embedding_dim) DataFrame indexed by df.index.

    If a valid cache exists, load from disk. Otherwise compute and cache.
    """
    if VERBATIM_COL not in df.columns:
        raise ValueError(
            f"Column '{VERBATIM_COL}' missing from df. "
            "Run `make load-verbatims` to integrate Phase 5 output."
        )
    if cache_path is None:
        cache_path = EMBEDDINGS_CACHE

    verbatims = df[VERBATIM_COL]
    h = _content_hash(verbatims, model_name)

    # Check cache
    meta_path = Path(str(cache_path).replace(".parquet", ".meta.json"))
    if cache_path.exists() and meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("content_hash") == h:
                if verbose:
                    print(f"  ✓ Using cached embeddings: {cache_path.name}")
                return pd.read_parquet(cache_path)
            else:
                if verbose:
                    print(f"  ⚠ Cache stale (input changed), recomputing...")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Cache unreadable ({e}), recomputing...")

    # Compute
    if verbose:
        print(f"  → Computing embeddings with {model_name}")
        print(f"    n_verbatims = {len(verbatims):,}")
    arr = compute_embeddings(
        verbatims, model_name=model_name, show_progress_bar=verbose,
    )
    if verbose:
        print(f"    shape = {arr.shape}")

    cols = [f"emb_{i:03d}" for i in range(arr.shape[1])]
    emb_df = pd.DataFrame(arr, index=df.index.copy(), columns=cols)

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(cache_path)
    meta_path.write_text(json.dumps({
        "model_name": model_name,
        "content_hash": h,
        "shape": list(arr.shape),
        "n_customers": int(arr.shape[0]),
        "embedding_dim": int(arr.shape[1]),
    }, indent=2))
    if verbose:
        print(f"  ✓ Saved {cache_path.name} (and meta)")

    return emb_df


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence-transformers model id (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if a valid cache exists.",
    )
    args = parser.parse_args()

    in_path = DATA_PROCESSED / "dataset_with_verbatims.parquet"
    if not in_path.exists():
        print(
            f"\n✗ {in_path} missing — run `make load-verbatims` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("COMPUTING VERBATIM EMBEDDINGS")
    print("=" * 60)

    df = pd.read_parquet(in_path)
    if VERBATIM_COL not in df.columns:
        print(
            f"\n✗ Column '{VERBATIM_COL}' not in {in_path.name}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n[1/2] Loaded {len(df):,} rows from {in_path.name}")
    n_missing = df[VERBATIM_COL].isna().sum()
    if n_missing > 0:
        print(f"  ⚠ {n_missing} customers have no verbatim (will get zero vectors)")

    print("\n[2/2] Embedding...")
    emb_df = load_or_compute_embeddings(
        df, model_name=args.model, force=args.force, verbose=True,
    )
    print(f"\n✓ Final shape: {emb_df.shape}")
    print(f"  Memory: {emb_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
