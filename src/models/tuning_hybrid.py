"""
Phase 8 — Hybrid (tabular + verbatim embeddings) Optuna tuning.

For each (model, feature_space) combination, run an Optuna study optimizing
QWK on the val split. Save 4 models in models/hybrid/.

Feature spaces:
    * "concat" : [tab_encoded(~94) | embedding(384)]                 ≈ 478 dims
    * "pca32"  : [tab_encoded(~94) | PCA(embedding, 32, fit on train)] ≈ 126 dims

PCA is fitted on the TRAIN split only (no leakage), then applied to val,
respondent_test and silent_test.

Models tuned by default: lightgbm + logistic.
Ordinal (mord.LogisticAT) is skipped — it is numerically unstable on >100 dims
and Phase 7 already showed it plateaus far below the others.

CLI:
    python -m src.models.tuning_hybrid
    python -m src.models.tuning_hybrid --models lightgbm
    python -m src.models.tuning_hybrid --feature-spaces concat
    python -m src.models.tuning_hybrid --target NPS_baseline
"""

from __future__ import annotations

import argparse
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.config import (
    DATA_PROCESSED,
    DEFAULT_TARGET,
    MODELS_DIR,
    NPS_CLASS_TO_INT,
    RANDOM_SEED,
    RESULTS_DIR,
    TUNING_BUDGETS,
    TUNING_SAMPLER,
    TUNING_TIMEOUT_SEC,
)
from src.data.split import load_splits
from src.evaluation.metrics import evaluate
from src.features.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    load_or_compute_embeddings,
)
# Reuse Phase 7 objective factories — DRY
from src.models.tuning import OBJECTIVE_FACTORIES, _make_sampler, build_final_model

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================
HYBRID_DIR = MODELS_DIR / "hybrid"
HYBRID_DIR.mkdir(parents=True, exist_ok=True)

# PCA for the "pca32" feature space
PCA_N_COMPONENTS = 32

HYBRID_MODELS = ["lightgbm", "logistic"]
HYBRID_FEATURE_SPACES = ["concat", "pca32"]

# Columns that are added by Phase 5 verbatims integration but should NOT be
# fed to the model (they are either the text itself or its provenance metadata).
VERBATIM_AUX_COLS = ["verbatim", "counter_intuitive"]


# ============================================================
# Helpers
# ============================================================
def _to_int(y) -> np.ndarray:
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    if y.dtype.kind in {"U", "O"} or hasattr(y, "categories"):
        return np.array([NPS_CLASS_TO_INT[str(v)] for v in y], dtype=int)
    return np.asarray(y, dtype=int)


def _build_split_matrices(
    df: pd.DataFrame,
    splits: pd.Series,
    pipeline,
    embeddings_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    feature_space: str,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """
    Build hybrid feature matrices for every split.

    PCA (if applicable) is fitted on the TRAIN split only.

    Returns
    -------
    X_per_split : {split_name: np.ndarray (n_split, n_hybrid_dims)}
    y_per_split : {split_name: np.ndarray (n_split,)} — int labels
    info        : {"pca_explained_variance": float | None, "n_dims": int, ...}
    """
    X_tab_per: dict[str, np.ndarray] = {}
    y_per: dict[str, np.ndarray] = {}
    emb_per_raw: dict[str, np.ndarray] = {}

    for split_name in splits.unique():
        mask = splits == split_name
        idx = df.index[mask]
        X_tab = df.loc[mask, feature_cols]
        X_tab_per[split_name] = pipeline.transform(X_tab)
        emb_per_raw[split_name] = embeddings_df.loc[idx].to_numpy()
        y_per[split_name] = _to_int(df.loc[mask, target_col])

    info: dict = {"feature_space": feature_space}

    if feature_space == "concat":
        emb_per = emb_per_raw
        info["pca_explained_variance"] = None
    elif feature_space == "pca32":
        if "train" not in emb_per_raw:
            raise ValueError("'train' split missing — cannot fit PCA.")
        pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)
        pca.fit(emb_per_raw["train"])
        emb_per = {k: pca.transform(v) for k, v in emb_per_raw.items()}
        info["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
        info["pca_components"] = PCA_N_COMPONENTS
        if verbose:
            print(
                f"    PCA(32) — explained variance: "
                f"{info['pca_explained_variance']:.2%}"
            )
    else:
        raise ValueError(f"Unknown feature_space: {feature_space}")

    X_per: dict[str, np.ndarray] = {
        k: np.hstack([X_tab_per[k], emb_per[k]]).astype(np.float32)
        for k in X_tab_per
    }
    info["n_dims"] = X_per["train"].shape[1]
    return X_per, y_per, info


# ============================================================
# Tune one (model, feature_space)
# ============================================================
def tune_one(
    model_name: str,
    feature_space: str,
    X_per: dict[str, np.ndarray],
    y_per: dict[str, np.ndarray],
    n_trials: int,
    target_col: str,
    verbose: bool = True,
) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if verbose:
        print(
            f"\n--- Tuning {model_name} × {feature_space} "
            f"({n_trials} trials, sampler={TUNING_SAMPLER}, "
            f"dims={X_per['train'].shape[1]}) ---"
        )

    sampler = _make_sampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = OBJECTIVE_FACTORIES[model_name](
        X_per["train"], y_per["train"],
        X_per["val"], y_per["val"],
    )
    study.optimize(
        objective, n_trials=n_trials, timeout=TUNING_TIMEOUT_SEC,
        show_progress_bar=False,
    )

    if verbose:
        print(f"  Best QWK on val: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

    # Refit final model on TRAIN with best params + persist
    final_model = build_final_model(model_name, study.best_params)
    final_model.fit(X_per["train"], y_per["train"])
    out_path = HYBRID_DIR / f"{model_name}_{feature_space}.joblib"
    joblib.dump(final_model, out_path)
    if verbose:
        print(f"  ✓ Saved {out_path.name}")

    # Evaluate on every split
    eval_rows: list[dict] = []
    for split_name, X in X_per.items():
        y_int = y_per[split_name]
        y_pred = final_model.predict(X)
        try:
            y_proba = final_model.predict_proba(X)
        except Exception:
            y_proba = None
        result = evaluate(y_int, y_pred, y_proba, name=split_name)
        result["split"] = split_name
        result["model"] = model_name
        result["feature_space"] = feature_space
        result["phase"] = "hybrid"
        result["target"] = target_col
        eval_rows.append(result)

    # Trials dataframe
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.insert(0, "feature_space", feature_space)
    trials_df.insert(0, "model", model_name)

    return {
        "model": model_name,
        "feature_space": feature_space,
        "best_val_qwk": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": n_trials,
        "trials_df": trials_df,
        "eval_df": pd.DataFrame(eval_rows),
    }


# ============================================================
# Main runner
# ============================================================
def run_hybrid_tuning(
    models: list[str] | None = None,
    feature_spaces: list[str] | None = None,
    target_col: str = DEFAULT_TARGET,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    verbose: bool = True,
) -> list[dict]:
    if models is None:
        models = list(HYBRID_MODELS)
    if feature_spaces is None:
        feature_spaces = list(HYBRID_FEATURE_SPACES)

    if not target_col.startswith("NPS_"):
        target_col = f"NPS_{target_col}"

    if verbose:
        print("=" * 70)
        print(f"HYBRID TUNING — target={target_col}")
        print(f"Models: {models}  ×  Feature spaces: {feature_spaces}")
        print("=" * 70)

    # Load dataset + verbatims
    df_path = DATA_PROCESSED / "dataset_with_verbatims.parquet"
    if not df_path.exists():
        raise FileNotFoundError(
            f"{df_path} missing — run `make load-verbatims` first."
        )
    df = pd.read_parquet(df_path)
    splits = load_splits("response_biased")
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"{pipeline_path} missing — run `make build-features` first."
        )
    pipeline = joblib.load(pipeline_path)

    # Compute or load embeddings
    embeddings_df = load_or_compute_embeddings(
        df, model_name=embedding_model, verbose=verbose,
    )

    # Defensive index alignment
    common = df.index.intersection(embeddings_df.index)
    if len(common) < len(df):
        if verbose:
            print(f"  ⚠ {len(df) - len(common)} rows dropped (no embedding)")
        df = df.loc[common]
        splits = splits.loc[common]
        embeddings_df = embeddings_df.loc[common]

    # Feature columns: drop NPS_* and verbatim aux columns
    feature_cols = [
        c for c in df.columns
        if not c.startswith("NPS_") and c not in VERBATIM_AUX_COLS
    ]

    if verbose:
        train_n = (splits == "train").sum()
        val_n = (splits == "val").sum()
        print(
            f"\nTrain: {train_n}  Val: {val_n}  "
            f"Tab cols: {len(feature_cols)}  Emb dim: {embeddings_df.shape[1]}"
        )

    all_trials: list[pd.DataFrame] = []
    all_evals: list[pd.DataFrame] = []
    summary: list[dict] = []

    # Outer loop on feature_space → matrices built once, reused for both models
    for feature_space in feature_spaces:
        if verbose:
            print(f"\n{'=' * 70}\nFEATURE SPACE: {feature_space}\n{'=' * 70}")
        X_per, y_per, info = _build_split_matrices(
            df=df, splits=splits, pipeline=pipeline,
            embeddings_df=embeddings_df,
            target_col=target_col, feature_cols=feature_cols,
            feature_space=feature_space, verbose=verbose,
        )

        for model_name in models:
            n_trials = TUNING_BUDGETS.get(model_name, 30)
            try:
                out = tune_one(
                    model_name=model_name, feature_space=feature_space,
                    X_per=X_per, y_per=y_per,
                    n_trials=n_trials, target_col=target_col,
                    verbose=verbose,
                )
            except ImportError as e:
                print(f"⚠ Skipping {model_name}/{feature_space}: {e}")
                continue
            all_trials.append(out["trials_df"])
            all_evals.append(out["eval_df"])
            summary.append({
                "model": model_name,
                "feature_space": feature_space,
                "best_val_qwk": out["best_val_qwk"],
                "best_params": out["best_params"],
                "n_trials": out["n_trials"],
                "n_dims": info["n_dims"],
                "pca_explained_variance": info.get("pca_explained_variance"),
            })

    # Persist
    if all_trials:
        trials_all = pd.concat(all_trials, ignore_index=True)
        trials_all.to_parquet(
            RESULTS_DIR / "hybrid_tuning_trials.parquet", index=False,
        )
        if verbose:
            print(f"\n✓ hybrid_tuning_trials.parquet ({len(trials_all)} trials)")
    if all_evals:
        evals_all = pd.concat(all_evals, ignore_index=True)
        evals_all.to_parquet(
            RESULTS_DIR / "hybrid_results.parquet", index=False,
        )
        if verbose:
            print(f"✓ hybrid_results.parquet ({len(evals_all)} evaluations)")

    # Summary
    if summary:
        print("\n" + "=" * 70)
        print("HYBRID TUNING SUMMARY")
        print("=" * 70)
        for s in summary:
            line = (
                f"\n  {s['model']:<10} × {s['feature_space']:<8} "
                f"| dims={s['n_dims']:>3} | val QWK = {s['best_val_qwk']:.4f}"
            )
            if s["pca_explained_variance"] is not None:
                line += f" | PCA var = {s['pca_explained_variance']:.1%}"
            print(line)
            for k, v in s["best_params"].items():
                if isinstance(v, float):
                    print(f"    {k:<22} = {v:.4g}")
                else:
                    print(f"    {k:<22} = {v}")

    return summary


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None, choices=HYBRID_MODELS,
        help="Which models to tune (default: lightgbm + logistic).",
    )
    parser.add_argument(
        "--feature-spaces", nargs="+", default=None, choices=HYBRID_FEATURE_SPACES,
        help="Which feature spaces to tune (default: concat + pca32).",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    try:
        run_hybrid_tuning(
            models=args.models,
            feature_spaces=args.feature_spaces,
            target_col=args.target,
            embedding_model=args.embedding_model,
            verbose=True,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
