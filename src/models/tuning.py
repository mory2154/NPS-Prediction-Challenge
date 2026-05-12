"""
Hyperparameter tuning with Optuna.

Three studies, one per model family. Each study optimizes Quadratic Weighted
Kappa on the validation split (NOT the test sets — those stay sealed).

CLI:
    python -m src.models.tuning
    python -m src.models.tuning --models lightgbm
    python -m src.models.tuning --target NPS_baseline
"""

from __future__ import annotations

import argparse
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DEFAULT_TARGET,
    MODELS_DIR,
    NPS_CLASS_TO_INT,
    RANDOM_SEED,
    RESULTS_DIR,
    TUNED_DIR,
    TUNING_BUDGETS,
    TUNING_SAMPLER,
    TUNING_TIMEOUT_SEC,
)
from src.data.split import get_split, load_splits
from src.evaluation.metrics import (
    evaluate_on_splits,
    quadratic_weighted_kappa,
)
# Import OrdinalWrapper from wrappers (NOT defined here) so that joblib pickles
# it with a stable qualified name `src.models.wrappers.OrdinalWrapper`.
from src.models.wrappers import OrdinalWrapper

warnings.filterwarnings("ignore")


# ============================================================
# Helpers
# ============================================================
def _to_int(y) -> np.ndarray:
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    if y.dtype.kind in {"U", "O"} or hasattr(y, "categories"):
        return np.array([NPS_CLASS_TO_INT[str(v)] for v in y], dtype=int)
    return np.asarray(y, dtype=int)


def _make_sampler(seed: int):
    import optuna

    if TUNING_SAMPLER == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    elif TUNING_SAMPLER == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unknown sampler: {TUNING_SAMPLER}")


# ============================================================
# Search spaces
# ============================================================
def lightgbm_objective_factory(X_train_enc, y_train_int, X_val_enc, y_val_int):
    from lightgbm import LGBMClassifier

    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train_enc, y_train_int)
        y_pred = model.predict(X_val_enc)
        return quadratic_weighted_kappa(y_val_int, y_pred)

    return objective


def logistic_objective_factory(X_train_enc, y_train_int, X_val_enc, y_val_int):
    from sklearn.linear_model import LogisticRegression

    def objective(trial):
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "saga" if penalty == "l1" else trial.suggest_categorical("solver", ["lbfgs", "saga"])
        C = trial.suggest_float("C", 1e-3, 100.0, log=True)

        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        try:
            model.fit(X_train_enc, y_train_int)
        except Exception:
            return 0.0
        y_pred = model.predict(X_val_enc)
        return quadratic_weighted_kappa(y_val_int, y_pred)

    return objective


def ordinal_objective_factory(X_train_enc, y_train_int, X_val_enc, y_val_int):
    from mord import LogisticAT

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
        model = LogisticAT(alpha=alpha)
        model.fit(X_train_enc, y_train_int)
        y_pred = model.predict(X_val_enc)
        return quadratic_weighted_kappa(y_val_int, y_pred)

    return objective


OBJECTIVE_FACTORIES = {
    "lightgbm": lightgbm_objective_factory,
    "logistic": logistic_objective_factory,
    "ordinal": ordinal_objective_factory,
}


# ============================================================
# Build a final model from best params
# ============================================================
def build_final_model(name: str, best_params: dict):
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="multiclass", num_class=3,
            class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
            **best_params,
        )
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        params = best_params.copy()
        if params.get("penalty") == "l1":
            params["solver"] = "saga"
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
            **params,
        )
    if name == "ordinal":
        from mord import LogisticAT
        return LogisticAT(**best_params)
    raise ValueError(f"Unknown model name: {name}")


# ============================================================
# Run a single study
# ============================================================
def tune_one_model(
    name, X_train_enc, y_train_int, X_val_enc, y_val_int,
    n_trials: int, verbose: bool = True,
):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if verbose:
        print(f"\n--- Tuning {name} ({n_trials} trials, sampler={TUNING_SAMPLER}) ---")

    sampler = _make_sampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = OBJECTIVE_FACTORIES[name](X_train_enc, y_train_int, X_val_enc, y_val_int)

    study.optimize(objective, n_trials=n_trials, timeout=TUNING_TIMEOUT_SEC, show_progress_bar=False)

    if verbose:
        print(f"  Best QWK on val: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.insert(0, "model", name)
    return study.best_params, study, trials_df


# ============================================================
# Main pipeline
# ============================================================
def run_tuning(
    models: list[str] | None = None,
    target_col: str = DEFAULT_TARGET,
    verbose: bool = True,
):
    if models is None:
        models = list(TUNING_BUDGETS.keys())

    if not target_col.startswith("NPS_"):
        target_col = f"NPS_{target_col}"

    if verbose:
        print("=" * 70)
        print(f"TUNING ON TARGET: {target_col}")
        print("=" * 70)

    df = pd.read_parquet(DATA_PROCESSED / "dataset_with_features.parquet")
    splits = load_splits("response_biased")
    pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
    feature_cols = [c for c in df.columns if not c.startswith("NPS_")]

    X_train, y_train = get_split(df, splits, "train", target_col=target_col)
    X_val, y_val = get_split(df, splits, "val", target_col=target_col)
    X_train, X_val = X_train[feature_cols], X_val[feature_cols]
    X_train_enc = pipeline.transform(X_train)
    X_val_enc = pipeline.transform(X_val)
    y_train_int = _to_int(y_train)
    y_val_int = _to_int(y_val)

    if verbose:
        print(f"Train: {X_train_enc.shape}  Val: {X_val_enc.shape}")
        print(f"Class dist (train): {np.bincount(y_train_int)}")

    all_trials: list[pd.DataFrame] = []
    eval_rows: list[pd.DataFrame] = []
    best_params_summary: list[dict] = []

    for name in models:
        n_trials = TUNING_BUDGETS.get(name, 30)
        try:
            best_params, study, trials_df = tune_one_model(
                name=name,
                X_train_enc=X_train_enc, y_train_int=y_train_int,
                X_val_enc=X_val_enc, y_val_int=y_val_int,
                n_trials=n_trials, verbose=verbose,
            )
        except ImportError as e:
            print(f"⚠ Skipping {name}: {e}")
            continue
        all_trials.append(trials_df)

        final_model = build_final_model(name, best_params)
        if name == "ordinal":
            wrapped = OrdinalWrapper(final_model)
            wrapped.fit(X_train_enc, y_train_int)
            fitted = wrapped
        else:
            fitted = final_model.fit(X_train_enc, y_train_int)

        out_path = TUNED_DIR / f"{name}_tuned.joblib"
        joblib.dump(fitted, out_path)
        if verbose:
            print(f"  ✓ Saved {out_path.name}")

        results = evaluate_on_splits(
            model=fitted, df=df, splits=splits,
            target_col=target_col, pipeline=pipeline,
            feature_cols=feature_cols,
        )
        results["model"] = name
        results["target"] = target_col
        results["phase"] = "tuned"
        eval_rows.append(results)

        best_params_summary.append({
            "model": name,
            "target": target_col,
            "n_trials": n_trials,
            "best_val_qwk": float(study.best_value),
            "best_params": best_params,
        })

    if all_trials:
        trials_df = pd.concat(all_trials, ignore_index=True)
        trials_df.to_parquet(RESULTS_DIR / "tuning_results.parquet", index=False)
        if verbose:
            print(f"\n✓ Saved tuning_results.parquet ({len(trials_df)} trials)")
    if eval_rows:
        eval_df = pd.concat(eval_rows, ignore_index=True)
        eval_df.to_parquet(RESULTS_DIR / "tuned_results.parquet", index=False)
        if verbose:
            print(f"✓ Saved tuned_results.parquet ({len(eval_df)} evaluations)")

    if best_params_summary:
        print("\n" + "=" * 70)
        print("TUNING SUMMARY")
        print("=" * 70)
        for s in best_params_summary:
            print(f"\n  {s['model']:<10} | val QWK = {s['best_val_qwk']:.4f}")
            for k, v in s["best_params"].items():
                if isinstance(v, float):
                    print(f"    {k:<22} = {v:.4g}")
                else:
                    print(f"    {k:<22} = {v}")

    return best_params_summary


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=list(TUNING_BUDGETS),
        help="Which models to tune (default: all).",
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET,
        help="NPS target column name (default: NPS_alternative).",
    )
    args = parser.parse_args()

    try:
        run_tuning(models=args.models, target_col=args.target, verbose=True)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n✗ {e}", file=sys.stderr)
        print("\nMake sure you've run `make baseline` first.\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
