"""
Evaluation metrics aligned with the NPS prediction problem.

Primary metric: Quadratic Weighted Kappa (QWK) — captures the ordinal nature
of the target. A wrong-by-2-classes prediction is penalized more than wrong-by-1.

Business metrics:
    - recall@Detractor: how many actual detractors does the model catch?
    - lift@k: among the top-k highest predicted Detractor probabilities,
      how many are actually Detractors? Used for prioritization.

Diagnostic metrics:
    - macro F1, balanced accuracy, per-class precision/recall
    - confusion matrix
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.config import NPS_CLASSES, NPS_CLASS_TO_INT


# ============================================================
# Encoding helpers
# ============================================================
def _to_int_labels(y) -> np.ndarray:
    """Convert string/categorical labels to 0/1/2 ints."""
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    if y.dtype.kind in {"U", "O"} or hasattr(y, "categories"):
        out = np.array([NPS_CLASS_TO_INT[str(v)] for v in y], dtype=int)
    else:
        out = np.asarray(y, dtype=int)
    return out


# ============================================================
# Core metrics
# ============================================================
def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """
    Quadratic Weighted Kappa, the primary metric for ordinal NPS prediction.

    Uses sklearn's cohen_kappa_score with weights="quadratic".
    """
    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred)
    return float(cohen_kappa_score(yt, yp, weights="quadratic"))


def macro_f1(y_true, y_pred) -> float:
    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred)
    return float(f1_score(yt, yp, average="macro"))


def balanced_acc(y_true, y_pred) -> float:
    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred)
    return float(balanced_accuracy_score(yt, yp))


def detractor_recall(y_true, y_pred) -> float:
    """Recall on the Detractor class — primary business metric."""
    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred)
    return float(recall_score(yt, yp, labels=[0], average="macro", zero_division=0))


def per_class_recall(y_true, y_pred) -> dict[str, float]:
    """Recall per NPS class."""
    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred)
    recalls = recall_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
    return {NPS_CLASSES[i]: float(recalls[i]) for i in range(3)}


# ============================================================
# Lift@k — business prioritization
# ============================================================
def lift_at_k(y_true, y_proba, k_pct: float = 0.10) -> float:
    """
    Lift on the Detractor class for the top-k% predicted probabilities.

    A retention manager who calls the top 10% will catch X times more
    Detractors than a random 10% sample. lift_at_10 = 3.0 means 3× better.
    """
    yt = _to_int_labels(y_true)
    proba_det = np.asarray(y_proba)[:, 0]  # Detractor = class 0

    n = len(yt)
    k = max(1, int(n * k_pct))

    # Sort by descending Detractor probability, take top k
    top_idx = np.argsort(-proba_det)[:k]
    n_det_top = (yt[top_idx] == 0).sum()
    base_rate = (yt == 0).mean()

    if base_rate == 0:
        return 0.0
    return float((n_det_top / k) / base_rate)


# ============================================================
# Confusion matrix as DataFrame for readability
# ============================================================
def confusion_df(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix with NPS class labels (rows=true, cols=pred)."""
    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred)
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    return pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in NPS_CLASSES],
        columns=[f"pred_{c}" for c in NPS_CLASSES],
    )


# ============================================================
# Aggregated report
# ============================================================
def evaluate(
    y_true,
    y_pred,
    y_proba=None,
    name: str = "model",
) -> dict:
    """
    Compute all metrics and return as a flat dict — perfect for storing in
    a results DataFrame and comparing across models.
    """
    out = {
        "model": name,
        "n": int(len(y_true)),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "balanced_acc": balanced_acc(y_true, y_pred),
        "detractor_recall": detractor_recall(y_true, y_pred),
    }
    pcr = per_class_recall(y_true, y_pred)
    for cls, r in pcr.items():
        out[f"recall_{cls.lower()}"] = r

    if y_proba is not None:
        out["lift@10"] = lift_at_k(y_true, y_proba, k_pct=0.10)
        out["lift@20"] = lift_at_k(y_true, y_proba, k_pct=0.20)

    return out


def evaluate_on_splits(
    model,
    df: pd.DataFrame,
    splits,
    target_col: str,
    pipeline=None,
    feature_cols: list | None = None,
    proba_method: bool = True,
) -> pd.DataFrame:
    """
    Run evaluation on all available splits (train, val, respondent_test, silent_test).

    Returns
    -------
    DataFrame with one row per split.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if not c.startswith("NPS_")]

    rows: list[dict] = []
    for split_name in splits.unique():
        mask = splits == split_name
        if mask.sum() == 0:
            continue
        X_split = df.loc[mask, feature_cols]
        y_split = df.loc[mask, target_col]

        # Apply pipeline if provided
        if pipeline is not None:
            X_enc = pipeline.transform(X_split)
        else:
            X_enc = X_split.to_numpy()

        # Predict
        y_pred = model.predict(X_enc)

        # Get probabilities
        y_proba = None
        if proba_method and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_enc)
            except Exception:
                y_proba = None

        result = evaluate(y_split, y_pred, y_proba, name=split_name)
        result["split"] = split_name
        rows.append(result)

    return pd.DataFrame(rows)
