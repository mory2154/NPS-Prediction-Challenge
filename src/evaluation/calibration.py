"""
Phase 9 — Probability calibration analysis.

For an ordinal classifier with `predict_proba`, we check whether the predicted
probabilities match observed frequencies. Two outputs:

1. **Reliability curve per class (one-vs-rest)** : bin predictions by predicted
   probability, plot observed rate vs predicted mean. A perfectly calibrated
   model lies on the diagonal y = x.

2. **Brier score per class** : mean squared error between predicted probability
   and true binary outcome (one-vs-rest). Lower is better, 0 = perfect.

Why this matters for Phase 9
----------------------------
The retention manager will use `predict_proba(Detractor) >= threshold` (or top-k)
to prioritize who to call. If probabilities are mis-calibrated, the threshold
loses its meaning. lift@10 is interpretable only if calibration is reasonable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.config import NPS_CLASSES
from src.evaluation.metrics import _to_int_labels


# ============================================================
# Reliability curve (manual implementation — sklearn's only does binary)
# ============================================================
def reliability_curve_one_vs_rest(
    y_true,
    y_proba,
    class_idx: int,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """
    Compute the reliability curve for one class (one-vs-rest).

    Parameters
    ----------
    y_true : array of int labels (0/1/2)
    y_proba : (n, 3) array
    class_idx : 0, 1 or 2
    n_bins : number of bins
    strategy : "uniform" (equal-width bins) or "quantile" (equal-mass bins)

    Returns
    -------
    DataFrame with columns:
        bin_id, bin_lo, bin_hi, n, predicted_mean, observed_rate
    """
    yt = _to_int_labels(y_true)
    p = np.asarray(y_proba)[:, class_idx]
    y_bin = (yt == class_idx).astype(int)

    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf + 1e-9

    bin_ids = np.digitize(p, edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        n_in = int(mask.sum())
        if n_in == 0:
            rows.append({
                "bin_id": b, "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "n": 0, "predicted_mean": np.nan, "observed_rate": np.nan,
            })
            continue
        rows.append({
            "bin_id": b,
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": n_in,
            "predicted_mean": float(p[mask].mean()),
            "observed_rate": float(y_bin[mask].mean()),
        })
    return pd.DataFrame(rows)


# ============================================================
# Brier score per class (one-vs-rest)
# ============================================================
def brier_per_class(y_true, y_proba) -> dict[str, float]:
    """
    Brier score per class (one-vs-rest). Lower = better, 0 = perfect.

    Returns
    -------
    {class_name: brier_score}
    """
    yt = _to_int_labels(y_true)
    p = np.asarray(y_proba)
    out = {}
    for i, c in enumerate(NPS_CLASSES):
        y_bin = (yt == i).astype(int)
        out[c] = float(brier_score_loss(y_bin, p[:, i]))
    return out


# ============================================================
# Aggregated calibration report
# ============================================================
def calibration_report(
    y_true,
    y_proba,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """
    Run the full calibration analysis.

    Returns
    -------
    dict with keys:
        - "curves": {class_name: DataFrame from reliability_curve_one_vs_rest}
        - "brier": {class_name: float}
        - "ece": {class_name: float}  (Expected Calibration Error)
    """
    curves = {}
    ece = {}
    for i, c in enumerate(NPS_CLASSES):
        curves[c] = reliability_curve_one_vs_rest(
            y_true, y_proba, class_idx=i,
            n_bins=n_bins, strategy=strategy,
        )
        # ECE = weighted average |predicted_mean - observed_rate|, weighted by n
        df = curves[c].dropna(subset=["predicted_mean", "observed_rate"])
        if len(df) > 0 and df["n"].sum() > 0:
            weights = df["n"] / df["n"].sum()
            ece[c] = float(
                (weights * (df["predicted_mean"] - df["observed_rate"]).abs()).sum()
            )
        else:
            ece[c] = float("nan")

    return {
        "curves": curves,
        "brier": brier_per_class(y_true, y_proba),
        "ece": ece,
    }
