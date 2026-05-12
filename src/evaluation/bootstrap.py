"""
Phase 9 — Stratified bootstrap confidence intervals for evaluation metrics.

We resample WITHIN each NPS class to preserve the 58/25/17 marginal
distribution. This is the right approach for class-imbalanced ordinal
classification.

Usage
-----
>>> from src.evaluation.bootstrap import bootstrap_ci
>>> value, lo, hi = bootstrap_ci(
...     metric_fn=quadratic_weighted_kappa,
...     y_true=y_silent, y_pred=y_pred,
...     n_resamples=1000, ci=0.95, random_state=42,
... )
>>> print(f"QWK = {value:.3f}  [{lo:.3f}, {hi:.3f}]")

For lift@k metrics that need probabilities, pass `y_proba`:
>>> value, lo, hi = bootstrap_ci(
...     metric_fn=lift_at_k_factory(k_pct=0.10),
...     y_true=y_silent, y_pred=y_pred, y_proba=y_proba,
...     n_resamples=1000, ci=0.95, random_state=42,
... )
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    _to_int_labels,
    balanced_acc,
    detractor_recall,
    lift_at_k,
    macro_f1,
    quadratic_weighted_kappa,
)


# ============================================================
# Stratified bootstrap resampling
# ============================================================
def _stratified_resample_indices(
    y: np.ndarray,
    n_resamples: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate `n_resamples` stratified bootstrap index arrays.

    Each resample preserves the marginal class distribution of `y`
    (sampled WITH replacement WITHIN each class).

    Returns
    -------
    indices_matrix : np.ndarray, shape (n_resamples, n)
        Each row is one bootstrap sample of indices into the original array.
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}

    out = np.empty((n_resamples, n), dtype=np.int64)
    for b in range(n_resamples):
        chunks = []
        for c in classes:
            idx_c = class_indices[c]
            sampled = rng.choice(idx_c, size=len(idx_c), replace=True)
            chunks.append(sampled)
        out[b] = np.concatenate(chunks)
    return out


# ============================================================
# Generic single-metric bootstrap
# ============================================================
def bootstrap_ci(
    metric_fn: Callable,
    y_true,
    y_pred,
    y_proba=None,
    n_resamples: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
    needs_proba: bool = False,
) -> tuple[float, float, float]:
    """
    Stratified bootstrap CI for a single metric.

    Parameters
    ----------
    metric_fn : Callable
        Either f(y_true, y_pred) -> float  (needs_proba=False)
        or     f(y_true, y_proba) -> float (needs_proba=True)
    needs_proba : bool
        If True, metric_fn is called as f(y_true_b, y_proba_b).
        Used for lift@k which needs probabilities.

    Returns
    -------
    (point_estimate, ci_lo, ci_hi)
    """
    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred) if not needs_proba else None
    yproba = np.asarray(y_proba) if y_proba is not None else None

    # Point estimate
    if needs_proba:
        if yproba is None:
            raise ValueError("needs_proba=True requires y_proba")
        point = float(metric_fn(yt, yproba))
    else:
        point = float(metric_fn(yt, yp))

    # Bootstrap resamples
    idx_matrix = _stratified_resample_indices(yt, n_resamples, random_state)
    values = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = idx_matrix[b]
        yt_b = yt[idx]
        if needs_proba:
            values[b] = metric_fn(yt_b, yproba[idx])
        else:
            values[b] = metric_fn(yt_b, yp[idx])

    alpha = (1 - ci) / 2
    lo = float(np.quantile(values, alpha))
    hi = float(np.quantile(values, 1 - alpha))
    return point, lo, hi


# ============================================================
# Lift factory for bootstrap (k_pct fixed at call time)
# ============================================================
def lift_at_k_factory(k_pct: float):
    """Return a closure with k_pct baked in, for use with bootstrap_ci."""
    def _lift(y_true, y_proba):
        return lift_at_k(y_true, y_proba, k_pct=k_pct)
    _lift.__name__ = f"lift_at_{int(k_pct * 100)}"
    return _lift


# ============================================================
# Multi-metric bootstrap — produces the canonical row for the
# Phase 9 final_eval_summary.parquet
# ============================================================
DEFAULT_METRICS = {
    "qwk":              {"fn": quadratic_weighted_kappa, "needs_proba": False},
    "macro_f1":         {"fn": macro_f1,                 "needs_proba": False},
    "balanced_acc":     {"fn": balanced_acc,             "needs_proba": False},
    "detractor_recall": {"fn": detractor_recall,         "needs_proba": False},
    "lift@10":          {"fn": lift_at_k_factory(0.10),  "needs_proba": True},
    "lift@20":          {"fn": lift_at_k_factory(0.20),  "needs_proba": True},
}


def bootstrap_all_metrics(
    y_true,
    y_pred,
    y_proba=None,
    metrics: dict | None = None,
    n_resamples: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute every metric with its bootstrap CI and return a DataFrame.

    Returns
    -------
    pd.DataFrame with columns: metric | value | ci_lo | ci_hi | n_resamples | ci
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    rows = []
    for name, spec in metrics.items():
        needs_proba = spec["needs_proba"]
        if needs_proba and y_proba is None:
            # skip silently
            continue
        v, lo, hi = bootstrap_ci(
            metric_fn=spec["fn"],
            y_true=y_true, y_pred=y_pred, y_proba=y_proba,
            n_resamples=n_resamples, ci=ci, random_state=random_state,
            needs_proba=needs_proba,
        )
        rows.append({
            "metric": name,
            "value": v,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_resamples": n_resamples,
            "ci": ci,
        })
    return pd.DataFrame(rows)
