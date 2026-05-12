"""
Phase 11 — Bootstrap CIs computed WITHIN each subgroup.

For a fairness audit we need CIs around the recall ESTIMATE for each
(model, segment, group) cell. Resampling must happen within the group
(stratified by class) — otherwise we mix bias from across groups.

Used by `audit.py` to attach `[ci_lo, ci_hi]` to every per-group recall, so
the report can say "Recall Senior=Yes is 0.85 [0.81, 0.88] vs Senior=No
0.82 [0.80, 0.84] — CIs overlap, no significant disparity".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.evaluation.metrics import _to_int_labels


# ============================================================
# Helper: stratified resample WITHIN a group
# ============================================================
def _stratified_resample_within(
    y: np.ndarray,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    """
    Stratified bootstrap indices preserving class marginals.
    Returns (n_resamples, n) int matrix.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}
    out = np.empty((n_resamples, n), dtype=np.int64)
    for b in range(n_resamples):
        chunks = []
        for c in classes:
            idx_c = class_indices[c]
            chunks.append(rng.choice(idx_c, size=len(idx_c), replace=True))
        out[b] = np.concatenate(chunks)
    return out


# ============================================================
# Bootstrap recall + selection_rate for one class within one group
# ============================================================
def bootstrap_group_metrics(
    y_true_group: np.ndarray,
    y_pred_group: np.ndarray,
    class_idx: int,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = RANDOM_SEED,
    min_class_n: int = 5,
) -> dict:
    """
    Bootstrap recall and selection rate for one (group × class) cell.

    Parameters
    ----------
    min_class_n : int
        If the group has fewer than `min_class_n` true positives, return NaN
        CIs (resampling would be too unstable).

    Returns
    -------
    {
        "recall":              float,
        "recall_ci_lo":        float,
        "recall_ci_hi":        float,
        "selection_rate":      float,
        "selection_rate_ci_lo": float,
        "selection_rate_ci_hi": float,
        "n_total":             int,
        "n_class":             int,
    }
    """
    yt = _to_int_labels(y_true_group)
    yp = _to_int_labels(y_pred_group)

    mask_pos = yt == class_idx
    n_pos = int(mask_pos.sum())
    n_total = int(len(yt))

    point_recall = (
        float((mask_pos & (yp == class_idx)).sum() / n_pos) if n_pos > 0 else float("nan")
    )
    point_sel = float((yp == class_idx).mean()) if n_total > 0 else float("nan")

    if n_pos < min_class_n or n_total < min_class_n * 3:
        return {
            "recall":              point_recall,
            "recall_ci_lo":        float("nan"),
            "recall_ci_hi":        float("nan"),
            "selection_rate":      point_sel,
            "selection_rate_ci_lo": float("nan"),
            "selection_rate_ci_hi": float("nan"),
            "n_total":             n_total,
            "n_class":             n_pos,
        }

    # Bootstrap
    idx_matrix = _stratified_resample_within(yt, n_resamples, seed)
    recalls = np.empty(n_resamples)
    sel_rates = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = idx_matrix[b]
        ytb = yt[idx]
        ypb = yp[idx]
        m_pos_b = ytb == class_idx
        n_pos_b = m_pos_b.sum()
        recalls[b] = (m_pos_b & (ypb == class_idx)).sum() / n_pos_b if n_pos_b else np.nan
        sel_rates[b] = (ypb == class_idx).mean() if len(ypb) else np.nan

    alpha = (1 - ci) / 2
    return {
        "recall":              point_recall,
        "recall_ci_lo":        float(np.nanquantile(recalls, alpha)),
        "recall_ci_hi":        float(np.nanquantile(recalls, 1 - alpha)),
        "selection_rate":      point_sel,
        "selection_rate_ci_lo": float(np.nanquantile(sel_rates, alpha)),
        "selection_rate_ci_hi": float(np.nanquantile(sel_rates, 1 - alpha)),
        "n_total":             n_total,
        "n_class":             n_pos,
    }


# ============================================================
# Build the per-group breakdown DataFrame
# ============================================================
def per_group_breakdown(
    y_true,
    y_pred,
    groups,
    class_indices: list[int],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build the full per-group breakdown for the given target classes.

    Returns one row per (group × class) with:
        group, class, n_total, n_class, recall, recall_ci_lo, recall_ci_hi,
        selection_rate, selection_rate_ci_lo, selection_rate_ci_hi
    """
    from src.config import NPS_CLASSES

    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred)
    g = pd.Series(groups).astype(str).reset_index(drop=True)

    rows = []
    for grp in sorted(g.unique()):
        mask = (g == grp).to_numpy()
        yt_grp = yt[mask]
        yp_grp = yp[mask]
        for ci_idx in class_indices:
            stats = bootstrap_group_metrics(
                yt_grp, yp_grp, ci_idx,
                n_resamples=n_resamples, ci=ci, seed=seed,
            )
            stats["group"] = grp
            stats["class"] = NPS_CLASSES[ci_idx]
            stats["class_idx"] = ci_idx
            rows.append(stats)
    return pd.DataFrame(rows)
