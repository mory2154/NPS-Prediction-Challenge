"""
Phase 11 — Fairness metrics.

Three group-fairness indicators on top of the standard sklearn metrics:

    1. **Disparate Impact (DI)** — ratio of recall (or selection rate) between
       a group and a reference. Threshold rule of thumb (US EEOC "4/5 rule"):
       DI ∈ [0.8, 1.25] is considered fair.

    2. **Equal Opportunity Difference (EOD)** — recall(class | groupA) −
       recall(class | groupB). |EOD| < 0.10 commonly considered acceptable.

    3. **Demographic Parity Difference (DPD)** — P(ŷ = class | groupA) −
       P(ŷ = class | groupB). Not the primary criterion (legitimate business
       differentiation can produce DPD ≠ 0), but documented for transparency.

We compute these for two target classes:
    * **Detractor** (the retention-priority class, brief 4.7)
    * **Promoter** (symmetric audit — we shouldn't miss promoters either)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import NPS_CLASSES
from src.evaluation.metrics import _to_int_labels


# ============================================================
# Per-group recall for one target class (one-vs-rest)
# ============================================================
def recall_for_class(y_true, y_pred, class_idx: int) -> float:
    """Recall = TP / (TP + FN) for the given class index (one-vs-rest)."""
    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred)
    mask_pos = yt == class_idx
    n_pos = int(mask_pos.sum())
    if n_pos == 0:
        return float("nan")
    tp = int((mask_pos & (yp == class_idx)).sum())
    return tp / n_pos


def selection_rate_for_class(y_pred, class_idx: int) -> float:
    """P(ŷ = class) — used for Demographic Parity."""
    yp = _to_int_labels(y_pred)
    if len(yp) == 0:
        return float("nan")
    return float((yp == class_idx).mean())


# ============================================================
# Disparate Impact
# ============================================================
def disparate_impact(
    y_true,
    y_pred,
    groups,
    class_idx: int,
    reference_group: str | None = None,
) -> pd.DataFrame:
    """
    DI(group) = recall(group) / recall(reference_group)

    Returns
    -------
    DataFrame columns: group | n | recall | DI | DI_status
    """
    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred)
    g = pd.Series(groups).astype(str).reset_index(drop=True)

    rows = []
    for grp in sorted(g.unique()):
        mask = (g == grp).to_numpy()
        r = recall_for_class(yt[mask], yp[mask], class_idx)
        rows.append({"group": grp, "n": int(mask.sum()), "recall": r})
    df = pd.DataFrame(rows)

    if reference_group is None:
        # default: pick the group with highest recall (the "favored" group)
        reference_group = df.loc[df["recall"].idxmax(), "group"]
    ref_recall = df.loc[df["group"] == reference_group, "recall"].iloc[0]
    df["reference_group"] = reference_group
    df["DI"] = df["recall"] / ref_recall if ref_recall > 0 else np.nan
    df["DI_status"] = df["DI"].apply(
        lambda x: "fair" if 0.8 <= x <= 1.25 else "biased"
    )
    return df


# ============================================================
# Equal Opportunity Difference
# ============================================================
def equal_opportunity_difference(
    y_true,
    y_pred,
    groups,
    class_idx: int,
) -> dict:
    """
    EOD = max(recall) − min(recall) across groups.

    Returns
    -------
    {"max_diff", "max_group", "min_group", "per_group_recall"}
    """
    yt = _to_int_labels(y_true)
    yp = _to_int_labels(y_pred)
    g = pd.Series(groups).astype(str).reset_index(drop=True)

    per_group = {}
    for grp in sorted(g.unique()):
        mask = (g == grp).to_numpy()
        per_group[grp] = recall_for_class(yt[mask], yp[mask], class_idx)
    if not per_group:
        return {"max_diff": float("nan"), "max_group": None,
                "min_group": None, "per_group_recall": {}}
    max_grp = max(per_group, key=per_group.get)
    min_grp = min(per_group, key=per_group.get)
    return {
        "max_diff":        per_group[max_grp] - per_group[min_grp],
        "max_group":       max_grp,
        "min_group":       min_grp,
        "per_group_recall": per_group,
    }


# ============================================================
# Demographic Parity Difference
# ============================================================
def demographic_parity_difference(y_pred, groups, class_idx: int) -> dict:
    """
    DPD = max(P(ŷ=class)) − min(P(ŷ=class)) across groups.

    Returns
    -------
    {"max_diff", "max_group", "min_group", "per_group_selection_rate"}
    """
    yp = _to_int_labels(y_pred)
    g = pd.Series(groups).astype(str).reset_index(drop=True)

    per_group = {}
    for grp in sorted(g.unique()):
        mask = (g == grp).to_numpy()
        per_group[grp] = selection_rate_for_class(yp[mask], class_idx)
    if not per_group:
        return {"max_diff": float("nan"), "max_group": None,
                "min_group": None, "per_group_selection_rate": {}}
    max_grp = max(per_group, key=per_group.get)
    min_grp = min(per_group, key=per_group.get)
    return {
        "max_diff":                per_group[max_grp] - per_group[min_grp],
        "max_group":               max_grp,
        "min_group":               min_grp,
        "per_group_selection_rate": per_group,
    }


# ============================================================
# Combined audit summary for one (model, segment, class)
# ============================================================
def audit_one(
    y_true,
    y_pred,
    groups,
    class_idx: int,
) -> dict:
    """Run the 3 fairness indicators in one shot."""
    di = disparate_impact(y_true, y_pred, groups, class_idx)
    eod = equal_opportunity_difference(y_true, y_pred, groups, class_idx)
    dpd = demographic_parity_difference(y_pred, groups, class_idx)
    return {
        "class":                 NPS_CLASSES[class_idx],
        "class_idx":             class_idx,
        "disparate_impact":      di,
        "equal_opportunity_diff": eod["max_diff"],
        "eod_max_group":         eod["max_group"],
        "eod_min_group":         eod["min_group"],
        "per_group_recall":      eod["per_group_recall"],
        "demographic_parity_diff": dpd["max_diff"],
        "dpd_max_group":         dpd["max_group"],
        "dpd_min_group":         dpd["min_group"],
        "per_group_selection_rate": dpd["per_group_selection_rate"],
    }
