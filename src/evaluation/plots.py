"""
Plotting helpers for baseline evaluation.

Used by Phase 6 notebook and Phase 9 evaluation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import NPS_CLASSES
from src.evaluation.metrics import _to_int_labels, confusion_df

NPS_PALETTE = {"Detractor": "#d62728", "Passive": "#ff7f0e", "Promoter": "#2ca02c"}


def plot_confusion_matrix(y_true, y_pred, ax=None, title: str = "Confusion matrix"):
    """Plot a 3x3 confusion matrix with NPS class labels."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    cm_df = confusion_df(y_true, y_pred)
    # Normalize per row
    cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0).round(3)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        cbar=False, ax=ax, vmin=0, vmax=1,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels([c.replace("pred_", "") for c in cm_df.columns])
    ax.set_yticklabels([c.replace("true_", "") for c in cm_df.index])
    return ax


def plot_lift_curve(
    y_true, y_proba, ax=None, title: str = "Lift curve — Detractor class"
):
    """
    Plot the lift curve for the Detractor class:
    on x: % of customers contacted (sorted by descending P(Detractor))
    on y: % of all detractors captured
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    yt = _to_int_labels(y_true)
    proba_det = np.asarray(y_proba)[:, 0]

    n = len(yt)
    order = np.argsort(-proba_det)
    yt_sorted = yt[order]

    cum_det = np.cumsum(yt_sorted == 0)
    total_det = cum_det[-1]
    if total_det == 0:
        ax.text(0.5, 0.5, "No detractors in y_true", ha="center", va="center")
        return ax

    pct_pop = np.arange(1, n + 1) / n
    pct_det = cum_det / total_det

    # Random baseline
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.5, label="Random")
    ax.plot(pct_pop, pct_det, color="#d62728", linewidth=2, label="Model")
    # Vertical lines at k=10 % and k=20 %
    for k_pct in [0.10, 0.20]:
        idx = int(n * k_pct) - 1
        ax.axvline(k_pct, ls=":", color="grey", alpha=0.4)
        ax.scatter([k_pct], [pct_det[idx]], color="#d62728", zorder=5)
        ax.annotate(
            f"k={int(k_pct*100)}% → {pct_det[idx]:.0%}",
            xy=(k_pct, pct_det[idx]), xytext=(k_pct + 0.03, pct_det[idx] - 0.05),
            fontsize=9,
        )

    ax.set_xlabel("% of customers contacted (top-k by Detractor proba)")
    ax.set_ylabel("% of Detractors captured")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_results_comparison(
    results_df: pd.DataFrame,
    metrics: list[str] | None = None,
    save_path=None,
):
    """
    Bar plot: rows = models, columns = metrics, color = split.
    """
    if metrics is None:
        metrics = ["qwk", "macro_f1", "balanced_acc", "detractor_recall"]
    metrics = [m for m in metrics if m in results_df.columns]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        pivot = results_df.pivot_table(
            index="model", columns="split", values=metric,
        )
        pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_per_class_recall(results_df: pd.DataFrame, save_path=None):
    """Show per-class recall for each (model, split) combination."""
    cols = ["recall_detractor", "recall_passive", "recall_promoter"]
    cols = [c for c in cols if c in results_df.columns]
    if not cols:
        return None

    melted = results_df.melt(
        id_vars=["model", "split"], value_vars=cols,
        var_name="class", value_name="recall",
    )
    melted["class"] = melted["class"].str.replace("recall_", "").str.capitalize()

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.barplot(
        data=melted, x="model", y="recall",
        hue="class",
        palette=[NPS_PALETTE["Detractor"], NPS_PALETTE["Passive"], NPS_PALETTE["Promoter"]],
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Per-class recall — all models, all splits combined")
    ax.set_ylabel("Recall")
    ax.set_xlabel("")
    ax.legend(title="NPS class", loc="upper right", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig
