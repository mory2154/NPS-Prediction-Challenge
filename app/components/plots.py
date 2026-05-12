"""
Graphiques réutilisables pour l'app NPS.

Wrappers fins autour de matplotlib qui produisent une Figure prête à être
passée à st.pyplot(fig). On évite Plotly pour limiter les dépendances —
matplotlib est déjà utilisé dans les notebooks 7-11.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import NPS_CLASSES


# Traductions des noms de classes pour l'affichage
CLASS_LABELS_FR = {
    "Detractor": "Détracteur",
    "Passive":   "Passif",
    "Promoter":  "Promoteur",
}


# ============================================================
# Jauge de probabilité — barre empilée horizontale
# ============================================================
def probability_gauge(p_det: float, p_pas: float, p_pro: float, title: str = ""):
    fig, ax = plt.subplots(figsize=(8, 1.1))
    colors = ["#d62728", "#ff9800", "#2ca02c"]
    vals = [p_det, p_pas, p_pro]
    labels = ["Détracteur", "Passif", "Promoteur"]

    left = 0
    for v, c, lab in zip(vals, colors, labels):
        ax.barh(0, v, left=left, color=c, edgecolor="white", linewidth=0.5)
        if v >= 0.10:
            ax.text(left + v / 2, 0, f"{lab}\n{v:.1%}",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        left += v

    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=8)
    ax.set_title(title, fontsize=10, pad=4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# Cascade — explication locale (SHAP ou coef × valeur)
# ============================================================
def explanation_waterfall(
    contributions: pd.DataFrame,
    expected_value: float | None = None,
    title: str = "Pourquoi cette prédiction ?",
    top_k: int = 8,
):
    """
    Affiche un diagramme en cascade des contributions des features.

    Parameters
    ----------
    contributions : DataFrame
        Colonnes requises : 'feature', 'contribution' (float signé).
        Optionnelle : 'feature_value' (sera concaténée au label).
    expected_value : float, optionnel
        Baseline (valeur attendue du modèle). Si donnée, tracée comme ligne verticale.
    top_k : int
        Affiche les top-k contributions par |valeur|.
    """
    df = contributions.copy()
    df["abs"] = df["contribution"].abs()
    df = df.nlargest(top_k, "abs").sort_values("contribution")

    labels = df.apply(
        lambda r: (
            f"{r['feature']} = {r['feature_value']:.2g}"
            if "feature_value" in df.columns and pd.notna(r.get("feature_value"))
            else r["feature"]
        ),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(df) + 1)))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in df["contribution"]]
    ax.barh(range(len(df)), df["contribution"], color=colors,
            edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Contribution (signée)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")

    # Valeurs en bout de barre
    for i, (v, _) in enumerate(zip(df["contribution"], df["feature"])):
        offset = (0.02 * df["abs"].max()) * (1 if v > 0 else -1)
        ax.text(v + offset, i, f"{v:+.3f}",
                va="center", ha="left" if v > 0 else "right", fontsize=8)
    plt.tight_layout()
    return fig


# ============================================================
# Rappel par groupe avec barres d'erreur — utilisé dans Équité
# ============================================================
def per_group_recall_bars(
    per_group_df: pd.DataFrame,
    segment: str,
    target_class: str,
    champion: str,
    champion_color: str = "#d62728",
):
    sub = per_group_df[
        (per_group_df["segment"] == segment)
        & (per_group_df["class"] == target_class)
        & (per_group_df["champion"] == champion)
    ].sort_values("group")
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(sub))
    values = sub["recall"].values
    lo = (sub["recall"] - sub["recall_ci_lo"]).values
    hi = (sub["recall_ci_hi"] - sub["recall"]).values
    ax.bar(x, values, yerr=[lo, hi], capsize=6,
           color=champion_color, alpha=0.85,
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(sub["group"].tolist(), fontsize=10)
    target_class_fr = CLASS_LABELS_FR.get(target_class, target_class)
    ax.set_ylabel(f"Rappel {target_class_fr}"); ax.set_ylim(0, 1)
    ax.set_title(f"{champion} — Rappel {target_class_fr} par {segment}", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    return fig


# ============================================================
# Rang du client dans la population
# ============================================================
def rank_in_population(rank: int, total: int, target_class: str = "Detractor"):
    """Figure simple montrant où le client se situe dans la population silent."""
    fig, ax = plt.subplots(figsize=(8, 0.8))
    pct = (rank - 1) / total * 100
    target_class_fr = CLASS_LABELS_FR.get(target_class, target_class)
    ax.barh([0], [100], color="#cccccc", height=0.4)
    ax.barh([0], [pct], color="#d62728" if target_class == "Detractor" else "#2ca02c",
            height=0.4)
    ax.axvline(pct, color="black", lw=1)
    ax.set_xlim(0, 100); ax.set_ylim(-0.4, 0.4); ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=8)
    ax.set_title(
        f"Rang {rank:,} / {total:,} ({pct:.1f} %) selon P({target_class_fr})",
        fontsize=10,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig
