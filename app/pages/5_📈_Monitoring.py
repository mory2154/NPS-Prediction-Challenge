"""
Page : Monitoring (simulation).

Affiche les résultats du simulateur de drift Phase 13 et des alertes 2σ
basées sur les CIs Phase 9/11.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "app"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import RESULTS_DIR


st.set_page_config(page_title="Monitoring · NPS", page_icon="📈", layout="wide")
st.title("📈 Monitoring (simulation)")

st.markdown(
    "**Source : simulateur de dérive Phase 13.** silent_test découpé en batches "
    "mensuels (FIFO par Customer ID), métriques calculées par batch, alertes "
    "déclenchées quand une valeur sort des intervalles de confiance Phase 9/11."
)


# ============================================================
# Chargement
# ============================================================
try:
    drift = pd.read_parquet(RESULTS_DIR / "drift_simulation.parquet")
    alerts = pd.read_parquet(RESULTS_DIR / "monitoring_alerts.parquet")
    cal_before_after = pd.read_parquet(RESULTS_DIR / "calibration_before_after.parquet")
    cal_headline = pd.read_parquet(RESULTS_DIR / "calibration_headline.parquet")
except FileNotFoundError as e:
    st.error(
        f"Artefact de monitoring manquant : {e.filename}. Exécuter depuis la racine :\n\n"
        "```bash\nmake recalibrate\nmake simulate-drift\n```"
    )
    st.stop()


# ============================================================
# Carte recalibration C2
# ============================================================
st.subheader("1. Recalibration de C2 (isotonic, fit sur val)")

st.markdown(
    "Phase 9 a documenté que les probabilités de C2 sont mal calibrées sur "
    "Passif/Promoteur (ECE > 0,10). La recalibration isotonic corrige ça **sans "
    "changer le classement** (préservée par construction)."
)

# Two columns: Brier and ECE before/after
col_b, col_e = st.columns(2)

with col_b:
    st.markdown("**Brier (plus bas = mieux)**")
    fig, ax = plt.subplots(figsize=(5, 3))
    classes_fr = {"Detractor": "Détracteur", "Passive": "Passif", "Promoter": "Promoteur"}
    x = np.arange(len(cal_before_after))
    width = 0.35
    ax.bar(x - width/2, cal_before_after["brier_before"], width,
           label="Avant", color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.bar(x + width/2, cal_before_after["brier_after"], width,
           label="Après", color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([classes_fr.get(c, c) for c in cal_before_after["class"]],
                       fontsize=10)
    ax.set_ylabel("Brier")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig)

with col_e:
    st.markdown("**ECE — Expected Calibration Error (plus bas = mieux)**")
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(cal_before_after))
    width = 0.35
    ax.bar(x - width/2, cal_before_after["ece_before"], width,
           label="Avant", color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.bar(x + width/2, cal_before_after["ece_after"], width,
           label="Après", color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([classes_fr.get(c, c) for c in cal_before_after["class"]],
                       fontsize=10)
    ax.set_ylabel("ECE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig)

with st.expander("Tableau détaillé Avant / Après"):
    display_cal = cal_before_after.copy()
    display_cal["class"] = display_cal["class"].map(classes_fr)
    display_cal = display_cal.rename(columns={
        "class": "Classe",
        "brier_before": "Brier avant",
        "brier_after":  "Brier après",
        "brier_delta":  "Δ Brier",
        "ece_before":   "ECE avant",
        "ece_after":    "ECE après",
        "ece_delta":    "Δ ECE",
    })
    st.dataframe(display_cal, hide_index=True, use_container_width=True)

with st.expander("Métriques globales — la recalibration affecte-t-elle le ranking ?"):
    st.caption(
        "L'isotonic est monotone non-décroissante : les **rangs** sont strictement "
        "préservés, donc `lift@10` doit être identique. QWK et F1 macro peuvent "
        "bouger légèrement car l'argmax peut basculer entre classes adjacentes."
    )
    display_head = cal_headline.copy().rename(columns={
        "metric": "Métrique",
        "before": "Avant",
        "after":  "Après",
        "delta":  "Δ",
    })
    st.dataframe(display_head, hide_index=True, use_container_width=True)


# ============================================================
# Carte 2 — métriques globales par batch
# ============================================================
st.divider()
st.subheader("2. Évolution des KPIs sur les batches mensuels")

champion_choice = st.radio(
    "Champion à suivre",
    options=["C2 (Production)", "C1 (Champion QWK)"],
    horizontal=True,
)
champion_key = "C2" if champion_choice.startswith("C2") else "C1"
color = "#d62728" if champion_key == "C2" else "#1f77b4"


# Filter global metrics (no segment)
global_drift = drift[(drift["champion"] == champion_key) & drift["segment"].isna()]

# Get reference CIs from alerts (those have ref_lo, ref_hi precomputed)
alert_global = alerts[
    (alerts["champion"] == champion_key)
    & alerts["segment"].isna()
]

metrics_to_plot = ["qwk", "detractor_recall", "macro_f1", "lift@10"]
metric_labels = {
    "qwk": "QWK",
    "detractor_recall": "Rappel Détracteur",
    "macro_f1": "F1 macro",
    "lift@10": "Lift @10",
}
severity_colors = {"ok": "#2ca02c", "warn": "#ff9800", "alert": "#d62728"}

fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
for ax, m in zip(axes.flat, metrics_to_plot):
    g = global_drift[global_drift["metric"] == m].sort_values("batch_id")
    a = alert_global[alert_global["metric"] == m].sort_values("batch_id")
    if g.empty:
        ax.set_title(f"{metric_labels[m]} — pas de données")
        continue

    # Reference band
    if not a.empty:
        ref_lo = a["ref_lo"].iloc[0]
        ref_hi = a["ref_hi"].iloc[0]
        if not (np.isnan(ref_lo) or np.isnan(ref_hi)):
            ax.axhspan(ref_lo, ref_hi, color="green", alpha=0.12,
                       label="IC réf. (P9/P11)")

    # Line
    ax.plot(g["batch_id"], g["value"], "-", color=color, lw=1.5, alpha=0.85)

    # Points colored by severity
    for _, r in a.iterrows():
        ax.scatter(r["batch_id"], r["observed"],
                   c=severity_colors.get(r["severity"], "grey"),
                   s=60, edgecolor="black", linewidth=0.4, zorder=10)

    ax.set_title(f"{metric_labels[m]} — {champion_key}")
    ax.set_xlabel("Batch")
    ax.set_ylabel(metric_labels[m])
    ax.grid(True, alpha=0.3)
    if not a.empty and not (np.isnan(a["ref_lo"].iloc[0])):
        ax.legend(loc="best", fontsize=8)

plt.tight_layout()
st.pyplot(fig)

st.caption(
    "🟢 dans IC · 🟠 sortie d'IC ponctuelle (warn) · 🔴 deux batches "
    "consécutifs dans la même direction (alert)"
)


# ============================================================
# Carte 3 — Rappel Détracteur par segment (priorité fairness Phase 11)
# ============================================================
st.divider()
st.subheader("3. Rappel Détracteur par segment (suivi équité Phase 11)")

seg_drift = drift[
    (drift["champion"] == champion_key)
    & drift["segment"].notna()
    & (drift["metric"] == "detractor_recall")
]
seg_alerts = alerts[
    (alerts["champion"] == champion_key)
    & alerts["segment"].notna()
    & (alerts["metric"] == "detractor_recall")
]

segments_list = ["Senior", "Gender", "Married"]
segment_labels_fr = {"Senior": "Senior", "Gender": "Genre", "Married": "Marié(e)"}

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, seg in zip(axes, segments_list):
    sub_d = seg_drift[seg_drift["segment"] == seg]
    sub_a = seg_alerts[seg_alerts["segment"] == seg]
    if sub_d.empty:
        ax.set_title(f"{segment_labels_fr[seg]} : pas de données")
        continue
    groups = sorted(sub_d["group"].astype(str).unique())
    palette = {g: c for g, c in zip(groups, ["#4477AA", "#EE6677", "#228833"])}

    for grp in groups:
        g_data = sub_d[sub_d["group"].astype(str) == grp].sort_values("batch_id")
        g_alert = sub_a[sub_a["group"].astype(str) == grp].sort_values("batch_id")
        ax.plot(g_data["batch_id"], g_data["value"], "-", lw=1.4,
                color=palette[grp], alpha=0.85, label=grp)
        if not g_alert.empty:
            for _, r in g_alert.iterrows():
                ax.scatter(r["batch_id"], r["observed"],
                           c=severity_colors.get(r["severity"], "grey"),
                           s=40, edgecolor="black", linewidth=0.3, zorder=10)
            # Reference band
            ref_lo = g_alert["ref_lo"].iloc[0]; ref_hi = g_alert["ref_hi"].iloc[0]
            if not (np.isnan(ref_lo) or np.isnan(ref_hi)):
                ax.axhspan(ref_lo, ref_hi, color=palette[grp], alpha=0.06)

    ax.set_xlabel("Batch")
    if seg == segments_list[0]:
        ax.set_ylabel("Rappel Détracteur")
    ax.set_title(segment_labels_fr[seg])
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

plt.tight_layout()
st.pyplot(fig)


# ============================================================
# Carte 4 — tableau des alertes
# ============================================================
st.divider()
st.subheader("4. Alertes déclenchées")

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    sev_filter = st.multiselect(
        "Sévérité", options=["ok", "warn", "alert"], default=["warn", "alert"]
    )
with filter_col2:
    champ_filter = st.multiselect(
        "Champion", options=["C1", "C2"], default=["C1", "C2"]
    )
with filter_col3:
    metric_filter = st.multiselect(
        "Métrique",
        options=sorted(alerts["metric"].unique()),
        default=sorted(alerts["metric"].unique()),
    )

filtered = alerts[
    alerts["severity"].isin(sev_filter)
    & alerts["champion"].isin(champ_filter)
    & alerts["metric"].isin(metric_filter)
].copy()

# Pretty columns
filtered = filtered.rename(columns={
    "champion": "Champion",
    "batch_id": "Batch",
    "metric": "Métrique",
    "segment": "Segment",
    "group": "Groupe",
    "observed": "Observé",
    "ref_lo": "Réf. bas",
    "ref_hi": "Réf. haut",
    "ref_source": "Source réf.",
    "status": "Statut",
    "severity": "Sévérité",
    "batch_size": "N",
})

cols = st.columns(3)
summary = {
    "ok":    int((alerts["severity"] == "ok").sum()),
    "warn":  int((alerts["severity"] == "warn").sum()),
    "alert": int((alerts["severity"] == "alert").sum()),
}
cols[0].metric("✓ OK (dans IC)", summary["ok"])
cols[1].metric("⚠ Warn (sortie ponctuelle)", summary["warn"])
cols[2].metric("🚨 Alert (2 batches consécutifs)", summary["alert"])

st.dataframe(filtered, hide_index=True, use_container_width=True)

# Download
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Télécharger les alertes en CSV",
    csv_bytes,
    file_name="alertes_monitoring.csv",
    mime="text/csv",
)


# ============================================================
# Note de bas de page
# ============================================================
st.divider()
st.markdown(
    """
**Comment lire ce dashboard ?**

* Le **band vert** sur chaque graphique correspond à l'intervalle de confiance
  95 % calculé en Phase 9 (métriques globales) ou Phase 11 (rappels par segment).
* Un point **vert** indique que le batch reste dans la zone acceptable.
* Un point **orange** = sortie d'IC ponctuelle (1 batch) → enquête à programmer.
* Un point **rouge** = sortie d'IC sur 2 batches consécutifs dans la même
  direction → alerte production, escalade au lead data.

**Limites de cette simulation** :

* Le découpage est FIFO par Customer ID, pas par date réelle. Sur les données
  IBM Telco, l'ordre des Customer ID n'a pas de sémantique temporelle — donc
  ce simulateur teste surtout la **stabilité** des KPIs sur sous-échantillons,
  pas un vrai drift temporel.
* En production réelle, le découpage serait par mois calendaire et le drift
  viendrait du marketing, des nouvelles offres, du contexte économique.
* Les IC de référence (Phase 9/11) sont elles-mêmes calculées sur silent_test
  via bootstrap : recalculer trimestriellement pour qu'elles restent à jour.
"""
)
