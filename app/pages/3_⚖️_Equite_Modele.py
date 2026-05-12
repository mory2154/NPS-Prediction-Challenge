"""
Page : Équité du Modèle.

Surface visuelle pour les résultats de l'audit Phase 11 : rappel par groupe
avec IC, heatmap *Disparate Impact*, analyse contrefactuelle.
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
import seaborn as sns
import streamlit as st

from components.data_loaders import (
    load_fairness_counterfactual,
    load_fairness_disparities,
    load_fairness_per_group,
)
from components.plots import per_group_recall_bars
from components.styling import fairness_badge


st.set_page_config(page_title="Équité · NPS", page_icon="⚖️", layout="wide")
st.title("⚖️ Tableau de bord — Équité du modèle")

st.markdown(
    "**Source : audit Phase 11.** Les deux champions ont été audités sur 3 "
    "segments protégés {Senior, Genre, Marié(e)} × 2 classes cibles "
    "{Détracteur, Promoteur}. Bootstrap IC 95 %, n=1000."
)


# ============================================================
# Chargement
# ============================================================
try:
    per_group = load_fairness_per_group()
    disparity = load_fairness_disparities()
    cf = load_fairness_counterfactual()
except FileNotFoundError as e:
    st.error(f"Artefacts d'équité manquants : {e}. Exécuter `make fairness` depuis la racine du projet.")
    st.stop()


# ============================================================
# Verdict global (nombre de disparités)
# ============================================================
disparity = disparity.copy()
disparity["verdict"] = disparity.apply(
    lambda r: "fair" if (
        0.8 <= r["DI_worst"] <= 1.25 and abs(r["EOD"]) < 0.10
    ) else "disparity",
    axis=1,
)
n_total = len(disparity)
n_disp = (disparity["verdict"] == "disparity").sum()

cols = st.columns(3)
cols[0].metric("Cellules auditées", f"{n_total}", help="2 champions × 3 segments × 2 classes")
cols[1].metric("Cellules équitables", f"{n_total - n_disp}",
               delta=f"{(n_total - n_disp) / n_total:.0%}")
cols[2].metric("⚠ Cellules en disparité", f"{n_disp}",
               delta=f"{n_disp / n_total:.0%}", delta_color="inverse")


# ============================================================
# Tableau de disparité
# ============================================================
st.divider()
st.subheader("Indicateurs de disparité")

display_table = disparity.copy()
display_table["DI worst"] = display_table["DI_worst"].apply(lambda x: f"{x:.3f}")
display_table["EOD"]      = display_table["EOD"].apply(lambda x: f"{x:+.3f}")
display_table["DPD"]      = display_table["DPD"].apply(lambda x: f"{x:+.3f}")
display_table["verdict"]  = display_table["verdict"].map(
    {"fair": "✓ Équitable", "disparity": "⚠ Disparité"}
)
# Traduction des noms de classes
display_table["class"] = display_table["class"].map(
    {"Detractor": "Détracteur", "Passive": "Passif", "Promoter": "Promoteur"}
)
display_table = display_table[
    ["champion", "segment", "class", "DI worst", "DI_worst_group", "DI_reference_group",
     "EOD", "DPD", "verdict"]
].rename(columns={
    "champion": "Champion",
    "segment": "Segment",
    "class": "Classe",
    "DI worst": "DI pire",
    "DI_worst_group": "Groupe pire",
    "DI_reference_group": "Groupe réf.",
})

st.dataframe(display_table, hide_index=True, use_container_width=True)

with st.expander("Comment lire ces indicateurs ?"):
    st.markdown(
        "* **DI (*Disparate Impact*)** : ratio du rappel par rapport au groupe le plus favorisé. "
        "Équitable si DI ∈ [0,8 ; 1,25] (règle 4/5 de l'EEOC). "
        "DI=0,40 signifie que le groupe défavorisé est *capté 60 % moins* que le groupe favorisé.\n\n"
        "* **EOD (*Equal Opportunity Difference*)** : max(rappel) − min(rappel) entre les groupes. "
        "Considéré acceptable si |EOD| < 0,10.\n\n"
        "* **DPD (*Demographic Parity Difference*)** : différence des *taux de prédiction*. "
        "Information uniquement — une différenciation business légitime peut produire DPD ≠ 0."
    )


# ============================================================
# Comparaison rappel par segment
# ============================================================
st.divider()
st.subheader("Rappel par segment (avec IC bootstrap 95 %)")

champion_choice = st.radio(
    "Champion à inspecter",
    options=["C2 (Production)", "C1 (Champion QWK)"],
    horizontal=True,
)
champion_key = "C2_safe" if champion_choice.startswith("C2") else "C1_qwk"
color = "#d62728" if champion_key == "C2_safe" else "#1f77b4"

segments = ["Senior", "Gender", "Married"]
segment_labels_fr = {"Senior": "Senior", "Gender": "Genre", "Married": "Marié(e)"}
classes = ["Detractor", "Promoter"]
class_labels_fr = {"Detractor": "Détracteur", "Promoter": "Promoteur"}

for target_class in classes:
    st.markdown(f"#### Rappel **{class_labels_fr[target_class]}**")
    cols_seg = st.columns(len(segments))
    for col, seg in zip(cols_seg, segments):
        with col:
            fig = per_group_recall_bars(
                per_group, segment=seg, target_class=target_class,
                champion=champion_key, champion_color=color,
            )
            if fig is None:
                col.info(f"{segment_labels_fr[seg]} : pas de données")
                continue
            col.pyplot(fig, use_container_width=True)

            # Verdict inline
            row = disparity[
                (disparity["champion"] == champion_key)
                & (disparity["segment"] == seg)
                & (disparity["class"] == target_class)
            ]
            if not row.empty:
                r = row.iloc[0]
                badge = fairness_badge(r["DI_worst"], r["EOD"])
                col.caption(badge)


# ============================================================
# Heatmap DI
# ============================================================
st.divider()
st.subheader("Heatmap du *Disparate Impact*")
st.caption(
    "Chaque cellule : DI du groupe le plus défavorisé. Vert = zone équitable [0,8 ; 1,25]. "
    "L'annotation donne le DI."
)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)

for ax, ck, label in zip(axes,
                         ["C2_safe", "C1_qwk"],
                         ["C2 — Production", "C1 — Champion QWK"]):
    sub = disparity[disparity["champion"] == ck]
    heat = sub.pivot_table(
        index="class", columns="segment", values="DI_worst", aggfunc="first",
    ).reindex(index=["Detractor", "Promoter"], columns=["Senior", "Gender", "Married"])
    # Traduire les labels
    heat.index = [class_labels_fr.get(c, c) for c in heat.index]
    heat.columns = [segment_labels_fr.get(s, s) for s in heat.columns]
    sns.heatmap(heat, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=0.5, vmax=1.5, center=1.0,
                cbar_kws={"label": "DI (pire)"},
                ax=ax, linewidth=0.5)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("")

plt.tight_layout()
st.pyplot(fig)


# ============================================================
# Analyse contrefactuelle
# ============================================================
st.divider()
st.subheader("Analyse contrefactuelle (échange de l'attribut)")
st.markdown(
    "On échange l'attribut protégé (ex : `Senior=Oui ↔ Non`) sur chaque "
    "client de silent_test, puis on re-prédit. Le **taux de changement** "
    "quantifie à quel point le modèle utilise *directement* l'attribut. "
    "Un taux < 2 % signifie que la disparité (si elle existe) passe par des "
    "*variables proxy* ; un taux plus élevé signifie que le modèle utilise "
    "lui-même l'attribut."
)

cf_display = cf.copy()
cf_display["change_rate"] = cf_display["change_rate"].apply(lambda x: f"{x:.1%}")
cf_display = cf_display[[
    "champion_label", "segment_label", "n", "n_changed", "change_rate",
    "detractor_to_other", "other_to_detractor",
    "promoter_to_other", "other_to_promoter",
]].rename(columns={
    "champion_label":     "Champion",
    "segment_label":      "Segment",
    "n":                  "N",
    "n_changed":          "Modifiés",
    "change_rate":        "Δ taux",
    "detractor_to_other": "Dét → Autre",
    "other_to_detractor": "Autre → Dét",
    "promoter_to_other":  "Pro → Autre",
    "other_to_promoter":  "Autre → Pro",
})
st.dataframe(cf_display, hide_index=True, use_container_width=True)


# ============================================================
# Conclusions clés
# ============================================================
st.divider()
st.subheader("Conclusions clés pour le déploiement")

st.success(
    "✅ **C2 est équitable sur le rappel Détracteur pour les 3 segments**. "
    "DI ∈ [0,79 ; 0,92] et |EOD| < 0,10. Le retention manager peut utiliser "
    "C2 en confiance pour sa mission principale (détecter les churners)."
)

st.error(
    "🚨 **C2 présente une forte disparité de rappel Promoteur sur Senior** "
    "(DI = 0,40, rappel 0,23 pour les seniors vs 0,58 pour les non-seniors). "
    "Et le taux de changement contrefactuel de 9,3 % montre que c'est "
    "**causalement dû** à l'attribut Senior lui-même, pas à des proxies. "
    "**Ne pas utiliser C2 pour cibler les promoteurs sans tenir compte des segments.** "
    "Soit retirer Senior des features et ré-entraîner, soit utiliser un seuil "
    "par segment (Phase 13)."
)

st.warning(
    "⚠ **C1 présente une disparité Marié(e) × Détracteur** (DI = 0,72) mais "
    "seulement 0,2 % de taux de changement contrefactuel — donc cette "
    "disparité passe par des proxies (Nombre de parrainages, Nombre de "
    "personnes à charge). Moins actionnable, mais à surveiller (Phase 13)."
)
