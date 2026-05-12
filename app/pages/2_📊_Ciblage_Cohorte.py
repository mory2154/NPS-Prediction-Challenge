"""
Page : Ciblage Cohorte.

Construit la liste top-N de clients à appeler aujourd'hui. Filtrable par :
    - segment (Senior/Gender/Married)
    - type de contrat
    - type d'internet
    - plage tenure, plage charges mensuelles
    - classe cible (Détracteur pour rétention ; Promoteur pour parrainage — avec warning)

Output : table exportable CSV avec customer_id + features clés + probas + rang.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "app"))

import pandas as pd
import streamlit as st

from components.data_loaders import load_predictions
from components.styling import (
    calibration_note,
    champion_selector,
    promoter_use_warning,
    top_k_note,
)


st.set_page_config(page_title="Ciblage Cohorte · NPS", page_icon="📊", layout="wide")
st.title("📊 Ciblage de Cohorte")

st.markdown(
    "**Objectif** : produire une *liste d'appels* pour l'équipe rétention. "
    "Filtrer la population silent et classer par probabilité prédite de la classe cible."
)


# ============================================================
# Sidebar
# ============================================================
selected = champion_selector()

st.sidebar.divider()
st.sidebar.subheader("Classe cible")
target_class = st.sidebar.radio(
    "Clients à trouver :",
    options=["Détracteur (priorité rétention)", "Promoteur (parrainage / advocacy)"],
    index=0,
)
target_class_clean = "Detractor" if target_class.startswith("Détracteur") else "Promoter"
target_class_fr = "Détracteur" if target_class_clean == "Detractor" else "Promoteur"

if target_class_clean == "Promoter" and selected == "C2":
    promoter_use_warning()


# ============================================================
# Chargement des prédictions
# ============================================================
try:
    pred = load_predictions()
except FileNotFoundError:
    st.error("Prédictions non pré-calculées. Exécuter `make batch-score`.")
    st.stop()


# ============================================================
# Filtres
# ============================================================
st.subheader("Filtres")

filter_cols = st.columns(3)

with filter_cols[0]:
    st.markdown("**Attributs protégés**")
    seg_senior = st.multiselect(
        "Senior Citizen",
        options=sorted(pred["Senior Citizen"].dropna().astype(str).unique()) if "Senior Citizen" in pred.columns else [],
        default=[],
    )
    seg_gender = st.multiselect(
        "Genre",
        options=sorted(pred["Gender"].dropna().astype(str).unique()) if "Gender" in pred.columns else [],
        default=[],
    )
    seg_married = st.multiselect(
        "Marié(e)",
        options=sorted(pred["Married"].dropna().astype(str).unique()) if "Married" in pred.columns else [],
        default=[],
    )

with filter_cols[1]:
    st.markdown("**Contrat / facturation**")
    contract = st.multiselect(
        "Contrat",
        options=sorted(pred["Contract"].dropna().unique()) if "Contract" in pred.columns else [],
        default=[],
    )
    internet = st.multiselect(
        "Type d'internet",
        options=sorted(pred["Internet Type"].dropna().unique()) if "Internet Type" in pred.columns else [],
        default=[],
    )
    payment = st.multiselect(
        "Mode de paiement",
        options=sorted(pred["Payment Method"].dropna().unique()) if "Payment Method" in pred.columns else [],
        default=[],
    )

with filter_cols[2]:
    st.markdown("**Plages numériques**")
    if "Tenure Months" in pred.columns:
        tmin, tmax = float(pred["Tenure Months"].min()), float(pred["Tenure Months"].max())
        tenure_range = st.slider("Ancienneté (mois)", tmin, tmax, (tmin, tmax))
    else:
        tenure_range = None

    if "Monthly Charges" in pred.columns:
        cmin, cmax = float(pred["Monthly Charges"].min()), float(pred["Monthly Charges"].max())
        charges_range = st.slider("Charges mensuelles", cmin, cmax, (cmin, cmax))
    else:
        charges_range = None


# ============================================================
# Application des filtres
# ============================================================
filtered = pred.copy()
if seg_senior and "Senior Citizen" in filtered.columns:
    filtered = filtered[filtered["Senior Citizen"].astype(str).isin(seg_senior)]
if seg_gender and "Gender" in filtered.columns:
    filtered = filtered[filtered["Gender"].astype(str).isin(seg_gender)]
if seg_married and "Married" in filtered.columns:
    filtered = filtered[filtered["Married"].astype(str).isin(seg_married)]
if contract and "Contract" in filtered.columns:
    filtered = filtered[filtered["Contract"].isin(contract)]
if internet and "Internet Type" in filtered.columns:
    filtered = filtered[filtered["Internet Type"].isin(internet)]
if payment and "Payment Method" in filtered.columns:
    filtered = filtered[filtered["Payment Method"].isin(payment)]
if tenure_range and "Tenure Months" in filtered.columns:
    filtered = filtered[filtered["Tenure Months"].between(*tenure_range)]
if charges_range and "Monthly Charges" in filtered.columns:
    filtered = filtered[filtered["Monthly Charges"].between(*charges_range)]


# ============================================================
# Curseur Top-K
# ============================================================
st.divider()
st.subheader(f"Top clients par P({target_class_fr}) prédite")

max_k = min(500, len(filtered))
if max_k == 0:
    st.warning("Aucun client ne correspond aux filtres.")
    st.stop()
k = st.slider("Top K", min_value=5, max_value=max(5, max_k), value=min(50, max_k), step=5)

# Tri
proba_col = f"proba_{selected}_{target_class_clean.lower()}"
rank_col = f"rank_{selected}_{target_class_clean.lower()}"
top_df = filtered.nlargest(k, proba_col)

top_k_note(k, len(filtered))

# Colonnes à afficher
show_cols = [
    "customer_id",
    proba_col,
    rank_col,
    f"pred_{selected}",
    "nps_true",
]
extra = ["Tenure Months", "Monthly Charges", "Contract", "Internet Type",
         "Number of Referrals", "Senior Citizen", "Gender", "Married"]
show_cols += [c for c in extra if c in top_df.columns]

display = top_df[show_cols].copy()
display = display.rename(columns={
    proba_col: f"P({target_class_fr})",
    rank_col: "rang",
    f"pred_{selected}": "prédiction",
    "nps_true": "vraie_classe",
    "customer_id": "id_client",
    "Tenure Months": "Ancienneté (mois)",
    "Monthly Charges": "Charges mensuelles",
    "Contract": "Contrat",
    "Internet Type": "Type internet",
    "Number of Referrals": "Nb parrainages",
})
st.dataframe(display, use_container_width=True, hide_index=True)
calibration_note()


# ============================================================
# Statistiques de la cohorte
# ============================================================
st.divider()
st.subheader("Synthèse de la cohorte")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Clients dans la cohorte", f"{k:,}")
col_b.metric(f"P({target_class_fr}) moyenne", f"{top_df[proba_col].mean():.2%}")
col_c.metric(
    f"% prédits {target_class_fr}",
    f"{(top_df[f'pred_{selected}'] == target_class_clean).mean():.1%}",
)
if "nps_true" in top_df.columns:
    col_d.metric(
        f"% réels {target_class_fr} (vérité terrain, pour monitoring)",
        f"{(top_df['nps_true'] == target_class_clean).mean():.1%}",
        help="Disponible parce que le dataset IBM Telco est entièrement labellisé. En production réelle, "
             "cette information ne serait PAS visible au moment du scoring.",
    )


# ============================================================
# Téléchargement
# ============================================================
csv_bytes = display.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Télécharger la cohorte en CSV",
    csv_bytes,
    file_name=f"cohorte_{selected}_{target_class_clean.lower()}_top{k}.csv",
    mime="text/csv",
)
