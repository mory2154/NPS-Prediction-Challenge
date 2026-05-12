"""
Page : Scorer Client.

Recherche d'un client dans silent_test → classe prédite avec explication
(diagramme en cascade des contributions des features, SHAP pour C1 ou
coefficient × valeur pour C2).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "app"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from components.data_loaders import load_predictions
from components.plots import (
    explanation_waterfall,
    probability_gauge,
    rank_in_population,
)
from components.styling import (
    calibration_note,
    champion_selector,
)
from src.config import DEFAULT_TARGET, MODELS_DIR

# Traductions des classes pour affichage
CLASS_FR = {"Detractor": "Détracteur", "Passive": "Passif", "Promoter": "Promoteur"}


st.set_page_config(page_title="Scorer Client · NPS", page_icon="🎯", layout="wide")
st.title("🎯 Scorer un Client")


# ============================================================
# Sidebar — sélection du modèle
# ============================================================
selected = champion_selector()


# ============================================================
# Chargement des prédictions
# ============================================================
try:
    pred = load_predictions()
except FileNotFoundError:
    st.error("Prédictions non pré-calculées. Exécuter `make batch-score` depuis la racine du projet.")
    st.stop()


# ============================================================
# Recherche client
# ============================================================
st.subheader("Trouver un client")
all_ids = pred["customer_id"].tolist()

col_search, col_random = st.columns([3, 1])
with col_search:
    cust_id = st.selectbox(
        "ID Client",
        options=[""] + all_ids,
        index=0,
        help="Tapez pour filtrer.",
    )
with col_random:
    st.write("")  # spacer
    st.write("")
    if st.button("🎲 Client aléatoire"):
        st.session_state["random_cust"] = np.random.choice(all_ids)
        st.rerun()

if "random_cust" in st.session_state and not cust_id:
    cust_id = st.session_state["random_cust"]
    st.info(f"Tiré au sort : **{cust_id}**")

if not cust_id:
    st.info("Sélectionnez un ID client pour voir sa prédiction.")
    st.stop()


# ============================================================
# Bloc client
# ============================================================
row = pred[pred["customer_id"] == cust_id].iloc[0]
proba_cols = [f"proba_{selected}_{c.lower()}" for c in ["detractor", "passive", "promoter"]]
p_det, p_pas, p_pro = [float(row[c]) for c in proba_cols]
pred_class = row[f"pred_{selected}"]
pred_class_fr = CLASS_FR.get(pred_class, pred_class)
true_class = row["nps_true"]
true_class_fr = CLASS_FR.get(true_class, true_class) if pd.notna(true_class) else true_class

st.divider()
left, right = st.columns([1, 1])

with left:
    st.subheader("Prédiction")
    color_map = {"Detractor": "🔴", "Passive": "🟠", "Promoter": "🟢"}
    icon = color_map.get(pred_class, "❓")
    st.markdown(f"### {icon} **{pred_class_fr}**  (modèle {selected})")

    fig = probability_gauge(p_det, p_pas, p_pro, title=f"Probabilités {selected}")
    st.pyplot(fig, use_container_width=True)
    calibration_note()

    # Position dans la population (Détracteur + Promoteur)
    rank_det = int(row[f"rank_{selected}_detractor"])
    rank_pro = int(row[f"rank_{selected}_promoter"])
    st.pyplot(rank_in_population(rank_det, len(pred), "Detractor"),
              use_container_width=True)
    st.pyplot(rank_in_population(rank_pro, len(pred), "Promoter"),
              use_container_width=True)

with right:
    st.subheader("Profil client")
    # Affichage des colonnes les plus pertinentes
    display_cols = [
        "Tenure Months", "Monthly Charges", "Contract", "Internet Type",
        "Payment Method", "Number of Referrals", "Number of Dependents",
        "Senior Citizen", "Gender", "Married", "Age",
    ]
    profile = {c: row[c] for c in display_cols if c in pred.columns}
    profile_df = pd.DataFrame(
        {"Feature": list(profile.keys()), "Valeur": list(profile.values())}
    )
    st.dataframe(profile_df, hide_index=True, use_container_width=True)

    # Validation contre la vraie classe (disponible car dataset entièrement labellisé)
    if pd.notna(true_class) and true_class != "nan":
        match = "✓ correct" if pred_class == true_class else "✗ incorrect"
        st.caption(
            f"Vraie classe (validation uniquement — non utilisée pour ré-entraîner) : "
            f"**{true_class_fr}** {match}"
        )

    # Note d'accord C1 vs C2
    other = "C1" if selected == "C2" else "C2"
    other_pred = row[f"pred_{other}"]
    other_pred_fr = CLASS_FR.get(other_pred, other_pred)
    if other_pred != pred_class:
        st.warning(
            f"⚠ Désaccord entre modèles : **{selected}** prédit **{pred_class_fr}**, "
            f"**{other}** prédit **{other_pred_fr}**. Inspectez l'explication ci-dessous."
        )


# ============================================================
# Explication (par client)
# ============================================================
st.divider()
st.subheader("Pourquoi cette prédiction ?")

st.caption(
    f"Contributions des features à la classe prédite ({pred_class_fr}). "
    f"{'TreeSHAP sur LightGBM' if selected == 'C1' else 'Coefficient × valeur sur Régression Logistique'}."
)

# Pour C2 (linéaire), c'est immédiat : contribution_j = coef[class, j] * x[j].
# Pour C1 (LightGBM hybride), on appelle shap.TreeExplainer sur la seule ligne.
try:
    from src.data.split import load_splits
    from src.features.embeddings import load_or_compute_embeddings
    from src.models.tuning_hybrid import VERBATIM_AUX_COLS, _build_split_matrices
    from src.config import DATA_PROCESSED, NPS_CLASS_TO_INT

    df = pd.read_parquet(DATA_PROCESSED / "dataset_with_verbatims.parquet")
    splits = load_splits("response_biased")
    pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")

    if cust_id not in df.index:
        st.error(f"Client {cust_id} introuvable dans le jeu de données.")
        st.stop()

    feature_cols = [
        c for c in df.columns
        if not c.startswith("NPS_") and c not in VERBATIM_AUX_COLS
    ]

    if selected == "C2":
        # Modèle linéaire — rapide, pas besoin de SHAP
        model = joblib.load(MODELS_DIR / "tuned" / "logistic_tuned.joblib")
        feature_names = list(pipeline.get_feature_names_out())
        x = pipeline.transform(df.loc[[cust_id], feature_cols])[0]
        class_idx = NPS_CLASS_TO_INT[pred_class]
        coef = model.coef_[class_idx]
        contrib_df = pd.DataFrame({
            "feature":        feature_names,
            "feature_value":  x,
            "contribution":   coef * x,
        })
    else:
        # C1 — SHAP nécessaire, mais sur 1 ligne → instantané
        import shap
        model = joblib.load(MODELS_DIR / "hybrid" / "lightgbm_pca32.joblib")
        embeddings_df = load_or_compute_embeddings(df, verbose=False)
        common = df.index.intersection(embeddings_df.index)
        df_aligned = df.loc[common]
        splits_aligned = splits.loc[common]
        embeddings_aligned = embeddings_df.loc[common]
        X_per, _, info = _build_split_matrices(
            df=df_aligned, splits=splits_aligned, pipeline=pipeline,
            embeddings_df=embeddings_aligned,
            target_col=DEFAULT_TARGET, feature_cols=feature_cols,
            feature_space="pca32", verbose=False,
        )
        cust_pos = df_aligned.index.get_loc(cust_id)
        which_split = splits_aligned.loc[cust_id]
        # Position du client DANS son split (pas dans le df complet)
        split_idx = df_aligned.index[splits_aligned == which_split]
        local_pos = split_idx.get_loc(cust_id)
        X_row = X_per[which_split][local_pos : local_pos + 1]
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_row)
        if isinstance(raw, list):
            shap_arr = np.array(raw)  # shape (3, 1, p)
            sv = shap_arr[NPS_CLASS_TO_INT[pred_class]][0]
        else:
            sv = raw[0, :, NPS_CLASS_TO_INT[pred_class]]
        # Feature names : tab_names + PCnn
        try:
            tab_names = list(pipeline.get_feature_names_out())
        except Exception:
            tab_names = [f"tab_{i:03d}" for i in range(X_row.shape[1] - 32)]
        feature_names = tab_names + [f"PC{i:02d}" for i in range(32)]
        contrib_df = pd.DataFrame({
            "feature":        feature_names,
            "feature_value":  X_row[0],
            "contribution":   sv,
        })

    top_k = st.slider("Nombre de contributions à afficher ?", 5, 20, 8, 1)
    fig = explanation_waterfall(
        contrib_df, top_k=top_k,
        title=f"Top-{top_k} contributions pour '{pred_class_fr}' (modèle {selected})",
    )
    st.pyplot(fig, use_container_width=False)

    with st.expander("Toutes les contributions (tableau brut)"):
        contrib_df["abs"] = contrib_df["contribution"].abs()
        st.dataframe(
            contrib_df.sort_values("abs", ascending=False).drop(columns="abs"),
            use_container_width=True,
        )

except Exception as e:
    st.error(f"Impossible de calculer l'explication : {e}")
    st.exception(e)
