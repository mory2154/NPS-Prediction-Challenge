"""
NPS Prediction — Interface Streamlit (Phase 12).

Page d'accueil = tableau de bord global :
    - Métriques clés sur silent_test pour les deux champions (avec IC)
    - Disponibilité des artefacts
    - 3 liens rapides vers les pages d'action

Navigation multi-pages gérée nativement par Streamlit via `pages/`.
"""

# ============================================================
# Path bootstrap — make `src/` importable when launched by Streamlit
# ============================================================
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "app"))
# ============================================================

import streamlit as st

from components.data_loaders import (
    check_artifacts_available,
    load_final_eval_summary,
    load_predictions,
)
from src.config import DEFAULT_TARGET, NPS_CLASSES


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Prédiction NPS — Rétention Telco",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# En-tête
# ============================================================
st.title("Prédiction NPS — Rétention Telco")
st.markdown(
    "**Prédire la classe NPS (Détracteur / Passif / Promoteur) pour les 85 % "
    "de clients qui n'ont pas répondu à l'enquête NPS, afin que l'équipe "
    "rétention puisse prioriser ses appels.**"
)
st.caption(f"Cible : `{DEFAULT_TARGET}` · Interface Phase 12")


# ============================================================
# Disponibilité des artefacts
# ============================================================
st.subheader("État du système")
availability = check_artifacts_available()
missing = [name for name, ok in availability.items() if not ok]
cols = st.columns(3)
for i, (name, ok) in enumerate(availability.items()):
    col = cols[i % 3]
    if ok:
        col.success(f"✓ {name}")
    else:
        col.error(f"✗ {name} manquant")
if missing:
    st.warning(
        "Certains artefacts sont manquants — certaines pages auront des "
        "informations partielles. Depuis la racine du projet, exécuter :\n"
        "```\n"
        "make tune                  # Phase 7 → modèle C2\n"
        "make tune-hybrid           # Phase 8 → modèle C1\n"
        "make final-eval            # Phase 9 → intervalles de confiance\n"
        "make interpret             # Phase 10 → SHAP + coefficients\n"
        "make fairness              # Phase 11 → audit par segment\n"
        "make batch-score           # Phase 12 → pré-calcul des prédictions\n"
        "```"
    )
    st.stop()


# ============================================================
# Chiffres principaux (évaluation finale Phase 9)
# ============================================================
st.subheader("Performance des champions sur `silent_test` (Phase 9)")
st.caption(
    "Échantillon retenu, jamais utilisé pour le tuning. Bootstrap IC 95 % (n=1000). "
    "Les chiffres ci-dessous estiment la **performance en production**."
)

try:
    final = load_final_eval_summary()
    silent = final[final["split"] == "silent_test"]

    def _fmt(v, lo, hi):
        return f"{v:.3f}  [{lo:.3f}, {hi:.3f}]"

    def _row(champion_key):
        sub = silent[silent["champion"] == champion_key]
        row = {r["metric"]: r for _, r in sub.iterrows()}
        return {
            "QWK":                _fmt(row["qwk"]["value"], row["qwk"]["ci_lo"], row["qwk"]["ci_hi"]),
            "Rappel Détracteur":  _fmt(row["detractor_recall"]["value"], row["detractor_recall"]["ci_lo"], row["detractor_recall"]["ci_hi"]),
            "F1 macro":           _fmt(row["macro_f1"]["value"], row["macro_f1"]["ci_lo"], row["macro_f1"]["ci_hi"]),
            "Lift@10":             _fmt(row["lift@10"]["value"], row["lift@10"]["ci_lo"], row["lift@10"]["ci_hi"]),
            "Lift@20":             _fmt(row["lift@20"]["value"], row["lift@20"]["ci_lo"], row["lift@20"]["ci_hi"]),
        }

    import pandas as pd
    headline = pd.DataFrame({
        "C2 — Production (par défaut)":  _row("C2_safe"),
        "C1 — Champion QWK (expérimental)": _row("C1_qwk"),
    }).T
    st.dataframe(headline, use_container_width=True)

    st.info(
        "**Pourquoi C2 est le modèle déployé** (recommandation Phase 9) : "
        "(a) Lift@10 et Lift@20 sont *statistiquement équivalents* à ceux de C1 "
        "(intervalles de confiance se chevauchant) — même valeur business pour le ciblage par cohorte ; "
        "(b) Le rappel Détracteur est significativement plus élevé (0,84 vs 0,64) — "
        "C2 attrape plus de détracteurs, ce qui est l'objectif du *retention manager* ; "
        "(c) Aucun risque de fuite via les verbatims synthétiques. "
        "L'avantage QWK de C1 est une borne supérieure issue du texte généré par LLM, "
        "et non un gain réel en production."
    )
except FileNotFoundError:
    st.warning("Résumé d'évaluation finale introuvable. Exécuter `make final-eval`.")
except Exception as e:
    st.error(f"Impossible de charger l'évaluation finale : {e}")


# ============================================================
# Statistiques sur la population silent
# ============================================================
try:
    pred = load_predictions()
    st.subheader("Population silent à scorer")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients silent", f"{len(pred):,}")
    col2.metric(
        "Détracteurs prédits (C2)",
        f"{(pred['pred_C2'] == 'Detractor').sum():,}",
        f"{(pred['pred_C2'] == 'Detractor').mean():.1%}",
    )
    col3.metric(
        "Promoteurs prédits (C2)",
        f"{(pred['pred_C2'] == 'Promoter').sum():,}",
        f"{(pred['pred_C2'] == 'Promoter').mean():.1%}",
    )
    col4.metric(
        "Accord C1 ↔ C2",
        f"{pred['agreement'].mean():.1%}",
        help="Part des clients pour lesquels les deux champions prédisent la même classe.",
    )
except FileNotFoundError:
    st.warning("Prédictions non pré-calculées. Exécuter `make batch-score`.")


# ============================================================
# Navigation rapide
# ============================================================
st.divider()
st.subheader("Où voulez-vous aller ?")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        "### 🎯 Scorer Client\n"
        "Rechercher un ID client, voir sa classe prédite avec une "
        "explication feature par feature (diagramme en cascade)."
    )
with col_b:
    st.markdown(
        "### 📊 Ciblage Cohorte\n"
        "Générer la liste des top-N clients à appeler aujourd'hui, "
        "filtrable par segment, mode de paiement, type de contrat, etc."
    )
with col_c:
    st.markdown(
        "### ⚖️ Équité du modèle\n"
        "Audit Phase 11 — rappel par segment avec IC bootstrap, "
        "*Disparate Impact*, analyse contrefactuelle. **À lire avant "
        "tout ciblage des Promoteurs.**"
    )

st.caption("Utilisez la barre latérale pour naviguer.")
