"""
Page : À Propos du Modèle.

Transparence méthodologique :
    - Vue d'ensemble de la pipeline (Phases 1-11)
    - Caveats (verbatims synthétiques, calibration, équité)
    - Rapport d'évaluation finale (Phase 9) — inline
    - Rapport d'audit d'équité (Phase 11) — inline
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "app"))

import streamlit as st

from components.data_loaders import (
    load_fairness_report,
    load_final_eval_report,
)
from components.styling import synthetic_verbatim_caveat


st.set_page_config(page_title="À Propos · NPS", page_icon="📖", layout="wide")
st.title("📖 À Propos du Modèle")


# ============================================================
# Vue d'ensemble de la pipeline
# ============================================================
st.subheader("Vue d'ensemble de la pipeline")
st.markdown(
    """
| Phase | Étape | Sortie |
|---|---|---|
| 1 | Données brutes IBM Telco (6 fichiers Excel, ~7 000 clients, satisfaction 1-5) | `data/interim/telco_raw.parquet` |
| 2 | Dérivation de la cible NPS depuis le score de satisfaction (mapping alternatif : 1-2 → Détracteur, 3 → Passif, 4-5 → Promoteur) ; audit & suppression de 8 features qui fuient | `data/processed/dataset.parquet` |
| 3 | Split train/val/test biaisé selon la réponse (15 % répondants, 85 % silent — simule le biais de réponse réel) | `data/processed/splits/` |
| 4 | 12 features dérivées (`n_services`, `has_security_bundle`, etc.) + fit `ColumnTransformer` | `models/preprocessing_pipeline.joblib` |
| 5 | Génération de verbatims synthétiques via Qwen2.5-7B-Instruct sur GPU Colab (~30 min) | `data/external/verbatims.parquet` |
| 6 | Baselines : LogReg, LightGBM, Ordinal (mord). Logistic gagne (QWK silent = 0,344) | `models/baselines/*.joblib` |
| 7 | Tuning Optuna TPE, 100 essais. Logistic reste champion (QWK silent = **0,355**) | `models/tuned/*.joblib` |
| 8 | Hybride texte+tabulaire (PCA(32) sur les embeddings + concat). LightGBM/pca32 monte à **QWK 0,659** (caveat ci-dessous) | `models/hybrid/*.joblib` |
| 9 | Évaluation finale avec IC bootstrap sur 6 métriques + calibration + contrôle de covariate shift | `reports/final_eval_report.md` |
| 10 | SHAP sur C1 + analyse des coefficients sur C2 + loadings PCA | `reports/figures/50-53*.png` |
| 11 | Audit d'équité : rappel par groupe × {Senior, Gender, Married} × {Détracteur, Promoteur} + contrefactuel | `reports/fairness_audit.md` |
| 12 | Interface Streamlit + scoring par lot | cette UI |
"""
)


# ============================================================
# Caveats
# ============================================================
st.divider()
st.subheader("Caveats méthodologiques")

synthetic_verbatim_caveat()

st.warning(
    "**Calibration des probabilités (Phase 9).** Les probabilités de C2 sont "
    "bien classées mais pas parfaitement calibrées sur Passif/Promoteur "
    "(Brier 0,29 sur Passif, ECE 0,29). Effet du `class_weight='balanced'` + "
    "régularisation L2 (C=0,027) qui compresse les probabilités vers le centre. "
    "→ Préférer les **rangs** (top-X %) aux probabilités absolues pour l'usage "
    "opérationnel. La Phase 13 devrait recalibrer C2 avec "
    "`CalibratedClassifierCV(method='isotonic', cv=5)`."
)

st.warning(
    "**Équité — Senior × Promoteur sur C2 (Phase 11).** *Disparate Impact* = 0,40. "
    "Analyse contrefactuelle : 9,3 % des prédictions changent quand on inverse "
    "la valeur de Senior. **Le modèle utilise directement l'attribut Senior "
    "et pénalise les seniors pour la prédiction Promoteur.** Ne pas utiliser "
    "C2 dans un workflow de ciblage Promoteur sans ré-entraînement (retirer "
    "Senior des features) ou tuning du seuil par groupe. Voir Équité du modèle."
)


# ============================================================
# Justification du choix de champion
# ============================================================
st.divider()
st.subheader("Pourquoi C2 est le modèle déployé")
st.markdown(
    """
**C1 vs C2 — ils sont statistiquement équivalents sur les métriques business**.

Analyse bootstrap Phase 9 sur silent_test (n=5 987) :

| Métrique | C1 — hybrid/LGBM/pca32 | C2 — tuned/Logistic/tabular | Chevauchement IC → verdict |
|---|---|---|---|
| QWK | 0,659 [0,644 ; 0,677] | 0,355 [0,334 ; 0,375] | disjoint → C1 gagne |
| F1 macro | 0,739 [0,728 ; 0,750] | 0,482 [0,471 ; 0,494] | disjoint → C1 gagne |
| Rappel Détracteur | 0,636 [0,608 ; 0,661] | **0,840 [0,820 ; 0,859]** | disjoint → **C2 gagne** |
| **Lift@10** | 2,85 [2,69 ; 3,02] | 2,76 [2,59 ; 2,92] | **chevauchant → équivalent** |
| **Lift@20** | 2,58 [2,47 ; 2,68] | 2,48 [2,37 ; 2,59] | **chevauchant → équivalent** |

**Sur les métriques qui comptent opérationnellement** (lift@10/20 — capacité du
modèle à prioriser les top-X % de détracteurs), **C1 et C2 ont la même
performance**. C2 gagne là où c'est important (rappel Détracteur — le
retention manager attrape plus de churners), n'a pas de fuite issue du
texte synthétique, et est interprétable via les coefficients linéaires (pas
besoin de SHAP pour le métier).

L'avantage QWK de C1 provient principalement de PC01 (une des 32 composantes
PCA), qui encode le sentiment généré par le LLM dans le verbatim synthétique.
PC01 contribue **3,9× plus que la feature n°2** pour la prédiction Promoteur
(analyse SHAP Phase 10). C'est un artefact de fuite confirmé, pas une valeur
prédictive réelle. Si nous avions de vrais verbatims clients, l'avantage
de C1 sur C2 serait plus faible (et inconnaissable sans la donnée).
"""
)


# ============================================================
# Rapport d'évaluation finale (inline)
# ============================================================
st.divider()
with st.expander("📑 Rapport complet d'évaluation finale (Phase 9)", expanded=False):
    st.markdown(load_final_eval_report())


# ============================================================
# Rapport d'audit d'équité (inline)
# ============================================================
with st.expander("⚖️ Rapport complet d'audit d'équité (Phase 11)", expanded=False):
    st.markdown(load_fairness_report())


# ============================================================
# Roadmap
# ============================================================
st.divider()
st.subheader("Roadmap — la suite")
st.markdown(
    """
**Phase 13 (Monitoring)** — à planifier une fois déployé :

1. **Recalibrer C2** avec `CalibratedClassifierCV(method='isotonic', cv=5)` pour
   que `predict_proba` soit utilisable opérationnellement, et pas seulement
   correct en rang.
2. **Suivre le rappel Détracteur × segment** mensuellement. Alerte si le rappel
   d'un segment dérive de plus de 2σ par rapport à l'IC Phase 11.
3. **Ré-entraîner C2 sans Senior Citizen** comme feature pour mitiger la
   disparité Promoteur. Compromis attendu : −2 à −3 points de rappel
   Détracteur en échange d'un rappel Promoteur équitable sur Senior.
4. **Si de vrais verbatims clients deviennent disponibles** (enquêtes avec
   champ libre, transcriptions d'appels support, avis en ligne) : ré-embedder
   → re-tuner `tune-hybrid` → mesurer honnêtement le Δ QWK sur des vraies
   données, et plus la borne supérieure synthétique.

**Questions méthodologiques ouvertes** documentées pour le rapport :

- Analyse de sensibilité sur le % de verbatims synthétiques « contre-intuitifs »
  (actuel : 15 %). À quoi ressemble le QWK de C1 à 30 % ou 50 % de bruit ?
- Comparaison avec un hybride sans verbatims (embeddings purs extraits d'autres
  champs textuels si disponibles) pour mieux isoler l'effet de la génération LLM.
- Optimisation des seuils par segment via `fairlearn.postprocessing` pour le
  cas Senior × Promoteur.
"""
)
