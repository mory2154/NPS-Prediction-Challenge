# Phase 13 — Spec monitoring lightweight (v3)

> Spec de référence pour passer du simulateur Phase 13 à une vraie infrastructure de monitoring en production. Cible Section 4.9 du brief Artefact ("optionnelle").

## Objectif business

Détecter dans les 30 jours :
- une dérive de performance du modèle déployé (C2)
- une régression d'équité sur l'un des 3 segments protégés (Senior, Genre, Marié)
- une dérive de distribution des features d'entrée (covariate shift réel)

Permettre une **escalade automatisée** vers le lead data quand un seuil est franchi.

---

## ⚠️ Section critique — Comment utiliser le modèle recalibré en production

Phase 13.1 a démontré que la recalibration isotonic de C2 :
- ✅ **Réduit massivement l'ECE** (Détracteur 0.143 → 0.019, Passif 0.292 → 0.031, Promoteur 0.148 → 0.010) — les probas deviennent vraiment interprétables comme « 73 % de chance d'être détracteur ».
- ❌ **Casse l'argmax** si on l'utilise tel quel pour `predict()` (Detractor recall chute de 0.84 à 0.30).

**Cause** : C2 a été entraîné avec `class_weight='balanced'` (Phase 7). Cela gonfle artificiellement `predict_proba(Detractor)` pour pousser le modèle à attraper plus de détracteurs. L'isotonic recalibration corrige *correctement* ce gonflement, ce qui ramène les probas à leur vraie valeur postérieure (où Passif domine à 58 % dans la marginale réelle). Du coup, `argmax(predict_proba_calibré)` favorise Passif et rate les détracteurs.

### Mode HYBRIDE = production-recommended

L'audit 3-modes de `recalibrate.py` produit 3 colonnes dans `calibration_headline.parquet` :

| Mode | predict_proba | predict | ECE | Detractor recall | Usage |
|---|---|---|---|---|---|
| `before` | original | original | 0.143 | 0.84 | Phase 12 (en place actuellement) |
| `after_full` | calibré | calibré (argmax) | 0.019 | **0.30 ⚠** | À **éviter** en prod |
| **`after_hybrid`** | **calibré** | **original (argmax)** | **0.019** | **0.84** | **À déployer** |

→ En production : utiliser le **mode hybride**. Calibrated proba pour l'affichage et le lift, original predict pour la décision binaire.

### Implémentation pratique en production

```python
# Chargement
orig_C2 = joblib.load("logistic_tuned.joblib")
cal_C2  = joblib.load("logistic_C2_calibrated.joblib")

# Au scoring
proba_displayed = cal_C2.predict_proba(X)      # pour l'app Streamlit (proba %)
ranks           = (-proba_displayed[:, 0]).argsort()  # pour top-K targeting
pred_decision   = orig_C2.predict(X)            # pour "appeler / pas appeler"
```

Le `drift_simulator --use-calibrated` (Phase 13.2) applique exactement ce pattern.

---

## KPIs à tracker (fréquence : mensuel)

| KPI | Source de la proba | Source de la prédiction | Seuil warn (1 batch) | Seuil alert (2 batches consécutifs) |
|---|---|---|---|---|
| **QWK global** | n/a | argmax(original) | sortie d'IC P9 [0,33 ; 0,38] | 2 batches dans le même sens |
| **Rappel Détracteur global** | n/a | argmax(original) | sortie d'IC P9 [0,82 ; 0,86] | idem |
| **Rappel Détracteur × Senior=Yes** | n/a | argmax(original) | sortie d'IC P11 [0,88 ; 0,94] | idem |
| **Rappel Détracteur × Senior=No** | n/a | argmax(original) | sortie d'IC P11 [0,79 ; 0,84] | idem |
| **Rappel Détracteur × Gender** | n/a | argmax(original) | idem | idem |
| **Rappel Détracteur × Married** | n/a | argmax(original) | idem | idem |
| **Lift@10** | calibrée | n/a (ranking) | sortie d'IC P9 [2,59 ; 2,92] | idem |
| **Lift@20** | calibrée | n/a (ranking) | sortie d'IC P9 [2,37 ; 2,59] | idem |
| **ECE Détracteur** (calibration) | calibrée | n/a | > 0,08 | > 0,10 deux mois de suite |
| **% prédits Détracteur** (selection rate) | n/a | argmax(original) | hors [16 % ; 26 %] | hors [13 % ; 30 %] |

Le **band de référence** vient des intervalles de confiance bootstrap calculés en Phase 9 (KPIs globaux) et Phase 11 (KPIs par segment). On recalcule ces IC trimestriellement avec les données fraîches.

---

## Architecture minimale en production

```
                            ┌───────────────────────────────┐
                            │ Production scoring (daily)    │
                            │ Use HYBRID pattern:           │
                            │   proba   ← cal_C2            │
                            │   predict ← orig_C2           │
                            │ → write to:                   │
                            │   scoring_events.parquet      │
                            └──────────┬────────────────────┘
                                       │
                                       ▼
       ┌───────────────────────────────────────────┐
       │ Monthly cron job (Airflow/Prefect)        │
       │   1. concat 30 days of scoring events     │
       │   2. join with ground truth NPS survey    │
       │      (15% who answered)                   │
       │   3. compute KPIs via `metrics.py`        │
       │   4. compute alerts via `alerts.py`       │
       │   5. write `monitoring_run_YYYYMM.parquet`│
       │   6. if any 'alert' severity:             │
       │      → trigger PagerDuty / Slack          │
       └──────────────┬────────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────┐
       │ Streamlit / Grafana dashboard        │
       │   reads monitoring_run_*.parquet     │
       │   trend chart + alert table          │
       └──────────────────────────────────────┘
```

---

## Source des labels vérité terrain

Le scoring quotidien est sur **silent customers** (qui n'ont pas répondu à l'enquête NPS). Pour calculer les KPIs (Recall, F1, QWK), il faut la vraie classe NPS.

Solutions par ordre de fiabilité :

1. **Sondage de validation** : appeler 100-500 silent customers/mois pour leur demander leur NPS. Permet de calculer QWK et Recall sur cet échantillon. **Coût** : ~2 ETP-jours/mois pour les appels. **Recommandé.**
2. **Churn comme proxy de Détracteur** : assumer que les clients qui résilient dans les 90 jours étaient Détracteurs. Imparfait (un Promoteur peut churner pour des raisons exogènes), mais **gratuit** et utilisable sans sondage.
3. **NPS volontaires reçus** : si l'entreprise relance la campagne NPS, les répondants enrichissent l'échantillon labellisé. Pas garanti.

**Recommandation** : combinaison 1 + 2. Sondage de 200 clients/mois (gold) + tracking churn 90j (silver). Comparer les KPIs des deux sources pour détecter une régression cachée.

---

## Logique d'alerte 2σ — implémentation

Le module `src.monitoring.alerts` matérialise déjà :

```python
build_alert_table(drift_df, fairness_per_group, final_eval_summary) → DataFrame
```

Inputs :
- `drift_df` : KPIs calculés sur le batch mensuel (sortie de `drift_simulator.py` adapté)
- `fairness_per_group` : IC bootstrap Phase 11 (un par cellule champion × segment × groupe × classe)
- `final_eval_summary` : IC bootstrap Phase 9 (un par cellule champion × split × métrique)

Output : 1 ligne par cellule monitorée, avec colonnes `observed`, `ref_lo`, `ref_hi`, `status` (in_band/drift_low/drift_high), `severity` (ok/warn/alert).

Règles :
- **warn** dès qu'un batch sort de [ref_lo, ref_hi]
- **alert** quand 2 batches consécutifs sortent **dans la même direction** (drift_low + drift_low, ou drift_high + drift_high)
- Un drift_low suivi d'un drift_high reste **warn** (peut être du bruit, pas du drift directionnel)

---

## Cas particuliers — quand le seuil est franchi

| Cas | Causes plausibles | Action recommandée |
|---|---|---|
| QWK ↘ sur 2 batches | drift entrée, modèle obsolète | Vérifier distribution features, re-tuner si confirmé |
| Rappel Détracteur ↘ uniquement chez Senior=Yes | covariate shift segment-spécifique | Vérifier l'évolution démographique de la base, re-fitter avec données récentes |
| ECE Détracteur ↑ | calibration dérive | Refit `CalibratedClassifierCV` sur les 3 derniers mois de données labellisées |
| % prédits Détracteur ↑↑ | changement de comportement client (campagne tarifaire ?) | Aligner avec marketing avant de toucher au modèle |
| Lift@10 stable, QWK ↘ | les classes Passif/Promoteur dégradent, mais le ranking Détracteur tient | Pas urgent pour la rétention ; à traiter si extension Promoteur |
| Detractor recall ↘ uniquement après recalibration | mauvais usage du modèle recalibré (mode `after_full` au lieu de `after_hybrid`) | Vérifier que le scoring code utilise `orig_C2.predict()` pour l'argmax |

---

## Recalibration périodique

Phase 13.1 a recalibré C2 avec `CalibratedClassifierCV(isotonic, fit sur val)` une fois. **En production, refit trimestriellement** sur les 90 derniers jours de données labellisées. Coût : <1 seconde de compute, automatisable dans le cron.

```python
# Pseudo-code du cron de recalibration
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

labels_90d = load_labelled_scoring_events(days=90)
frozen_c2 = FrozenEstimator(load_C2())
new_cal = CalibratedClassifierCV(estimator=frozen_c2, method="isotonic", cv=5)
new_cal.fit(labels_90d.X, labels_90d.y)
save(new_cal, "logistic_C2_calibrated_YYYYMMDD.joblib")
```

⚠ Si à terme on **retire `class_weight='balanced'`** de C2 (option B du write-up), la calibration en mode FULL devient utilisable directement (plus de collapse argmax). Ça simplifie le code de scoring. Trade-off : Detractor recall global baisse de quelques points.

---

## Plan de rollback

Si une alerte `alert` se déclenche sur QWK ou Detractor recall :

1. **Stop scoring automatisé** (l'app Streamlit affiche un bandeau "Modèle suspendu").
2. **Lead data investigation** dans les 48h : reproduire le batch, vérifier les distributions de features.
3. **Décision binaire** :
   - drift confirmé → ré-entraîner C2 sur le périmètre récent → re-déployer après audit complet (Phases 9 + 11)
   - faux positif → resserrer le seuil d'alerte ou élargir l'IC de référence

---

## Indicateurs hors scope Phase 13

- **Détection automatique de drift d'entrée** (KS test, MMD) — possible avec `evidently` mais hors brief.
- **Re-tuning automatique** — non, le ré-entraînement reste une décision humaine validée.
- **Multi-modèle A/B** — ce serait une Phase 14+ (champion C2 vs nouveau challenger).

---

## Définition de "Done" pour la Phase 13

- ✅ C2 recalibré et persisté (`logistic_C2_calibrated.joblib`)
- ✅ ECE Détracteur et Promoteur < 0,05 sur silent_test (mode HYBRID)
- ✅ Audit 3-modes documenté (`before` / `after_full` / `after_hybrid`) → trace claire de l'interaction `class_weight` × isotonic
- ✅ `drift_simulator.py` produit un parquet de KPIs par batch (mode hybride par défaut quand `--use-calibrated`)
- ✅ `alerts.py` produit un parquet d'alertes basé sur les CIs Phase 9/11
- ✅ Page Streamlit "Monitoring" affiche le tout
- ✅ Cette spec écrite, prête à passer aux ops
- ✅ Cible Makefile `make build-alerts` + chaîne automatique depuis `make simulate-drift`
