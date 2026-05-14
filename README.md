# NPS Prediction for Telco Retention

> **Predict NPS class (Detractor / Passive / Promoter) for the 85% of customers who didn't answer the satisfaction survey, so the retention team can prioritize outreach.**
>
> Take-home challenge for **Artefact** · Dataset: IBM Telco Customer Churn v11.1.3 · Stack: Python 3.11, scikit-learn, LightGBM, Optuna, SHAP, Fairlearn, Streamlit.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-~210%20invariants-brightgreen.svg)](#testing)
[![Phases](https://img.shields.io/badge/phases-14%2F14%20delivered-success.svg)](#project-phases)
[![Report](https://img.shields.io/badge/report-6%20pages%20FR-blueviolet.svg)](reports/rapport_final.pdf)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## TL;DR

A **rigorously evaluated** NPS prediction system with two competing champions, a Streamlit retention manager app, fairness audit on 3 protected segments, isotonic recalibration, and a lightweight monitoring simulator. **Deployed model = C2** (regularized Logistic Regression on tabular features), selected on a **triple convergence** of evidence rather than on raw QWK:

1. **Statistical equivalence** with the QWK champion C1 on business metrics (Lift@10/20 confidence intervals overlap).
2. **Fairness validated** on the Detractor mission across Senior, Gender, Married segments (DI ∈ [0.79, 0.92]).
3. **Temporal stability** on 12 FIFO monthly batches (Detractor recall range = 0.085 vs 0.159 for C1).

The hybrid model C1 (LightGBM + sentence-transformer embeddings) reaches QWK 0.659 but **its +0.30 advantage is shown empirically to be a leakage artifact** from synthetic verbatim generation — PC01 alone contributes 3.9× more than feature #2 for Promoter prediction (SHAP). This caveat is documented, not hidden.

---

## Key results

All numbers on **`silent_test`** (n = 5,987, never used during tuning). Bootstrap 95% CI, n=1000 stratified resamples.

| Metric | **C1** — hybrid/LGBM/pca32 | **C2** — tuned/Logistic/tabular | Verdict |
|---|---|---|---|
| QWK | **0.659** [0.644, 0.677] | 0.355 [0.334, 0.375] | C1 ≫ C2 (disjoint CIs) |
| Macro F1 | **0.739** [0.728, 0.750] | 0.482 [0.471, 0.494] | C1 ≫ C2 |
| Detractor recall | 0.636 [0.608, 0.661] | **0.840** [0.820, 0.859] | **C2 ≫ C1** |
| **Lift@10** | **2.850** [2.685, 3.023] | **2.756** [2.591, 2.921] | **Equivalent** (overlapping CIs) |
| **Lift@20** | **2.577** [2.471, 2.683] | **2.475** [2.373, 2.589] | **Equivalent** |

**Recommendation: deploy C2.** On the business metrics that drive retention (lift@K — how well the model concentrates detractors in the top-X% call list), the two champions are statistically equivalent. C2 wins on what matters most (Detractor recall = 0.84 vs 0.64), carries no leakage risk, is natively interpretable through linear coefficients, and is simpler to maintain.

---

## Quick start

### Prerequisites

- Python 3.11
- ~2 GB free disk (data + models + cached embeddings)
- The IBM Telco Customer Churn v11.1.3 Excel files are **included in this repository** under `data/raw/` (~30 MB total). [Original source: IBM Cognos sample data](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
- The 7,043 synthetic verbatims generated via Qwen2.5-7B are **included** under `data/external/verbatims.parquet` (~3 MB) — no need to re-run the Colab GPU generation pipeline unless you want to regenerate them. The Colab notebook is in [`colab/`](colab/) for reference.

### Install

```bash
git clone <this-repo>
cd nps-prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce the full pipeline

```bash
make data                          # load raw + audit leaks
make build-dataset build-splits    # build modelable dataset
make build-features                # ColumnTransformer + 12 derived features
# Verbatims: see colab/ to regenerate via Qwen2.5-7B on Colab GPU
make compute-embeddings            # cache sentence-transformer embeddings

make baseline                      # 3 baseline models
make tune tune-hybrid              # C2 + C1 with Optuna (100 trials each)
make final-eval                    # bootstrap CIs on silent_test
make interpret                     # SHAP + linear coef + PCA loadings
make fairness                      # 3 segments × 2 classes + counterfactual
make batch-score                   # pre-compute predictions for the app
make recalibrate                   # isotonic recalibration of C2
make simulate-drift                # 12 batches FIFO + alerts

make test                          # ~210 invariants, all passing
make app                           # launch Streamlit on localhost:8501
```

### Just want to see the app

```bash
make app
```

Five French pages (Score Customer, Cohort Targeting, Fairness, About, Monitoring) backed by all pre-computed artifacts in `models/results/`.

---

## Repository structure

```
nps-prediction/
├── src/
│   ├── data/              # Phase 1-3: load, audit leaks, build dataset, splits
│   ├── features/          # Phase 4-5: ColumnTransformer + embeddings (cached)
│   ├── models/            # Phase 6-8: baselines, tuning, hybrid text+tabular
│   ├── evaluation/        # Phase 9: bootstrap CI, calibration, final eval
│   ├── interpretation/    # Phase 10: SHAP, linear coef, PCA loadings
│   ├── fairness/          # Phase 11: per-segment metrics + counterfactual
│   ├── inference/         # Phase 12: batch_score for the Streamlit app
│   └── monitoring/        # Phase 13: recalibrate, drift_simulator, alerts
├── app/
│   ├── main.py            # Streamlit home dashboard
│   ├── components/        # data_loaders, plots, styling
│   └── pages/             # 5 pages (FR): Scorer Client, Cohorte, Équité, ...
├── notebooks/
│   ├── 01_eda.ipynb            # Phase 1
│   ├── 02_dataset_build.ipynb  # Phase 2
│   ├── ...
│   └── 13_monitoring.ipynb     # Phase 13
├── tests/                 # ~210 pytest invariants, organized by phase
├── reports/
│   ├── rapport_final.pdf            # 6-page write-up (FR)
│   ├── rapport_NPS_phases_0_7.docx  # detailed report, phases 0-7
│   ├── rapport_NPS_phases_8_14.docx # detailed report, phases 8-14
│   ├── phase13_monitoring_spec.md   # production spec
│   ├── final_eval_report.md         # Phase 9 markdown report
│   ├── fairness_audit.md            # Phase 11 markdown report
│   └── figures/                     # PNG figures from each phase
├── data/
│   ├── raw/               # IBM Telco Excel files (gitignored)
│   ├── interim/           # raw merge → parquet
│   ├── processed/         # dataset_with_verbatims.parquet + splits/
│   └── external/          # verbatims.parquet (Qwen-generated)
├── models/
│   ├── preprocessing_pipeline.joblib
│   ├── baselines/         # Phase 6
│   ├── tuned/             # Phase 7-8 + recalibrated C2
│   ├── hybrid/            # Phase 8 (LightGBM/pca32 = C1)
│   └── results/           # all parquet outputs feeding the app
├── colab/                 # Phase 5 verbatim generation notebook (GPU)
├── Makefile               # orchestrates the full pipeline
└── requirements.txt
```

---

## Methodology highlights

The full methodology is documented in three layers:

- **6-page summary** — [`reports/rapport_final.pdf`](reports/rapport_final.pdf) (FR), the argumentative pitch.
- **Detailed reports** — [`reports/rapport_NPS_phases_0_7.docx`](reports/rapport_NPS_phases_0_7.docx) and [`reports/rapport_NPS_phases_8_14.docx`](reports/rapport_NPS_phases_8_14.docx), one section per phase with *what was done / results / choices / lesson learned*.
- **Notebooks** — `notebooks/0X_*.ipynb`, fully reproducible.

The non-negotiable methodological commitments that shape every result below:

### 1. Leak audit before any modeling (Phase 1)

Eight columns of IBM Telco are leakers — they encode the target directly or by construction. They were identified by a programmatic audit (Pearson correlation for numerics, eta² for categoricals) and removed before any split:

- `Satisfaction Score` (source of the NPS target — always drop)
- `Churn Value`, `Customer Status`, `Churn Label`, `Churn Score`, `Churn Reason`, `Churn Category` (six churn-related variables, all with assoc ≥ 0.4)
- `CLTV` (assoc = 0.076, statistically clean, **dropped by precaution** because it's an a-posteriori IBM-computed metric — potential temporal leak)

### 2. Response-biased split simulating real-world non-response (Phase 3)

A naive 80/20 random split would estimate the model on respondents only, but the production setting scores **silent customers** — those who never answered the NPS survey. We simulate this:

- 15% respondents (sampled with probability increasing with `Tenure Months` — long-tenured customers respond more)
- 85% **silent_test**, never touched during training or tuning

All metrics reported in the headline table above are on `silent_test`. Phase 9 validates the design by checking covariate shift: Δ QWK = −0.012 for C1 and +0.009 for C2 between `respondent_test` and `silent_test` — well below the 0.05 practical threshold.

### 3. The synthetic verbatim caveat (Phases 5, 8, 10)

The IBM Telco dataset has no customer verbatims. To test a hybrid text+tabular pipeline (C1), we generated 7,043 verbatims with **Qwen2.5-7B-Instruct conditioned on customer features and satisfaction score** (Colab GPU, ~30 min). 15% are intentionally "counter-intuitive" to avoid perfect correlation.

**Caveat**: the LLM knew the target when writing the verbatim. C1's gain over C2 is therefore an **upper bound** on what real verbatims could deliver.

Phase 10 confirms this empirically via TreeSHAP: **PC01 (first PCA component of embeddings) contributes 3.9× more than the second feature** to Promoter prediction. On the Promoter archetype customer (`6112-KTHFQ`), PC01 alone contributes +3.24 logit. The +0.30 QWK gain of C1 is largely a leakage artifact — quantified, documented, and reported.

### 4. Bootstrap CIs as the decision-making lever (Phase 9)

The decision to deploy C2 hinges on the bootstrap analysis. Without confidence intervals, "QWK 0.659 vs 0.355" looks like a clear win for C1. **With CIs, Lift@10/20 are statistically equivalent**, which is the metric that matters for the retention manager. Bootstrap is stratified by class (preserves the marginal 21/58/21) and uses 1,000 resamples with fixed seed.

### 5. Fairness audit on 3 segments × 2 classes (Phase 11)

The brief mentions Detractor only. We also audit Promoter to find the strongest disparity in the project: **C2 × Senior × Promoter has DI = 0.40, EOD = +0.35, counterfactual rate = 9.3%** — meaning the disparity is causally due to the Senior attribute itself (not proxies). C2 misses three-quarters of senior Promoters. The Detractor mission of C2 is fair across all 3 segments (DI ∈ [0.79, 0.92], |EOD| < 0.10) — green light for deployment on the primary mission.

The counterfactual swap analysis (flip the protected attribute, re-predict, count changes) is an **original method of the project** for distinguishing direct-causal from proxy-mediated disparities. Implementation in [`src/fairness/audit.py`](src/fairness/audit.py).

### 6. The HYBRID recalibration pattern (Phase 13)

C2's `predict_proba` is poorly calibrated on Passive/Promoter (ECE Passive = 0.292 in Phase 9, qualified as "catastrophic"). Isotonic recalibration via `CalibratedClassifierCV(method='isotonic', cv=5)` reduces ECE by 86–93%:

| Class | ECE before | ECE after | Reduction |
|---|---|---|---|
| Detractor | 0.143 | **0.020** | −86% |
| Passive | 0.292 | **0.031** | −89% |
| Promoter | 0.148 | **0.010** | −93% |

**But — `class_weight='balanced'` × isotonic recalibration is a non-trivial pitfall.** The reweighting inflates `predict_proba(Detractor)` to catch more detractors. Isotonic correctly maps these inflated probabilities back to their true posteriors → ECE drops massively (the win we wanted), but `argmax(predict_proba)` then favors the majority class (Passive) and Detractor recall collapses from 0.84 to 0.30.

**Resolution = HYBRID mode**: calibrated `predict_proba` for display and lift, original `predict` for argmax decisions. Preserves Phase 9 metrics exactly (Δ recall = 0.0000) while delivering the calibration win. Documented in [`reports/phase13_monitoring_spec.md`](reports/phase13_monitoring_spec.md). This is a **methodological contribution** of the project, rarely covered in standard calibration literature.

---

## The Streamlit app

Launch with `make app`. Six pages (UI in French, code in English):

| Page | Purpose | Audience |
|---|---|---|
| 🏠 **Accueil** | Phase 9 headline metrics + artifact status | Manager, first-look |
| 🎯 **Scorer Client** | Per-customer prediction + live SHAP/coef explanation | Retention analyst |
| 📊 **Ciblage Cohorte** | Top-N call list with filters, CSV export | Retention manager |
| ⚖️ **Équité** | Fairness audit dashboard with all 12 cells | Compliance, lead |
| 📖 **À Propos** | Methodology + caveats + inline reports | Technical reviewer |
| 📈 **Monitoring** | Calibration before/after + 12-batch drift simulation | Ops, lead data |

**UX choices that materialize the methodology**:

- **C2 is the default**; C1 is selectable but triggers a red banner: *"experimental model, synthetic embeddings, Married × Detractor disparity"*.
- **Ranks are preferred over absolute probabilities** for operational decisions (because C2 calibration is poor in non-HYBRID mode).
- **A red banner pops up automatically** when the user selects "Promoter" as the targeting class on C2 (because of the Senior × Promoter disparity, DI = 0.40).
- Each prediction displays a per-customer waterfall explanation: TreeSHAP on C1 (one row, instant) or `coef × x_std` on C2 (zero compute).

---

## Project phases

The project is structured in 14 phases, each with code modules, a notebook, pytest invariants, and a parquet/joblib artifact consumed by the next phase. The full chronology:

| # | Phase | Output |
|---|---|---|
| 0 | Project scaffold | `src/`, `Makefile`, `requirements.txt`, 7 tests |
| 1 | EDA + leak audit | 8 leakers identified |
| 2 | NPS target + processed dataset | `dataset.parquet`, 25 tests |
| 3 | Response-biased split | `splits/response_biased.parquet`, 17 tests |
| 4 | Feature engineering | 12 derived features + ColumnTransformer, 22 tests |
| 5 | Synthetic verbatims (Colab GPU) | 7,043 verbatims via Qwen2.5-7B, 17 tests |
| 6 | Three disciplined baselines | LogReg wins on QWK silent, 20 tests |
| 7 | Optuna tuning, 100 trials × 3 models | C2 = `tuned/logistic_tuned.joblib`, 9 tests |
| 8 | Hybrid text+tabular | C1 = `hybrid/lightgbm_pca32.joblib`, 17 tests |
| 9 | Bootstrap CIs + calibration + covariate shift | `final_eval_summary.parquet`, 17 tests |
| 10 | TreeSHAP + linear coef + PCA loadings | Leakage confirmed (PC01 = 3.9× feature #2), 22 tests |
| 11 | Fairness audit + counterfactual | 3 disparities identified, 26 tests |
| 12 | Streamlit app (5 pages FR) | App + `silent_predictions.parquet`, 19 tests |
| 13 | Recalibration HYBRID + drift simulator + alerts | ECE ÷ 9 to 15, 27 tests |
| 14 | Final write-up (md + docx + pdf) | 6-page FR report |

Total: ~210 pytest invariants, all passing as of phase 14 delivery.

---

## Testing

```bash
make test                # full suite (~210 invariants)
make test-phase9         # just one phase
pytest tests/ -v -k bootstrap   # filter by name
```

The pytest invariants are not just code validation — they document the **methodological constraints** that must hold:

- No leak columns in the modeling dataframe (`test_no_leak_in_X`)
- PCA fitted on train only (`test_pca_fitted_on_train_only`)
- Bootstrap CIs are in `[0, 1]` for bounded metrics (`test_ci_within_kappa_range`)
- Predictions sum to 1 per model (`test_probabilities_sum_to_one_per_model`)
- C2 calibration after recalibration is at most as bad as before (`test_recalibration_helps_or_neutral`)
- Severities in the alert table are in `{ok, warn, alert}` (`test_severity_values_valid`)

When you return to the code three months later, these tests will tell you what you committed to never break.

---

## The triple-confirmation argument for C2

The deployment decision was not made on a single metric. It rests on **three independent confirmations** obtained through different methodologies:

| Confirmation | Phase | Method | Finding |
|---|---|---|---|
| **Business equivalence** | 9 | Bootstrap 95% CI on lift@K | C1 and C2 CIs overlap on Lift@10 and Lift@20 → same retention value |
| **Fairness validated** | 11 | DI, EOD, counterfactual swap | C2 fair on Detractor across 3 segments (DI ∈ [0.79, 0.92], EOD < 0.10) |
| **Temporal stability** | 13 | 12 FIFO monthly batches | C2 Detractor recall range = 0.085 (vs 0.159 for C1, ~2× less stable) |

These three arguments do not derive from each other. A single argument could be challenged ("you optimized for lift, of course it's lifted"). Three convergent arguments from independent methodologies make the recommendation robust to challenge.

---

## Limits and roadmap

**Limits acknowledged in the final report**:

- The +0.30 QWK gain of C1 is conditional on the synthetic verbatim generation process. Real-world value with authentic verbatims is unknowable without the data.
- IBM Telco Customer IDs have no temporal semantics — the Phase 13 monitoring simulator validates the *plumbing* of the alerting pipeline, not its responsiveness to real drift.
- Calibration was fit on `val` (n = 211 only — small). Production should refit quarterly on 90 days of labeled scoring events (≥ 2,000 lines).
- The Promoter × Senior disparity on C2 is unresolved. The current decision is to display an explicit warning in the UI; a clean fix requires either retraining without Senior or per-group threshold tuning.

**Roadmap proposed**:

1. **Month 1** — Deploy C2 in HYBRID mode for Detractor mission. Set up validation survey (200 customers/month, ~2 FTE-days/month).
2. **Months 2-3** — Implement the monitoring cron on real infrastructure (Airflow/Prefect). ~5 dev-days.
3. **Months 4-6** — First quarterly recalibration on fresh data. Recompute reference CIs on monthly windows (will eliminate the false-positive alerts observed on the Phase 13 simulator).
4. **If real customer verbatims become available** — Retrain C1 on this real data to measure the true text contribution. Re-run Phase 11 + Phase 13 audits.

---

## License

This repository is released under the **MIT License** — see [`LICENSE`](LICENSE).

### Data licensing notice

**IBM Telco Customer Churn v11.1.3** Excel files under `data/raw/` are © IBM
Corporation, redistributed here for educational reproducibility as part of an
Artefact take-home challenge. All rights to the dataset remain with IBM.
[Original source on IBM Cognos](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

**Synthetic verbatims** under `data/external/verbatims.parquet` were generated
by the author using [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
(Alibaba, Apache 2.0). The generated text and the generation methodology are
released under MIT alongside the rest of this repository.

If you reuse the methodological patterns documented here (response-biased
split, counterfactual swap for fairness, HYBRID recalibration mode,
triple-confirmation deployment argument), a citation/link back is appreciated
but not required.

---

## Documentation index

| Document | Audience | Length |
|---|---|---|
| [`reports/rapport_final.pdf`](reports/rapport_final.pdf) | Artefact reviewers, hiring committee | 6 pages (FR) |
| [`reports/rapport_NPS_phases_0_7.docx`](reports/rapport_NPS_phases_0_7.docx) | Technical reviewer, future self | ~30 pages (FR) |
| [`reports/rapport_NPS_phases_8_14.docx`](reports/rapport_NPS_phases_8_14.docx) | Technical reviewer, future self | ~23 pages (FR) |
| [`reports/final_eval_report.md`](reports/final_eval_report.md) | Phase 9 deep-dive | inline markdown |
| [`reports/fairness_audit.md`](reports/fairness_audit.md) | Compliance | inline markdown |
| [`reports/phase13_monitoring_spec.md`](reports/phase13_monitoring_spec.md) | Ops, infra | inline markdown |
| `notebooks/0X_*.ipynb` | Reproducibility | 13 notebooks |

The 6-page PDF is the right starting point for evaluators. The detailed Word reports are for whoever wants to understand *why* each choice was made.

---

## Acknowledgments

Challenge designed by **Artefact**. Dataset by **IBM** (Telco Customer Churn v11.1.3 via Cognos sample data). LLM for verbatim generation: **Qwen2.5-7B-Instruct** (Alibaba), run on **Google Colab** GPU.
