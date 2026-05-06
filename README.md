# NPS Prediction Challenge — Telecom Operator

Predict customer NPS category (Detractor / Passive / Promoter) from account and behavioural data, so the retention team can prioritise outreach toward predicted Detractors before they churn.

Built on the **IBM Telco Customer Churn dataset (v11.1.3+)**, using the `Satisfaction Score` column as the basis for a derived NPS label.

---

## Quick start

```bash
# 1. Clone
git clone <your-repo-url> nps-prediction
cd nps-prediction

# 2. Set up environment
make install                  # creates .venv and installs dependencies
source .venv/bin/activate

# 3. Configure secrets
cp .env.example .env
# Edit .env to add your Kaggle and (optional) Anthropic keys

# 4. Download and prepare data
bash scripts/download_data.sh
make data

# 5. Train and evaluate
make baseline
make train
make evaluate

# 6. Launch the app
make app
```

---

## Project structure

```
nps-prediction/
├── data/
│   ├── raw/          # IBM Telco original files (gitignored)
│   ├── interim/      # Cleaned, no feature engineering yet
│   ├── processed/    # Final analytical dataset with NPS target
│   └── external/     # Verbatims, enrichment data
├── notebooks/        # Exploratory work (one per phase)
├── src/
│   ├── config.py     # Paths, seeds, constants, leakage drop list
│   ├── data/         # Load, build target, splits, leak audit
│   ├── features/     # Feature engineering pipeline
│   ├── models/       # Baseline, train, predict
│   ├── evaluation/   # Metrics and plots
│   ├── interpretation/ # SHAP utilities
│   ├── fairness/     # Per-segment audit
│   └── verbatims/    # LLM-generated customer notes (bonus)
├── app/              # Streamlit UI
├── models/           # Trained model artifacts (gitignored)
├── reports/          # Final write-up + figures
├── tests/            # pytest suite
├── scripts/          # Bash helpers
├── Makefile          # All commands routed here
├── requirements.txt
└── .env.example
```

---

## Approach summary

1. **Target** — derive 3-class NPS from `Satisfaction Score` (1–5). Document mapping, run sensitivity analysis, drop all label-source columns.
2. **Leakage** — explicitly drop `Satisfaction Score`, `Churn Score`, `Churn Value`, `Churn Reason`, `Churn Category`. Audit remaining features for label correlation.
3. **Validation** — split simulating the 15% respondents / 85% silent gap, not naïve random.
4. **Models** — baseline (logistic, ordinal logistic, LightGBM default) → tuned LightGBM → optional ordinal CORAL / TabPFN. Compare honestly.
5. **Metrics** — Quadratic Weighted Kappa (primary), recall@Detractor (business), macro-F1, calibration, lift curve.
6. **Interpretability** — SHAP global + per-segment, distinction actionable vs non-actionable drivers.
7. **Fairness** — per-group recall on Detractor across senior, gender, dependents. Trade-off analysis vs accuracy.
8. **Verbatims (bonus)** — LLM-generated customer notes conditioned on tabular features, combined via embeddings.
9. **App** — Streamlit interface for retention managers: pick a customer, see prediction + top drivers.

See `reports/final_report.md` for the full write-up.

---

## Reproducibility

- All randomness seeded via `src.config.RANDOM_SEED` (= 42).
- Data pipeline is deterministic: same raw input → same processed dataset.
- Verbatim generation logs prompt and seed in `data/external/verbatims_log.json`.
- Trained models persisted in `models/` with metadata (hyperparams, train commit).

To re-run the full pipeline from scratch:

```bash
make clean
make data
make train
make evaluate
```

---

## Honesty checklist

- [ ] All `Churn*` and `Satisfaction Score` features dropped from training.
- [ ] Validation split simulates non-respondent gap (not pure random).
- [ ] Baseline trained before any tuning.
- [ ] Metrics include macro-F1, QWK, per-class recall, calibration.
- [ ] SHAP analysis includes per-segment views.
- [ ] Fairness audit on at least 3 demographic axes.
- [ ] Limitations of the derived label discussed in the report.
- [ ] LLM use disclosed in the report (verbatims, code scaffolding, etc.).

---

## Stack

Python 3.11 · pandas · scikit-learn · LightGBM · XGBoost · mord · SHAP · Fairlearn · Streamlit · imbalanced-learn

Optional: TabPFN, sentence-transformers, Anthropic SDK, Evidently, MLflow.

---

## License

This is a private take-home challenge. Not for redistribution.
