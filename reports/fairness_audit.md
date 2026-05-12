# Phase 11 — Fairness Audit Report

Target: `NPS_alternative` · Bootstrap n = 1000 · CI = 95% · Seed = 42

Two champions audited (Phase 9 selection):
- **C1 — QWK champion** — `hybrid/lightgbm/pca32`
- **C2 — Production-safe** — `tuned/logistic/tabular`

Three protected segments (brief 4.7): **Senior**, **Gender**, **Married**.

Two audited classes: **Detractor** (retention priority) and **Promoter** (symmetric check).

## Per-group recall on silent_test (bootstrap 95% CI)

### C1 — QWK champion

**Gender**

| Group | n_total | n_Detractor | Detractor recall | n_Promoter | Promoter recall |
|---|---|---|---|---|---|
| Female | 2989 | 630 | 0.629 [0.594, 0.667] | 485 | 0.847 [0.814, 0.878] |
| Male | 2998 | 645 | 0.643 [0.608, 0.681] | 484 | 0.847 [0.816, 0.876] |

**Married**

| Group | n_total | n_Detractor | Detractor recall | n_Promoter | Promoter recall |
|---|---|---|---|---|---|
| No | 3192 | 821 | 0.706 [0.675, 0.739] | 526 | 0.817 [0.783, 0.848] |
| Yes | 2795 | 454 | 0.509 [0.463, 0.555] | 443 | 0.883 [0.853, 0.914] |

**Senior**

| Group | n_total | n_Detractor | Detractor recall | n_Promoter | Promoter recall |
|---|---|---|---|---|---|
| No | 5035 | 953 | 0.637 [0.604, 0.668] | 852 | 0.853 [0.827, 0.876] |
| Yes | 952 | 322 | 0.634 [0.578, 0.686] | 117 | 0.803 [0.726, 0.872] |


### C2 — Production-safe

**Gender**

| Group | n_total | n_Detractor | Detractor recall | n_Promoter | Promoter recall |
|---|---|---|---|---|---|
| Female | 2989 | 630 | 0.867 [0.841, 0.894] | 485 | 0.532 [0.489, 0.577] |
| Male | 2998 | 645 | 0.814 [0.784, 0.845] | 484 | 0.537 [0.494, 0.583] |

**Married**

| Group | n_total | n_Detractor | Detractor recall | n_Promoter | Promoter recall |
|---|---|---|---|---|---|
| No | 3192 | 821 | 0.865 [0.839, 0.887] | 526 | 0.473 [0.430, 0.517] |
| Yes | 2795 | 454 | 0.795 [0.756, 0.830] | 443 | 0.607 [0.560, 0.655] |

**Senior**

| Group | n_total | n_Detractor | Detractor recall | n_Promoter | Promoter recall |
|---|---|---|---|---|---|
| No | 5035 | 953 | 0.816 [0.790, 0.839] | 852 | 0.576 [0.543, 0.608] |
| Yes | 952 | 322 | 0.910 [0.882, 0.938] | 117 | 0.231 [0.154, 0.308] |


## Disparity indicators

* **DI** (Disparate Impact) ratio of recall to the favored group. Fair if DI ∈ [0.8, 1.25] (EEOC 4/5 rule).
* **EOD** (Equal Opportunity Difference) = max(recall) − min(recall) across groups. Commonly acceptable if |EOD| < 0.10.
* **DPD** (Demographic Parity Difference) = max − min of P(ŷ = class) across groups. Informational only — legitimate business differentiation can produce DPD ≠ 0.

| Champion | Segment | Class | DI (worst) | EOD | DPD | Verdict |
|---|---|---|---|---|---|---|
| C1 — QWK champion | Senior | Detractor | 0.995 (Yes vs No) | +0.003 | +0.118 | ✓ fair |
| C1 — QWK champion | Senior | Promoter | 0.942 (Yes vs No) | +0.050 | +0.056 | ✓ fair |
| C1 — QWK champion | Gender | Detractor | 0.977 (Female vs Male) | +0.015 | +0.006 | ✓ fair |
| C1 — QWK champion | Gender | Promoter | 1.000 (Male vs Female) | +0.000 | +0.002 | ✓ fair |
| C1 — QWK champion | Married | Detractor | 0.720 (Yes vs No) | +0.198 | +0.238 | ⚠ disparity |
| C1 — QWK champion | Married | Promoter | 0.926 (No vs Yes) | +0.065 | +0.009 | ✓ fair |
| C2 — Production-safe | Senior | Detractor | 0.897 (No vs Yes) | +0.094 | +0.243 | ✓ fair |
| C2 — Production-safe | Senior | Promoter | 0.400 (Yes vs No) | +0.346 | +0.242 | ⚠ disparity |
| C2 — Production-safe | Gender | Detractor | 0.939 (Male vs Female) | +0.053 | +0.017 | ✓ fair |
| C2 — Production-safe | Gender | Promoter | 0.990 (Female vs Male) | +0.005 | +0.017 | ✓ fair |
| C2 — Production-safe | Married | Detractor | 0.919 (Yes vs No) | +0.070 | +0.310 | ✓ fair |
| C2 — Production-safe | Married | Promoter | 0.780 (No vs Yes) | +0.134 | +0.050 | ⚠ disparity |

## Counterfactual swap analysis

We flip the protected attribute on silent_test (e.g. Senior=Yes → No and vice-versa), re-predict with the unchanged model, and measure the rate of changed predictions. A model that does NOT directly use the protected attribute will see ~0 % change. Any non-zero rate quantifies the *direct causal* dependence on the attribute (vs indirect, through correlated features).

| Champion | Segment | n | Δ predictions | Δ rate | Detractor → other | Other → Detractor | Promoter → other | Other → Promoter |
|---|---|---|---|---|---|---|---|---|
| C1 — QWK champion | Senior | 5987 | 92 | 1.5% | 58 | 17 | 13 | 4 |
| C1 — QWK champion | Gender | 5987 | 42 | 0.7% | 14 | 16 | 8 | 6 |
| C1 — QWK champion | Married | 5987 | 10 | 0.2% | 6 | 4 | 0 | 1 |
| C2 — Production-safe | Senior | 5987 | 557 | 9.3% | 185 | 32 | 308 | 38 |
| C2 — Production-safe | Gender | 5987 | 87 | 1.5% | 32 | 24 | 26 | 32 |
| C2 — Production-safe | Married | 5987 | 100 | 1.7% | 14 | 15 | 69 | 29 |

**Interpretation rule of thumb**: change_rate < 2 % means the model barely uses the attribute directly. If a disparity is observed in DI/EOD but change_rate ≈ 0, it operates through *proxy variables* (e.g. Tenure correlates with Senior). Reducing such proxy-mediated disparity requires either feature engineering or a fairness-aware training objective — both beyond Phase 11.

## Verdict

Out of 12 (champion × segment × class) cells, **3 flag a disparity** (DI outside [0.8, 1.25] OR |EOD| ≥ 0.10).

### Recommendations
* Investigate which proxy features drive the disparity (use Phase 10's `shap_segment_*.parquet` and `linear_coef_segment_*.parquet`).
* Consider Phase 13 monitoring: track per-segment recall monthly with alert at 2σ deviation.
* Per-group threshold tuning is technically possible but may be legally/ethically inadvisable — document risk before adopting.
