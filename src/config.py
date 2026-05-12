"""
Project-wide configuration.
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
BASELINES_DIR = MODELS_DIR / "baselines"
TUNED_DIR = MODELS_DIR / "tuned"
RESULTS_DIR = MODELS_DIR / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _dir in [
    DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_EXTERNAL,
    MODELS_DIR, BASELINES_DIR, TUNED_DIR, RESULTS_DIR,
    REPORTS_DIR, FIGURES_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Target definition
# ============================================================
NPS_CLASSES = ["Detractor", "Passive", "Promoter"]
NPS_CLASS_TO_INT = {"Detractor": 0, "Passive": 1, "Promoter": 2}
NPS_INT_TO_CLASS = {v: k for k, v in NPS_CLASS_TO_INT.items()}

NPS_MAPPINGS: dict[str, dict[int, str]] = {
    "baseline": {1: "Detractor", 2: "Detractor", 3: "Detractor",
                 4: "Passive", 5: "Promoter"},
    "alternative": {1: "Detractor", 2: "Detractor",
                    3: "Passive", 4: "Passive", 5: "Promoter"},
}
SATISFACTION_TO_NPS_BASELINE = NPS_MAPPINGS["baseline"]

# Phase 6 result: NPS_alternative wins (QWK 0.293 vs 0.212).
# This is now the primary target for tuning.
DEFAULT_TARGET = "NPS_alternative"

# ============================================================
# Leakage
# ============================================================
LEAKY_FEATURES = [
    "Satisfaction Score", "Churn Score", "Churn Value", "Churn Label",
    "Churn Reason", "Churn Category", "Customer Status", "CLTV",
]
ID_AND_CONSTANT_FEATURES = ["Count", "Country", "State", "Quarter", "Lat Long", "ID"]
INDEX_COLUMN = "Customer ID"
DEMOGRAPHIC_FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Married",
    "Dependents", "Number of Dependents", "Age", "Under 30",
]

# ============================================================
# Modelling
# ============================================================
LIGHTGBM_DEFAULT_PARAMS = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
    "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
    "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1,
}

PRIMARY_METRIC = "qwk"

# Phase 7 — Optuna tuning budgets per model
TUNING_BUDGETS = {
    "lightgbm": 50,
    "logistic": 30,
    "ordinal":  20,
}
TUNING_SAMPLER = "tpe"  # 'tpe' or 'random'
TUNING_TIMEOUT_SEC = 600  # safety net per study

# Validation strategy
TEST_SIZE = 0.2
VAL_SIZE = 0.2
N_SPLITS_CV = 5
