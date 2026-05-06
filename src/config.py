"""
Project-wide configuration.

Single source of truth for paths, random seeds, target definition,
and the explicit list of leaky features to drop before modelling.

Import as:
    from src.config import RANDOM_SEED, DATA_PROCESSED, LEAKY_FEATURES
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
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Make sure all directories exist on import — first-run safety
for _dir in [
    DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_EXTERNAL,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
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

# Baseline mapping from Satisfaction Score (1-5) to NPS class.
# This is the starting point; alternative mappings tested in src/data/target.py.
SATISFACTION_TO_NPS_BASELINE = {
    1: "Detractor",
    2: "Detractor",
    3: "Detractor",
    4: "Passive",
    5: "Promoter",
}

# ============================================================
# Leakage — features to ALWAYS drop before modelling
# ============================================================
# Reasoning for each:
#   Satisfaction Score → source of the label, trivial leak (f(x)=y)
#   Churn Score        → IBM-computed propensity, leaks via correlation
#   Churn Value        → outcome variable, calculated post-hoc
#   Churn Label        → human-readable Churn Value
#   Churn Reason       → only filled for churners → presence/absence leaks
#   Churn Category     → same logic as Churn Reason
#   CLTV               → debated, often computed with future data → drop by default

LEAKY_FEATURES = [
    "Satisfaction Score",
    "Churn Score",
    "Churn Value",
    "Churn Label",
    "Churn Reason",
    "Churn Category",
    "CLTV",
]

# ============================================================
# Identifiers and constants — drop from features but keep in dataset
# ============================================================
ID_FEATURES = [
    "CustomerID",
    "Customer ID",
    "Count",       # always 1
    "Country",     # always "United States"
    "State",       # always "California"
    "Quarter",     # always "Q3"
    "Lat Long",    # redundant with Latitude / Longitude
]

# ============================================================
# Demographic features — kept for fairness audit, used cautiously in modelling
# ============================================================
DEMOGRAPHIC_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Age",                # if present in v11.1.3+
    "Under 30",           # if present
    "Number of Dependents",
]

# ============================================================
# Default hyperparameters — overridden by tuning later
# ============================================================
LIGHTGBM_DEFAULT_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# ============================================================
# Validation strategy
# ============================================================
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # of the remaining train pool
N_SPLITS_CV = 5
