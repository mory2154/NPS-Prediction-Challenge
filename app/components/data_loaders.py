"""
Cached parquet loaders for the Streamlit app.

We use @st.cache_data on parquet reads so the app doesn't re-read the same
files on every interaction. The cache invalidates automatically when the file
content changes (Streamlit hashes the file).

Phase 9-11 outputs surfaced in the app:
    silent_predictions.parquet         — Phase 12 batch_score output
    final_eval_summary.parquet         — Phase 9 (CIs for headline metrics)
    fairness_per_group.parquet         — Phase 11 per-group recalls
    fairness_disparities.parquet       — Phase 11 DI/EOD/DPD
    fairness_counterfactual.parquet    — Phase 11 swap analysis
    shap_global_C1.parquet             — Phase 10 SHAP global C1
    shap_local_C1.parquet              — Phase 10 SHAP per-archetype
    linear_coef_C2.parquet             — Phase 10 C2 coefficients
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import REPORTS_DIR, RESULTS_DIR


# ============================================================
# Predictions
# ============================================================
@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    """Load silent_test pre-scored predictions (Phase 12 batch_score)."""
    return pd.read_parquet(RESULTS_DIR / "silent_predictions.parquet")


# ============================================================
# Phase 9 — final eval headline numbers
# ============================================================
@st.cache_data(show_spinner=False)
def load_final_eval_summary() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "final_eval_summary.parquet")


# ============================================================
# Phase 11 — fairness audit
# ============================================================
@st.cache_data(show_spinner=False)
def load_fairness_per_group() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "fairness_per_group.parquet")


@st.cache_data(show_spinner=False)
def load_fairness_disparities() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "fairness_disparities.parquet")


@st.cache_data(show_spinner=False)
def load_fairness_counterfactual() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "fairness_counterfactual.parquet")


# ============================================================
# Phase 10 — interpretability
# ============================================================
@st.cache_data(show_spinner=False)
def load_shap_global() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "shap_global_C1.parquet")


@st.cache_data(show_spinner=False)
def load_shap_local() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "shap_local_C1.parquet")


@st.cache_data(show_spinner=False)
def load_linear_coef_C2() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "linear_coef_C2.parquet")


# ============================================================
# Phase 9 — markdown report (for About page)
# ============================================================
@st.cache_data(show_spinner=False)
def load_final_eval_report() -> str:
    p = REPORTS_DIR / "final_eval_report.md"
    if p.exists():
        return p.read_text()
    return "Final eval report not generated yet. Run `make final-eval`."


@st.cache_data(show_spinner=False)
def load_fairness_report() -> str:
    p = REPORTS_DIR / "fairness_audit.md"
    if p.exists():
        return p.read_text()
    return "Fairness audit report not generated yet. Run `make fairness`."


# ============================================================
# Availability check (used by main.py to warn if artifacts missing)
# ============================================================
def check_artifacts_available() -> dict[str, bool]:
    """Return a dict mapping artifact_name → bool (exists)."""
    files = {
        "Predictions (Phase 12)":   RESULTS_DIR / "silent_predictions.parquet",
        "Final eval (Phase 9)":     RESULTS_DIR / "final_eval_summary.parquet",
        "Fairness (Phase 11)":      RESULTS_DIR / "fairness_per_group.parquet",
        "SHAP global C1 (Phase 10)":RESULTS_DIR / "shap_global_C1.parquet",
        "SHAP local C1 (Phase 10)": RESULTS_DIR / "shap_local_C1.parquet",
        "Coef C2 (Phase 10)":       RESULTS_DIR / "linear_coef_C2.parquet",
    }
    return {name: p.exists() for name, p in files.items()}
