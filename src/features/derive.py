"""
Derived features for the NPS prediction model.

Pure-pandas helpers — no scikit-learn here. Each function adds one or more
columns to the DataFrame, with semantic meaning, ready for the encoding
pipeline in `src.features.pipeline`.

Design principles:
    * One function per feature family.
    * Idempotent (safe to call twice).
    * Robust to missing columns (skip rather than crash).
    * Document business meaning in each docstring.

Public entry point:
    add_all_derived_features(df) → df with ~12 new columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# 1. Financial ratios
# ============================================================
def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    avg_monthly_charge : Total Charges / max(Tenure Months, 1)
        Sanity check on Monthly Charges. A divergence between the two
        suggests price changes, promos, or downgrades over time.

    charges_per_service : Monthly Charges / max(n_services, 1)
        How much each subscribed service "costs". High ratio → potential
        frustration (paying premium for few services).
    """
    out = df.copy()

    if {"Total Charges", "Tenure Months"}.issubset(out.columns):
        tenure_safe = out["Tenure Months"].fillna(0).clip(lower=1)
        out["avg_monthly_charge"] = (
            out["Total Charges"].astype(float) / tenure_safe
        ).round(2)

    # Count subscribed services (computed in add_service_bundle, see below)
    if "Monthly Charges" in out.columns and "n_services" in out.columns:
        n_services_safe = out["n_services"].clip(lower=1)
        out["charges_per_service"] = (
            out["Monthly Charges"].astype(float) / n_services_safe
        ).round(2)

    return out


# ============================================================
# 2. Service bundle counts
# ============================================================
SERVICE_COLUMNS = [
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",      # v11.1.3+
    "Unlimited Data",       # v11.1.3+
]

ADDON_COLUMNS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
]


def _is_yes(s: pd.Series) -> pd.Series:
    """Robust 'yes' detector across the variations IBM uses."""
    return (
        s.astype(str).str.strip().str.lower()
        .isin(["yes", "y", "true", "1"])
    )


def add_service_bundle(df: pd.DataFrame) -> pd.DataFrame:
    """
    n_services : count of services flagged 'Yes' across SERVICE_COLUMNS.
        Higher = more engaged, more revenue, but more surface for problems.

    n_addons : count of premium add-ons (Online Security, Online Backup,
        Device Protection, Tech Support).
        Indicates security-conscious, value-buying customers.

    has_security_bundle : binary flag if customer has all 4 add-ons.
        Strong loyalty signal in telco.
    """
    out = df.copy()

    available_services = [c for c in SERVICE_COLUMNS if c in out.columns]
    if available_services:
        flags = pd.DataFrame({c: _is_yes(out[c]) for c in available_services})
        out["n_services"] = flags.sum(axis=1).astype(int)

    available_addons = [c for c in ADDON_COLUMNS if c in out.columns]
    if available_addons:
        addon_flags = pd.DataFrame({c: _is_yes(out[c]) for c in available_addons})
        out["n_addons"] = addon_flags.sum(axis=1).astype(int)
        out["has_security_bundle"] = (
            out["n_addons"] == len(available_addons)
        ).astype(int)

    return out


# ============================================================
# 3. Tenure buckets
# ============================================================
TENURE_BUCKETS = [
    (0, 6, "0-6m"),       # newcomers — high churn risk
    (7, 24, "7-24m"),     # critical retention period
    (25, 48, "25-48m"),   # established
    (49, np.inf, "49m+"),  # loyal
]


def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    tenure_bucket : ordinal lifecycle stage from Tenure Months.
        Captures non-linear NPS patterns across the customer lifecycle.
    """
    out = df.copy()
    if "Tenure Months" not in out.columns:
        return out

    bins = [b[0] for b in TENURE_BUCKETS] + [TENURE_BUCKETS[-1][1] + 1]
    labels = [b[2] for b in TENURE_BUCKETS]
    out["tenure_bucket"] = pd.cut(
        out["Tenure Months"], bins=bins, labels=labels,
        right=False, include_lowest=True,
    )
    out["tenure_bucket"] = out["tenure_bucket"].astype(str)
    return out


# ============================================================
# 4. Digital engagement proxy
# ============================================================
def add_digital_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_paperless_autopay : binary flag — customer is on paperless billing
        AND on automatic payment.
        Strong digital engagement proxy. Such customers tend to respond
        more to surveys (cf. response propensity in Phase 3).
    """
    out = df.copy()

    paperless = (
        _is_yes(out["Paperless Billing"])
        if "Paperless Billing" in out.columns
        else pd.Series(False, index=out.index)
    )

    if "Payment Method" in out.columns:
        autopay = (
            out["Payment Method"].astype(str).str.lower()
            .str.contains("automatic", na=False)
        )
    else:
        autopay = pd.Series(False, index=out.index)

    out["is_paperless_autopay"] = (paperless & autopay).astype(int)
    return out


# ============================================================
# 5. Geographic — population density bucket
# ============================================================
def add_population_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    pop_density_bucket : ZIP population quartile (Q1=rural ... Q4=urban).
        Aggregated geography proxy that does NOT use ZIP code directly,
        avoiding the socio-economic proxying problem (cf. Phase 11 fairness).
    """
    out = df.copy()
    if "Population" not in out.columns:
        return out

    try:
        out["pop_density_bucket"] = pd.qcut(
            out["Population"], q=4,
            labels=["Q1_rural", "Q2", "Q3", "Q4_urban"],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        # not enough unique values — fall back
        out["pop_density_bucket"] = "unknown"
    return out


# ============================================================
# 6. Financial friction
# ============================================================
def add_friction_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    has_extra_charges : binary flag — customer paid extra data charges.
        Direct friction signal: hit a data cap.

    has_refund : binary flag — customer received refunds.
        Either a service issue (negative) or proactive customer care
        (positive) — let SHAP disentangle.
    """
    out = df.copy()

    if "Total Extra Data Charges" in out.columns:
        out["has_extra_charges"] = (
            out["Total Extra Data Charges"].fillna(0).astype(float) > 0
        ).astype(int)

    if "Total Refunds" in out.columns:
        out["has_refund"] = (
            out["Total Refunds"].fillna(0).astype(float) > 0
        ).astype(int)

    return out


# ============================================================
# 7. Loyalty / referrals
# ============================================================
def add_loyalty_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_referrer : binary — has referred at least one friend.
        The Phase 1 EDA showed this is the strongest loyalty signal.
    """
    out = df.copy()
    if "Number of Referrals" in out.columns:
        out["is_referrer"] = (
            out["Number of Referrals"].fillna(0).astype(int) > 0
        ).astype(int)
    return out


# ============================================================
# 8. Marketing — offer received
# ============================================================
def add_offer_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    received_offer : binary — has been offered a marketing offer.
        Useful to measure ROI of marketing actions on NPS.
        The detail (Offer A / B / ...) is kept as a categorical feature.
    """
    out = df.copy()
    if "Offer" in out.columns:
        out["received_offer"] = out["Offer"].notna().astype(int)
    return out


# ============================================================
# Public API
# ============================================================
def add_all_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply every derivation in the right order.

    Order matters: financial ratios depend on n_services from service_bundle.
    """
    out = df.copy()
    out = add_service_bundle(out)         # creates n_services, n_addons, has_security_bundle
    out = add_financial_ratios(out)       # uses n_services
    out = add_tenure_bucket(out)
    out = add_digital_engagement(out)
    out = add_population_bucket(out)
    out = add_friction_signals(out)
    out = add_loyalty_signals(out)
    out = add_offer_signal(out)
    return out


def list_derived_features() -> list[str]:
    """The exact list of columns this module adds (when all source data is present)."""
    return [
        "n_services", "n_addons", "has_security_bundle",
        "avg_monthly_charge", "charges_per_service",
        "tenure_bucket",
        "is_paperless_autopay",
        "pop_density_bucket",
        "has_extra_charges", "has_refund",
        "is_referrer",
        "received_offer",
    ]
