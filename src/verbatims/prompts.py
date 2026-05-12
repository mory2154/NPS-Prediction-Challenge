"""
Build prompts conditioned on each customer's tabular features.

Generates one prompt per customer that asks the LLM to produce a 1-3 sentence
"last interaction note" — what the customer said or wrote during their last
contact with customer support.

The prompt encodes:
    * The expected NPS class (Detractor / Passive / Promoter)
    * Tenure, contract, services, charges as factual context
    * **Imperfect correlation** : we deliberately ask for ~15 % counter-intuitive
      cases (a Detractor who sounds calm, a Promoter who has a minor complaint),
      to mimic real-world noise and avoid trivial label↔text correlation.

Reproducibility: the prompt set is deterministic given the random seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED

SYSTEM_PROMPT = """You are simulating customer feedback for a US telecom company.
Your task is to generate ONE short, realistic 'last interaction note' (1 to 3 sentences)
that the customer said during their last contact with customer support, on a chat,
or in an app review.

Rules:
  - Stay in character. A Detractor sounds frustrated. A Promoter sounds satisfied.
    A Passive is matter-of-fact, neither hot nor cold.
  - When the persona is COUNTER-INTUITIVE, blend signals: a Detractor who sounds
    calm but mentions a real issue, a Promoter who is happy but flags a small annoyance.
  - Use natural conversational English — no formal sign-offs, no "Dear sir".
  - NEVER mention the words "Detractor", "Passive", "Promoter", "NPS", or
    "satisfaction score" in the output. Just the customer voice.
  - 1 to 3 sentences. No emoji. No quotes around the message.

Output ONLY the message text, nothing else."""

# Templates that get filled with customer-specific facts.
USER_TEMPLATE = """Customer profile:
- Tenure: {tenure} months ({tenure_phase})
- Contract: {contract}
- Internet: {internet}
- Total monthly charges: ~${monthly_charges:.0f}
- Services bundle: {services_summary}
- Recent friction: {friction_summary}
- Has referred friends: {has_referrer}

Persona to write as: {persona}

Generate the message now (1-3 sentences, customer voice only)."""


@dataclass
class PromptBuildConfig:
    """Knobs for prompt construction."""
    counter_intuitive_rate: float = 0.15  # ~15 % noisy cases, per brief
    seed: int = RANDOM_SEED


# ============================================================
# Helpers
# ============================================================
def _tenure_phase(months: float) -> str:
    if months <= 6:
        return "newly joined"
    if months <= 24:
        return "early-stage relationship"
    if months <= 48:
        return "established customer"
    return "long-time loyal customer"


def _services_summary(row: pd.Series) -> str:
    """Compose a short summary like 'Phone + Internet + 2 add-ons'."""
    parts: list[str] = []
    if str(row.get("Phone Service", "No")).lower() == "yes":
        parts.append("Phone")
    if str(row.get("Internet Service", "No")).lower() == "yes":
        parts.append("Internet")
    n_addons = int(row.get("n_addons", 0))
    if n_addons > 0:
        parts.append(f"{n_addons} add-on{'s' if n_addons > 1 else ''}")
    streaming = []
    if str(row.get("Streaming TV", "No")).lower() == "yes":
        streaming.append("TV")
    if str(row.get("Streaming Movies", "No")).lower() == "yes":
        streaming.append("Movies")
    if streaming:
        parts.append("Streaming " + "+".join(streaming))
    return " + ".join(parts) if parts else "minimal services"


def _friction_summary(row: pd.Series) -> str:
    """Summarise customer's recent friction signals (or 'none')."""
    points: list[str] = []
    if int(row.get("has_extra_charges", 0)) == 1:
        points.append("extra data charges")
    if int(row.get("has_refund", 0)) == 1:
        points.append("had a refund recently")
    internet_type = str(row.get("Internet Type", "None"))
    if internet_type.lower() == "fiber":
        points.append("fiber service (premium)")
    if not points:
        return "no recent issues flagged"
    return ", ".join(points)


def _persona_for_class(nps_class: str, counter_intuitive: bool) -> str:
    """
    Compose a one-line persona description matching the target NPS class,
    optionally with a counter-intuitive twist.
    """
    if not counter_intuitive:
        return {
            "Detractor": "frustrated customer who is unhappy with the service",
            "Passive": "neutral customer, neither very happy nor angry",
            "Promoter": "happy and satisfied customer who would recommend the brand",
        }[nps_class]

    # Counter-intuitive: mix the dominant signal
    return {
        "Detractor": "customer who is calm and polite on the surface but actually disappointed",
        "Passive": "customer with mixed feelings — one strong positive AND one real annoyance",
        "Promoter": "loyal customer who is overall happy but flags one small annoyance",
    }[nps_class]


# ============================================================
# Public API
# ============================================================
def build_prompts(
    df: pd.DataFrame,
    target_col: str = "NPS_baseline",
    config: PromptBuildConfig | None = None,
) -> pd.DataFrame:
    """
    For each row in df, build the (system_prompt, user_prompt, persona, is_counter_intuitive)
    tuple. Returns a DataFrame indexed like df.

    Required columns in df:
        Tenure Months, Contract, Internet Type, Monthly Charges,
        Phone Service, Internet Service, Streaming TV, Streaming Movies,
        n_addons, has_extra_charges, has_refund, is_referrer, NPS_baseline
    """
    config = config or PromptBuildConfig()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from df")

    rng = np.random.default_rng(config.seed)
    is_ci = rng.random(len(df)) < config.counter_intuitive_rate

    rows: list[dict] = []
    for (idx, row), counter_intuitive in zip(df.iterrows(), is_ci):
        nps_class = str(row[target_col])
        tenure = float(row.get("Tenure Months", 0))

        user_prompt = USER_TEMPLATE.format(
            tenure=int(tenure),
            tenure_phase=_tenure_phase(tenure),
            contract=str(row.get("Contract", "month-to-month")),
            internet=str(row.get("Internet Type", "None")),
            monthly_charges=float(row.get("Monthly Charges", 0.0)),
            services_summary=_services_summary(row),
            friction_summary=_friction_summary(row),
            has_referrer=("yes" if int(row.get("is_referrer", 0)) == 1 else "no"),
            persona=_persona_for_class(nps_class, bool(counter_intuitive)),
        )

        rows.append({
            "Customer ID": idx,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "expected_class": nps_class,
            "counter_intuitive": bool(counter_intuitive),
        })

    return pd.DataFrame(rows).set_index("Customer ID")


# ============================================================
# Diagnostics
# ============================================================
def summarize_prompts(prompts_df: pd.DataFrame) -> pd.DataFrame:
    """Quick stats on the generated prompts."""
    summary = pd.DataFrame({
        "count": prompts_df.groupby("expected_class").size(),
        "ci_count": prompts_df.groupby("expected_class")["counter_intuitive"].sum(),
        "user_prompt_avg_chars": prompts_df.groupby("expected_class")["user_prompt"].apply(
            lambda s: int(s.str.len().mean())
        ),
    })
    summary["ci_rate"] = (summary["ci_count"] / summary["count"]).round(3)
    return summary
