"""
Phase 5 invariant tests for verbatim generation pipeline.

Most tests use synthetic data; the integration tests skip if the Colab-generated
verbatims file isn't present yet.

Run with:
    pytest tests/test_phase5.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED, NPS_CLASSES
from src.verbatims.inspect import quality_audit, sample_verbatims, FORBIDDEN_WORDS
from src.verbatims.prompts import (
    SYSTEM_PROMPT,
    PromptBuildConfig,
    build_prompts,
    summarize_prompts,
)


# ============================================================
# Synthetic dataset for prompt building
# ============================================================
@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "Customer ID": [f"C{i:04d}" for i in range(n)],
        "Tenure Months": rng.integers(0, 72, n),
        "Monthly Charges": rng.uniform(20, 120, n).round(2),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "Internet Type": rng.choice(["Fiber", "DSL", "Cable", "None"], n),
        "Phone Service": rng.choice(["Yes", "No"], n),
        "Internet Service": rng.choice(["Yes", "No"], n),
        "Streaming TV": rng.choice(["Yes", "No"], n),
        "Streaming Movies": rng.choice(["Yes", "No"], n),
        "n_addons": rng.integers(0, 5, n),
        "has_extra_charges": rng.integers(0, 2, n),
        "has_refund": rng.integers(0, 2, n),
        "is_referrer": rng.integers(0, 2, n),
        "NPS_baseline": pd.Categorical(
            rng.choice(NPS_CLASSES, n), categories=NPS_CLASSES, ordered=True
        ),
    }).set_index("Customer ID")
    return df


# ============================================================
# Prompt building
# ============================================================
class TestPromptBuilding:
    def test_one_prompt_per_row(self, synthetic_df):
        prompts = build_prompts(synthetic_df)
        assert len(prompts) == len(synthetic_df)

    def test_required_columns_present(self, synthetic_df):
        prompts = build_prompts(synthetic_df)
        for col in ["system_prompt", "user_prompt", "expected_class", "counter_intuitive"]:
            assert col in prompts.columns

    def test_system_prompt_constant(self, synthetic_df):
        prompts = build_prompts(synthetic_df)
        # All rows share the same system prompt
        assert prompts["system_prompt"].nunique() == 1
        assert prompts["system_prompt"].iloc[0] == SYSTEM_PROMPT

    def test_user_prompt_varies(self, synthetic_df):
        prompts = build_prompts(synthetic_df)
        # Each user prompt must be unique-ish (high variability)
        assert prompts["user_prompt"].nunique() > 0.5 * len(prompts)

    def test_counter_intuitive_rate_close_to_target(self, synthetic_df):
        cfg = PromptBuildConfig(counter_intuitive_rate=0.15, seed=42)
        prompts = build_prompts(synthetic_df, config=cfg)
        rate = prompts["counter_intuitive"].mean()
        # Allow ±5 pts variance on n=100
        assert 0.10 <= rate <= 0.25

    def test_reproducible_same_seed(self, synthetic_df):
        cfg = PromptBuildConfig(seed=42)
        p1 = build_prompts(synthetic_df, config=cfg)
        p2 = build_prompts(synthetic_df, config=cfg)
        # Same prompts string-for-string
        assert (p1["user_prompt"] == p2["user_prompt"]).all()
        assert (p1["counter_intuitive"] == p2["counter_intuitive"]).all()

    def test_expected_class_matches_target(self, synthetic_df):
        prompts = build_prompts(synthetic_df, target_col="NPS_baseline")
        # Each prompt's expected_class should match the row's NPS_baseline
        joined = prompts.join(synthetic_df["NPS_baseline"])
        assert (joined["expected_class"] == joined["NPS_baseline"].astype(str)).all()

    def test_user_prompt_does_not_leak_class_name(self, synthetic_df):
        """The system prompt mentions the persona but the user prompt
        should NOT contain a literal 'Promoter' / 'Detractor' / 'Passive' label
        in plain factual context — it should describe the persona via traits."""
        prompts = build_prompts(synthetic_df)
        # The persona line in the user prompt is allowed to phrase the persona,
        # but the words should never appear as standalone labels (the prompt is
        # designed not to use them, see src/verbatims/prompts.py).
        # We assert no row has any of these forbidden NPS keywords.
        forbidden = ["Detractor", "Passive", "Promoter"]
        for word in forbidden:
            count = prompts["user_prompt"].str.contains(word, regex=False).sum()
            assert count == 0, (
                f"User prompt leaks the literal label '{word}' in {count} rows"
            )


# ============================================================
# Prompt summary
# ============================================================
class TestSummary:
    def test_summary_has_one_row_per_class(self, synthetic_df):
        prompts = build_prompts(synthetic_df)
        summary = summarize_prompts(prompts)
        assert len(summary) == prompts["expected_class"].nunique()

    def test_summary_columns(self, synthetic_df):
        prompts = build_prompts(synthetic_df)
        summary = summarize_prompts(prompts)
        for col in ["count", "ci_count", "ci_rate"]:
            assert col in summary.columns


# ============================================================
# Quality audit (uses fake verbatims)
# ============================================================
class TestQualityAudit:
    @pytest.fixture
    def fake_verbatims_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "verbatim": [
                "I'm really happy with the service, my internet is super fast.",
                "Honestly, the customer support was terrible last week.",
                "It's fine I guess, nothing to complain about.",
                "Service is OK overall, no major issues.",
            ],
            "expected_class": ["Promoter", "Detractor", "Passive", "Passive"],
            "counter_intuitive": [False, False, False, False],
        })

    def test_audit_returns_expected_keys(self, fake_verbatims_df):
        report = quality_audit(fake_verbatims_df)
        for key in ["total", "char_length", "word_length", "forbidden_words", "duplicates"]:
            assert key in report

    def test_audit_counts(self, fake_verbatims_df):
        report = quality_audit(fake_verbatims_df)
        assert report["total"] == 4
        assert report["duplicates"] == 0

    def test_audit_detects_forbidden_words(self):
        bad_df = pd.DataFrame({
            "verbatim": ["I am a Detractor and that's that."],
            "expected_class": ["Detractor"],
        })
        report = quality_audit(bad_df)
        assert "detractor" in report["forbidden_words"]

    def test_sample_verbatims_returns_per_class(self, fake_verbatims_df):
        samples = sample_verbatims(fake_verbatims_df, n_per_class=1)
        # Has one per class present
        classes_in_samples = set(samples["expected_class"].unique())
        assert len(classes_in_samples) > 0


# ============================================================
# Integration with real Colab output (skipped if missing)
# ============================================================
class TestColabIntegration:
    @pytest.fixture
    def real_verbatims(self):
        path = DATA_PROCESSED / "dataset_with_verbatims.parquet"
        if not path.exists():
            pytest.skip(f"{path} missing — generate verbatims on Colab first.")
        return pd.read_parquet(path)

    def test_one_verbatim_per_customer(self, real_verbatims):
        # Allow up to 5 % missing (Colab might fail on a few prompts)
        n_missing = real_verbatims["verbatim"].isna().sum()
        assert n_missing < 0.05 * len(real_verbatims)

    def test_no_forbidden_words_in_real_output(self, real_verbatims):
        text = real_verbatims["verbatim"].dropna().str.lower()
        for w in FORBIDDEN_WORDS:
            n = text.str.contains(w, regex=False).sum()
            # Allow rare slips from the model
            assert n / len(text) < 0.01, (
                f"'{w}' appears in {n / len(text):.2%} of verbatims (>1 %)"
            )

    def test_realistic_length_distribution(self, real_verbatims):
        lens = real_verbatims["verbatim"].dropna().str.len()
        # 1-3 sentences ≈ 50-300 chars roughly
        assert lens.median() > 30
        assert lens.median() < 500
