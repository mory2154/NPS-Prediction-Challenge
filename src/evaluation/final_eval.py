"""
Phase 9 — Final evaluation orchestrator.

Consolidates the 2 champions identified in Phases 7-8:

    C1 (QWK ordinal champion)   = hybrid  / lightgbm / pca32
    C2 (production-safe)         = tuned   / logistic / tabular

Produces the canonical Phase 9 deliverables:

    models/results/final_eval_summary.parquet
        — one row per (champion × split × metric) with bootstrap CI

    models/results/final_eval_calibration.parquet
        — one row per (champion × class × bin) for reliability curves

    reports/final_eval_report.md
        — executive summary ready to paste into the final write-up

The silent_test is reported as the canonical headline number. The other 3
splits (train, val, respondent_test) are reported for diagnostics:
    * train  → confirms there is no critical underfit
    * val    → consistency check with Optuna optimization target
    * respondent_test → covariate-shift control vs silent_test

CLI
---
    python -m src.evaluation.final_eval
    python -m src.evaluation.final_eval --n-resamples 500   # faster
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DEFAULT_TARGET,
    MODELS_DIR,
    NPS_CLASS_TO_INT,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_DIR,
)
from src.data.split import load_splits
from src.evaluation.bootstrap import bootstrap_all_metrics
from src.evaluation.calibration import calibration_report
from src.features.embeddings import load_or_compute_embeddings
from src.models.tuning_hybrid import VERBATIM_AUX_COLS, _build_split_matrices

# ============================================================
# Champion registry — frozen from Phase 8
# ============================================================
CHAMPIONS = {
    "C1_qwk": {
        "label":          "C1 — QWK champion",
        "phase":          "hybrid",
        "model":          "lightgbm",
        "feature_space":  "pca32",
        "path":           MODELS_DIR / "hybrid" / "lightgbm_pca32.joblib",
        "uses_text":      True,
        "narrative": (
            "Maximum ordinal accuracy. Uses synthetic verbatim embeddings — "
            "the +0.30 QWK gain vs C2 reflects the LLM-generated verbatims' "
            "fidelity to the NPS class, not real-world predictive value."
        ),
    },
    "C2_safe": {
        "label":          "C2 — Production-safe",
        "phase":          "tuned",
        "model":          "logistic",
        "feature_space":  "tabular",
        "path":           MODELS_DIR / "tuned" / "logistic_tuned.joblib",
        "uses_text":      False,
        "narrative": (
            "Deployable today on tabular signals only. No leakage concern. "
            "Best Detractor recall (0.840) — the metric the retention "
            "manager actually optimises."
        ),
    },
}


# ============================================================
# Helpers — build X for a given champion on a given split
# ============================================================
def _build_X_for_split(
    champion_key: str,
    split_name: str,
    df: pd.DataFrame,
    splits: pd.Series,
    pipeline,
    embeddings_df: pd.DataFrame | None,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y_int) for one champion on one split.

    For C1 (hybrid) we reuse Phase 8's `_build_split_matrices` so that
    PCA is fit on TRAIN only — same code path as during tuning.
    """
    champ = CHAMPIONS[champion_key]
    fs = champ["feature_space"]

    if champ["uses_text"]:
        assert embeddings_df is not None
        feature_cols = [
            c for c in df.columns
            if not c.startswith("NPS_") and c not in VERBATIM_AUX_COLS
        ]
        X_per, y_per, _ = _build_split_matrices(
            df=df, splits=splits, pipeline=pipeline,
            embeddings_df=embeddings_df,
            target_col=target_col, feature_cols=feature_cols,
            feature_space=fs, verbose=False,
        )
        return X_per[split_name], y_per[split_name]
    else:
        # Tab-only: drop verbatim aux cols if present (defensive)
        feature_cols = [
            c for c in df.columns
            if not c.startswith("NPS_") and c not in VERBATIM_AUX_COLS
        ]
        mask = splits == split_name
        X_tab = pipeline.transform(df.loc[mask, feature_cols])
        y_int = np.array(
            [NPS_CLASS_TO_INT[str(v)]
             for v in df.loc[mask, target_col].to_numpy()],
            dtype=int,
        )
        return X_tab, y_int


# ============================================================
# Main orchestrator
# ============================================================
def run_final_eval(
    n_resamples: int = 1000,
    ci: float = 0.95,
    target_col: str = DEFAULT_TARGET,
    verbose: bool = True,
) -> dict:
    if not target_col.startswith("NPS_"):
        target_col = f"NPS_{target_col}"

    if verbose:
        print("=" * 70)
        print("PHASE 9 — FINAL EVALUATION")
        print("=" * 70)
        print(f"Target           : {target_col}")
        print(f"Bootstrap n      : {n_resamples}")
        print(f"CI               : {ci:.0%}")
        print(f"Random seed      : {RANDOM_SEED}")
        for k, c in CHAMPIONS.items():
            print(f"\n  {c['label']}")
            print(f"    {c['phase']} / {c['model']} / {c['feature_space']}")
            print(f"    uses_text = {c['uses_text']}")

    # Load assets
    df_path = DATA_PROCESSED / "dataset_with_verbatims.parquet"
    if not df_path.exists():
        raise FileNotFoundError(f"{df_path} — run `make load-verbatims`")

    df = pd.read_parquet(df_path)
    splits = load_splits("response_biased")
    pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")

    # Embeddings needed only if at least one champion uses text
    embeddings_df = None
    if any(c["uses_text"] for c in CHAMPIONS.values()):
        embeddings_df = load_or_compute_embeddings(df, verbose=False)
        # Align (defensive)
        common = df.index.intersection(embeddings_df.index)
        df = df.loc[common]
        splits = splits.loc[common]
        embeddings_df = embeddings_df.loc[common]

    # Verify champions exist on disk
    missing = [k for k, c in CHAMPIONS.items() if not c["path"].exists()]
    if missing:
        raise FileNotFoundError(
            f"Champion artifact(s) missing: {missing}. "
            "Run Phases 7 and 8 first (`make tune` and `make tune-hybrid`)."
        )

    # Containers
    summary_rows: list[dict] = []
    calibration_rows: list[dict] = []
    calibration_summary: list[dict] = []

    # ============================================================
    # Loop on champions × splits
    # ============================================================
    for ck, champ in CHAMPIONS.items():
        if verbose:
            print(f"\n{'=' * 70}\n{champ['label']}\n{'=' * 70}")
        model = joblib.load(champ["path"])

        for split_name in ["train", "val", "respondent_test", "silent_test"]:
            if split_name not in splits.values:
                continue

            X, y_int = _build_X_for_split(
                champion_key=ck,
                split_name=split_name,
                df=df, splits=splits, pipeline=pipeline,
                embeddings_df=embeddings_df,
                target_col=target_col,
            )
            y_pred = model.predict(X)
            try:
                y_proba = model.predict_proba(X)
            except Exception:
                y_proba = None

            # Bootstrap CIs on all metrics (including lift if y_proba)
            metrics_df = bootstrap_all_metrics(
                y_true=y_int,
                y_pred=y_pred,
                y_proba=y_proba,
                n_resamples=n_resamples,
                ci=ci,
                random_state=RANDOM_SEED,
            )
            metrics_df.insert(0, "split", split_name)
            metrics_df.insert(0, "champion", ck)
            metrics_df["champion_label"] = champ["label"]
            metrics_df["n_split"] = int(len(y_int))
            summary_rows.append(metrics_df)

            if verbose:
                head = metrics_df[metrics_df["metric"].isin(
                    ["qwk", "detractor_recall", "macro_f1"]
                )]
                print(f"\n  [{split_name}] (n={len(y_int)})")
                for _, r in head.iterrows():
                    print(
                        f"    {r['metric']:<18} = {r['value']:.4f}  "
                        f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
                    )

            # Calibration only on silent_test (the canonical reporting split)
            if split_name == "silent_test" and y_proba is not None:
                cal = calibration_report(y_int, y_proba, n_bins=10)
                for cls, curve_df in cal["curves"].items():
                    tmp = curve_df.copy()
                    tmp["champion"] = ck
                    tmp["champion_label"] = champ["label"]
                    tmp["class"] = cls
                    calibration_rows.append(tmp)
                for cls in cal["brier"]:
                    calibration_summary.append({
                        "champion": ck,
                        "champion_label": champ["label"],
                        "class": cls,
                        "brier": cal["brier"][cls],
                        "ece":   cal["ece"][cls],
                    })
                if verbose:
                    print(f"\n    Calibration silent_test:")
                    for cls in cal["brier"]:
                        print(
                            f"      {cls:<10} Brier = {cal['brier'][cls]:.4f}  "
                            f"ECE = {cal['ece'][cls]:.4f}"
                        )

    # ============================================================
    # Persist results
    # ============================================================
    summary_df = pd.concat(summary_rows, ignore_index=True)
    out_summary = RESULTS_DIR / "final_eval_summary.parquet"
    summary_df.to_parquet(out_summary, index=False)
    if verbose:
        print(f"\n✓ {out_summary.name}  ({len(summary_df)} rows)")

    if calibration_rows:
        calib_df = pd.concat(calibration_rows, ignore_index=True)
        out_calib = RESULTS_DIR / "final_eval_calibration.parquet"
        calib_df.to_parquet(out_calib, index=False)
        if verbose:
            print(f"✓ {out_calib.name}  ({len(calib_df)} rows)")

    if calibration_summary:
        calib_sum = pd.DataFrame(calibration_summary)
        out_calib_sum = RESULTS_DIR / "final_eval_calibration_summary.parquet"
        calib_sum.to_parquet(out_calib_sum, index=False)
        if verbose:
            print(f"✓ {out_calib_sum.name}  ({len(calib_sum)} rows)")
    else:
        calib_sum = pd.DataFrame()

    # ============================================================
    # Write executive markdown report
    # ============================================================
    md = _build_markdown_report(
        summary_df=summary_df,
        calibration_summary=calib_sum,
        target_col=target_col,
        n_resamples=n_resamples,
        ci=ci,
    )
    out_md = REPORTS_DIR / "final_eval_report.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    if verbose:
        print(f"✓ {out_md}")

    return {
        "summary_df": summary_df,
        "calibration_summary": calib_sum,
        "report_path": out_md,
    }


# ============================================================
# Markdown report builder
# ============================================================
def _fmt_ci(v: float, lo: float, hi: float) -> str:
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def _build_markdown_report(
    summary_df: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    target_col: str,
    n_resamples: int,
    ci: float,
) -> str:
    lines: list[str] = []
    lines.append(f"# Phase 9 — Final Evaluation Report")
    lines.append("")
    lines.append(
        f"Target: `{target_col}` · Bootstrap n = {n_resamples} · CI = {ci:.0%} · "
        f"Seed = {RANDOM_SEED}"
    )
    lines.append("")

    # ----- Headline table (silent_test only) -----
    lines.append("## Headline — silent_test (n_silent = "
                 f"{int(summary_df[summary_df['split'] == 'silent_test']['n_split'].iloc[0])})")
    lines.append("")
    lines.append(
        "| Champion | QWK | Detractor recall | Macro F1 | "
        "Lift@10 | Lift@20 |"
    )
    lines.append("|---|---|---|---|---|---|")
    silent = summary_df[summary_df["split"] == "silent_test"]
    for ck in ["C1_qwk", "C2_safe"]:
        sub = silent[silent["champion"] == ck]
        if sub.empty:
            continue
        row = {r["metric"]: r for _, r in sub.iterrows()}
        label = sub["champion_label"].iloc[0]
        line = f"| **{label}** "
        for m in ["qwk", "detractor_recall", "macro_f1", "lift@10", "lift@20"]:
            if m in row:
                line += f"| {_fmt_ci(row[m]['value'], row[m]['ci_lo'], row[m]['ci_hi'])} "
            else:
                line += "| — "
        line += "|"
        lines.append(line)
    lines.append("")

    # ----- Per-split diagnostics -----
    lines.append("## Per-split diagnostics (QWK only)")
    lines.append("")
    lines.append("| Split | C1 — QWK champion | C2 — Production-safe |")
    lines.append("|---|---|---|")
    for split in ["train", "val", "respondent_test", "silent_test"]:
        row = {"split": split}
        for ck in ["C1_qwk", "C2_safe"]:
            sub = summary_df[
                (summary_df["split"] == split)
                & (summary_df["champion"] == ck)
                & (summary_df["metric"] == "qwk")
            ]
            if sub.empty:
                row[ck] = "—"
            else:
                r = sub.iloc[0]
                row[ck] = _fmt_ci(r["value"], r["ci_lo"], r["ci_hi"])
        lines.append(f"| {row['split']} | {row['C1_qwk']} | {row['C2_safe']} |")
    lines.append("")

    # ----- Covariate-shift control -----
    lines.append("## Covariate-shift control (respondent_test vs silent_test)")
    lines.append("")
    for ck in ["C1_qwk", "C2_safe"]:
        resp = summary_df[
            (summary_df["split"] == "respondent_test")
            & (summary_df["champion"] == ck)
            & (summary_df["metric"] == "qwk")
        ]
        sil = summary_df[
            (summary_df["split"] == "silent_test")
            & (summary_df["champion"] == ck)
            & (summary_df["metric"] == "qwk")
        ]
        if resp.empty or sil.empty:
            continue
        d = sil["value"].iloc[0] - resp["value"].iloc[0]
        label = sub["champion_label"].iloc[0] if not sub.empty else ck
        verdict = "low shift" if abs(d) < 0.05 else "non-trivial shift"
        lines.append(
            f"- **{ck}** — Δ QWK (silent − respondent) = {d:+.3f}  →  *{verdict}*."
        )
    lines.append("")
    lines.append(
        "If both deltas are small (<0.05), the response-biased split design "
        "successfully simulates the silent population: the model generalises."
    )
    lines.append("")

    # ----- Calibration -----
    if not calibration_summary.empty:
        lines.append("## Calibration on silent_test")
        lines.append("")
        lines.append("Brier score per class (lower = better, 0 = perfect).")
        lines.append("")
        lines.append("| Champion | Detractor | Passive | Promoter |")
        lines.append("|---|---|---|---|")
        for ck in ["C1_qwk", "C2_safe"]:
            row = calibration_summary[calibration_summary["champion"] == ck]
            if row.empty:
                continue
            d = {r["class"]: r["brier"] for _, r in row.iterrows()}
            label = row["champion_label"].iloc[0]
            lines.append(
                f"| **{label}** | "
                f"{d.get('Detractor', float('nan')):.3f} | "
                f"{d.get('Passive', float('nan')):.3f} | "
                f"{d.get('Promoter', float('nan')):.3f} |"
            )
        lines.append("")
        lines.append(
            "Expected Calibration Error (ECE, weighted |mean predicted − "
            "observed rate|):"
        )
        lines.append("")
        lines.append("| Champion | Detractor | Passive | Promoter |")
        lines.append("|---|---|---|---|")
        for ck in ["C1_qwk", "C2_safe"]:
            row = calibration_summary[calibration_summary["champion"] == ck]
            if row.empty:
                continue
            d = {r["class"]: r["ece"] for _, r in row.iterrows()}
            label = row["champion_label"].iloc[0]
            lines.append(
                f"| **{label}** | "
                f"{d.get('Detractor', float('nan')):.3f} | "
                f"{d.get('Passive', float('nan')):.3f} | "
                f"{d.get('Promoter', float('nan')):.3f} |"
            )
        lines.append("")

    # ----- Verdict -----
    lines.append("## Verdict")
    lines.append("")
    for ck, c in CHAMPIONS.items():
        lines.append(f"### {c['label']}")
        lines.append(f"`{c['phase']} / {c['model']} / {c['feature_space']}`")
        lines.append("")
        lines.append(c["narrative"])
        lines.append("")

    lines.append("## Methodological caveats")
    lines.append("")
    lines.append(
        "* C1 uses verbatim embeddings. The verbatims were synthetically "
        "generated by Qwen2.5-7B-Instruct from customer features + Sat Score "
        "(15% counter-intuitive cases). Its QWK gain vs C2 should be read as "
        "a **performance upper bound**, not an estimate of real-world value "
        "for genuine customer verbatims."
    )
    lines.append(
        "* C2 has no leakage concern: tabular features only, satisfaction-derived "
        "leaky columns audited and dropped in Phase 1. Its numbers are a fair "
        "estimate of production performance."
    )
    lines.append(
        "* The silent_test was never used for tuning or model selection — every "
        "hyperparameter choice was made on val. The numbers above are an "
        "**unbiased** held-out estimate."
    )
    lines.append("")
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()

    try:
        run_final_eval(
            n_resamples=args.n_resamples,
            ci=args.ci,
            target_col=args.target,
            verbose=True,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
