"""
Phase 11 — Fairness audit orchestrator.

Audits both Phase 9 champions on the 3 protected segments declared in the
brief 4.7: {Senior, Gender, Married}.

What is computed (per champion × segment):
    1. **Per-group breakdown** with bootstrap CIs (Detractor recall + Promoter
       recall + selection rates).
    2. **Disparate Impact ratio** for Detractor and Promoter (4/5 rule).
    3. **Equal Opportunity Difference** for Detractor and Promoter.
    4. **Demographic Parity Difference** for Detractor predictions.
    5. **Counterfactual swap analysis** — flip the protected attribute on
       silent_test (Senior=Yes ↔ No, etc.), re-predict, measure how many
       customers see their predicted class change. The lower the rate, the
       less the model "directly causally uses" the attribute. Any remaining
       disparity in (1)-(4) is then through *proxy variables*, not direct use.

Outputs
-------
    models/results/fairness_per_group.parquet           (~108 rows)
    models/results/fairness_disparities.parquet         (~36 rows)
    models/results/fairness_counterfactual.parquet      (~12 rows)
    reports/fairness_audit.md                            (executive summary)

CLI
---
    python -m src.fairness.audit
    python -m src.fairness.audit --n-resamples 500
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DEFAULT_TARGET,
    MODELS_DIR,
    NPS_CLASS_TO_INT,
    NPS_CLASSES,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_DIR,
)
from src.data.split import load_splits
from src.evaluation.metrics import _to_int_labels
from src.features.embeddings import load_or_compute_embeddings
from src.fairness.bootstrap_per_group import per_group_breakdown
from src.fairness.metrics import (
    demographic_parity_difference,
    disparate_impact,
    equal_opportunity_difference,
)
from src.models.tuning_hybrid import VERBATIM_AUX_COLS, _build_split_matrices


# ============================================================
# Configuration
# ============================================================
CHAMPIONS = {
    "C1_qwk": {
        "label":         "C1 — QWK champion",
        "phase":         "hybrid",
        "model":         "lightgbm",
        "feature_space": "pca32",
        "path":          MODELS_DIR / "hybrid" / "lightgbm_pca32.joblib",
        "uses_text":     True,
    },
    "C2_safe": {
        "label":         "C2 — Production-safe",
        "phase":         "tuned",
        "model":         "logistic",
        "feature_space": "tabular",
        "path":          MODELS_DIR / "tuned" / "logistic_tuned.joblib",
        "uses_text":     False,
    },
}

SEGMENTS = {
    "Senior":  "Senior Citizen",
    "Gender":  "Gender",
    "Married": "Married",
}

CLASSES_TO_AUDIT = [
    NPS_CLASS_TO_INT["Detractor"],
    NPS_CLASS_TO_INT["Promoter"],
]


# ============================================================
# Build X for one champion on silent_test (reuses Phase 9 helper logic)
# ============================================================
def _build_X_for_champion(
    champion_key: str,
    df: pd.DataFrame,
    splits: pd.Series,
    pipeline,
    embeddings_df,
    target_col: str,
    split_name: str = "silent_test",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns (X, y_int, df_split) for the requested champion on the split."""
    champ = CHAMPIONS[champion_key]
    feature_cols = [
        c for c in df.columns
        if not c.startswith("NPS_") and c not in VERBATIM_AUX_COLS
    ]
    if champ["uses_text"]:
        X_per, y_per, _ = _build_split_matrices(
            df=df, splits=splits, pipeline=pipeline,
            embeddings_df=embeddings_df,
            target_col=target_col, feature_cols=feature_cols,
            feature_space=champ["feature_space"], verbose=False,
        )
        X = X_per[split_name]
        y = y_per[split_name]
    else:
        mask = splits == split_name
        X = pipeline.transform(df.loc[mask, feature_cols])
        y = np.array(
            [NPS_CLASS_TO_INT[str(v)] for v in df.loc[mask, target_col].to_numpy()],
            dtype=int,
        )
    df_split = df.loc[splits == split_name]
    return X, y, df_split


# ============================================================
# Counterfactual swap analysis
# ============================================================
def counterfactual_flip(
    champion_key: str,
    segment_col: str,
    df: pd.DataFrame,
    splits: pd.Series,
    pipeline,
    embeddings_df,
    target_col: str,
) -> dict:
    """
    Flip the protected attribute on silent_test and measure how many predictions
    change. A model that "truly" doesn't use the attribute will have ~0 changes.

    Strategy:
    * For binary segments (Yes/No, Male/Female), swap each value to its opposite.
    * Re-build features (pipeline.transform → potentially PCA via _build_split_matrices).
    * Re-predict, compare to original predictions.

    Returns
    -------
    {
        "segment":             str,
        "n":                   int,
        "n_changed":           int,
        "change_rate":         float,
        "detractor_to_other":  int,
        "other_to_detractor":  int,
        "promoter_to_other":   int,
        "other_to_promoter":   int,
    }
    """
    champ = CHAMPIONS[champion_key]
    model = joblib.load(champ["path"])

    # Original predictions
    X_orig, y_int, df_silent = _build_X_for_champion(
        champion_key, df, splits, pipeline, embeddings_df, target_col,
    )
    y_pred_orig = model.predict(X_orig)

    if segment_col not in df_silent.columns:
        return {"segment": segment_col, "error": "column missing"}

    # Build the flipped dataframe (limited to silent_test rows)
    df_flip = df.copy()
    mask_silent = splits == "silent_test"
    series = df_flip.loc[mask_silent, segment_col]

    unique_vals = sorted(series.dropna().unique())
    if len(unique_vals) != 2:
        # Skip non-binary or empty segments
        return {
            "segment":            segment_col,
            "n":                  int(mask_silent.sum()),
            "n_changed":          0,
            "change_rate":        0.0,
            "skip_reason":        f"segment has {len(unique_vals)} values (need 2)",
        }
    v0, v1 = unique_vals
    flipped = series.where(series != v0, "__TMP__").replace({v1: v0, "__TMP__": v1})
    df_flip.loc[mask_silent, segment_col] = flipped

    # Recompute features on the flipped dataframe — re-fit pipeline transform
    # (pipeline is FIT once; .transform applies the stored encoding consistently)
    X_flip, _, _ = _build_X_for_champion(
        champion_key, df_flip, splits, pipeline, embeddings_df, target_col,
    )
    y_pred_flip = model.predict(X_flip)

    n = len(y_pred_orig)
    n_changed = int((y_pred_orig != y_pred_flip).sum())

    # Class-flip breakdown for retention-relevant transitions
    det_idx = NPS_CLASS_TO_INT["Detractor"]
    pro_idx = NPS_CLASS_TO_INT["Promoter"]
    det_to_other = int(((y_pred_orig == det_idx) & (y_pred_flip != det_idx)).sum())
    other_to_det = int(((y_pred_orig != det_idx) & (y_pred_flip == det_idx)).sum())
    pro_to_other = int(((y_pred_orig == pro_idx) & (y_pred_flip != pro_idx)).sum())
    other_to_pro = int(((y_pred_orig != pro_idx) & (y_pred_flip == pro_idx)).sum())

    return {
        "segment":             segment_col,
        "n":                   n,
        "n_changed":           n_changed,
        "change_rate":         n_changed / n if n else 0.0,
        "detractor_to_other":  det_to_other,
        "other_to_detractor":  other_to_det,
        "promoter_to_other":   pro_to_other,
        "other_to_promoter":   other_to_pro,
    }


# ============================================================
# Main runner
# ============================================================
def _normalize_value(v):
    if isinstance(v, (int, np.integer)):
        return "Yes" if v == 1 else "No"
    return str(v)


def run_fairness_audit(
    n_resamples: int = 1000,
    ci: float = 0.95,
    target_col: str = DEFAULT_TARGET,
    do_counterfactual: bool = True,
    verbose: bool = True,
) -> dict:
    if not target_col.startswith("NPS_"):
        target_col = f"NPS_{target_col}"

    if verbose:
        print("=" * 70)
        print("PHASE 11 — FAIRNESS AUDIT")
        print("=" * 70)
        print(f"Target           : {target_col}")
        print(f"Bootstrap n      : {n_resamples}, CI {ci:.0%}")
        print(f"Champions        : {list(CHAMPIONS)}")
        print(f"Segments         : {list(SEGMENTS)}")
        print(f"Classes audited  : Detractor + Promoter")
        print(f"Counterfactual   : {do_counterfactual}")

    # Load data once
    df = pd.read_parquet(DATA_PROCESSED / "dataset_with_verbatims.parquet")
    splits = load_splits("response_biased")
    pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
    embeddings_df = load_or_compute_embeddings(df, verbose=False)
    common = df.index.intersection(embeddings_df.index)
    df = df.loc[common]; splits = splits.loc[common]
    embeddings_df = embeddings_df.loc[common]

    # Verify champions
    missing = [k for k, c in CHAMPIONS.items() if not c["path"].exists()]
    if missing:
        raise FileNotFoundError(
            f"Champion artifact(s) missing: {missing}. "
            "Run Phases 7 and 8 first."
        )

    per_group_rows: list[pd.DataFrame] = []
    disparity_rows: list[dict] = []
    cf_rows: list[dict] = []

    # ========================================================
    # Loop on champions × segments
    # ========================================================
    for ck, champ in CHAMPIONS.items():
        if verbose:
            print(f"\n{'=' * 70}\n{champ['label']}\n{'=' * 70}")
        model = joblib.load(champ["path"])
        X, y_int, df_silent = _build_X_for_champion(
            ck, df, splits, pipeline, embeddings_df, target_col,
        )
        y_pred = model.predict(X)

        for seg_label, col_name in SEGMENTS.items():
            if col_name not in df_silent.columns:
                if verbose:
                    print(f"  ⚠ segment '{seg_label}' ({col_name}) not in dataset, skip")
                continue
            groups_raw = df_silent[col_name].apply(_normalize_value).to_numpy()

            if verbose:
                print(f"\n  [{seg_label}]")
                vc = pd.Series(groups_raw).value_counts().to_dict()
                print(f"    distribution: {vc}")

            # 1) Per-group breakdown with bootstrap CIs
            pgb = per_group_breakdown(
                y_int, y_pred, groups_raw,
                class_indices=CLASSES_TO_AUDIT,
                n_resamples=n_resamples, ci=ci, seed=RANDOM_SEED,
            )
            pgb.insert(0, "segment", seg_label)
            pgb.insert(0, "champion", ck)
            pgb["champion_label"] = champ["label"]
            per_group_rows.append(pgb)

            # 2) Disparity indicators
            for class_idx in CLASSES_TO_AUDIT:
                cls_name = NPS_CLASSES[class_idx]
                di = disparate_impact(y_int, y_pred, groups_raw, class_idx)
                eod = equal_opportunity_difference(
                    y_int, y_pred, groups_raw, class_idx,
                )
                dpd = demographic_parity_difference(y_pred, groups_raw, class_idx)

                # Pick the "worst" DI cell (furthest from 1.0)
                di_dist = (di["DI"] - 1.0).abs()
                worst_row = di.loc[di_dist.idxmax()]

                disparity_rows.append({
                    "champion":              ck,
                    "champion_label":        champ["label"],
                    "segment":               seg_label,
                    "class":                 cls_name,
                    "DI_worst":              float(worst_row["DI"]),
                    "DI_worst_group":        str(worst_row["group"]),
                    "DI_reference_group":    str(worst_row["reference_group"]),
                    "DI_status":             str(worst_row["DI_status"]),
                    "EOD":                   float(eod["max_diff"]) if eod["max_diff"] == eod["max_diff"] else float("nan"),
                    "EOD_max_group":         eod["max_group"],
                    "EOD_min_group":         eod["min_group"],
                    "DPD":                   float(dpd["max_diff"]) if dpd["max_diff"] == dpd["max_diff"] else float("nan"),
                    "DPD_max_group":         dpd["max_group"],
                    "DPD_min_group":         dpd["min_group"],
                })

                if verbose:
                    eod_val = eod["max_diff"]
                    print(
                        f"    {cls_name:<10} DI={float(worst_row['DI']):.3f} "
                        f"({worst_row['DI_status']:<6}) "
                        f"EOD={eod_val:+.3f}  DPD={dpd['max_diff']:+.3f}"
                    )

            # 3) Counterfactual flip
            if do_counterfactual:
                cf = counterfactual_flip(
                    ck, col_name, df, splits, pipeline,
                    embeddings_df, target_col,
                )
                cf["champion"] = ck
                cf["champion_label"] = champ["label"]
                cf["segment_label"] = seg_label
                cf_rows.append(cf)
                if verbose and "change_rate" in cf:
                    print(
                        f"    counterfactual: {cf['n_changed']}/{cf['n']} "
                        f"({cf['change_rate']:.1%}) predictions change when "
                        f"flipping {col_name}"
                    )

    # ========================================================
    # Persist
    # ========================================================
    per_group_df = pd.concat(per_group_rows, ignore_index=True)
    out_pg = RESULTS_DIR / "fairness_per_group.parquet"
    per_group_df.to_parquet(out_pg, index=False)
    if verbose:
        print(f"\n✓ {out_pg.name}  ({len(per_group_df)} rows)")

    disparity_df = pd.DataFrame(disparity_rows)
    out_d = RESULTS_DIR / "fairness_disparities.parquet"
    disparity_df.to_parquet(out_d, index=False)
    if verbose:
        print(f"✓ {out_d.name}  ({len(disparity_df)} rows)")

    cf_df = pd.DataFrame(cf_rows) if cf_rows else pd.DataFrame()
    if not cf_df.empty:
        out_cf = RESULTS_DIR / "fairness_counterfactual.parquet"
        cf_df.to_parquet(out_cf, index=False)
        if verbose:
            print(f"✓ {out_cf.name}  ({len(cf_df)} rows)")

    # ========================================================
    # Markdown report
    # ========================================================
    md = _build_markdown_report(
        per_group_df=per_group_df,
        disparity_df=disparity_df,
        cf_df=cf_df,
        target_col=target_col,
        n_resamples=n_resamples,
        ci=ci,
    )
    out_md = REPORTS_DIR / "fairness_audit.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    if verbose:
        print(f"✓ {out_md}")

    return {
        "per_group_df": per_group_df,
        "disparity_df": disparity_df,
        "cf_df":        cf_df,
        "report_path":  out_md,
    }


# ============================================================
# Markdown report builder
# ============================================================
def _fmt_ci(v, lo, hi):
    if v != v:
        return "—"
    if lo != lo:
        return f"{v:.3f}"
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def _build_markdown_report(
    per_group_df,
    disparity_df,
    cf_df,
    target_col,
    n_resamples,
    ci,
) -> str:
    lines: list[str] = []
    lines.append("# Phase 11 — Fairness Audit Report")
    lines.append("")
    lines.append(
        f"Target: `{target_col}` · Bootstrap n = {n_resamples} · "
        f"CI = {ci:.0%} · Seed = {RANDOM_SEED}"
    )
    lines.append("")
    lines.append(
        "Two champions audited (Phase 9 selection):"
    )
    for k, c in CHAMPIONS.items():
        lines.append(f"- **{c['label']}** — `{c['phase']}/{c['model']}/{c['feature_space']}`")
    lines.append("")
    lines.append("Three protected segments (brief 4.7): **Senior**, **Gender**, **Married**.")
    lines.append("")
    lines.append("Two audited classes: **Detractor** (retention priority) and **Promoter** (symmetric check).")
    lines.append("")

    # ----- Per-group recall tables -----
    lines.append("## Per-group recall on silent_test (bootstrap 95% CI)")
    lines.append("")
    for ck in CHAMPIONS:
        label = CHAMPIONS[ck]["label"]
        lines.append(f"### {label}")
        lines.append("")
        sub = per_group_df[per_group_df["champion"] == ck]
        for seg in sorted(sub["segment"].unique()):
            seg_sub = sub[sub["segment"] == seg]
            groups = sorted(seg_sub["group"].unique())
            header = "| Group | n_total | n_Detractor | Detractor recall | n_Promoter | Promoter recall |"
            divider = "|---|---|---|---|---|---|"
            lines.append(f"**{seg}**")
            lines.append("")
            lines.append(header)
            lines.append(divider)
            for grp in groups:
                grp_sub = seg_sub[seg_sub["group"] == grp]
                det = grp_sub[grp_sub["class"] == "Detractor"]
                pro = grp_sub[grp_sub["class"] == "Promoter"]
                if det.empty or pro.empty:
                    continue
                det = det.iloc[0]; pro = pro.iloc[0]
                lines.append(
                    f"| {grp} | {int(det['n_total'])} | {int(det['n_class'])} | "
                    f"{_fmt_ci(det['recall'], det['recall_ci_lo'], det['recall_ci_hi'])} | "
                    f"{int(pro['n_class'])} | "
                    f"{_fmt_ci(pro['recall'], pro['recall_ci_lo'], pro['recall_ci_hi'])} |"
                )
            lines.append("")
        lines.append("")

    # ----- Disparity indicators -----
    lines.append("## Disparity indicators")
    lines.append("")
    lines.append(
        "* **DI** (Disparate Impact) ratio of recall to the favored group. "
        "Fair if DI ∈ [0.8, 1.25] (EEOC 4/5 rule).\n"
        "* **EOD** (Equal Opportunity Difference) = max(recall) − min(recall) "
        "across groups. Commonly acceptable if |EOD| < 0.10.\n"
        "* **DPD** (Demographic Parity Difference) = max − min of "
        "P(ŷ = class) across groups. Informational only — legitimate "
        "business differentiation can produce DPD ≠ 0."
    )
    lines.append("")
    lines.append("| Champion | Segment | Class | DI (worst) | EOD | DPD | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in disparity_df.iterrows():
        # Verdict: fair if DI in [0.8, 1.25] AND |EOD| < 0.10
        di_ok = 0.8 <= r["DI_worst"] <= 1.25
        eod_ok = abs(r["EOD"]) < 0.10 if r["EOD"] == r["EOD"] else True
        verdict = "✓ fair" if (di_ok and eod_ok) else "⚠ disparity"
        lines.append(
            f"| {r['champion_label']} | {r['segment']} | {r['class']} | "
            f"{r['DI_worst']:.3f} ({r['DI_worst_group']} vs {r['DI_reference_group']}) | "
            f"{r['EOD']:+.3f} | {r['DPD']:+.3f} | {verdict} |"
        )
    lines.append("")

    # ----- Counterfactual analysis -----
    if not cf_df.empty:
        lines.append("## Counterfactual swap analysis")
        lines.append("")
        lines.append(
            "We flip the protected attribute on silent_test (e.g. Senior=Yes → "
            "No and vice-versa), re-predict with the unchanged model, and "
            "measure the rate of changed predictions. A model that does NOT "
            "directly use the protected attribute will see ~0 % change. Any "
            "non-zero rate quantifies the *direct causal* dependence on the "
            "attribute (vs indirect, through correlated features)."
        )
        lines.append("")
        lines.append(
            "| Champion | Segment | n | Δ predictions | Δ rate | "
            "Detractor → other | Other → Detractor | "
            "Promoter → other | Other → Promoter |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in cf_df.iterrows():
            if "error" in r and isinstance(r.get("error"), str):
                continue
            if "skip_reason" in r and isinstance(r.get("skip_reason"), str):
                continue
            lines.append(
                f"| {r['champion_label']} | {r['segment_label']} | "
                f"{int(r['n'])} | {int(r['n_changed'])} | "
                f"{r['change_rate']:.1%} | "
                f"{int(r['detractor_to_other'])} | {int(r['other_to_detractor'])} | "
                f"{int(r['promoter_to_other'])} | {int(r['other_to_promoter'])} |"
            )
        lines.append("")
        lines.append(
            "**Interpretation rule of thumb**: change_rate < 2 % means the "
            "model barely uses the attribute directly. If a disparity is "
            "observed in DI/EOD but change_rate ≈ 0, it operates through "
            "*proxy variables* (e.g. Tenure correlates with Senior). Reducing "
            "such proxy-mediated disparity requires either feature engineering "
            "or a fairness-aware training objective — both beyond Phase 11."
        )
        lines.append("")

    # ----- Verdict + recommendations -----
    lines.append("## Verdict")
    lines.append("")
    n_unfair = int((
        ~((disparity_df["DI_worst"].between(0.8, 1.25))
          & (disparity_df["EOD"].abs() < 0.10))
    ).sum())
    lines.append(f"Out of {len(disparity_df)} (champion × segment × class) cells, "
                 f"**{n_unfair} flag a disparity** (DI outside [0.8, 1.25] OR |EOD| ≥ 0.10).")
    lines.append("")

    lines.append("### Recommendations")
    if n_unfair == 0:
        lines.append("* The model is fair on all audited segments and classes. "
                     "No mitigation needed.")
    else:
        lines.append("* Investigate which proxy features drive the disparity "
                     "(use Phase 10's `shap_segment_*.parquet` and "
                     "`linear_coef_segment_*.parquet`).")
        lines.append("* Consider Phase 13 monitoring: track per-segment recall "
                     "monthly with alert at 2σ deviation.")
        lines.append("* Per-group threshold tuning is technically possible but "
                     "may be legally/ethically inadvisable — document risk "
                     "before adopting.")
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
    parser.add_argument("--no-counterfactual", action="store_true")
    args = parser.parse_args()

    try:
        run_fairness_audit(
            n_resamples=args.n_resamples,
            ci=args.ci,
            target_col=args.target,
            do_counterfactual=not args.no_counterfactual,
            verbose=True,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
