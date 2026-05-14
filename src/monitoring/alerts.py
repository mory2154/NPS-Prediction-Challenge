"""
Phase 13 — Alert rules for monthly monitoring.

Alert philosophy (Phase 9 promise):
    "Alerte si dérive >2σ par rapport aux CIs Phase 11."

We interpret this concretely as follows:

    * For each batch, compute the monitoring KPI (e.g. Detractor recall on
      segment Senior=Yes for champion C2).
    * Compare the observed value to the **bootstrap CI** computed in
      Phase 11 (this is the "static reference" of acceptable variation).
    * A drift is flagged if the batch value falls *outside* the 95 % CI
      bracket — equivalent to a ~2σ excursion under Gaussian assumption.
    * Two severity levels:
        - **warn**  : single batch outside CI
        - **alert** : two consecutive batches in the same direction

The reference CI is read from `fairness_per_group.parquet` (Phase 11) and
`final_eval_summary.parquet` (Phase 9). When neither contains the exact
(metric × segment × group × champion) tuple, we fall back to the per-
champion silent_test CI (Phase 9 headline) — coarser but always available.

This module is **pure functions**, no I/O. The Streamlit page or a cron
job consumes the alerts dataframe.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Reference CI lookup
# ============================================================
def _normalize_champion_key(ck: str) -> str:
    """The Phase 9/11 parquets use "C1_qwk" and "C2_safe"; the simulator uses
    "C1" and "C2". We map between them."""
    if ck in ("C1", "C1_qwk"):
        return "C1_qwk"
    if ck in ("C2", "C2_safe"):
        return "C2_safe"
    return ck


def lookup_reference_ci(
    fairness_per_group: pd.DataFrame,
    final_eval_summary: pd.DataFrame,
    champion: str,
    metric: str,
    segment: Optional[str] = None,
    group: Optional[str] = None,
) -> tuple[float, float, str]:
    """
    Find the appropriate reference (ci_lo, ci_hi) for this monitored cell.

    Strategy:
        1. If segment+group given AND fairness_per_group has the row → use it
           (Phase 11 per-segment CI).
        2. Else, use final_eval_summary on silent_test for this metric+champion
           (Phase 9 headline CI).
        3. Else, return (NaN, NaN, "no_reference").
    """
    ck = _normalize_champion_key(champion)

    # 1) Per-segment CI from Phase 11
    if segment is not None and group is not None and not fairness_per_group.empty:
        # Phase 11 only audits Detractor + Promoter, so this only fires for those metrics
        cls_for_metric = (
            "Detractor" if metric == "detractor_recall"
            else "Promoter" if metric == "promoter_recall"
            else None
        )
        if cls_for_metric is not None:
            sub = fairness_per_group[
                (fairness_per_group["champion"] == ck)
                & (fairness_per_group["segment"] == segment)
                & (fairness_per_group["class"] == cls_for_metric)
                & (fairness_per_group["group"].astype(str) == str(group))
            ]
            if not sub.empty:
                r = sub.iloc[0]
                lo, hi = float(r["recall_ci_lo"]), float(r["recall_ci_hi"])
                if not (np.isnan(lo) or np.isnan(hi)):
                    return lo, hi, "phase11_per_segment"

    # 2) Headline CI from Phase 9 (silent_test)
    if not final_eval_summary.empty:
        sub = final_eval_summary[
            (final_eval_summary["champion"] == ck)
            & (final_eval_summary["split"] == "silent_test")
            & (final_eval_summary["metric"] == metric)
        ]
        if not sub.empty:
            r = sub.iloc[0]
            return float(r["ci_lo"]), float(r["ci_hi"]), "phase9_headline"

    return float("nan"), float("nan"), "no_reference"


# ============================================================
# Single-batch alert
# ============================================================
def evaluate_alert(
    observed: float,
    ref_lo: float,
    ref_hi: float,
) -> str:
    """
    Classify the observed value relative to the reference CI.

    Returns one of: "in_band" | "drift_low" | "drift_high" | "no_reference"
    """
    if np.isnan(observed) or np.isnan(ref_lo) or np.isnan(ref_hi):
        return "no_reference"
    if observed < ref_lo:
        return "drift_low"
    if observed > ref_hi:
        return "drift_high"
    return "in_band"


# ============================================================
# Sequential escalation (two consecutive batches → alert level "alert")
# ============================================================
def escalate_alerts(per_batch_status: pd.Series) -> pd.Series:
    """
    Take a chronological series of {in_band, drift_low, drift_high, no_reference}
    and produce a severity series in {ok, warn, alert}.

    Rules:
        - "in_band" / "no_reference"           → "ok"
        - First drift, no prior drift           → "warn"
        - Drift in same direction as previous batch → "alert"
    """
    severities: list[str] = []
    prev_status = None
    for status in per_batch_status:
        if status in ("in_band", "no_reference"):
            severities.append("ok")
            prev_status = status
            continue
        # status is drift_low or drift_high
        if prev_status == status:
            severities.append("alert")
        else:
            severities.append("warn")
        prev_status = status
    return pd.Series(severities, index=per_batch_status.index)


# ============================================================
# Full alert table — consume drift_simulation.parquet
# ============================================================
def build_alert_table(
    drift_df: pd.DataFrame,
    fairness_per_group: pd.DataFrame,
    final_eval_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one row per (champion × metric × segment × group × batch) with
    {observed, ref_lo, ref_hi, status, severity}.
    """
    out_rows = []
    # Group by (champion, metric, segment, group) and walk batches chronologically
    group_cols = ["champion", "metric", "segment", "group"]
    drift_df = drift_df.copy()
    # Normalise NaN segment/group to a sentinel so groupby works
    drift_df["segment"] = drift_df["segment"].where(
        drift_df["segment"].notna(), "__GLOBAL__"
    )
    drift_df["group"] = drift_df["group"].where(
        drift_df["group"].notna(), "__GLOBAL__"
    )

    for (champ, metric, seg, grp), sub in drift_df.groupby(group_cols, sort=False):
        sub = sub.sort_values("batch_id")
        seg_ref = None if seg == "__GLOBAL__" else seg
        grp_ref = None if grp == "__GLOBAL__" else grp
        lo, hi, ref_source = lookup_reference_ci(
            fairness_per_group, final_eval_summary,
            champion=champ, metric=metric,
            segment=seg_ref, group=grp_ref,
        )
        statuses = sub["value"].apply(lambda v: evaluate_alert(v, lo, hi))
        severities = escalate_alerts(statuses)
        for (_, row), status, sev in zip(sub.iterrows(), statuses, severities):
            out_rows.append({
                "champion":  row["champion"],
                "batch_id":  row["batch_id"],
                "metric":    row["metric"],
                "segment":   None if seg == "__GLOBAL__" else seg,
                "group":     None if grp == "__GLOBAL__" else grp,
                "observed":  row["value"],
                "ref_lo":    lo,
                "ref_hi":    hi,
                "ref_source": ref_source,
                "status":    status,
                "severity":  sev,
                "batch_size": row["batch_size"],
            })
    return pd.DataFrame(out_rows)


# ============================================================
# Top-level summary — counts by severity, useful for a Streamlit KPI
# ============================================================
def summarize_alerts(alert_df: pd.DataFrame) -> dict:
    if alert_df.empty:
        return {"total": 0, "ok": 0, "warn": 0, "alert": 0}
    counts = alert_df["severity"].value_counts().to_dict()
    return {
        "total": int(len(alert_df)),
        "ok":    int(counts.get("ok", 0)),
        "warn":  int(counts.get("warn", 0)),
        "alert": int(counts.get("alert", 0)),
    }

# ============================================================
# CLI entrypoint
# ============================================================
def main() -> None:
    """
    Build monitoring_alerts.parquet from drift_simulation + Phase 9/11 CIs.

    Usage:
        python -m src.monitoring.alerts
    """
    import sys
    from src.config import RESULTS_DIR

    drift_path    = RESULTS_DIR / "drift_simulation.parquet"
    fairness_path = RESULTS_DIR / "fairness_per_group.parquet"
    final_path    = RESULTS_DIR / "final_eval_summary.parquet"

    for p in (drift_path, fairness_path, final_path):
        if not p.exists():
            print(f"✗ {p.name} missing — run earlier phases first.", file=sys.stderr)
            sys.exit(1)

    drift      = pd.read_parquet(drift_path)
    fairness   = pd.read_parquet(fairness_path)
    final_eval = pd.read_parquet(final_path)

    print("=" * 60)
    print("PHASE 13 — BUILD ALERTS TABLE")
    print("=" * 60)
    print(f"  drift_simulation     : {drift.shape}")
    print(f"  fairness_per_group   : {fairness.shape}")
    print(f"  final_eval_summary   : {final_eval.shape}")

    alerts = build_alert_table(drift, fairness, final_eval)
    out = RESULTS_DIR / "monitoring_alerts.parquet"
    alerts.to_parquet(out, index=False)

    summary = summarize_alerts(alerts)
    print(f"\n  ✓ saved {out.name}  ({len(alerts)} rows)")
    print(f"  Summary: {summary}")
    print(f"\n  Severity counts:")
    print(f"    ok    : {summary['ok']}")
    print(f"    warn  : {summary['warn']}")
    print(f"    alert : {summary['alert']}")


if __name__ == "__main__":
    main()
