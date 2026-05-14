"""
Phase 13 — Drift simulator.

We don't have a real production stream of incoming customers, but we can
simulate one by chunking silent_test in monthly batches (FIFO order by
Customer ID, the natural index of the IBM Telco dataset) and computing the
KPIs that a production monitoring job would surface.

This serves two purposes:
    1. Demonstrate that the alerting pipeline (Phase 13 `alerts.py`) can
       process monthly batches without crashing.
    2. Establish a baseline: how stable ARE the KPIs across batches if the
       data has no real drift?

`--use-calibrated` semantics (HYBRID mode)
------------------------------------------
When `--use-calibrated` is set, we substitute ONLY the probability columns
of C2 from the recalibrated model, and KEEP the original `pred_C2`
(argmax of the original probas).

Why: C2 was trained with `class_weight='balanced'`. Recalibrating shifts the
argmax toward the majority class (Passive) — which collapses Detractor recall
from 0.84 to 0.30 if we naively replace `pred_C2`. The HYBRID mode is the
production-recommended pattern: enjoy the calibrated probabilities (for
display, lift, threshold-based decisions) while keeping the original
detector-friendly argmax for binary classification.

Output schema (one row per (champion × batch × metric)):
    champion, batch_id, batch_size, batch_start_customer_id, batch_end_customer_id,
    metric, value, segment, group (NaN if global)

CLI:
    python -m src.monitoring.drift_simulator
    python -m src.monitoring.drift_simulator --n-batches 12 --use-calibrated
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_TARGET,
    MODELS_DIR,
    NPS_CLASS_TO_INT,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.evaluation.metrics import (
    detractor_recall as _det_recall,
    lift_at_k,
    macro_f1,
    quadratic_weighted_kappa,
)
from src.monitoring.recalibrate import C2_CALIBRATED_PATH

PREDICTIONS_PATH = RESULTS_DIR / "silent_predictions.parquet"

# Segments to track per batch (matches Phase 11)
SEGMENT_COLUMNS = {"Senior": "Senior Citizen", "Gender": "Gender", "Married": "Married"}


# ============================================================
# Metric helpers — re-using src.evaluation
# ============================================================
def _batch_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, float]:
    """Compute the 4 monitoring KPIs on one batch."""
    out = {
        "qwk":               quadratic_weighted_kappa(y_true, y_pred),
        "macro_f1":          macro_f1(y_true, y_pred),
        "detractor_recall":  _det_recall(y_true, y_pred),
    }
    if y_proba is not None:
        out["lift@10"] = lift_at_k(y_true, y_proba, k_pct=0.10)
    return out


def _per_segment_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    class_idx: int,
) -> dict[str, float]:
    """Recall for the given class within each group of `groups`."""
    out = {}
    for grp in sorted(pd.Series(groups).dropna().astype(str).unique()):
        mask = (groups.astype(str) == grp).to_numpy()
        if mask.sum() == 0:
            continue
        yt = y_true[mask]; yp = y_pred[mask]
        n_pos = int((yt == class_idx).sum())
        if n_pos == 0:
            out[grp] = float("nan")
        else:
            out[grp] = float(((yt == class_idx) & (yp == class_idx)).sum()) / n_pos
    return out


# ============================================================
# Build the batches
# ============================================================
def _make_batches(
    customer_ids: pd.Series,
    n_batches: int = 12,
) -> list[tuple[int, np.ndarray]]:
    """FIFO chunks of the customer_ids by their natural order."""
    n = len(customer_ids)
    batch_size = int(np.ceil(n / n_batches))
    batches = []
    for b in range(n_batches):
        lo, hi = b * batch_size, min((b + 1) * batch_size, n)
        if lo >= hi:
            continue
        batches.append((b + 1, np.arange(lo, hi)))
    return batches


# ============================================================
# Main runner
# ============================================================
def run_drift_simulation(
    n_batches: int = 12,
    use_calibrated: bool = False,
    target_col: str = DEFAULT_TARGET,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the drift simulator on the silent_predictions parquet.

    use_calibrated : if True, substitute the C2 probability columns with
                     scores from the recalibrated model. ⚠ `pred_C2` is NOT
                     substituted — we keep the original argmax to avoid the
                     class_weight='balanced' × isotonic collapse (see module
                     docstring).
    """
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{PREDICTIONS_PATH} missing — run `make batch-score` first."
        )

    if verbose:
        print("=" * 70)
        print(f"PHASE 13 — DRIFT SIMULATOR  (n_batches={n_batches}, "
              f"calibrated={use_calibrated})")
        if use_calibrated:
            print("  Mode HYBRIDE : probas calibrées + argmax original (Phase 13 spec)")
        print("=" * 70)

    pred = pd.read_parquet(PREDICTIONS_PATH).copy()

    # ============================================================
    # HYBRID substitution: calibrated proba, original predict
    # ============================================================
    if use_calibrated:
        if not C2_CALIBRATED_PATH.exists():
            raise FileNotFoundError(
                f"{C2_CALIBRATED_PATH} missing — run `make recalibrate` first."
            )
        if verbose:
            print(f"\n  Re-scoring C2 probas with {C2_CALIBRATED_PATH.name}")
            print(f"  Keeping `pred_C2` from the original model (HYBRID mode)\n")
        from src.config import DATA_PROCESSED
        from src.data.split import load_splits
        from src.models.tuning_hybrid import VERBATIM_AUX_COLS

        cal_model = joblib.load(C2_CALIBRATED_PATH)
        df = pd.read_parquet(DATA_PROCESSED / "dataset_with_verbatims.parquet")
        splits = load_splits("response_biased")
        pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
        feature_cols = [
            c for c in df.columns
            if not c.startswith("NPS_") and c not in VERBATIM_AUX_COLS
        ]
        mask = splits == "silent_test"
        X = pipeline.transform(df.loc[mask, feature_cols])
        proba_new = cal_model.predict_proba(X)

        # Align by customer_id
        new_ids = df.index[mask].tolist()
        new_map = dict(zip(new_ids, range(len(new_ids))))
        order = pred["customer_id"].map(new_map).values

        from src.config import NPS_CLASSES
        # Substitute proba ONLY
        for k, cls in enumerate(NPS_CLASSES):
            pred[f"proba_C2_{cls.lower()}"] = proba_new[order, k]
        # NB: pred_C2 is NOT overwritten — it remains the original argmax

    # Force chronological FIFO order by Customer ID
    pred = pred.sort_values("customer_id").reset_index(drop=True)
    if verbose:
        print(f"  Customers in stream    : {len(pred):,}")
        print(f"  Approx. batch size     : {len(pred) // n_batches}")

    batches = _make_batches(pred["customer_id"], n_batches=n_batches)

    y_true_all = pred["nps_true"].map(NPS_CLASS_TO_INT).values

    rows: list[dict] = []
    for batch_id, idxs in batches:
        sub = pred.iloc[idxs]
        y_true = y_true_all[idxs]
        first_id = str(sub["customer_id"].iloc[0])
        last_id  = str(sub["customer_id"].iloc[-1])

        for champ in ("C1", "C2"):
            y_pred = np.array([
                NPS_CLASS_TO_INT[c] for c in sub[f"pred_{champ}"].values
            ])
            y_proba = sub[[
                f"proba_{champ}_detractor",
                f"proba_{champ}_passive",
                f"proba_{champ}_promoter",
            ]].to_numpy()

            # Global metrics
            kpis = _batch_metrics(y_true, y_pred, y_proba)
            for m, v in kpis.items():
                rows.append({
                    "champion":   champ,
                    "batch_id":   batch_id,
                    "batch_size": len(sub),
                    "first_id":   first_id,
                    "last_id":    last_id,
                    "metric":     m,
                    "value":      float(v),
                    "segment":    None,
                    "group":      None,
                })

            # Per-segment Detractor recall (key fairness KPI)
            for seg_label, col in SEGMENT_COLUMNS.items():
                if col not in sub.columns:
                    continue
                groups = sub[col].astype(str)
                det_recalls = _per_segment_recall(
                    y_true, y_pred, groups, class_idx=NPS_CLASS_TO_INT["Detractor"]
                )
                for grp, v in det_recalls.items():
                    rows.append({
                        "champion":   champ,
                        "batch_id":   batch_id,
                        "batch_size": len(sub),
                        "first_id":   first_id,
                        "last_id":    last_id,
                        "metric":     "detractor_recall",
                        "value":      float(v),
                        "segment":    seg_label,
                        "group":      grp,
                    })

    out_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "drift_simulation.parquet"
    out_df.to_parquet(out_path, index=False)
    if verbose:
        print(f"\n  ✓ saved {out_path.name}  ({len(out_df)} rows)")
        # Summary: range of QWK across batches
        for ch in ("C2", "C1"):
            qwks = out_df[
                (out_df["champion"] == ch)
                & (out_df["metric"] == "qwk")
                & (out_df["segment"].isna())
            ]["value"]
            recalls = out_df[
                (out_df["champion"] == ch)
                & (out_df["metric"] == "detractor_recall")
                & (out_df["segment"].isna())
            ]["value"]
            if not qwks.empty:
                print(
                    f"  {ch} QWK across batches: "
                    f"min={qwks.min():.3f}, max={qwks.max():.3f}, "
                    f"range={qwks.max() - qwks.min():.3f}"
                )
            if not recalls.empty:
                print(
                    f"  {ch} Detractor recall  : "
                    f"min={recalls.min():.3f}, max={recalls.max():.3f}, "
                    f"range={recalls.max() - recalls.min():.3f}"
                )

    return out_df


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=12)
    parser.add_argument("--use-calibrated", action="store_true",
                        help="HYBRID mode: substitute C2 probas from calibrated "
                             "model, but keep `pred_C2` from the original.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()

    try:
        run_drift_simulation(
            n_batches=args.n_batches,
            use_calibrated=args.use_calibrated,
            target_col=args.target,
            verbose=True,
        )
    except FileNotFoundError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
