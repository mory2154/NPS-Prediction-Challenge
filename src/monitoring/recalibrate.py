"""
Phase 13 — Isotonic recalibration of C2.

Why
---
Phase 9 documented that C2's `predict_proba` is well-ranked but poorly
calibrated on Passive/Promoter (ECE Passive = 0.29, ECE Promoter = 0.15).
Effect of `class_weight='balanced'` + L2 regularization (C=0.027) that
compresses probabilities toward the centre.

Important caveat — `class_weight='balanced'` × isotonic recalibration
---------------------------------------------------------------------
C2 was trained with `class_weight='balanced'` which intentionally inflates
`predict_proba(Detractor)` so the model catches more detractors. When we
recalibrate, isotonic correctly maps those inflated probabilities back to
their true posteriors → ECE drops massively (good!) — but `predict()` is
`argmax(predict_proba)`, and the argmax of *correctly* calibrated probas
favours the majority class (Passive at ~58%). So Detractor recall via
argmax drops catastrophically.

**Resolution (recommended usage in production)**:
    * Use **calibrated proba** for display, top-K ranking, lift@K metrics
      and any threshold-based decision.
    * Use the **original C2** for argmax-based `predict()` decisions
      (catching detractors in the retention workflow).

The audit below computes 3 modes side-by-side to make this clear:
    * `before`:        original model.predict_proba + original predict
    * `after_full`:    calibrated.predict_proba + calibrated.predict (= argmax of calibrated)
    * `after_hybrid`:  calibrated.predict_proba + original predict (= recommended)

CLI:
    python -m src.monitoring.recalibrate
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    DATA_PROCESSED,
    DEFAULT_TARGET,
    MODELS_DIR,
    NPS_CLASSES,
    NPS_CLASS_TO_INT,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.data.split import load_splits
from src.evaluation.calibration import calibration_report
from src.evaluation.metrics import evaluate
from src.models.tuning_hybrid import VERBATIM_AUX_COLS


C2_PATH = MODELS_DIR / "tuned" / "logistic_tuned.joblib"
C2_CALIBRATED_PATH = MODELS_DIR / "tuned" / "logistic_C2_calibrated.joblib"


# ============================================================
# Build X matrices (val + silent_test) for C2 (tabular only)
# ============================================================
def _build_C2_matrices(target_col: str) -> dict:
    if not target_col.startswith("NPS_"):
        target_col = f"NPS_{target_col}"

    df = pd.read_parquet(DATA_PROCESSED / "dataset_with_verbatims.parquet")
    splits = load_splits("response_biased")
    pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")

    feature_cols = [
        c for c in df.columns
        if not c.startswith("NPS_") and c not in VERBATIM_AUX_COLS
    ]

    out = {}
    for split_name in ("train", "val", "respondent_test", "silent_test"):
        mask = splits == split_name
        if not mask.any():
            continue
        X = pipeline.transform(df.loc[mask, feature_cols])
        y = np.array(
            [NPS_CLASS_TO_INT[str(v)] for v in df.loc[mask, target_col].to_numpy()],
            dtype=int,
        )
        out[split_name] = {"X": X, "y": y, "index": df.index[mask]}
    return out


# ============================================================
# Recalibrate
# ============================================================
def recalibrate_C2(
    method: str = "isotonic",
    cv: int = 5,
    target_col: str = DEFAULT_TARGET,
    verbose: bool = True,
) -> dict:
    """
    Fit a `CalibratedClassifierCV` on top of the trained C2, using val for
    isotonic regression. Save the result + compute Brier/ECE before/after.

    Returns
    -------
    dict with the calibrated model + before/after metrics DataFrame.
    """
    if not C2_PATH.exists():
        raise FileNotFoundError(
            f"{C2_PATH} missing — run `make tune` first (Phase 7)."
        )
    if verbose:
        print("=" * 70)
        print(f"PHASE 13 — RECALIBRATE C2 ({method}, cv={cv})")
        print("=" * 70)

    raw_model = joblib.load(C2_PATH)
    mats = _build_C2_matrices(target_col)

    if "val" not in mats:
        raise ValueError("val split missing — cannot recalibrate without it.")
    if verbose:
        print(
            f"  Training data for calibration : val "
            f"(n={len(mats['val']['y'])}, class dist = "
            f"{np.bincount(mats['val']['y']).tolist()})"
        )

    # sklearn ≥ 1.6 : wrap with FrozenEstimator; older : cv='prefit'
    try:
        from sklearn.frozen import FrozenEstimator
        frozen = FrozenEstimator(raw_model)
        calibrated = CalibratedClassifierCV(
            estimator=frozen, method=method, cv=cv,
        )
    except ImportError:
        calibrated = CalibratedClassifierCV(
            estimator=raw_model, method=method, cv="prefit",
        )
    calibrated.fit(mats["val"]["X"], mats["val"]["y"])

    # Save
    C2_CALIBRATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, C2_CALIBRATED_PATH)
    if verbose:
        print(f"  ✓ saved {C2_CALIBRATED_PATH.name}")

    # Before/after audit on silent_test
    silent = mats["silent_test"]
    X_s, y_s = silent["X"], silent["y"]

    proba_before = raw_model.predict_proba(X_s)
    proba_after  = calibrated.predict_proba(X_s)
    pred_before  = raw_model.predict(X_s)         # original argmax (kept!)
    pred_after   = calibrated.predict(X_s)        # calibrated argmax (collapses!)

    # ============================================================
    # Calibration (depends only on proba)
    # ============================================================
    cal_before = calibration_report(y_s, proba_before, n_bins=10)
    cal_after  = calibration_report(y_s, proba_after, n_bins=10)

    rows = []
    for cls in NPS_CLASSES:
        rows.append({
            "class":         cls,
            "brier_before":  cal_before["brier"][cls],
            "brier_after":   cal_after["brier"][cls],
            "brier_delta":   cal_after["brier"][cls] - cal_before["brier"][cls],
            "ece_before":    cal_before["ece"][cls],
            "ece_after":     cal_after["ece"][cls],
            "ece_delta":     cal_after["ece"][cls] - cal_before["ece"][cls],
        })
    cal_df = pd.DataFrame(rows)

    # ============================================================
    # Headline metrics — THREE MODES
    # ============================================================
    eval_before = evaluate(y_s, pred_before, proba_before, name="before")
    eval_after  = evaluate(y_s, pred_after,  proba_after,  name="after_full")
    # Hybrid = calibrated proba + original predict (RECOMMENDED IN PRODUCTION)
    eval_hybrid = evaluate(y_s, pred_before, proba_after, name="after_hybrid")

    headline_rows = []
    for metric in ("qwk", "macro_f1", "detractor_recall", "lift@10", "lift@20"):
        if metric in eval_before:
            headline_rows.append({
                "metric":             metric,
                "before":             float(eval_before[metric]),
                "after_full":         float(eval_after[metric]),
                "after_hybrid":       float(eval_hybrid[metric]),
                "delta_full":         float(eval_after[metric])  - float(eval_before[metric]),
                "delta_hybrid":       float(eval_hybrid[metric]) - float(eval_before[metric]),
            })
    head_df = pd.DataFrame(headline_rows)

    # ============================================================
    # Pretty-print
    # ============================================================
    if verbose:
        print("\n  Calibration (per class):")
        for _, r in cal_df.iterrows():
            print(
                f"    {r['class']:<10} "
                f"Brier {r['brier_before']:.3f} → {r['brier_after']:.3f}  "
                f"(Δ {r['brier_delta']:+.3f})  | "
                f"ECE {r['ece_before']:.3f} → {r['ece_after']:.3f}  "
                f"(Δ {r['ece_delta']:+.3f})"
            )

        print("\n  Headline metrics — 3 modes:")
        print(f"    {'metric':<18} {'before':>10}  {'after_FULL':>10} {'(Δ)':>9}  "
              f"{'after_HYBRID':>12} {'(Δ)':>9}")
        print(f"    {'─'*18} {'─'*10}  {'─'*10} {'─'*9}  {'─'*12} {'─'*9}")
        for _, r in head_df.iterrows():
            print(
                f"    {r['metric']:<18} "
                f"{r['before']:>10.4f}  "
                f"{r['after_full']:>10.4f} {r['delta_full']:+9.4f}  "
                f"{r['after_hybrid']:>12.4f} {r['delta_hybrid']:+9.4f}"
            )

        print("\n  Reading guide:")
        print("    • 'after_FULL'   = calibrated.predict_proba + calibrated.predict")
        print("                       (argmax shifts due to class_weight='balanced'")
        print("                       interaction — Detractor recall drops)")
        print("    • 'after_HYBRID' = calibrated.predict_proba + original.predict")
        print("                       (RECOMMENDED — preserves argmax metrics)")
        print("    • Calibration ECE drops in both cases (= the win we wanted).")

    # Persist
    out_cal = RESULTS_DIR / "calibration_before_after.parquet"
    cal_df.to_parquet(out_cal, index=False)
    out_head = RESULTS_DIR / "calibration_headline.parquet"
    head_df.to_parquet(out_head, index=False)
    if verbose:
        print(f"\n  ✓ saved {out_cal.name} ({len(cal_df)} rows)")
        print(f"  ✓ saved {out_head.name} ({len(head_df)} rows)")

    return {
        "model": calibrated,
        "calibration_audit": cal_df,
        "headline_audit": head_df,
    }


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="isotonic", choices=["isotonic", "sigmoid"])
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()

    try:
        recalibrate_C2(method=args.method, target_col=args.target, verbose=True)
    except FileNotFoundError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
