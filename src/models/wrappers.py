"""
Pickle-safe wrappers for non-sklearn models.

This module exists for ONE reason: classes used inside `joblib.dump()` must
live in a module that is ALWAYS imported as a package (e.g. `src.models.wrappers`),
NEVER as `__main__`.

If you put them inside `src/models/tuning.py` and run that file with
`python -m src.models.tuning`, the qualified name becomes `__main__.OrdinalWrapper`
and the resulting joblib file can't be loaded from any other context (notably pytest).

So all wrapper classes go HERE.
"""

from __future__ import annotations

import numpy as np

from src.config import NPS_CLASS_TO_INT


def _to_int(y) -> np.ndarray:
    """Convert string/categorical labels to 0/1/2 ints."""
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    if y.dtype.kind in {"U", "O"} or hasattr(y, "categories"):
        return np.array([NPS_CLASS_TO_INT[str(v)] for v in y], dtype=int)
    return np.asarray(y, dtype=int)


class OrdinalWrapper:
    """
    Wrap a `mord.LogisticAT` model to expose a uniform sklearn-like interface
    with `predict_proba`.

    mord's predict() returns hard labels. We approximate predict_proba with
    a one-hot smoothed by 0.05 — sufficient for ranking (lift curves) but not
    for true probability calibration. If proper proba is needed downstream,
    apply Platt scaling or isotonic regression on top.
    """

    def __init__(self, model):
        self.model = model
        self.classes_ = None

    def fit(self, X, y):
        y_int = _to_int(y)
        self.model.fit(X, y_int)
        self.classes_ = np.unique(y_int)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        preds = self.predict(X).astype(int)
        n = len(preds)
        proba = np.zeros((n, 3))
        proba[np.arange(n), preds] = 1.0
        # Smooth slightly to give lift_at_k a continuous ranking signal
        proba = proba * 0.9 + 0.05
        return proba
