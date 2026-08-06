"""Feature normalization fit on the training split and applied to all splits.

We deliberately keep this dependency-light (numpy only) so it works identically
inside the container and on the login node.
"""
from typing import Optional

import numpy as np


class FeatureScaler:
    """Standard (z-score) or min-max scaler with train-only statistics."""

    def __init__(self, mode: str = "standard"):
        assert mode in ("standard", "minmax", "none")
        self.mode = mode
        self.a: Optional[np.ndarray] = None  # center / min
        self.b: Optional[np.ndarray] = None  # scale / range

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if self.mode == "standard":
            self.a = X.mean(axis=0)
            self.b = X.std(axis=0)
            self.b[self.b < 1e-8] = 1.0
        elif self.mode == "minmax":
            self.a = X.min(axis=0)
            self.b = X.max(axis=0) - self.a
            self.b[self.b < 1e-8] = 1.0
        else:  # none
            self.a = np.zeros(X.shape[1])
            self.b = np.ones(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.a) / self.b).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
