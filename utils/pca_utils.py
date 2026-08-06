"""PCA helpers for constructing reproducible gene latent targets.

The PCA basis is ALWAYS fit on the training split only and then applied to
val/test, so the latent target never leaks evaluation information.
"""
from typing import Optional

import numpy as np


class PCAReducer:
    def __init__(self, n_components: int = 32, whiten: bool = True):
        self.n_components = n_components
        self.whiten = whiten
        self._pca = None
        self._mean = None

    def fit(self, X: np.ndarray):
        from sklearn.decomposition import PCA

        n = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self.n_components = n
        self._pca = PCA(n_components=n, whiten=self.whiten, random_state=0)
        self._pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._pca.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        return None if self._pca is None else self._pca.explained_variance_ratio_
