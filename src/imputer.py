import numpy as np
from abc import ABC, abstractmethod
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class BaseImputer(ABC):
    """Common interface for all imputation strategies."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseImputer":
        """Learn imputation parameters from X (with NaNs)."""
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a copy of X with NaNs filled in."""
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Naive zero imputation — the paper's subject
# ---------------------------------------------------------------------------


class ZeroImputer(BaseImputer):
    """
    Naive zero imputation. Fills each feature with zero.
    This is the method whose OLS solution the paper shows is equivalent
    to ridge regression on complete data under Ho-MCAR.
    """

    def __init__(self):
        pass

    def fit(self, X: np.ndarray) -> "ZeroImputer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_filled = X.copy()
        X_filled[np.isnan(X_filled)] = 0.0
        return X_filled


# ---------------------------------------------------------------------------
# Mean imputation
# ---------------------------------------------------------------------------


class MeanImputer(BaseImputer):
    """Fills each feature with its observed column mean."""

    def __init__(self):
        self._imputer = SimpleImputer(strategy="mean")

    def fit(self, X: np.ndarray) -> "MeanImputer":
        self._imputer.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._imputer.transform(X)


# ---------------------------------------------------------------------------
# KNN imputation
# ---------------------------------------------------------------------------


class KNNImputer_(BaseImputer):
    """
    Fills missing values using the mean of the k nearest neighbors
    (measured on observed features only).

    Args:
        n_neighbors: number of neighbors to use.
    """

    def __init__(self, n_neighbors: int = 5):
        self._imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X: np.ndarray) -> "KNNImputer_":
        self._imputer.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._imputer.transform(X)


# ---------------------------------------------------------------------------
# MICE — Multiple Imputation by Chained Equations
# ---------------------------------------------------------------------------


class MICEImputer(BaseImputer):
    """
    Iterative imputation (sklearn's implementation of ICE).
    Each feature is modeled as a function of the others in a round-robin fashion.

    Args:
        max_iter: number of imputation rounds.
        random_state: for reproducibility.
    """

    def __init__(self, max_iter: int = 10, random_state: int = 0):
        self._imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=random_state,
            verbose=0,
        )

    def fit(self, X: np.ndarray) -> "MICEImputer":
        self._imputer.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._imputer.transform(X)


# ---------------------------------------------------------------------------
# MissForest
# ---------------------------------------------------------------------------


class MissForestImputer(BaseImputer):
    """
    Random-forest-based iterative imputation (MissForest).
    Uses sklearn's IterativeImputer with a RandomForestRegressor estimator.

    Args:
        n_estimators: number of trees in each random forest.
        max_iter: number of imputation rounds.
        random_state: for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 5,
        random_state: int = 0,
    ):
        from sklearn.ensemble import RandomForestRegressor

        self._imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            ),
            max_iter=max_iter,
            random_state=random_state,
            verbose=0,
        )

    def fit(self, X: np.ndarray) -> "MissForestImputer":
        self._imputer.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._imputer.transform(X)


def get_all_imputers(random_state: int = 0) -> dict[str, BaseImputer]:
    """
    Returns all imputers keyed by name.
    Pass this dict directly to the benchmark loop.
    """
    return {
        "Zero": ZeroImputer(),
        "Mean": MeanImputer(),
        "KNN": KNNImputer_(n_neighbors=5),
        "MICE": MICEImputer(random_state=random_state),
        "MissForest": MissForestImputer(random_state=random_state),
    }
