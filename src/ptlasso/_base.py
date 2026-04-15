from abc import abstractmethod

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score


class BasePretrainedLasso(BaseEstimator):
    """Abstract base class for pretrained Lasso estimators."""

    @abstractmethod
    def fit(self, X, y, groups):
        """Fit the model."""

    @abstractmethod
    def predict(self, X, groups):
        """Predict using the fitted model.

        Note: ``groups`` is required because predictions depend on which
        group-specific model to apply. This deviates from the standard
        sklearn ``predict(X)`` signature — PTLasso is a group-aware model.
        """

    def score(self, X, y, groups):
        """Return R² on the given data."""
        return r2_score(y, self.predict(X=X, groups=groups))
