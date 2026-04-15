"""Abstract base class shared by PretrainedLasso and PretrainedLassoCV."""

from abc import abstractmethod

from sklearn.base import BaseEstimator


class BasePretrainedLasso(BaseEstimator):
    """Abstract base for pretrained Lasso estimators."""

    @abstractmethod
    def fit(self, X, y, groups):
        """Fit the model."""

    @abstractmethod
    def predict(self, X, groups):
        """Predict using the fitted model.

        Note: ``groups`` is required because predictions depend on which
        group-specific model to apply.  This intentionally deviates from the
        standard sklearn ``predict(X)`` signature.
        """
