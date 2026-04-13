import numpy as np
import pytest

from ptlasso import PretrainedLasso, PretrainedLassoCV


def make_data(n=100, p=20, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    groups = rng.integers(0, k, size=n)
    return X, y, groups


def test_ptlasso_fit_predict():
    X, y, groups = make_data()
    model = PretrainedLasso(alpha=0.5)
    model.fit(X, y, groups)
    y_pred = model.predict(X, groups)
    assert y_pred.shape == (len(y),)


def test_ptlasso_score():
    X, y, groups = make_data()
    model = PretrainedLasso(alpha=0.5).fit(X, y, groups)
    score = model.score(X, y, groups)
    assert isinstance(score, float)


def test_ptlasso_alpha_zero_equals_individual():
    """alpha=0 means no pretraining: pretrain and individual predictions should match."""
    X, y, groups = make_data()
    model = PretrainedLasso(alpha=0).fit(X, y, groups)
    pred_pretrain = model.predict(X, groups, model="pretrain")
    pred_individual = model.predict(X, groups, model="individual")
    np.testing.assert_allclose(pred_pretrain, pred_individual)


def test_ptlasso_fitted_attributes():
    X, y, groups = make_data(k=2)
    model = PretrainedLasso().fit(X, y, groups)
    assert hasattr(model, "overall_model_")
    assert hasattr(model, "overall_coef_")
    assert hasattr(model, "pretrain_models_")
    assert hasattr(model, "individual_models_")
    assert set(model.groups_) == set(np.unique(groups))


def test_ptlassocv_fit_predict():
    X, y, groups = make_data()
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0])
    model.fit(X, y, groups)
    y_pred = model.predict(X, groups)
    assert y_pred.shape == (len(y),)
    assert hasattr(model, "alpha_")
    assert hasattr(model, "cv_results_")
