import warnings

import numpy as np
import pytest

from ptlasso import (
    PretrainedLasso,
    PretrainedLassoCV,
    get_overall_support,
    get_pretrain_support,
    get_pretrain_support_split,
    get_individual_support,
    make_data,
    gaussian_example_data,
    binomial_example_data,
    plot_cv,
    plot_paths,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _gaussian_data(n=120, p=20, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p) * (rng.random(p) < 0.3)
    y = X @ beta + rng.standard_normal(n) * 0.5
    groups = np.repeat(np.arange(k), n // k)
    return X, y, groups


def _binary_data(n=120, p=20, k=3, seed=1):
    X, yc, groups = _gaussian_data(n, p, k, seed)
    prob = 1 / (1 + np.exp(-yc))
    y = (np.random.default_rng(seed).random(n) < prob).astype(float)
    return X, y, groups


def _multi_data(n=150, p=20, k=3, K=3, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    W = rng.standard_normal((p, K))
    eta = X @ W
    y = eta.argmax(axis=1).astype(float)
    groups = np.repeat(np.arange(k), n // k)
    return X, y, groups


# ------------------------------------------------------------------
# Simulate helpers
# ------------------------------------------------------------------


def test_gaussian_example_data():
    d = gaussian_example_data(k=2, seed=42)
    assert d["X"].shape[0] == d["y"].shape[0] == d["groups"].shape[0]
    assert set(np.unique(d["groups"])) == {0, 1}


def test_binomial_example_data():
    d = binomial_example_data(k=2, seed=42)
    assert set(np.unique(d["y"])).issubset({0.0, 1.0})


def test_make_data_feature_layout():
    d = make_data(
        k=2, class_sizes=[50, 50], s_common=3, s_indiv=4, beta_common=1.0, beta_indiv=0.5, seed=7
    )
    assert d["X"].shape == (100, 3 + 4 * 2)


# ------------------------------------------------------------------
# PretrainedLasso — gaussian
# ------------------------------------------------------------------


def test_fit_predict_gaussian():
    X, y, groups = _gaussian_data()
    model = PretrainedLasso(alpha=0.5).fit(X, y, groups)
    y_pred = model.predict(X, groups)
    assert y_pred.shape == (len(y),)


def test_alpha_one_equals_individual():
    # alpha=1 → no pretraining → pretrain predictions == individual predictions
    X, y, groups = _gaussian_data()
    model = PretrainedLasso(alpha=1).fit(X, y, groups)
    np.testing.assert_allclose(
        model.predict(X, groups, model="pretrain"),
        model.predict(X, groups, model="individual"),
    )


def test_score_returns_float():
    X, y, groups = _gaussian_data()
    s = PretrainedLasso(alpha=0.5).fit(X, y, groups).score(X, y, groups)
    assert isinstance(s, float)


def test_evaluate_keys():
    X, y, groups = _gaussian_data()
    result = PretrainedLasso(alpha=0.5).fit(X, y, groups).evaluate(X, y, groups)
    assert set(result) == {"pretrain", "individual", "overall"}
    for v in result.values():
        assert "predictions" in v and "score" in v


def test_fitted_attributes():
    X, y, groups = _gaussian_data(k=2)
    model = PretrainedLasso().fit(X, y, groups)
    for attr in ("overall_model_", "overall_coef_", "pretrain_models_", "individual_models_"):
        assert hasattr(model, attr)
    assert set(model.groups_) == set(np.unique(groups))


def test_overall_lambda_min():
    X, y, groups = _gaussian_data()
    model = PretrainedLasso(alpha=0.5, overall_lambda="lambda.min").fit(X, y, groups)
    assert model.overall_lmda_idx_ >= 0


def test_feature_names():
    X, y, groups = _gaussian_data(p=5)
    names = [f"feat_{i}" for i in range(5)]
    model = PretrainedLasso(alpha=0.5).fit(X, y, groups, feature_names=names)
    assert list(model.feature_names_in_) == names


def test_get_coef_structure():
    X, y, groups = _gaussian_data(k=2)
    model = PretrainedLasso(alpha=0.5).fit(X, y, groups)
    coefs = model.get_coef()
    assert set(coefs) == {"overall", "pretrain", "individual"}
    assert set(coefs["overall"]) == {"coef", "intercept"}
    assert coefs["overall"]["coef"].shape == (X.shape[1],)
    # pretrain and individual are keyed by group label
    for section in ("pretrain", "individual"):
        assert len(coefs[section]) == len(model.groups_)
        for v in coefs[section].values():
            assert set(v) == {"coef", "intercept"}
            assert v["coef"].shape == (X.shape[1],)
    # single-model mode
    assert set(model.get_coef(model="overall")) == {"coef", "intercept"}


def test_get_pretrain_support_split():
    X, y, groups = _gaussian_data()
    model = PretrainedLasso(alpha=0.5).fit(X, y, groups)
    common, indiv = get_pretrain_support_split(model)
    # common = overall support; individual = stage-2 only
    ov = get_overall_support(model)
    assert set(common) == set(ov)
    # common and indiv must be disjoint
    assert len(set(common) & set(indiv)) == 0


# ------------------------------------------------------------------
# PretrainedLasso — binomial
# ------------------------------------------------------------------


def test_fit_predict_binomial():
    X, y, groups = _binary_data()
    model = PretrainedLasso(alpha=0.5, family="binomial").fit(X, y, groups)
    y_pred = model.predict(X, groups)
    assert y_pred.shape == (len(y),)
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_evaluate_binomial():
    X, y, groups = _binary_data()
    result = PretrainedLasso(alpha=0.5, family="binomial").fit(X, y, groups).evaluate(X, y, groups)
    for v in result.values():
        assert 0.0 <= v["score"] <= 1.0


# ------------------------------------------------------------------
# PretrainedLasso — multinomial
# ------------------------------------------------------------------


def test_fit_predict_multinomial():
    X, y, groups = _multi_data()
    model = PretrainedLasso(alpha=0.5, family="multinomial").fit(X, y, groups)
    y_pred = model.predict(X, groups)
    assert y_pred.shape == (len(y), model.n_classes_)
    np.testing.assert_allclose(y_pred.sum(axis=1), 1.0, atol=1e-6)


def test_evaluate_multinomial():
    X, y, groups = _multi_data()
    result = (
        PretrainedLasso(alpha=0.5, family="multinomial").fit(X, y, groups).evaluate(X, y, groups)
    )
    assert set(result) == {"pretrain", "individual", "overall"}


# ------------------------------------------------------------------
# Support functions
# ------------------------------------------------------------------


def test_support_functions():
    X, y, groups = _gaussian_data()
    model = PretrainedLasso(alpha=0.5).fit(X, y, groups)
    ov = get_overall_support(model)
    pre = get_pretrain_support(model)
    ind = get_individual_support(model)
    common, indiv = get_pretrain_support_split(model)
    assert isinstance(ov, np.ndarray)
    assert isinstance(pre, np.ndarray)
    assert isinstance(ind, np.ndarray)
    # common must be subset of pretrain support
    assert set(common).issubset(set(pre))


# ------------------------------------------------------------------
# PretrainedLassoCV
# ------------------------------------------------------------------


def test_cv_fit_predict():
    X, y, groups = _gaussian_data()
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3)
    model.fit(X, y, groups)
    y_pred = model.predict(X, groups)
    assert y_pred.shape == (len(y),)
    assert hasattr(model, "alpha_")
    assert model.alpha_ in [0.0, 0.5, 1.0]


def test_cv_results_keys():
    X, y, groups = _gaussian_data()
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3).fit(X, y, groups)
    for a in [0.0, 0.5, 1.0]:
        assert a in model.cv_results_
        assert a in model.cv_results_se_
        assert a in model.cv_results_per_group_
        assert a in model.cv_results_mean_
        assert a in model.cv_results_wtd_mean_
    assert hasattr(model, "cv_results_individual_")
    assert hasattr(model, "cv_results_overall_")


def test_cv_varying_alphahat():
    X, y, groups = _gaussian_data()
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3).fit(X, y, groups)
    assert hasattr(model, "varying_alphahat_")
    for g in model.groups_:
        assert model.varying_alphahat_[g] in [0.0, 0.5, 1.0]


def test_cv_all_estimators():
    X, y, groups = _gaussian_data()
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3).fit(X, y, groups)
    assert hasattr(model, "all_estimators_")
    # best alpha must be in all_estimators_
    assert model.alpha_ in model.all_estimators_


def test_cv_predict_varying():
    X, y, groups = _gaussian_data()
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3).fit(X, y, groups)
    y_pred = model.predict(X, groups, alphatype="varying")
    assert y_pred.shape == (len(y),)


def test_cv_evaluate():
    X, y, groups = _gaussian_data()
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3).fit(X, y, groups)
    result = model.evaluate(X, y, groups)
    assert set(result) == {"pretrain", "individual", "overall"}
    result_v = model.evaluate(X, y, groups, alphatype="varying")
    assert set(result_v) == {"pretrain", "individual", "overall"}


def test_cv_foldid():
    X, y, groups = _gaussian_data(n=90, k=3)
    foldid = np.tile(np.arange(3), 30)  # 3 folds of 30 samples each
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], foldid=foldid).fit(X, y, groups)
    assert hasattr(model, "alpha_")


def test_cv_alphahat_choice_mean():
    X, y, groups = _gaussian_data()
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3, alphahat_choice="mean").fit(
        X, y, groups
    )
    assert model.alpha_ in [0.0, 0.5, 1.0]


def test_cv_binomial():
    X, y, groups = _binary_data()
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3, family="binomial").fit(X, y, groups)
    y_pred = model.predict(X, groups)
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_cv_multinomial():
    X, y, groups = _multi_data()
    model = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3, family="multinomial").fit(X, y, groups)
    y_pred = model.predict(X, groups)
    assert y_pred.shape == (len(y), model.n_classes_)


# ------------------------------------------------------------------
# Plot smoke tests
# ------------------------------------------------------------------


def test_plot_paths_no_warnings():
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for CI
    X, y, groups = _gaussian_data(k=2)
    model = PretrainedLasso(alpha=0.5).fit(X, y, groups)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_paths(model)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_cv_no_warnings():
    import matplotlib

    matplotlib.use("Agg")
    X, y, groups = _gaussian_data()
    cv = PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3).fit(X, y, groups)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig, ax = plot_cv(cv)
    import matplotlib.pyplot as plt

    plt.close(fig)


# ------------------------------------------------------------------
# Repr smoke tests
# ------------------------------------------------------------------


def test_repr_unfitted():
    assert "not fitted" in repr(PretrainedLasso())
    assert "not fitted" in repr(PretrainedLassoCV())


def test_repr_fitted():
    X, y, groups = _gaussian_data()
    r = repr(PretrainedLasso(alpha=0.5).fit(X, y, groups))
    assert "overall" in r
    r_cv = repr(PretrainedLassoCV(alphas=[0.0, 0.5, 1.0], cv=3).fit(X, y, groups))
    assert "alpha_" in r_cv


# ------------------------------------------------------------------
# Validation errors
# ------------------------------------------------------------------


def test_bad_alpha():
    with pytest.raises(ValueError):
        PretrainedLasso(alpha=1.5).fit(*_gaussian_data())


def test_bad_family():
    with pytest.raises(ValueError):
        PretrainedLasso(family="poisson").fit(*_gaussian_data())


def test_bad_overall_lambda():
    with pytest.raises(ValueError):
        PretrainedLasso(overall_lambda="lambda.2se").fit(*_gaussian_data())


def test_single_group_raises():
    X, y, _ = _gaussian_data()
    groups = np.zeros(len(y), dtype=int)
    with pytest.raises(ValueError):
        PretrainedLasso().fit(X, y, groups)
