"""Core estimators for the Pretrained Lasso.

Classes
-------
PretrainedLasso
    Two-step estimator: overall Lasso followed by per-group Lasso with offset.
PretrainedLassoCV
    Same estimator with cross-validation over the pretraining strength alpha.
"""

import contextlib
import os
import time

import adelie as ad
import numpy as np

from sklearn.base import RegressorMixin
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._base import BasePretrainedLasso
from ._constants import (
    FAMILIES,
    LMDA_MODES,
    PREDICT_MODELS,
    PREDICT_TYPES,
    COEF_MODELS,
    ALPHATYPES,
    DEFAULT_ALPHAS,
)


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------


class _StateProxy:
    """Picklable substitute for an adelie grpnet state object.

    Adelie state objects hold C++ pybind11-bound internals that cannot be
    pickled.  This class stores only the numpy arrays needed by ptlasso and
    exposes the same ``betas``, ``lmdas``, and ``intercepts`` attributes so
    that the rest of the code works without modification.
    """

    __slots__ = ("betas", "lmdas", "intercepts")

    def __init__(self, betas, lmdas, intercepts):
        self.betas = betas  # (L, p) or (L, p*K) float64 ndarray
        self.lmdas = lmdas  # (L,) float64 ndarray
        self.intercepts = intercepts  # (L,) or (L, K) float64 ndarray


def _state_to_proxy(state):
    """Convert an adelie state object to a picklable :class:`_StateProxy`."""
    b = state.betas
    betas = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
    return _StateProxy(
        betas=betas,
        lmdas=np.asarray(state.lmdas),
        intercepts=np.asarray(state.intercepts),
    )


def _proxify_models(d):
    """Replace any raw adelie state dicts with proxy dicts in place."""
    if "overall_model_" in d:
        d["overall_model_"] = _state_to_proxy(d["overall_model_"])
    for key in ("pretrain_models_", "individual_models_"):
        if key in d:
            d[key] = {g: _state_to_proxy(s) for g, s in d[key].items()}


# ------------------------------------------------------------------
# adelie helpers
# ------------------------------------------------------------------


@contextlib.contextmanager
def _silence():
    """Suppress stdout/stderr at the file-descriptor level.

    Redirects fds 1 and 2 to /dev/null so output from C extensions
    (such as adelie's solver messages) is always swallowed.
    """
    null_fd = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(1), os.dup(2)]
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        os.close(null_fd)


def _coef_at(state, lmda_idx):
    """Dense coefficient vector from an adelie state at a given lambda index."""
    beta = state.betas[lmda_idx]
    return beta.toarray().ravel() if hasattr(beta, "toarray") else np.asarray(beta).ravel()


def _lmda_1se_idx(cv_result):
    """Largest-lambda (most regularised) index within 1 SE of the CV minimum.

    adelie stores lambdas in decreasing order, so a smaller index means more
    regularisation.  lambda.1se is the leftmost index whose mean CV loss is
    still within 1 SE of the minimum.
    """
    avg = np.asarray(cv_result.avg_losses)
    folds = np.asarray(cv_result.losses)  # (n_folds, L)
    se = folds.std(axis=0, ddof=1) / np.sqrt(folds.shape[0])
    threshold = avg[cv_result.best_idx] + se[cv_result.best_idx]
    candidates = np.where(avg <= threshold)[0]
    return int(candidates[0]) if len(candidates) else cv_result.best_idx


# ------------------------------------------------------------------
# Family helpers
# ------------------------------------------------------------------


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(eta):
    """Row-wise softmax of an (n, K) array."""
    e = np.exp(eta - eta.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _make_glm(family, y):
    """Construct an adelie GLM object for the given family and response."""
    if family == "gaussian":
        return ad.glm.gaussian(y)
    if family == "binomial":
        return ad.glm.binomial(y)
    if family == "multinomial":
        y = np.asarray(y, dtype=int)
        K = int(y.max()) + 1
        mat = np.zeros((len(y), K), dtype=np.float64, order="F")
        mat[np.arange(len(y)), y] = 1.0
        return ad.glm.multinomial(mat)
    raise ValueError(f"family must be one of {FAMILIES}, got '{family}'")


def _eta_from_state(state, X, lmda_idx, family, n_classes=None):
    """Linear predictor (before link): (n,) for gaussian/binomial, (n, K) for multinomial."""
    if family == "multinomial":
        coef_mat = _coef_at(state, lmda_idx).reshape(X.shape[1], n_classes, order="F")
        intercepts = np.asarray(state.intercepts[lmda_idx]).ravel()
        return X @ coef_mat + intercepts
    coef = _coef_at(state, lmda_idx)
    intercept = float(state.intercepts[lmda_idx])
    return X @ coef + intercept


def _apply_link(eta, family):
    """Apply the inverse link function to a linear predictor."""
    if family == "multinomial":
        return _softmax(eta)
    if family == "binomial":
        return _sigmoid(eta)
    return eta  # gaussian: identity


def _eta_to_output(eta, family, type):
    """Convert a linear predictor to the requested output type.

    Parameters
    ----------
    eta : ndarray
        Linear predictor, shape ``(n,)`` or ``(n, K)`` for multinomial.
    family : str
        One of ``"gaussian"``, ``"binomial"``, ``"multinomial"``.
    type : {"response", "link", "class"}
    """
    if type == "link":
        return eta
    proba = _apply_link(eta, family)
    if type == "response":
        return proba
    # type == "class"
    if family == "binomial":
        return (proba >= 0.5).astype(int)
    return proba.argmax(axis=1).astype(int)  # multinomial


def _model_score(y_true, y_pred, family):
    """Scalar performance metric, higher = better (R² or accuracy)."""
    if family == "gaussian":
        return r2_score(y_true, y_pred)
    if family == "binomial":
        return float(np.mean((y_pred >= 0.5) == np.asarray(y_true, dtype=bool)))
    return float(np.mean(y_pred.argmax(axis=1) == np.asarray(y_true, dtype=int)))


def _fold_loss(y_true, y_pred, family):
    """Scalar loss, lower = better (MSE or log-loss)."""
    if family == "gaussian":
        return float(np.mean((y_true - y_pred) ** 2))
    eps = 1e-15
    if family == "binomial":
        p = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    p = np.clip(y_pred, eps, 1)  # multinomial: (n, K)
    idx = y_true.astype(int)
    return float(-np.mean(np.log(p[np.arange(len(idx)), idx])))


# ------------------------------------------------------------------
# Scorer registry (used by PretrainedLassoCV)
# ------------------------------------------------------------------

# All built-in scorers return higher = better, matching the sklearn convention.
_SCORERS = {
    "roc_auc": roc_auc_score,
    "neg_log_loss": lambda yt, yp: -log_loss(yt, yp),
    "accuracy": lambda yt, yp: accuracy_score(
        yt,
        (yp >= 0.5).astype(int) if np.ndim(yp) == 1 else np.argmax(yp, axis=1),
    ),
    "neg_mean_squared_error": lambda yt, yp: -mean_squared_error(yt, yp),
    "r2": r2_score,
}

_VALID_SCORERS = tuple(_SCORERS.keys())


def _resolve_scorer(scoring):
    """Return a callable ``(y_true, y_pred) -> float`` (higher = better), or ``None``."""
    if scoring is None or callable(scoring):
        return scoring
    return _SCORERS[scoring]


def _cv_loss(scorer_fn, y_true, y_pred, family):
    """Unified CV loss (lower = better) used throughout the CV loop.

    When *scorer_fn* is ``None`` the family-based loss is used.  Otherwise the
    scorer (higher = better) is negated so that the existing minimisation logic
    is unchanged.
    """
    if scorer_fn is None:
        return _fold_loss(y_true, y_pred, family)
    return -float(scorer_fn(y_true, y_pred))


# ------------------------------------------------------------------
# PretrainedLasso
# ------------------------------------------------------------------


class PretrainedLasso(RegressorMixin, BasePretrainedLasso):
    """Pretrained Lasso estimator.

    Two-step training:
    1. Fit an overall Lasso on all samples (lambda selected by internal CV).
    2. For each group, fit a group-specific Lasso with offset
       ``(1 - alpha) * eta_overall``, where ``eta_overall`` is the overall
       linear predictor (before the link function).  Features not selected
       by the overall model receive a stronger penalty of ``1 / alpha``.

    Parameters
    ----------
    alpha : float in [0, 1], default=0.5
        Pretraining strength.  ``0`` = overall model with fine-tuning
        (maximum pretraining); ``1`` = individual per-group Lasso
        (no pretraining).  Matches the R ptLasso convention.
    family : {"gaussian", "binomial", "multinomial"}, default="gaussian"
        Response distribution.
    overall_lambda : {"lambda.1se", "lambda.min"}, default="lambda.1se"
        Lambda selection rule for the stage-1 overall model.
        ``"lambda.1se"`` (default, matching R) gives a sparser offset;
        ``"lambda.min"`` uses the CV minimum.
    fit_intercept : bool, default=True
        Whether to fit an intercept in every sub-model.
    lmda_path_size : int, default=100
        Number of lambdas in the regularisation path.
    min_ratio : float, default=0.0001
        Ratio of the smallest to largest lambda on the path.
    verbose : bool, default=True
        Whether to display fitting progress and a summary after training.
        Adelie's internal output is always suppressed regardless of this setting.
    n_threads : int, default=-1
        Number of threads passed to adelie's solver.  Set to a higher value
        to parallelise the coordinate descent within each model fit.
        ``-1`` uses all available CPU cores (``os.cpu_count()``).

    Attributes
    ----------
    overall_model_ : adelie state
        Fitted overall Lasso (stage 1).
    overall_coef_ : ndarray of shape (n_features,) or (n_features, K)
        Coefficients from the overall model at the selected lambda.
        Shape is ``(n_features, K)`` for multinomial.
    overall_intercept_ : float
        Intercept from the overall model.  Not set for multinomial.
    overall_lmda_idx_ : int
        Index into ``overall_model_.lmdas`` for the selected lambda.
    pretrain_models_ : dict {group -> adelie state}
        Per-group fitted Lasso models (stage 2, with pretraining offset).
    pretrain_lmda_idx_ : dict {group -> int}
        CV-selected lambda index for each pretrain group model (``lambda.min``).
    individual_models_ : dict {group -> adelie state}
        Per-group fitted Lasso models without any pretraining offset.
    individual_lmda_idx_ : dict {group -> int}
        CV-selected lambda index for each individual group model (``lambda.min``).
    groups_ : ndarray
        Unique group labels seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of str or None
        Feature names, if provided.
    n_classes_ : int or None
        Number of classes for multinomial; ``None`` otherwise.

    References
    ----------
    Craig, E., Pilanci, M., Le Menestrel, T., Narasimhan, B., Rivas, M. A.,
    Gullaksen, S. E., & Tibshirani, R. (2025). Pretraining and the lasso.
    *Journal of the Royal Statistical Society Series B*, qkaf050.
    """

    def __init__(
        self,
        alpha=0.5,
        family="gaussian",
        overall_lambda="lambda.1se",
        fit_intercept=True,
        lmda_path_size=100,
        min_ratio=0.0001,
        verbose=True,
        n_threads=-1,
        standardize=True,
    ):
        self.alpha = alpha
        self.family = family
        self.overall_lambda = overall_lambda
        self.fit_intercept = fit_intercept
        self.lmda_path_size = lmda_path_size
        self.min_ratio = min_ratio
        self.verbose = verbose
        self.n_threads = n_threads
        self.standardize = standardize

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_params(self):
        if not isinstance(self.alpha, (int, float)):
            raise TypeError(f"alpha must be a number, got {type(self.alpha).__name__}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got '{self.family}'")
        if self.overall_lambda not in LMDA_MODES:
            raise ValueError(
                f"overall_lambda must be one of {LMDA_MODES}, got '{self.overall_lambda}'"
            )
        if not isinstance(self.lmda_path_size, int) or self.lmda_path_size < 1:
            raise ValueError(
                f"lmda_path_size must be a positive integer, got {self.lmda_path_size}"
            )
        if not isinstance(self.min_ratio, (int, float)):
            raise TypeError(f"min_ratio must be a number, got {type(self.min_ratio).__name__}")
        if not (0.0 < self.min_ratio < 1.0):
            raise ValueError(f"min_ratio must be in (0, 1), got {self.min_ratio}")

    def _label(self, g):
        return self.group_labels_.get(g, g)

    def _names_or_indices(self, indices):
        if self.feature_names_in_ is not None:
            return self.feature_names_in_[indices]
        return indices

    def _make_onehot(self, groups):
        """Build (n, k-1) group indicator matrix.

        The first group in ``self.groups_`` is the reference category (all
        zeros), matching R's ``model.matrix(~groups - 1)[, 2:k]`` convention.
        These columns are prepended to X in the overall model with zero
        penalty so they act as free group-specific intercepts (R's
        ``group.intercepts = TRUE``).
        """
        k = len(self.groups_)
        n = len(groups)
        if k <= 1:
            return np.empty((n, 0), dtype=np.float64, order="F")
        onehot = np.zeros((n, k - 1), dtype=np.float64, order="F")
        for col, g in enumerate(self.groups_[1:]):  # groups_[0] is reference
            onehot[groups == g, col] = 1.0
        return onehot

    def _grpnet_kwargs(self):
        n_threads = os.cpu_count() if self.n_threads == -1 else self.n_threads
        return dict(
            alpha=1,  # pure lasso (no group penalty mixing)
            intercept=self.fit_intercept,
            lmda_path_size=self.lmda_path_size,
            min_ratio=self.min_ratio,
            progress_bar=False,  # replaced by ptlasso's own tqdm output
            n_threads=n_threads,
        )

    def _wrap_matrix(self, X):
        # adelie incurs ~20x slowdown when matrix and solver use different
        # n_threads (OpenMP switching cost). Wrap X so both use the same value.
        n_threads = os.cpu_count() if self.n_threads == -1 else self.n_threads
        return ad.matrix.dense(X, method="naive", n_threads=n_threads)

    def _overall_eta(self, X, groups):
        """Overall linear predictor at the selected lambda.

        The overall model was trained on ``[onehot | X]``, so we reconstruct
        the augmented matrix before computing the linear predictor.  This
        matches R's predict behaviour where group-intercept columns are always
        included.
        """
        onehot = self._make_onehot(groups)
        X_aug = (
            np.asfortranarray(np.hstack([onehot, X]))
            if onehot.shape[1] > 0
            else np.asfortranarray(X)
        )
        return _eta_from_state(
            self.overall_model_, X_aug, self.overall_lmda_idx_, self.family, self.n_classes_
        )

    def _compute_oof_eta(self, X_overall, y, overall_pf, groups):
        """Compute out-of-fold linear predictors for the overall model.

        Mirrors R's ``cv.glmnet(..., keep=TRUE)`` which stores prevalidated
        (out-of-fold) predictions.  Each sample's prediction comes from a
        model that never saw that sample during training, so the offset used
        in stage-2 fitting is free of in-sample optimism.

        Parameters
        ----------
        X_overall : (n, k-1+p) ndarray
            Augmented feature matrix including group-indicator columns.
        y : (n,) ndarray
            Target values.
        overall_pf : (k-1+p,) ndarray
            Penalty factors (0 for group-dummy columns, 1 for features).
        groups : (n,) ndarray
            Group membership for each sample, used to create stratified folds.

        Returns
        -------
        oof_eta : (n,) or (n, K) ndarray
            Out-of-fold linear predictors in the same shape as the overall
            model's ``_eta_from_state`` output.
        """
        n = X_overall.shape[0]
        lamhat = float(self.overall_model_.lmdas[self.overall_lmda_idx_])

        if self.family == "multinomial":
            oof_eta = np.zeros((n, self.n_classes_))
        else:
            oof_eta = np.zeros(n)

        # Determine number of folds: cap at min group size, floor at 2.
        n_folds = min(10, min(int(np.sum(groups == g)) for g in self.groups_))
        n_folds = max(2, n_folds)

        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        _show_oof = getattr(self, "_show_overall_progress", False)
        _oof_pbar = None
        if _show_oof:
            from tqdm import tqdm as _tqdm

            _oof_pbar = _tqdm(total=n_folds, desc="  OOF folds", unit="fold", leave=False)
            _oof_pbar.refresh()
        for train_idx, test_idx in splitter.split(X_overall, groups):
            X_tr = np.asfortranarray(X_overall[train_idx])
            y_tr = y[train_idx]
            X_te = np.asfortranarray(X_overall[test_idx])

            glm_fold = _make_glm(self.family, y_tr)
            with _silence():
                fold_state = ad.grpnet(
                    self._wrap_matrix(X_tr), glm_fold, penalty=overall_pf, **self._grpnet_kwargs()
                )

            # Find the lambda in this fold's path closest to the full-data lamhat.
            lmdas = np.asarray(fold_state.lmdas)
            lmda_idx = int(np.argmin(np.abs(lmdas - lamhat))) if len(lmdas) > 0 else 0

            oof_eta[test_idx] = _eta_from_state(
                fold_state, X_te, lmda_idx, self.family, self.n_classes_
            )
            if _oof_pbar is not None:
                _oof_pbar.update(1)
                _oof_pbar.refresh()
        if _oof_pbar is not None:
            _oof_pbar.close()

        return oof_eta

    # ------------------------------------------------------------------
    # Progress / display helpers
    # ------------------------------------------------------------------

    def _support_size(self, state, lmda_idx):
        """Number of nonzero features in a fitted state at a given lambda index."""
        c = _coef_at(state, lmda_idx)
        if self.n_classes_ is not None:
            c = c.reshape(self.n_features_in_, self.n_classes_, order="F")
            return int(np.sum(np.any(c != 0, axis=1)))
        return int(np.sum(c != 0))

    def _print_fit_summary(self, elapsed):
        SEP = "─" * 54
        if self.family == "multinomial":
            n_ov = int(np.sum(np.any(self.overall_coef_ != 0, axis=1)))
        else:
            n_ov = int(np.sum(self.overall_coef_ != 0))
        pre_parts = "   ".join(
            f"{self._label(g)}: |S|={self._support_size(self.pretrain_models_[g], self.pretrain_lmda_idx_[g])}"
            for g in self.groups_
        )
        ind_parts = "   ".join(
            f"{self._label(g)}: |S|={self._support_size(self.individual_models_[g], self.individual_lmda_idx_[g])}"
            for g in self.groups_
        )
        print(SEP)
        print(f"  {'overall':<13}|S| = {n_ov}")
        print(f"  {'pretrain':<13}{pre_parts}")
        print(f"  {'individual':<13}{ind_parts}")
        print(SEP)
        print(f"  Fitted in {elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, groups, group_labels=None, feature_names=None):
        """Fit the pretrained Lasso.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.  For ``family="binomial"``, must contain only 0
            and 1.  For ``family="multinomial"``, must contain non-negative
            integer class labels 0..K-1.
        groups : array-like of shape (n_samples,)
            Group membership for each sample.  Must contain at least two
            distinct values.
        group_labels : dict or None, default=None
            Optional mapping from group values to display names used in
            ``__repr__`` and ``get_coef()``.
        feature_names : array-like of str or None, default=None
            Feature names.  Inferred from ``X.columns`` when ``X`` is a
            DataFrame and this argument is ``None``.

        Returns
        -------
        self : PretrainedLasso
            Fitted estimator.
        """
        t0 = time.time()
        self._validate_params()

        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)

        X, y = check_X_y(X, y, dtype=np.float64, order="F")
        self.n_features_in_ = X.shape[1]

        # Standardize features to zero mean / unit std, matching glmnet's
        # default (standardize=TRUE).  The one-hot group-indicator columns
        # added later are built after this step and left un-standardized.
        self.scaler_ = StandardScaler(with_mean=True, with_std=self.standardize)
        X = self.scaler_.fit_transform(X)

        if feature_names is not None:
            feature_names = np.asarray(feature_names)
            if len(feature_names) != self.n_features_in_:
                raise ValueError(
                    f"feature_names has {len(feature_names)} entries "
                    f"but X has {self.n_features_in_} features"
                )
        self.feature_names_in_ = feature_names

        if self.family == "binomial":
            unique_y = np.unique(y)
            if not np.all(np.isin(unique_y, [0.0, 1.0])):
                raise ValueError(
                    f"For family='binomial', y must contain only 0 and 1, "
                    f"got unique values {unique_y}"
                )
        if self.family == "multinomial":
            y_int = np.asarray(y)
            if not (np.all(y_int == np.round(y_int)) and np.all(y_int >= 0)):
                raise ValueError(
                    "For family='multinomial', y must contain non-negative integer class labels"
                )

        groups = np.asarray(groups)
        if len(groups) != X.shape[0]:
            raise ValueError(f"groups has {len(groups)} elements but X has {X.shape[0]} samples")
        self.groups_ = np.unique(groups)
        if len(self.groups_) < 2:
            raise ValueError("groups must contain at least 2 unique values")

        if group_labels is not None:
            if not isinstance(group_labels, dict):
                raise TypeError(f"group_labels must be a dict, got {type(group_labels).__name__}")
            unknown = set(group_labels) - set(self.groups_)
            if unknown:
                raise ValueError(f"group_labels contains unknown group keys: {unknown}")
        self.group_labels_ = group_labels or {}

        self.n_classes_ = (
            int(np.asarray(y, dtype=int).max()) + 1 if self.family == "multinomial" else None
        )

        if self.verbose:
            k = len(self.groups_)
            glabels = [str(self._label(g)) for g in self.groups_]
            gstr = ", ".join(glabels[:4]) + (f" +{k - 4} more" if k > 4 else "")
            print(
                f"\nPretrainedLasso  {self.family}  ·  {k} groups ({gstr})"
                f"  ·  {self.n_features_in_} features  ·  α={self.alpha}"
            )

        # ----------------------------------------------------------
        # Build augmented X for the overall model (group intercepts).
        # Matches R's group.intercepts = TRUE: one-hot group dummies (k-1
        # columns, first group = reference) are prepended to X with zero
        # penalty so they act as free intercepts per group.
        # ----------------------------------------------------------
        onehot = self._make_onehot(groups)  # (n, k-1)
        n_onehot = onehot.shape[1]  # = k-1 >= 1
        self._n_onehot_ = n_onehot
        X_overall = (
            np.asfortranarray(np.hstack([onehot, X])) if n_onehot > 0 else np.asfortranarray(X)
        )
        # Zero penalty for group-dummy columns; unit penalty for X features.
        overall_pf = np.concatenate([np.zeros(n_onehot), np.ones(self.n_features_in_)])

        # ----------------------------------------------------------
        # Step 1: overall model (lambda selected by CV)
        # ----------------------------------------------------------
        glm_all = _make_glm(self.family, y)
        if self.verbose:
            print("  [1/2] Overall model (CV) ...", flush=True)
            _t1 = time.time()

        # cv_grpnet (lambda selection via internal CV folds) — always silent.
        with _silence():
            cv_overall = ad.cv_grpnet(
                self._wrap_matrix(X_overall), glm_all, penalty=overall_pf, **self._grpnet_kwargs()
            )
            self.overall_lmda_idx_ = (
                cv_overall.best_idx
                if self.overall_lambda == "lambda.min"
                else _lmda_1se_idx(cv_overall)
            )

        # Final refit on full data — show adelie's lambda-path progress bar when
        # either verbose=True (standalone use) or the CV loop opts in via
        # _show_overall_progress (so the user sees progress during the long fit).
        _show_progress = self.verbose or getattr(self, "_show_overall_progress", False)
        if _show_progress:
            self.overall_model_ = cv_overall.fit(
                self._wrap_matrix(X_overall),
                glm_all,
                penalty=overall_pf,
                **{**self._grpnet_kwargs(), "progress_bar": True},
            )
        else:
            with _silence():
                self.overall_model_ = cv_overall.fit(
                    self._wrap_matrix(X_overall),
                    glm_all,
                    penalty=overall_pf,
                    **self._grpnet_kwargs(),
                )

        # Extract X-feature coefficients only (skip the k-1 onehot columns).
        # overall_coef_ has shape (p,) or (p, K) — identical to the no-group-
        # intercepts case from the caller's perspective.
        if self.family == "multinomial":
            flat = _coef_at(self.overall_model_, self.overall_lmda_idx_)
            p_aug = X_overall.shape[1]
            coef_mat = flat.reshape(p_aug, self.n_classes_, order="F")
            self.overall_coef_ = coef_mat[n_onehot:, :]  # (p, K)
        else:
            coef_full = _coef_at(self.overall_model_, self.overall_lmda_idx_)
            self.overall_coef_ = coef_full[n_onehot:]  # (p,)
            self.overall_intercept_ = float(self.overall_model_.intercepts[self.overall_lmda_idx_])

        # ----------------------------------------------------------
        # Compute OOF overall linear predictor (matches R's keep=TRUE).
        # Using in-sample predictions as the offset introduces optimism:
        # the overall model has already seen the training samples, so its
        # predictions are sharper than true held-out predictions.  OOF
        # predictions avoid this leakage.
        # ----------------------------------------------------------
        preval_offset = self._compute_oof_eta(X_overall, y, overall_pf, groups)
        self._preval_offset = preval_offset  # cached for PretrainedLassoCV reuse

        # ----------------------------------------------------------
        # Step 2: per-group models
        # ----------------------------------------------------------
        # Penalty factor: features NOT in the overall X-support get penalty
        # 1/alpha, steering the group model to prefer features already
        # selected overall.  Matches R's fac = rep(1/alpha, p); fac[supall] = 1.
        if self.family == "multinomial":
            overall_support = np.where(np.any(self.overall_coef_ != 0, axis=1))[0]
        else:
            overall_support = np.where(self.overall_coef_ != 0)[0]

        _alpha_pf = self.alpha if self.alpha > 0 else 1e-9
        pf = np.full(self.n_features_in_, 1.0 / _alpha_pf)
        pf[overall_support] = 1.0

        if self.verbose:
            if self.family == "multinomial":
                n_ov = int(np.sum(np.any(self.overall_coef_ != 0, axis=1)))
            else:
                n_ov = int(np.sum(self.overall_coef_ != 0))
            print(f" done  ({time.time() - _t1:.1f}s)  |S|={n_ov}")

        self.pretrain_models_ = {}
        self.pretrain_lmda_idx_ = {}
        self.individual_models_ = {}
        self.individual_lmda_idx_ = {}

        if self.verbose:
            try:
                from tqdm import tqdm as _tqdm

                group_iter = _tqdm(
                    self.groups_, desc="  [2/2] Group models", unit="group", leave=True
                )
            except ImportError:
                print("  [2/2] Group models")
                group_iter = self.groups_
        else:
            group_iter = self.groups_

        for g in group_iter:
            mask = groups == g
            X_g = np.asfortranarray(X[mask])
            glm_g = _make_glm(self.family, y[mask])
            # OOF offset: each sample's overall prediction came from a model
            # trained without that sample — mirrors R's use of fit.preval.
            offset = np.asfortranarray((1 - self.alpha) * preval_offset[mask])

            # Mirrors R's cv.glmnet for per-group models: select lambda by CV
            # (lambda.min), then refit on the full group data at that lambda.
            with _silence():
                cv_pre = ad.cv_grpnet(
                    self._wrap_matrix(X_g),
                    glm_g,
                    offsets=offset,
                    penalty=pf,
                    **self._grpnet_kwargs(),
                )
                self.pretrain_lmda_idx_[g] = cv_pre.best_idx
                self.pretrain_models_[g] = cv_pre.fit(
                    self._wrap_matrix(X_g),
                    glm_g,
                    offsets=offset,
                    penalty=pf,
                    **self._grpnet_kwargs(),
                )

                cv_ind = ad.cv_grpnet(self._wrap_matrix(X_g), glm_g, **self._grpnet_kwargs())
                self.individual_lmda_idx_[g] = cv_ind.best_idx
                self.individual_models_[g] = cv_ind.fit(
                    self._wrap_matrix(X_g), glm_g, **self._grpnet_kwargs()
                )

        if self.verbose:
            self._print_fit_summary(time.time() - t0)

        return self

    def _fit_groups_only(self, X, y, groups, preval_offset):
        """Fit stage-2 group models, reusing the stage-1 state already on self.

        Called by PretrainedLassoCV to avoid re-running the expensive overall
        model + OOF computation for every alpha within the same CV fold.
        Requires that all stage-1 attributes (overall_model_, overall_coef_,
        overall_lmda_idx_, n_features_in_, groups_, _n_onehot_, etc.) are
        already set (typically copied from a template estimator).
        """
        X = np.asfortranarray(self.scaler_.transform(np.asarray(X, dtype=np.float64)))
        y = np.asarray(y, dtype=np.float64)
        groups = np.asarray(groups)

        if self.family == "multinomial":
            overall_support = np.where(np.any(self.overall_coef_ != 0, axis=1))[0]
        else:
            overall_support = np.where(self.overall_coef_ != 0)[0]

        _alpha_pf = self.alpha if self.alpha > 0 else 1e-9
        pf = np.full(self.n_features_in_, 1.0 / _alpha_pf)
        pf[overall_support] = 1.0

        self.pretrain_models_ = {}
        self.pretrain_lmda_idx_ = {}
        self.individual_models_ = {}
        self.individual_lmda_idx_ = {}

        for g in self.groups_:
            mask = groups == g
            X_g = np.asfortranarray(X[mask])
            glm_g = _make_glm(self.family, y[mask])
            offset = np.asfortranarray((1 - self.alpha) * preval_offset[mask])

            with _silence():
                cv_pre = ad.cv_grpnet(
                    self._wrap_matrix(X_g),
                    glm_g,
                    offsets=offset,
                    penalty=pf,
                    **self._grpnet_kwargs(),
                )
                self.pretrain_lmda_idx_[g] = cv_pre.best_idx
                self.pretrain_models_[g] = cv_pre.fit(
                    self._wrap_matrix(X_g),
                    glm_g,
                    offsets=offset,
                    penalty=pf,
                    **self._grpnet_kwargs(),
                )

                cv_ind = ad.cv_grpnet(self._wrap_matrix(X_g), glm_g, **self._grpnet_kwargs())
                self.individual_lmda_idx_[g] = cv_ind.best_idx
                self.individual_models_[g] = cv_ind.fit(
                    self._wrap_matrix(X_g), glm_g, **self._grpnet_kwargs()
                )

    # ------------------------------------------------------------------
    # predict / score / evaluate
    # ------------------------------------------------------------------

    def predict(self, X, groups, model="pretrain", type="response", lmda_idx=None):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        groups : array-like of shape (n_samples,)
        model : {"pretrain", "individual", "overall"}, default="pretrain"
        type : {"response", "link", "class"}, default="response"
            Scale of the returned predictions.

            ``"response"`` — fitted values on the response scale: probabilities
            for binomial/multinomial, fitted values for gaussian.

            ``"link"`` — raw linear predictor before the link function
            (``η = Xβ + intercept``).  For gaussian this is identical to
            ``"response"``.

            ``"class"`` — predicted class label: ``0`` or ``1`` for binomial
            (threshold 0.5), integer argmax for multinomial.  Not valid for
            gaussian.
        lmda_idx : int or None
            Lambda index for group models.  ``None`` uses the CV-selected
            lambda for each group (``lambda.min``, matching R).

        Returns
        -------
        y_pred : ndarray
            Shape ``(n,)`` for gaussian/binomial and for multinomial
            ``"class"``.  Shape ``(n, K)`` for multinomial ``"response"``
            or ``"link"``.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, order="F")
        X = np.asfortranarray(self.scaler_.transform(X))
        groups = np.asarray(groups)

        if model not in PREDICT_MODELS:
            raise ValueError(f"model must be one of {PREDICT_MODELS}, got '{model}'")
        if type not in PREDICT_TYPES:
            raise ValueError(f"type must be one of {PREDICT_TYPES}, got '{type}'")
        if type == "class" and self.family == "gaussian":
            raise ValueError("type='class' is not valid for family='gaussian'")
        if len(groups) != X.shape[0]:
            raise ValueError(f"groups has {len(groups)} elements but X has {X.shape[0]} samples")
        unknown = set(np.unique(groups)) - set(self.groups_)
        if unknown:
            raise ValueError(f"predict received groups not seen during fit: {unknown}")

        if model == "overall":
            return _eta_to_output(self._overall_eta(X, groups), self.family, type)

        n_out = (X.shape[0], self.n_classes_) if self.family == "multinomial" else (X.shape[0],)
        eta_out = np.empty(n_out)

        # Overall eta is only needed when combining with a group model (pretrain).
        eta_ov = self._overall_eta(X, groups) if model == "pretrain" else None

        for g in self.groups_:
            mask = groups == g
            X_g = X[mask]

            if model == "pretrain":
                g_idx = self.pretrain_lmda_idx_.get(g, -1) if lmda_idx is None else lmda_idx
                eta_group = _eta_from_state(
                    self.pretrain_models_[g], X_g, g_idx, self.family, self.n_classes_
                )
                eta_out[mask] = (1 - self.alpha) * eta_ov[mask] + eta_group
            else:
                g_idx = self.individual_lmda_idx_.get(g, -1) if lmda_idx is None else lmda_idx
                eta_out[mask] = _eta_from_state(
                    self.individual_models_[g], X_g, g_idx, self.family, self.n_classes_
                )

        return _eta_to_output(eta_out, self.family, type)

    def score(self, X, y, groups):
        """Return a scalar performance metric using the pretrained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        groups : array-like of shape (n_samples,)

        Returns
        -------
        score : float
            R² for gaussian; classification accuracy for binomial/multinomial.
        """
        return _model_score(y, self.predict(X, groups), self.family)

    def evaluate(self, X, y, groups):
        """Predict and score with all three sub-models.

        Convenience method matching R's ``predict(fit, xtest, ytest=ytest)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        groups : array-like of shape (n_samples,)

        Returns
        -------
        result : dict
            Keys are ``"pretrain"``, ``"individual"``, ``"overall"``.
            Each value is a dict with entries:

            ``"predictions"`` : ndarray
                Predicted values, shape ``(n,)`` or ``(n, K)`` for multinomial.
            ``"score"`` : float
                R² for gaussian; accuracy for binomial/multinomial.
        """
        check_is_fitted(self)
        result = {}
        for m in ("pretrain", "individual", "overall"):
            preds = self.predict(X, groups, model=m)
            result[m] = {"predictions": preds, "score": _model_score(y, preds, self.family)}
        return result

    # ------------------------------------------------------------------
    # repr / get_coef
    # ------------------------------------------------------------------

    def __repr__(self):
        header = (
            f"PretrainedLasso(alpha={self.alpha}, family='{self.family}', "
            f"overall_lambda='{self.overall_lambda}', "
            f"fit_intercept={self.fit_intercept}, lmda_path_size={self.lmda_path_size})"
        )
        if not hasattr(self, "overall_model_"):
            return header + "\n  [not fitted]"

        coef = self.overall_coef_
        nz_idx = (
            np.where(np.any(coef != 0, axis=1))[0]
            if self.family == "multinomial"
            else np.where(coef != 0)[0]
        )
        names = self._names_or_indices(nz_idx)
        preview = list(names[:5]) + (["..."] if len(nz_idx) > 5 else [])
        ov_str = (
            f"|Ŝ| = {len(nz_idx)} / {self.n_features_in_}  [{', '.join(str(v) for v in preview)}]"
        )

        group_sizes = {}
        for g in self.groups_:
            c = _coef_at(self.pretrain_models_[g], self.pretrain_lmda_idx_.get(g, -1))
            if self.family == "multinomial":
                c = c.reshape(self.n_features_in_, self.n_classes_, order="F")
                group_sizes[self._label(g)] = int(np.sum(np.any(c != 0, axis=1)))
            else:
                group_sizes[self._label(g)] = int(np.sum(c != 0))
        group_str = ", ".join(f"{lbl}: |Ŝ|={v}" for lbl, v in group_sizes.items())

        return (
            f"{header}\n"
            f"  family       : {self.family}\n"
            f"  n_features   : {self.n_features_in_}\n"
            f"  n_groups     : {len(self.groups_)}\n"
            f"  overall |Ŝ|  : {ov_str}\n"
            f"  pretrain |Ŝ| : {group_str}\n"
            f"  overall λ    : {self.overall_lambda}  (idx {self.overall_lmda_idx_})"
        )

    def get_coef(self, model="all", lmda_idx=None):
        """Return fitted coefficients as a nested dict.

        Parameters
        ----------
        model : {"all", "overall", "pretrain", "individual"}, default="all"
            Which sub-model(s) to return.
        lmda_idx : int or None
            Lambda index for group models.  ``None`` uses the last lambda (``-1``).

        Returns
        -------
        coefs : dict
            When ``model="all"``, keys are ``"overall"``, ``"pretrain"``,
            ``"individual"``.  For ``"overall"``, value is
            ``{"coef": ndarray, "intercept": ndarray}``.  For ``"pretrain"``
            and ``"individual"``, value is a dict keyed by group label, each
            containing ``{"coef": ndarray, "intercept": ndarray}``.
            When a specific model is requested, only that sub-dict is returned.
        """
        if model not in COEF_MODELS:
            raise ValueError(f"model must be one of {COEF_MODELS}, got '{model}'")
        check_is_fitted(self)

        def _group_coefs(models, lmda_idxs):
            result = {}
            for g, state in models.items():
                use_idx = lmda_idxs.get(g, -1) if lmda_idx is None else lmda_idx
                result[self._label(g)] = {
                    "coef": _coef_at(state, use_idx),
                    "intercept": np.asarray(state.intercepts[use_idx]).ravel(),
                }
            return result

        result = {}
        if model in ("all", "overall"):
            result["overall"] = {
                "coef": self.overall_coef_,
                "intercept": np.asarray(
                    self.overall_model_.intercepts[self.overall_lmda_idx_]
                ).ravel(),
            }
        if model in ("all", "pretrain"):
            result["pretrain"] = _group_coefs(self.pretrain_models_, self.pretrain_lmda_idx_)
        if model in ("all", "individual"):
            result["individual"] = _group_coefs(self.individual_models_, self.individual_lmda_idx_)
        return result if model == "all" else result[model]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def __getstate__(self):
        d = self.__dict__.copy()
        _proxify_models(d)
        return d


# ------------------------------------------------------------------
# PretrainedLassoCV
# ------------------------------------------------------------------


class PretrainedLassoCV(RegressorMixin, BasePretrainedLasso):
    """Pretrained Lasso with cross-validation over alpha.

    Parameters
    ----------
    alphas : array-like or None, default=None
        Candidate pretraining strengths.  Defaults to [0, 0.25, 0.5, 0.75, 1.0].
        ``0`` = maximum pretraining; ``1`` = no pretraining (individual models).
    cv : int, default=5
        Number of CV folds.
    alphahat_choice : {"overall", "mean"}, default="overall"
        ``"overall"`` minimises the global CV error; ``"mean"`` minimises the
        unweighted mean of per-group CV errors.
    family : {"gaussian", "binomial", "multinomial"}, default="gaussian"
        Response distribution.
    overall_lambda : {"lambda.1se", "lambda.min"}, default="lambda.1se"
        Lambda selection rule for the stage-1 overall model.
    fit_intercept : bool, default=True
        Whether to fit an intercept in every sub-model.
    lmda_path_size : int, default=100
        Number of lambdas in the regularisation path.
    min_ratio : float, default=0.0001
        Ratio of the smallest to largest lambda on the path.
    verbose : bool, default=True
        Whether to display fitting progress and a summary after training.
        Adelie's internal output is always suppressed regardless of this setting.
    foldid : array-like of int or None, default=None
        Fold assignments, one integer per sample.  When provided, overrides
        the internal ``StratifiedKFold`` splitter.
    n_threads : int, default=-1
        Number of threads passed to adelie's solver.  Set to a higher value
        to parallelise the coordinate descent within each model fit.
        ``-1`` uses all available CPU cores (``os.cpu_count()``).
    scoring : str, callable, or None, default=None
        CV criterion used to select the best alpha.  All values follow the
        **higher = better** convention (matching sklearn).

        - ``None`` — family-based loss: MSE for gaussian, log-loss for
          binomial/multinomial.  Matches the current default behaviour.
        - ``"roc_auc"`` — area under the ROC curve (binomial only).
          Matches R's ``type.measure="auc"``.
        - ``"neg_log_loss"`` — negative log-loss (binomial/multinomial).
        - ``"accuracy"`` — classification accuracy (binomial/multinomial).
        - ``"neg_mean_squared_error"`` — negative MSE (gaussian).
        - ``"r2"`` — coefficient of determination (gaussian).
        - callable — any function ``(y_true, y_pred) -> float`` where
          **higher is better**, e.g. a custom sklearn scorer.

        ``cv_results_`` always stores values as *losses* (lower = better),
        so AUC will appear as ``-AUC``.

    Attributes
    ----------
    alpha_ : float
        Best alpha selected by CV (based on ``alphahat_choice``).
    varying_alphahat_ : dict {group -> float}
        Per-group best alpha.
    cv_results_ : dict {alpha -> float}
        Global mean CV loss per alpha.
    cv_results_se_ : dict {alpha -> float}
        Standard error of global CV loss.
    cv_results_per_group_ : dict {alpha -> {group -> float}}
        Mean CV loss per alpha per group.
    cv_results_mean_ : dict {alpha -> float}
        Unweighted mean of per-group CV losses.
    cv_results_wtd_mean_ : dict {alpha -> float}
        Group-size-weighted mean of per-group CV losses.
    cv_results_individual_ : float
        Global CV loss for the individual (no-pretraining) model.
    cv_results_overall_ : float
        Global CV loss for the overall model.
    best_estimator_ : PretrainedLasso
        Full-data refit at the globally selected ``alpha_``.
    all_estimators_ : dict {alpha -> PretrainedLasso}
        Full-data refits for each unique alpha needed by ``varying_alphahat_``.

    References
    ----------
    Craig, E., Pilanci, M., Le Menestrel, T., Narasimhan, B., Rivas, M. A.,
    Gullaksen, S. E., & Tibshirani, R. (2025). Pretraining and the lasso.
    *Journal of the Royal Statistical Society Series B*, qkaf050.
    """

    def __init__(
        self,
        alphas=DEFAULT_ALPHAS,
        cv=5,
        alphahat_choice="overall",
        family="gaussian",
        overall_lambda="lambda.1se",
        fit_intercept=True,
        lmda_path_size=100,
        min_ratio=0.0001,
        verbose=True,
        foldid=None,
        scoring=None,
        n_threads=-1,
        standardize=True,
    ):
        self.alphas = alphas
        self.cv = cv
        self.alphahat_choice = alphahat_choice
        self.family = family
        self.overall_lambda = overall_lambda
        self.fit_intercept = fit_intercept
        self.lmda_path_size = lmda_path_size
        self.min_ratio = min_ratio
        self.verbose = verbose
        self.foldid = foldid
        self.scoring = scoring
        self.n_threads = n_threads
        self.standardize = standardize

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_params(self):
        alphas = self.alphas
        bad = [a for a in alphas if not (0.0 <= a <= 1.0)]
        if bad:
            raise ValueError(f"All alphas must be in [0, 1], got {bad}")
        if len(set(alphas)) != len(alphas):
            raise ValueError(f"alphas must not contain duplicates, got {alphas}")
        if not isinstance(self.cv, int) or self.cv < 2:
            raise ValueError(f"cv must be an integer >= 2, got {self.cv}")
        if self.alphahat_choice not in ("overall", "mean"):
            raise ValueError(
                f"alphahat_choice must be 'overall' or 'mean', got '{self.alphahat_choice}'"
            )
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got '{self.family}'")
        if self.overall_lambda not in LMDA_MODES:
            raise ValueError(f"overall_lambda must be one of {LMDA_MODES}")
        if not isinstance(self.lmda_path_size, int) or self.lmda_path_size < 1:
            raise ValueError("lmda_path_size must be a positive integer")
        if not (0.0 < self.min_ratio < 1.0):
            raise ValueError("min_ratio must be in (0, 1)")
        if self.scoring is not None and not callable(self.scoring):
            if self.scoring not in _VALID_SCORERS:
                raise ValueError(
                    f"scoring must be None, a callable, or one of {_VALID_SCORERS}, "
                    f"got '{self.scoring}'"
                )

    def _get_alphas(self):
        """Return the list of alpha candidates."""
        return list(self.alphas if self.alphas is not None else DEFAULT_ALPHAS)

    def _base_estimator(self, alpha):
        """Return a configured PretrainedLasso for a given alpha."""
        return PretrainedLasso(
            alpha=alpha,
            family=self.family,
            overall_lambda=self.overall_lambda,
            fit_intercept=self.fit_intercept,
            lmda_path_size=self.lmda_path_size,
            min_ratio=self.min_ratio,
            verbose=False,  # CV sub-fits are silent; PretrainedLassoCV owns the progress bar
            n_threads=self.n_threads,
            standardize=self.standardize,
        )

    def _fold_iter(self, X, groups):
        """Yield (train_idx, test_idx) pairs for CV."""
        if self.foldid is not None:
            foldid = np.asarray(self.foldid)
            if len(foldid) != len(groups):
                raise ValueError(
                    f"foldid has {len(foldid)} elements but X has {len(groups)} samples"
                )
            if len(np.unique(foldid)) < 2:
                raise ValueError("foldid must contain at least 2 distinct fold values")
            for f in np.unique(foldid):
                yield np.where(foldid != f)[0], np.where(foldid == f)[0]
        else:
            splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=0)
            yield from splitter.split(X, groups)

    # ------------------------------------------------------------------
    # Progress / display helpers
    # ------------------------------------------------------------------

    def _print_cv_summary(self, elapsed, n_cv_folds):
        SEP = "─" * 54
        print(SEP)
        print(f"    {'α':<8} {'CV loss':<12} {'±SE'}")
        print(f"  {'─' * 34}")
        for a in self.alphalist_:
            marker = "►" if a == self.alpha_ else " "
            print(f" {marker}  {a:<8.2f} {self.cv_results_[a]:<12.4f} {self.cv_results_se_[a]:.4f}")
        print(f"  {'─' * 34}")
        print(f"  {'individual':<13}{self.cv_results_individual_:.4f}")
        print(f"  {'overall':<13}{self.cv_results_overall_:.4f}")
        print(SEP)
        best = self.best_estimator_
        stage2 = set()
        for g in self.groups_:
            c = _coef_at(best.pretrain_models_[g], best.pretrain_lmda_idx_[g])
            if best.n_classes_ is not None:
                c = c.reshape(best.n_features_in_, best.n_classes_, order="F")
                stage2 |= set(int(i) for i in np.where(np.any(c != 0, axis=1))[0])
            else:
                stage2 |= set(int(i) for i in np.where(c != 0)[0])
        print(
            f"  Best α = {self.alpha_:.2f}   |S| = {len(stage2)}"
            f"   Fitted in {elapsed:.1f}s"
            f"  ({len(self.alphalist_)} alphas × {n_cv_folds} folds)\n"
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, groups, group_labels=None, feature_names=None):
        """Fit PretrainedLassoCV.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.  For ``family="binomial"``, must contain only 0
            and 1.  For ``family="multinomial"``, must contain non-negative
            integer class labels 0..K-1.
        groups : array-like of shape (n_samples,)
            Group membership for each sample.  Must contain at least two
            distinct values.
        group_labels : dict or None, default=None
            Optional mapping from group values to display names used in
            ``__repr__`` and ``get_coef()``.
        feature_names : array-like of str or None, default=None
            Feature names.  Inferred from ``X.columns`` when ``X`` is a
            DataFrame and this argument is ``None``.

        Returns
        -------
        self : PretrainedLassoCV
            Fitted estimator.
        """
        t0 = time.time()
        self._validate_params()

        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)

        X, y = check_X_y(X, y, dtype=np.float64, order="F")
        groups = np.asarray(groups)

        self.n_features_in_ = X.shape[1]
        self.group_labels_ = group_labels or {}

        alphas = self._get_alphas()
        self.alphalist_ = np.asarray(alphas)

        unique_groups = np.unique(groups)
        group_sizes = {g: int(np.sum(groups == g)) for g in unique_groups}

        scorer_fn = _resolve_scorer(self.scoring)

        n_cv_folds = self.cv if self.foldid is None else len(np.unique(self.foldid))
        total_cv_fits = n_cv_folds * len(alphas)

        if self.verbose:
            print(
                f"\nPretrainedLassoCV  {self.family}  ·  {len(unique_groups)} groups"
                f"  ·  {self.n_features_in_} features"
                f"  ·  {len(alphas)} alphas × {n_cv_folds} folds = {total_cv_fits} fits"
            )
            from tqdm import tqdm as _tqdm

            _n_threads = os.cpu_count() if self.n_threads == -1 else self.n_threads

        # Fold-level accumulators
        fold_losses = {a: [] for a in alphas}
        fold_losses_grp = {a: {g: [] for g in unique_groups} for a in alphas}
        fold_losses_ind = []
        fold_losses_ov = []

        _fold_num = 0
        for train_idx, test_idx in self._fold_iter(X, groups):
            _fold_num += 1
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            g_tr, g_te = groups[train_idx], groups[test_idx]

            if self.verbose:
                print(f"\n── Fold {_fold_num}/{n_cv_folds} " + "─" * 44, flush=True)
                print("  Overall model (λ-path):", flush=True)

            # Stage 1 (overall model + OOF) is identical for every alpha within
            # a fold — fit it once and reuse, avoiding repeated identical
            # ad.cv_grpnet calls on the same data which hang on HPC clusters.
            # _show_overall_progress exposes adelie's lambda-path bar for this fit.
            _stage1 = self._base_estimator(alphas[0])
            _stage1._show_overall_progress = self.verbose
            _stage1.fit(X_tr, y_tr, g_tr)
            _preval_offset = _stage1._preval_offset

            if self.verbose:
                _n_support = (
                    int(np.sum(_stage1.overall_coef_ != 0))
                    if self.family != "multinomial"
                    else int(np.sum(np.any(_stage1.overall_coef_ != 0, axis=1)))
                )
                print(f"  Overall model done  |S|={_n_support}", flush=True)

            _pbar = None
            if self.verbose:
                _pbar = _tqdm(total=len(alphas), desc="  Group fits", unit="α", leave=True)
                _pbar.refresh()  # force initial render on buffered terminals

            for i, a in enumerate(alphas):
                if i == 0:
                    est = _stage1
                else:
                    est = self._base_estimator(a)
                    # Copy stage-1 state so _fit_groups_only can use it.
                    est.n_features_in_ = _stage1.n_features_in_
                    est.groups_ = _stage1.groups_
                    est.group_labels_ = _stage1.group_labels_
                    est.n_classes_ = _stage1.n_classes_
                    est.feature_names_in_ = _stage1.feature_names_in_
                    est._n_onehot_ = _stage1._n_onehot_
                    est.scaler_ = _stage1.scaler_
                    est.overall_model_ = _stage1.overall_model_
                    est.overall_lmda_idx_ = _stage1.overall_lmda_idx_
                    est.overall_coef_ = _stage1.overall_coef_
                    if self.family != "multinomial":
                        est.overall_intercept_ = _stage1.overall_intercept_
                    est._fit_groups_only(X_tr, y_tr, g_tr, _preval_offset)

                y_pred = est.predict(X_te, g_te)
                fold_losses[a].append(_cv_loss(scorer_fn, y_te, y_pred, self.family))

                for g in unique_groups:
                    mask = g_te == g
                    if mask.any():
                        fold_losses_grp[a][g].append(
                            _cv_loss(scorer_fn, y_te[mask], y_pred[mask], self.family)
                        )

                # Individual and overall baselines don't depend on alpha — compute
                # once per fold using the first alpha's fitted estimator.
                if i == 0:
                    y_ind = est.predict(X_te, g_te, model="individual")
                    y_ov = est.predict(X_te, g_te, model="overall")
                    fold_losses_ind.append(_cv_loss(scorer_fn, y_te, y_ind, self.family))
                    fold_losses_ov.append(_cv_loss(scorer_fn, y_te, y_ov, self.family))

                if _pbar is not None:
                    _pbar.update(1)
                    _pbar.set_postfix(alpha=f"{a:.2f}", threads=_n_threads, refresh=False)

            if _pbar is not None:
                _pbar.close()

        # Aggregate CV results
        self.cv_results_ = {a: float(np.mean(errs)) for a, errs in fold_losses.items()}
        self.cv_results_se_ = {
            a: float(np.std(errs, ddof=1) / np.sqrt(len(errs))) for a, errs in fold_losses.items()
        }
        self.cv_results_per_group_ = {
            a: {
                g: float(np.mean(fold_losses_grp[a][g])) if fold_losses_grp[a][g] else np.nan
                for g in unique_groups
            }
            for a in alphas
        }
        self.cv_results_mean_ = {
            a: float(np.nanmean(list(self.cv_results_per_group_[a].values()))) for a in alphas
        }
        self.cv_results_wtd_mean_ = {
            a: float(
                np.nansum(
                    [
                        self.cv_results_per_group_[a][g] * group_sizes[g] / len(y)
                        for g in unique_groups
                    ]
                )
            )
            for a in alphas
        }
        self.cv_results_individual_ = float(np.mean(fold_losses_ind))
        self.cv_results_overall_ = float(np.mean(fold_losses_ov))

        # Select global best alpha
        criterion = self.cv_results_mean_ if self.alphahat_choice == "mean" else self.cv_results_
        self.alpha_ = min(alphas, key=criterion.__getitem__)

        # Per-group best alpha
        self.varying_alphahat_ = {
            g: min(alphas, key=lambda a, _g=g: self.cv_results_per_group_[a].get(_g, np.inf))
            for g in unique_groups
        }

        # Refit on full data for each unique alpha required (best + varying)
        unique_alphas = set(self.varying_alphahat_.values()) | {self.alpha_}
        fit_kwargs = dict(group_labels=group_labels, feature_names=feature_names)
        if self.verbose:
            print(f"  Refitting at α={self.alpha_:.2f} (best) ...", end="", flush=True)
            _t_refit = time.time()
        self.all_estimators_ = {
            a: self._base_estimator(a).fit(X, y, groups, **fit_kwargs) for a in unique_alphas
        }
        self.best_estimator_ = self.all_estimators_[self.alpha_]
        if self.verbose:
            print(f" done  ({time.time() - _t_refit:.1f}s)")

        # Mirror fitted attributes from the best estimator for a uniform interface
        for attr in (
            "overall_model_",
            "overall_lmda_idx_",
            "overall_coef_",
            "pretrain_models_",
            "pretrain_lmda_idx_",
            "individual_models_",
            "individual_lmda_idx_",
            "groups_",
            "feature_names_in_",
            "n_classes_",
            "_n_onehot_",
        ):
            setattr(self, attr, getattr(self.best_estimator_, attr))
        if self.family != "multinomial":
            self.overall_intercept_ = self.best_estimator_.overall_intercept_

        if self.verbose:
            self._print_cv_summary(time.time() - t0, n_cv_folds)

        return self

    # ------------------------------------------------------------------
    # predict / score / evaluate
    # ------------------------------------------------------------------

    @property
    def alpha(self):
        """Selected alpha — mirrors PretrainedLasso.alpha for a uniform interface."""
        check_is_fitted(self, ["alpha_"])
        return self.alpha_

    def predict(
        self, X, groups, model="pretrain", type="response", alphatype="best", lmda_idx=None
    ):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        groups : array-like of shape (n_samples,)
        model : {"pretrain", "individual", "overall"}, default="pretrain"
        type : {"response", "link", "class"}, default="response"
            Scale of the returned predictions.  See :meth:`PretrainedLasso.predict`
            for full documentation.
        alphatype : {"best", "varying"}, default="best"
            ``"best"`` uses the globally selected ``alpha_``; ``"varying"``
            uses each group's own optimal alpha from ``varying_alphahat_``.
        lmda_idx : int or None

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, K)
            Shape is ``(n_samples, K)`` for multinomial ``"response"`` or
            ``"link"``.
        """
        check_is_fitted(self)
        groups = np.asarray(groups)

        if model not in PREDICT_MODELS:
            raise ValueError(f"model must be one of {PREDICT_MODELS}, got '{model}'")
        if type not in PREDICT_TYPES:
            raise ValueError(f"type must be one of {PREDICT_TYPES}, got '{type}'")
        if alphatype not in ALPHATYPES:
            raise ValueError(f"alphatype must be one of {ALPHATYPES}, got '{alphatype}'")

        if alphatype == "varying":
            X = check_array(X, dtype=np.float64, order="F")
            n_out = (X.shape[0], self.n_classes_) if self.family == "multinomial" else (X.shape[0],)
            y_pred = np.empty(n_out)
            for g in np.unique(groups):
                mask = groups == g
                a = self.varying_alphahat_.get(g, self.alpha_)
                y_pred[mask] = self.all_estimators_[a].predict(
                    X[mask], groups[mask], model=model, type=type, lmda_idx=lmda_idx
                )
            return y_pred

        return self.best_estimator_.predict(
            X=X, groups=groups, model=model, type=type, lmda_idx=lmda_idx
        )

    def evaluate(self, X, y, groups, alphatype="best"):
        """Predict and score with all three sub-models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        groups : array-like of shape (n_samples,)
        alphatype : {"best", "varying"}, default="best"
            ``"best"`` uses ``alpha_``; ``"varying"`` uses each group's own
            optimal alpha from ``varying_alphahat_``.

        Returns
        -------
        result : dict
            Keys are ``"pretrain"``, ``"individual"``, ``"overall"``.
            Each value is a dict with entries:

            ``"predictions"`` : ndarray
                Predicted values.
            ``"score"`` : float
                R² for gaussian; accuracy for binomial/multinomial.
        """
        check_is_fitted(self)
        result = {}
        for m in ("pretrain", "individual", "overall"):
            preds = self.predict(X, groups, model=m, alphatype=alphatype)
            result[m] = {"predictions": preds, "score": _model_score(y, preds, self.family)}
        return result

    def score(self, X, y, groups):
        """Return a scalar performance metric using the best estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        groups : array-like of shape (n_samples,)

        Returns
        -------
        score : float
            R² for gaussian; classification accuracy for binomial/multinomial.
        """
        check_is_fitted(self)
        return self.best_estimator_.score(X, y, groups)

    # ------------------------------------------------------------------
    # repr / get_coef
    # ------------------------------------------------------------------

    def __repr__(self):
        alphas = self._get_alphas()
        header = (
            f"PretrainedLassoCV(alphas={alphas}, cv={self.cv}, "
            f"alphahat_choice='{self.alphahat_choice}', "
            f"family='{self.family}', overall_lambda='{self.overall_lambda}')"
        )
        if not hasattr(self, "best_estimator_"):
            return header + "\n  [not fitted]"

        rows = [
            f"  alpha={a:.2f}  overall={self.cv_results_[a]:.4f}"
            f"  mean={self.cv_results_mean_[a]:.4f}"
            f"  wtd_mean={self.cv_results_wtd_mean_[a]:.4f}"
            for a in self.alphalist_
        ]
        return (
            f"{header}\n"
            f"  family     : {self.family}\n"
            f"  n_features : {self.n_features_in_}\n"
            f"  n_groups   : {len(self.groups_)}\n"
            f"  alpha_     : {self.alpha_}  ({self.alphahat_choice})\n"
            f"  individual : {self.cv_results_individual_:.4f}\n"
            f"  overall    : {self.cv_results_overall_:.4f}\n"
            f"  CV results :\n" + "\n".join(rows)
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def __getstate__(self):
        d = self.__dict__.copy()
        # Convert the adelie state refs mirrored directly onto this object.
        # all_estimators_ / best_estimator_ are PretrainedLasso instances and
        # go through PretrainedLasso.__getstate__ automatically.
        _proxify_models(d)
        return d

    # ------------------------------------------------------------------
    # get_coef
    # ------------------------------------------------------------------

    def get_coef(self, model="all", **kwargs):
        """Return fitted coefficients from the best estimator.

        Delegates to :meth:`PretrainedLasso.get_coef`.

        Parameters
        ----------
        model : {"all", "overall", "pretrain", "individual"}, default="all"
        **kwargs
            Forwarded to :meth:`PretrainedLasso.get_coef`.

        Returns
        -------
        coefs : dict
            See :meth:`PretrainedLasso.get_coef` for the structure.
        """
        check_is_fitted(self)
        return self.best_estimator_.get_coef(model=model, **kwargs)
