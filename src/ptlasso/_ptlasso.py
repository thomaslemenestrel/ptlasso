"""Core estimators for the Pretrained Lasso.

Classes
-------
PretrainedLasso
    Two-step estimator: overall Lasso followed by per-group Lasso with offset.
PretrainedLassoCV
    Same estimator with cross-validation over the pretraining strength alpha.
"""

import adelie as ad
import numpy as np

from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._base import BasePretrainedLasso
from ._constants import FAMILIES, LMDA_MODES, PREDICT_MODELS, COEF_MODELS, ALPHATYPES


# ------------------------------------------------------------------
# adelie helpers
# ------------------------------------------------------------------


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
    min_ratio : float, default=0.01
        Ratio of the smallest to largest lambda on the path.
    verbose : bool, default=False
        Whether to display adelie's progress bar during fitting.

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
    individual_models_ : dict {group -> adelie state}
        Per-group fitted Lasso models without any pretraining offset.
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
        min_ratio=0.01,
        verbose=False,
    ):
        self.alpha = alpha
        self.family = family
        self.overall_lambda = overall_lambda
        self.fit_intercept = fit_intercept
        self.lmda_path_size = lmda_path_size
        self.min_ratio = min_ratio
        self.verbose = verbose

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

    def _grpnet_kwargs(self):
        return dict(
            alpha=1,  # pure lasso (no group penalty mixing)
            intercept=self.fit_intercept,
            lmda_path_size=self.lmda_path_size,
            min_ratio=self.min_ratio,
            progress_bar=self.verbose,
        )

    def _overall_eta(self, X):
        """Overall linear predictor at the selected lambda."""
        return _eta_from_state(
            self.overall_model_, X, self.overall_lmda_idx_, self.family, self.n_classes_
        )

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
        self._validate_params()

        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)

        X, y = check_X_y(X, y, dtype=np.float64, order="F")
        self.n_features_in_ = X.shape[1]

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

        # ----------------------------------------------------------
        # Step 1: overall model (lambda selected by CV)
        # ----------------------------------------------------------
        glm_all = _make_glm(self.family, y)
        cv_overall = ad.cv_grpnet(X, glm_all, **self._grpnet_kwargs())

        self.overall_lmda_idx_ = (
            cv_overall.best_idx
            if self.overall_lambda == "lambda.min"
            else _lmda_1se_idx(cv_overall)
        )
        self.overall_model_ = cv_overall.fit(X, glm_all, **self._grpnet_kwargs())

        if self.family == "multinomial":
            flat = _coef_at(self.overall_model_, self.overall_lmda_idx_)
            self.overall_coef_ = flat.reshape(self.n_features_in_, self.n_classes_, order="F")
        else:
            self.overall_coef_ = _coef_at(self.overall_model_, self.overall_lmda_idx_)
            self.overall_intercept_ = float(self.overall_model_.intercepts[self.overall_lmda_idx_])

        # ----------------------------------------------------------
        # Step 2: per-group models
        # ----------------------------------------------------------
        # Offset = (1 - alpha) * overall linear predictor (eta space, before link).
        # Matches R convention: alpha=0 → full pretraining, alpha=1 → individual.
        # Penalty factor: features NOT in the overall support get penalty 1/alpha,
        # steering the group model to prefer features already selected overall.
        overall_eta = self._overall_eta(X)

        if self.family == "multinomial":
            overall_support = np.where(np.any(self.overall_coef_ != 0, axis=1))[0]
        else:
            overall_support = np.where(self.overall_coef_ != 0)[0]

        _alpha_pf = self.alpha if self.alpha > 0 else 1e-9
        pf = np.full(self.n_features_in_, 1.0 / _alpha_pf)
        pf[overall_support] = 1.0

        self.pretrain_models_ = {}
        self.individual_models_ = {}

        for g in self.groups_:
            mask = groups == g
            X_g = np.asfortranarray(X[mask])
            glm_g = _make_glm(self.family, y[mask])
            offset = np.asfortranarray((1 - self.alpha) * overall_eta[mask])

            self.pretrain_models_[g] = ad.grpnet(
                X_g, glm_g, offsets=offset, penalty=pf, **self._grpnet_kwargs()
            )
            self.individual_models_[g] = ad.grpnet(X_g, glm_g, **self._grpnet_kwargs())

        return self

    # ------------------------------------------------------------------
    # predict / score / evaluate
    # ------------------------------------------------------------------

    def predict(self, X, groups, model="pretrain", lmda_idx=None):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        groups : array-like of shape (n_samples,)
        model : {"pretrain", "individual", "overall"}, default="pretrain"
        lmda_idx : int or None
            Lambda index for group models.  None uses the last fitted lambda.

        Returns
        -------
        y_pred : ndarray, shape (n,) or (n, K) for multinomial
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, order="F")
        groups = np.asarray(groups)

        if model not in PREDICT_MODELS:
            raise ValueError(f"model must be one of {PREDICT_MODELS}, got '{model}'")
        if len(groups) != X.shape[0]:
            raise ValueError(f"groups has {len(groups)} elements but X has {X.shape[0]} samples")
        unknown = set(np.unique(groups)) - set(self.groups_)
        if unknown:
            raise ValueError(f"predict received groups not seen during fit: {unknown}")

        if model == "overall":
            return _apply_link(self._overall_eta(X), self.family)

        idx = -1 if lmda_idx is None else lmda_idx
        n_out = (X.shape[0], self.n_classes_) if self.family == "multinomial" else (X.shape[0],)
        y_pred = np.empty(n_out)

        # Overall eta is only needed when combining with a group model (pretrain).
        eta_ov = self._overall_eta(X) if model == "pretrain" else None

        for g in self.groups_:
            mask = groups == g
            X_g = X[mask]

            if model == "pretrain":
                # Total eta = (1-alpha) * eta_overall + eta_group.
                # The group model was trained with offset = (1-alpha) * eta_overall,
                # so its eta already captures the residual.
                eta_group = _eta_from_state(
                    self.pretrain_models_[g], X_g, idx, self.family, self.n_classes_
                )
                y_pred[mask] = _apply_link((1 - self.alpha) * eta_ov[mask] + eta_group, self.family)
            else:
                eta = _eta_from_state(
                    self.individual_models_[g], X_g, idx, self.family, self.n_classes_
                )
                y_pred[mask] = _apply_link(eta, self.family)

        return y_pred

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
            c = _coef_at(self.pretrain_models_[g], -1)
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
        idx = -1 if lmda_idx is None else lmda_idx

        def _group_coefs(models):
            return {
                self._label(g): {
                    "coef": _coef_at(state, idx),
                    "intercept": np.asarray(state.intercepts[idx]).ravel(),
                }
                for g, state in models.items()
            }

        result = {}
        if model in ("all", "overall"):
            result["overall"] = {
                "coef": self.overall_coef_,
                "intercept": np.asarray(
                    self.overall_model_.intercepts[self.overall_lmda_idx_]
                ).ravel(),
            }
        if model in ("all", "pretrain"):
            result["pretrain"] = _group_coefs(self.pretrain_models_)
        if model in ("all", "individual"):
            result["individual"] = _group_coefs(self.individual_models_)
        return result if model == "all" else result[model]


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
    min_ratio : float, default=0.01
        Ratio of the smallest to largest lambda on the path.
    verbose : bool, default=False
        Whether to display adelie's progress bar during fitting.
    foldid : array-like of int or None, default=None
        Fold assignments, one integer per sample.  When provided, overrides
        the internal ``StratifiedKFold`` splitter.

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

    _DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

    def __init__(
        self,
        alphas=None,
        cv=5,
        alphahat_choice="overall",
        family="gaussian",
        overall_lambda="lambda.1se",
        fit_intercept=True,
        lmda_path_size=100,
        min_ratio=0.01,
        verbose=False,
        foldid=None,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_alphas(self):
        return list(self.alphas) if self.alphas is not None else self._DEFAULT_ALPHAS

    def _validate_params(self):
        alphas = self._get_alphas()
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

    def _base_estimator(self, alpha):
        """Return a configured PretrainedLasso for a given alpha."""
        return PretrainedLasso(
            alpha=alpha,
            family=self.family,
            overall_lambda=self.overall_lambda,
            fit_intercept=self.fit_intercept,
            lmda_path_size=self.lmda_path_size,
            min_ratio=self.min_ratio,
            verbose=self.verbose,
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

        # Fold-level accumulators
        fold_losses = {a: [] for a in alphas}
        fold_losses_grp = {a: {g: [] for g in unique_groups} for a in alphas}
        fold_losses_ind = []
        fold_losses_ov = []

        for train_idx, test_idx in self._fold_iter(X, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            g_tr, g_te = groups[train_idx], groups[test_idx]

            for i, a in enumerate(alphas):
                est = self._base_estimator(a).fit(X_tr, y_tr, g_tr)
                y_pred = est.predict(X_te, g_te)
                fold_losses[a].append(_fold_loss(y_te, y_pred, self.family))

                for g in unique_groups:
                    mask = g_te == g
                    if mask.any():
                        fold_losses_grp[a][g].append(
                            _fold_loss(y_te[mask], y_pred[mask], self.family)
                        )

                # Individual and overall baselines don't depend on alpha — compute
                # once per fold using the first alpha's fitted estimator.
                if i == 0:
                    y_ind = est.predict(X_te, g_te, model="individual")
                    y_ov = est.predict(X_te, g_te, model="overall")
                    fold_losses_ind.append(_fold_loss(y_te, y_ind, self.family))
                    fold_losses_ov.append(_fold_loss(y_te, y_ov, self.family))

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
        self.all_estimators_ = {
            a: self._base_estimator(a).fit(X, y, groups, **fit_kwargs) for a in unique_alphas
        }
        self.best_estimator_ = self.all_estimators_[self.alpha_]

        # Mirror fitted attributes from the best estimator for a uniform interface
        for attr in (
            "overall_model_",
            "overall_lmda_idx_",
            "overall_coef_",
            "pretrain_models_",
            "individual_models_",
            "groups_",
            "feature_names_in_",
            "n_classes_",
        ):
            setattr(self, attr, getattr(self.best_estimator_, attr))
        if self.family != "multinomial":
            self.overall_intercept_ = self.best_estimator_.overall_intercept_

        return self

    # ------------------------------------------------------------------
    # predict / score / evaluate
    # ------------------------------------------------------------------

    @property
    def alpha(self):
        """Selected alpha — mirrors PretrainedLasso.alpha for a uniform interface."""
        check_is_fitted(self, ["alpha_"])
        return self.alpha_

    def predict(self, X, groups, model="pretrain", alphatype="best", lmda_idx=None):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        groups : array-like of shape (n_samples,)
        model : {"pretrain", "individual", "overall"}, default="pretrain"
        alphatype : {"best", "varying"}, default="best"
            ``"best"`` uses the globally selected ``alpha_``; ``"varying"``
            uses each group's own optimal alpha from ``varying_alphahat_``.
        lmda_idx : int or None

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, K)
            Shape is ``(n_samples, K)`` for multinomial.
        """
        check_is_fitted(self)
        groups = np.asarray(groups)

        if model not in PREDICT_MODELS:
            raise ValueError(f"model must be one of {PREDICT_MODELS}, got '{model}'")
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
                    X[mask], groups[mask], model=model, lmda_idx=lmda_idx
                )
            return y_pred

        return self.best_estimator_.predict(X=X, groups=groups, model=model, lmda_idx=lmda_idx)

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
