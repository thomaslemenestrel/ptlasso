import adelie as ad
import numpy as np

from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._base import BasePretrainedLasso
from ._constants import FAMILIES as _FAMILIES, LMDA_MODES as _LMDA_MODES


# ------------------------------------------------------------------
# Low-level adelie helpers
# ------------------------------------------------------------------


def _coef_at(state, lmda_idx):
    """Dense coefficient vector from adelie state at lambda index."""
    beta = state.betas[lmda_idx]
    if hasattr(beta, "toarray"):
        return beta.toarray().ravel()
    return np.asarray(beta).ravel()


def _intercept_at(state, lmda_idx):
    return float(state.intercepts[lmda_idx])


def _intercepts_at(state, lmda_idx):
    """All intercepts at a lambda index (vector for multinomial, scalar for others)."""
    raw = state.intercepts[lmda_idx]
    return np.asarray(raw).ravel()


def _predict_linear(state, X, lmda_idx):
    return X @ _coef_at(state, lmda_idx) + _intercept_at(state, lmda_idx)


def _lmda_1se_idx(cv_result):
    """Index of the largest lambda (most regularized) within 1 SE of CV minimum.

    adelie exposes ``avg_losses`` (L,) and ``losses`` (n_folds, L).
    Lambdas are in decreasing order, so smaller index = more regularized.
    lambda.1se is the leftmost (most regularized) index whose avg loss
    is within 1 SE of the minimum.
    """
    best = cv_result.best_idx
    avg = np.asarray(cv_result.avg_losses)
    folds = np.asarray(cv_result.losses)  # (n_folds, L)
    n_folds = folds.shape[0]
    se = folds.std(axis=0, ddof=1) / np.sqrt(n_folds)
    threshold = avg[best] + se[best]
    candidates = np.where(avg <= threshold)[0]
    return int(candidates[0]) if len(candidates) else best


# ------------------------------------------------------------------
# Family helpers
# ------------------------------------------------------------------


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(eta):
    """(n, K) → (n, K) row-wise softmax."""
    e = np.exp(eta - eta.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _make_glm(family, y):
    if family == "gaussian":
        return ad.glm.gaussian(y)
    if family == "binomial":
        return ad.glm.binomial(y)
    if family == "multinomial":
        # adelie multinomial expects (n, K) response matrix
        y = np.asarray(y, dtype=int)
        K = int(y.max()) + 1
        mat = np.zeros((len(y), K), dtype=np.float64, order="F")
        mat[np.arange(len(y)), y] = 1.0
        return ad.glm.multinomial(mat)
    raise ValueError(f"family must be one of {_FAMILIES}, got '{family}'")


def _eta_from_state(state, X, lmda_idx, family, n_classes=None):
    """Linear predictor (before link function).

    gaussian/binomial : (n,) array
    multinomial       : (n, K) array
    """
    if family == "multinomial":
        p = X.shape[1]
        K = n_classes
        coef_flat = _coef_at(state, lmda_idx)  # (p*K,)
        intercepts = _intercepts_at(state, lmda_idx)  # (K,)
        coef_mat = coef_flat.reshape(p, K, order="F")  # (p, K)
        return X @ coef_mat + intercepts  # (n, K)
    return _predict_linear(state, X, lmda_idx)


def _predict_from_state(state, X, lmda_idx, family, n_classes=None):
    """Apply the inverse link to produce predictions from an adelie state.

    Returns
    -------
    gaussian   : (n,) array of fitted values
    binomial   : (n,) array of P(y=1)
    multinomial: (n, K) array of class probabilities
    """
    eta = _eta_from_state(state, X, lmda_idx, family, n_classes)
    if family == "multinomial":
        return _softmax(eta)
    if family == "binomial":
        return _sigmoid(eta)
    return eta


def _fold_loss(y_true, y_pred, family):
    """Scalar loss for one fold (lower = better)."""
    if family == "gaussian":
        return float(np.mean((y_true - y_pred) ** 2))
    eps = 1e-15
    if family == "binomial":
        p = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    if family == "multinomial":
        p = np.clip(y_pred, eps, 1)  # (n, K)
        idx = y_true.astype(int)
        return float(-np.mean(np.log(p[np.arange(len(idx)), idx])))
    raise ValueError(family)


# ------------------------------------------------------------------
# PretrainedLasso
# ------------------------------------------------------------------


class PretrainedLasso(RegressorMixin, BasePretrainedLasso):
    """Pretrained Lasso estimator.

    Two-step training:
    1. Fit an overall Lasso on all samples.
    2. For each group k, fit a group-specific Lasso with offset
       ``alpha * (X_k @ beta_overall + intercept_overall)``.

    Parameters
    ----------
    alpha : float in [0, 1], default=0.5
        Pretraining strength. 0 = no pretraining, 1 = residuals only.
    family : {"gaussian", "binomial", "multinomial"}, default="gaussian"
        Response type.
    overall_lambda : {"lambda.1se", "lambda.min"}, default="lambda.1se"
        Lambda to use from the overall model for stage-1 offset.
        ``"lambda.1se"`` (default, matching R) gives a sparser overall model.
    fit_intercept : bool, default=True
    lmda_path_size : int, default=100
    min_ratio : float, default=0.01
    """

    def __init__(
        self,
        alpha=0.5,
        family="gaussian",
        overall_lambda="lambda.1se",
        fit_intercept=True,
        lmda_path_size=100,
        min_ratio=0.01,
    ):
        self.alpha = alpha
        self.family = family
        self.overall_lambda = overall_lambda
        self.fit_intercept = fit_intercept
        self.lmda_path_size = lmda_path_size
        self.min_ratio = min_ratio

    def _validate_params(self):
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.family not in _FAMILIES:
            raise ValueError(f"family must be one of {_FAMILIES}, got '{self.family}'")
        if self.overall_lambda not in _LMDA_MODES:
            raise ValueError(
                f"overall_lambda must be one of {_LMDA_MODES}, got '{self.overall_lambda}'"
            )
        if not isinstance(self.lmda_path_size, int) or self.lmda_path_size < 1:
            raise ValueError(
                f"lmda_path_size must be a positive integer, got {self.lmda_path_size}"
            )
        if not (0.0 < self.min_ratio < 1.0):
            raise ValueError(f"min_ratio must be in (0, 1), got {self.min_ratio}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _label(self, g):
        return self.group_labels_.get(g, g)

    def _names_or_indices(self, indices):
        if self.feature_names_in_ is not None:
            return self.feature_names_in_[indices]
        return indices

    def _grpnet_kwargs(self):
        return dict(
            alpha=1,  # pure lasso
            intercept=self.fit_intercept,
            lmda_path_size=self.lmda_path_size,
            min_ratio=self.min_ratio,
            progress_bar=False,
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, groups, group_labels=None, feature_names=None):
        """Fit the pretrained Lasso model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            For ``family="multinomial"``, integer class labels 0..K-1.
        groups : array-like of shape (n_samples,)
        group_labels : dict or None
        feature_names : array-like or None

        Returns
        -------
        self
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
                    f"feature_names has {len(feature_names)} entries but X has "
                    f"{self.n_features_in_} features"
                )
        self.feature_names_in_ = feature_names

        groups = np.asarray(groups)
        if len(groups) != X.shape[0]:
            raise ValueError(f"groups has {len(groups)} elements but X has {X.shape[0]} samples")
        self.groups_ = np.unique(groups)
        if len(self.groups_) < 2:
            raise ValueError("groups must contain at least 2 unique values")

        if group_labels is not None:
            unknown = set(group_labels) - set(self.groups_)
            if unknown:
                raise ValueError(f"group_labels contains unknown group keys: {unknown}")
        self.group_labels_ = group_labels if group_labels is not None else {}

        # Store number of classes for multinomial
        if self.family == "multinomial":
            self.n_classes_ = int(np.asarray(y, dtype=int).max()) + 1
        else:
            self.n_classes_ = None

        # ------------------------------------------------------------------
        # Step 1: overall model — lambda selected by CV
        # ------------------------------------------------------------------
        glm_all = _make_glm(self.family, y)
        cv_overall = ad.cv_grpnet(X, glm_all, **self._grpnet_kwargs())

        if self.overall_lambda == "lambda.min":
            self.overall_lmda_idx_ = cv_overall.best_idx
        else:
            self.overall_lmda_idx_ = _lmda_1se_idx(cv_overall)

        # Refit on full data — path goes from lambda_max down to lambda at best_idx.
        # overall_lmda_idx_ <= best_idx, so it is always within this path.
        self.overall_model_ = cv_overall.fit(X, glm_all, **self._grpnet_kwargs())

        if self.family == "multinomial":
            # overall_coef_ is (p, K) for convenience
            p, K = self.n_features_in_, self.n_classes_
            flat = _coef_at(self.overall_model_, self.overall_lmda_idx_)
            self.overall_coef_ = flat.reshape(p, K, order="F")
        else:
            self.overall_coef_ = _coef_at(self.overall_model_, self.overall_lmda_idx_)
            self.overall_intercept_ = _intercept_at(self.overall_model_, self.overall_lmda_idx_)

        # ------------------------------------------------------------------
        # Step 2: per-group models
        # ------------------------------------------------------------------
        self.pretrain_models_ = {}
        self.individual_models_ = {}

        # Stage-1 overall linear predictor used as offset for all groups.
        # Using eta (before link) is correct: the group model learns the
        # residual in linear space, and we add contributions back at predict time.
        overall_eta = _eta_from_state(
            self.overall_model_, X, self.overall_lmda_idx_, self.family, self.n_classes_
        )

        for g in self.groups_:
            mask = groups == g
            X_g = np.asfortranarray(X[mask])
            y_g = y[mask]
            glm_g = _make_glm(self.family, y_g)

            # offset shape must match glm.y shape:
            #   gaussian/binomial → (n_g,)
            #   multinomial       → (n_g, K)
            offset_g = np.asfortranarray(self.alpha * overall_eta[mask])
            self.pretrain_models_[g] = ad.grpnet(
                X_g, glm_g, offsets=offset_g, **self._grpnet_kwargs()
            )

            self.individual_models_[g] = ad.grpnet(X_g, glm_g, **self._grpnet_kwargs())

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X, groups, model="pretrain", lmda_idx=None):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        groups : array-like of shape (n_samples,)
        model : {"pretrain", "individual", "overall"}, default="pretrain"
        lmda_idx : int or None
            Lambda index for group models. None uses the last fitted lambda (-1).

        Returns
        -------
        y_pred : ndarray
            Shape (n_samples,) for gaussian/binomial, (n_samples, K) for multinomial.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, order="F")
        groups = np.asarray(groups)

        if model == "overall":
            return _predict_from_state(
                self.overall_model_, X, self.overall_lmda_idx_, self.family, self.n_classes_
            )

        if self.family == "multinomial":
            n_out = (X.shape[0], self.n_classes_)
        else:
            n_out = (X.shape[0],)
        y_pred = np.empty(n_out)

        # Precompute overall linear predictor (used for pretrain combination)
        overall_eta = _eta_from_state(
            self.overall_model_, X, self.overall_lmda_idx_, self.family, self.n_classes_
        )

        for g in self.groups_:
            mask = groups == g
            X_g = X[mask]
            state = self.pretrain_models_[g] if model == "pretrain" else self.individual_models_[g]
            idx = lmda_idx if lmda_idx is not None else -1

            if model == "pretrain":
                # The group model was trained with offset = alpha * eta_overall.
                # Total linear predictor = alpha * eta_overall + eta_group.
                eta_group = _eta_from_state(state, X_g, idx, self.family, self.n_classes_)
                eta_total = self.alpha * overall_eta[mask] + eta_group
                if self.family == "multinomial":
                    y_pred[mask] = _softmax(eta_total)
                elif self.family == "binomial":
                    y_pred[mask] = _sigmoid(eta_total)
                else:
                    y_pred[mask] = eta_total
            else:
                y_pred[mask] = _predict_from_state(state, X_g, idx, self.family, self.n_classes_)

        return y_pred

    def score(self, X, y, groups):
        """Return R² for gaussian, accuracy for classification families."""
        y_pred = self.predict(X, groups)
        if self.family == "gaussian":
            return r2_score(y, y_pred)
        if self.family == "binomial":
            return float(np.mean((y_pred >= 0.5).astype(int) == np.asarray(y)))
        if self.family == "multinomial":
            return float(np.mean(y_pred.argmax(axis=1) == np.asarray(y, dtype=int)))
        return r2_score(y, y_pred)

    # ------------------------------------------------------------------
    # repr / get_coef
    # ------------------------------------------------------------------

    def __repr__(self):
        params = (
            f"alpha={self.alpha}, family='{self.family}', "
            f"overall_lambda='{self.overall_lambda}', "
            f"fit_intercept={self.fit_intercept}, "
            f"lmda_path_size={self.lmda_path_size}"
        )
        r = f"PretrainedLasso({params})"

        if not hasattr(self, "overall_model_"):
            return r + "\n  [not fitted]"

        if self.family == "multinomial":
            ov_nz_idx = np.where(np.any(self.overall_coef_ != 0, axis=1))[0]
        else:
            ov_nz_idx = np.where(self.overall_coef_ != 0)[0]

        n_nz = len(ov_nz_idx)
        names = self._names_or_indices(ov_nz_idx)
        preview = list(names[:5]) + (["..."] if n_nz > 5 else [])
        ov_str = f"|Ŝ| = {n_nz} / {self.n_features_in_}  [{', '.join(str(v) for v in preview)}]"

        group_nz = {
            self._label(g): int(np.any(_coef_at(self.pretrain_models_[g], -1).reshape(-1) != 0))
            for g in self.groups_
        }
        # Count nonzero features per group
        group_nz = {}
        for g in self.groups_:
            c = _coef_at(self.pretrain_models_[g], -1)
            if self.family == "multinomial":
                K = self.n_classes_
                cm = c.reshape(self.n_features_in_, K, order="F")
                group_nz[self._label(g)] = int(np.sum(np.any(cm != 0, axis=1)))
            else:
                group_nz[self._label(g)] = int(np.sum(c != 0))

        group_str = ", ".join(f"{lbl}: |Ŝ|={v}" for lbl, v in group_nz.items())

        return (
            f"{r}\n"
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
        lmda_idx : int or None

        Returns
        -------
        dict
        """
        check_is_fitted(self)

        def _group_coefs(models):
            out = {}
            for g, state in models.items():
                idx = lmda_idx if lmda_idx is not None else -1
                out[self._label(g)] = {
                    "coef": _coef_at(state, idx),
                    "intercept": _intercepts_at(state, idx),
                }
            return out

        result = {}
        if model in ("all", "overall"):
            result["overall"] = {
                "coef": self.overall_coef_,
                "intercept": (_intercepts_at(self.overall_model_, self.overall_lmda_idx_)),
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
        Candidate pretraining strengths. Defaults to ``[0, 0.25, 0.5, 0.75, 1.0]``.
    cv : int, default=5
        Number of CV folds.
    alphahat_choice : {"overall", "mean"}, default="overall"
        How to select the best alpha.
        - ``"overall"`` : minimise the global CV error across all samples.
        - ``"mean"``    : minimise the unweighted mean of per-group CV errors.
    family : {"gaussian", "binomial", "multinomial"}, default="gaussian"
    overall_lambda : {"lambda.1se", "lambda.min"}, default="lambda.1se"
    fit_intercept : bool, default=True
    lmda_path_size : int, default=100
    min_ratio : float, default=0.01

    Attributes
    ----------
    alpha_ : float
        Best alpha selected by CV (based on ``alphahat_choice``).
    cv_results_ : dict {alpha -> float}
        Global mean CV loss per alpha.
    cv_results_se_ : dict {alpha -> float}
        Standard error of global CV loss.
    cv_results_per_group_ : dict {alpha -> {group -> float}}
        Mean CV loss per alpha per group.
    cv_results_mean_ : dict {alpha -> float}
        Simple mean of per-group CV loss (used when ``alphahat_choice="mean"``).
    cv_results_wtd_mean_ : dict {alpha -> float}
        Group-size-weighted mean of per-group CV loss.
    cv_results_individual_ : float
        Global CV loss for the individual (no-pretraining) model.
    cv_results_overall_ : float
        Global CV loss for the overall model.
    best_estimator_ : PretrainedLasso
    """

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
    ):
        self.alphas = alphas
        self.cv = cv
        self.alphahat_choice = alphahat_choice
        self.family = family
        self.overall_lambda = overall_lambda
        self.fit_intercept = fit_intercept
        self.lmda_path_size = lmda_path_size
        self.min_ratio = min_ratio

    def _validate_params(self):
        alphas = self.alphas if self.alphas is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
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
        if self.family not in _FAMILIES:
            raise ValueError(f"family must be one of {_FAMILIES}, got '{self.family}'")
        if self.overall_lambda not in _LMDA_MODES:
            raise ValueError(f"overall_lambda must be one of {_LMDA_MODES}")
        if not isinstance(self.lmda_path_size, int) or self.lmda_path_size < 1:
            raise ValueError("lmda_path_size must be a positive integer")
        if not (0.0 < self.min_ratio < 1.0):
            raise ValueError("min_ratio must be in (0, 1)")

    def fit(self, X, y, groups, group_labels=None, feature_names=None):
        """Fit PretrainedLassoCV.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        groups : array-like of shape (n_samples,)
        group_labels : dict or None
        feature_names : array-like or None

        Returns
        -------
        self
        """
        self._validate_params()

        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)

        X, y = check_X_y(X, y, dtype=np.float64, order="F")
        self.n_features_in_ = X.shape[1]
        self.group_labels_ = group_labels if group_labels is not None else {}
        groups = np.asarray(groups)

        alphas = self.alphas if self.alphas is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        self.alphalist_ = np.asarray(alphas)

        unique_groups = np.unique(groups)
        group_sizes = {g: int(np.sum(groups == g)) for g in unique_groups}
        total_n = len(y)

        # Per-fold losses: {alpha -> [fold_loss, ...]}, per-group: {alpha -> {g -> [...]}}
        fold_errors = {a: [] for a in alphas}
        fold_per_group = {a: {g: [] for g in unique_groups} for a in alphas}
        fold_errors_individual = []
        fold_errors_overall = []
        fold_per_group_ind = {g: [] for g in unique_groups}
        fold_per_group_ov = {g: [] for g in unique_groups}

        splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=0)

        for train_idx, test_idx in splitter.split(X, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            g_tr, g_te = groups[train_idx], groups[test_idx]

            for i, a in enumerate(alphas):
                est = PretrainedLasso(
                    alpha=a,
                    family=self.family,
                    overall_lambda=self.overall_lambda,
                    fit_intercept=self.fit_intercept,
                    lmda_path_size=self.lmda_path_size,
                    min_ratio=self.min_ratio,
                ).fit(X_tr, y_tr, g_tr)

                y_pred = est.predict(X_te, g_te)
                fold_errors[a].append(_fold_loss(y_te, y_pred, self.family))

                for g in unique_groups:
                    mask = g_te == g
                    if mask.any():
                        fold_per_group[a][g].append(
                            _fold_loss(y_te[mask], y_pred[mask], self.family)
                        )

                # Individual and overall only depend on training data, not alpha;
                # compute once per fold from the first alpha's fit.
                if i == 0:
                    y_ind = est.predict(X_te, g_te, model="individual")
                    y_ov = est.predict(X_te, g_te, model="overall")
                    fold_errors_individual.append(_fold_loss(y_te, y_ind, self.family))
                    fold_errors_overall.append(_fold_loss(y_te, y_ov, self.family))
                    for g in unique_groups:
                        mask = g_te == g
                        if mask.any():
                            fold_per_group_ind[g].append(
                                _fold_loss(y_te[mask], y_ind[mask], self.family)
                            )
                            fold_per_group_ov[g].append(
                                _fold_loss(y_te[mask], y_ov[mask], self.family)
                            )

        # Aggregate
        self.cv_results_ = {a: float(np.mean(errs)) for a, errs in fold_errors.items()}
        self.cv_results_se_ = {
            a: float(np.std(errs, ddof=1) / np.sqrt(len(errs))) for a, errs in fold_errors.items()
        }
        self.cv_results_per_group_ = {
            a: {
                g: float(np.mean(fold_per_group[a][g])) if fold_per_group[a][g] else np.nan
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
                        self.cv_results_per_group_[a][g] * group_sizes[g] / total_n
                        for g in unique_groups
                    ]
                )
            )
            for a in alphas
        }
        self.cv_results_individual_ = float(np.mean(fold_errors_individual))
        self.cv_results_overall_ = float(np.mean(fold_errors_overall))

        # Select best alpha
        if self.alphahat_choice == "mean":
            self.alpha_ = min(alphas, key=self.cv_results_mean_.__getitem__)
        else:
            self.alpha_ = min(alphas, key=self.cv_results_.__getitem__)

        # Refit on all data with selected alpha
        self.best_estimator_ = PretrainedLasso(
            alpha=self.alpha_,
            family=self.family,
            overall_lambda=self.overall_lambda,
            fit_intercept=self.fit_intercept,
            lmda_path_size=self.lmda_path_size,
            min_ratio=self.min_ratio,
        ).fit(X, y, groups, group_labels=group_labels, feature_names=feature_names)

        # Mirror fitted attributes
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

    @property
    def alpha(self):
        """Selected alpha — mirrors PretrainedLasso.alpha for a uniform interface."""
        check_is_fitted(self, ["alpha_"])
        return self.alpha_

    def predict(self, X, groups, **kwargs):
        check_is_fitted(self)
        return self.best_estimator_.predict(X=X, groups=groups, **kwargs)

    def score(self, X, y, groups):
        check_is_fitted(self)
        return self.best_estimator_.score(X, y, groups)

    def __repr__(self):
        alphas = self.alphas if self.alphas is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        params = (
            f"alphas={alphas}, cv={self.cv}, "
            f"alphahat_choice='{self.alphahat_choice}', "
            f"family='{self.family}', "
            f"overall_lambda='{self.overall_lambda}'"
        )
        r = f"PretrainedLassoCV({params})"

        if not hasattr(self, "best_estimator_"):
            return r + "\n  [not fitted]"

        rows = []
        for a in self.alphalist_:
            rows.append(
                f"  alpha={a:.2f}  overall={self.cv_results_[a]:.4f}"
                f"  mean={self.cv_results_mean_[a]:.4f}"
                f"  wtd_mean={self.cv_results_wtd_mean_[a]:.4f}"
            )
        cv_table = "\n".join(rows)

        return (
            f"{r}\n"
            f"  family       : {self.family}\n"
            f"  n_features   : {self.n_features_in_}\n"
            f"  n_groups     : {len(self.groups_)}\n"
            f"  alpha_       : {self.alpha_}  ({self.alphahat_choice})\n"
            f"  individual   : {self.cv_results_individual_:.4f}\n"
            f"  overall      : {self.cv_results_overall_:.4f}\n"
            f"  CV results   :\n{cv_table}"
        )

    def get_coef(self, model="all", **kwargs):
        check_is_fitted(self)
        return self.best_estimator_.get_coef(model=model, **kwargs)
