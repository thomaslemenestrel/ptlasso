import adelie as ad
import numpy as np

from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._base import BasePretrainedLasso


def _coef_at(state, lmda_idx):
    """Extract dense coefficient vector from an adelie state at a given lambda index.

    adelie stores betas as shape (L, p) — possibly sparse.
    """
    beta = state.betas[lmda_idx]
    if hasattr(beta, "toarray"):
        return beta.toarray().ravel()
    return np.asarray(beta).ravel()


def _intercept_at(state, lmda_idx):
    return float(state.intercepts[lmda_idx])


def _predict_linear(state, X, lmda_idx):
    return X @ _coef_at(state, lmda_idx) + _intercept_at(state, lmda_idx)


class PretrainedLasso(RegressorMixin, BasePretrainedLasso):
    """Pretrained Lasso estimator.

    Two-step training:
    1. Fit an overall Lasso on all samples.
    2. For each group k, fit a group-specific Lasso with offset
       ``alpha * X_k @ beta_overall``, where ``alpha`` controls
       how strongly the overall model is used as a prior.

    When ``alpha=0`` the group model is fit independently (no pretraining).
    When ``alpha=1`` the group model explains the residuals from the overall model.

    Parameters
    ----------
    alpha : float in [0, 1], default=0.5
        Pretraining strength.
    fit_intercept : bool, default=True
        Whether to fit an intercept in all sub-models.
    lmda_path_size : int, default=100
        Number of lambda values in the regularization path.
    min_ratio : float, default=0.01
        Ratio of smallest to largest lambda on the path.

    Attributes
    ----------
    overall_model_ : adelie state
        Fitted overall Lasso on all data.
    overall_lmda_idx_ : int
        Index into the overall model's lambda path selected by CV.
    overall_coef_ : ndarray of shape (n_features,)
        Coefficients of the overall model at ``overall_lmda_idx_``.
    overall_intercept_ : float
        Intercept of the overall model at ``overall_lmda_idx_``.
    pretrain_models_ : dict {group -> adelie state}
        Per-group Lasso fits with the pretraining offset.
    individual_models_ : dict {group -> adelie state}
        Per-group Lasso fits with no offset (alpha=0 baseline).
    groups_ : ndarray
        Unique group labels seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self, alpha=0.5, fit_intercept=True, lmda_path_size=100, min_ratio=0.01
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.lmda_path_size = lmda_path_size
        self.min_ratio = min_ratio

    def _validate_params(self):
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not isinstance(self.lmda_path_size, int) or self.lmda_path_size < 1:
            raise ValueError(f"lmda_path_size must be a positive integer, got {self.lmda_path_size}")
        if not (0.0 < self.min_ratio < 1.0):
            raise ValueError(f"min_ratio must be in (0, 1), got {self.min_ratio}")

    def _label(self, g):
        """Return the display label for group g."""
        return self.group_labels_.get(g, g)

    def _feature_name(self, i):
        """Return the feature name for index i, or the index itself if no names set."""
        if self.feature_names_in_ is not None:
            return self.feature_names_in_[i]
        return i

    def _names_or_indices(self, indices):
        """Convert an array of feature indices to names if available."""
        if self.feature_names_in_ is not None:
            return self.feature_names_in_[indices]
        return indices

    def fit(self, X, y, groups, group_labels=None, feature_names=None):
        """Fit the pretrained Lasso model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        groups : array-like of shape (n_samples,)
            Group identifier for each sample.
        group_labels : dict or None, default=None
            Human-readable names for each group, e.g. ``{0: "control", 1: "treated"}``.
            If None, the raw group values are used as keys everywhere.
        feature_names : array-like of shape (n_features,) or None, default=None
            Names for each feature. If None and X is a DataFrame, column names
            are used automatically.

        Returns
        -------
        self
        """
        self._validate_params()

        # Extract feature names from DataFrame before converting to array
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
            raise ValueError(
                f"groups has {len(groups)} elements but X has {X.shape[0]} samples"
            )
        self.groups_ = np.unique(groups)
        if len(self.groups_) < 2:
            raise ValueError("groups must contain at least 2 unique values")

        if group_labels is not None:
            unknown = set(group_labels) - set(self.groups_)
            if unknown:
                raise ValueError(f"group_labels contains unknown group keys: {unknown}")
        self.group_labels_ = group_labels if group_labels is not None else {}

        # ------------------------------------------------------------------
        # Step 1: overall model on all data, lambda selected by CV
        # ------------------------------------------------------------------
        cv_overall = ad.cv_grpnet(
            X,
            ad.glm.gaussian(y),
            alpha=1,  # lasso (no elastic net)
            intercept=self.fit_intercept,
            lmda_path_size=self.lmda_path_size,
            min_ratio=self.min_ratio,
            progress_bar=False,
        )
        self.overall_lmda_idx_ = cv_overall.best_idx  # index in the CV lambda path
        # fit() refits on all data stopping at the CV-selected lambda,
        # so that lambda is the last one in the returned state (index -1).
        self.overall_model_ = cv_overall.fit(
            X,
            ad.glm.gaussian(y),
            alpha=1,
            intercept=self.fit_intercept,
            progress_bar=False,
        )
        self.overall_coef_ = _coef_at(self.overall_model_, -1)
        self.overall_intercept_ = _intercept_at(self.overall_model_, -1)

        # ------------------------------------------------------------------
        # Step 2: per-group models
        # ------------------------------------------------------------------
        self.pretrain_models_ = {}
        self.individual_models_ = {}

        overall_pred = _predict_linear(self.overall_model_, X, -1)

        for g in self.groups_:
            mask = groups == g
            X_g = np.asfortranarray(X[mask])
            y_g = y[mask]

            glm_g = ad.glm.gaussian(y_g)
            shared_kwargs = dict(
                alpha=1,
                intercept=self.fit_intercept,
                lmda_path_size=self.lmda_path_size,
                min_ratio=self.min_ratio,
                progress_bar=False,
            )

            # Pretrained: offset = alpha * overall predictions for this group
            offset_g = self.alpha * overall_pred[mask]
            self.pretrain_models_[g] = ad.grpnet(
                X_g, glm_g, offsets=offset_g, **shared_kwargs
            )

            # Individual: no offset
            self.individual_models_[g] = ad.grpnet(X_g, glm_g, **shared_kwargs)

        return self

    def predict(self, X, groups, model="pretrain", lmda_idx=None):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        groups : array-like of shape (n_samples,)
        model : {"pretrain", "individual", "overall"}, default="pretrain"
        lmda_idx : int or None
            Lambda index to use for group models. If None, uses the index
            that minimizes CV error for each group model.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, order="F")
        groups = np.asarray(groups)
        y_pred = np.empty(X.shape[0])

        if model == "overall":
            return _predict_linear(self.overall_model_, X, -1)

        overall_pred = _predict_linear(self.overall_model_, X, -1)

        for g in self.groups_:
            mask = groups == g
            X_g = X[mask]

            if model == "pretrain":
                state_g = self.pretrain_models_[g]
            else:
                state_g = self.individual_models_[g]

            idx = (
                lmda_idx if lmda_idx is not None else len(state_g.lmdas) // 2
            )  # TODO: CV selection
            group_pred = _predict_linear(state_g, X_g, idx)

            if model == "pretrain":
                # Total prediction = offset contribution + group-specific
                y_pred[mask] = self.alpha * overall_pred[mask] + group_pred
            else:
                y_pred[mask] = group_pred

        return y_pred

    def score(self, X, y, groups):
        """Return R² on the given data."""
        return r2_score(y, self.predict(X, groups))

    def __repr__(self):
        params = (
            f"alpha={self.alpha}, "
            f"fit_intercept={self.fit_intercept}, "
            f"lmda_path_size={self.lmda_path_size}, "
            f"min_ratio={self.min_ratio}"
        )
        r = f"PretrainedLasso({params})"

        fitted = hasattr(self, "overall_model_")
        if not fitted:
            return r + "\n  [not fitted]"

        overall_nz_idx = np.where(self.overall_coef_ != 0)[0]
        n_overall_nz = len(overall_nz_idx)
        overall_nz_names = self._names_or_indices(overall_nz_idx)
        preview = list(overall_nz_names[:5])
        if n_overall_nz > 5:
            preview.append("...")
        overall_nz_str = f"{n_overall_nz} / {self.n_features_in_}  [{', '.join(str(v) for v in preview)}]"

        group_nz = {
            self._label(g): int(np.sum(_coef_at(self.pretrain_models_[g], -1) != 0))
            for g in self.groups_
        }
        group_nz_str = ", ".join(f"{lbl}: {v}" for lbl, v in group_nz.items())

        return (
            f"{r}\n"
            f"  n_features   : {self.n_features_in_}\n"
            f"  n_groups     : {len(self.groups_)}\n"
            f"  overall nz   : {overall_nz_str}\n"
            f"  pretrain nz  : {group_nz_str}\n"
            f"  overall λ    : idx {self.overall_lmda_idx_} in CV path, {len(self.overall_model_.lmdas)} lambdas fitted"
        )

    def get_coef(self, model="all", lmda_idx=None):
        """Return fitted coefficients.

        Parameters
        ----------
        model : {"all", "overall", "pretrain", "individual"}, default="all"
        lmda_idx : int or None

        Returns
        -------
        dict or ndarray
        """
        check_is_fitted(self)

        def _group_coefs(models):
            out = {}
            for g, state in models.items():
                idx = lmda_idx if lmda_idx is not None else len(state.lmdas) // 2
                out[self._label(g)] = {
                    "coef": _coef_at(state, idx),
                    "intercept": _intercept_at(state, idx),
                }
            return out

        result = {}
        if model in ("all", "overall"):
            result["overall"] = {
                "coef": self.overall_coef_,
                "intercept": self.overall_intercept_,
            }
        if model in ("all", "pretrain"):
            result["pretrain"] = _group_coefs(self.pretrain_models_)
        if model in ("all", "individual"):
            result["individual"] = _group_coefs(self.individual_models_)

        return result if model == "all" else result[model]


class PretrainedLassoCV(RegressorMixin, BasePretrainedLasso):
    """Pretrained Lasso with cross-validation over alpha (pretraining strength).

    Fits :class:`PretrainedLasso` for each candidate alpha and selects the one
    that minimises CV error on the group-specific step.

    Parameters
    ----------
    alphas : array-like or None, default=None
        Candidate pretraining strengths. If None, defaults to
        ``[0, 0.25, 0.5, 0.75, 1.0]``.
    cv : int, default=5
        Number of CV folds for alpha selection.
    fit_intercept : bool, default=True
    lmda_path_size : int, default=100
    min_ratio : float, default=0.01

    Attributes
    ----------
    alpha_ : float
        Best alpha chosen by CV.
    best_estimator_ : PretrainedLasso
        Fitted PretrainedLasso with ``alpha_``.
    cv_results_ : dict
        Mean CV error per alpha value.
    """

    def __init__(
        self, alphas=None, cv=5, fit_intercept=True, lmda_path_size=100, min_ratio=0.01
    ):
        self.alphas = alphas
        self.cv = cv
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
        if not isinstance(self.lmda_path_size, int) or self.lmda_path_size < 1:
            raise ValueError(f"lmda_path_size must be a positive integer, got {self.lmda_path_size}")
        if not (0.0 < self.min_ratio < 1.0):
            raise ValueError(f"min_ratio must be in (0, 1), got {self.min_ratio}")

    def fit(self, X, y, groups, group_labels=None, feature_names=None):
        """Fit PretrainedLassoCV.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        groups : array-like of shape (n_samples,)
        group_labels : dict or None, default=None
            Human-readable names for each group, e.g. ``{0: "control", 1: "treated"}``.
        feature_names : array-like of shape (n_features,) or None, default=None
            Names for each feature. If None and X is a DataFrame, column names
            are used automatically.

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

        # ------------------------------------------------------------------
        # K-fold CV over alpha candidates
        # StratifiedKFold on groups ensures every group appears in every
        # training fold, so no group model is ever asked to predict unseen
        # group IDs.
        # ------------------------------------------------------------------
        splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=0)
        fold_errors = {a: [] for a in alphas}

        for train_idx, test_idx in splitter.split(X, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            g_tr, g_te = groups[train_idx], groups[test_idx]

            for a in alphas:
                est = PretrainedLasso(
                    alpha=a,
                    fit_intercept=self.fit_intercept,
                    lmda_path_size=self.lmda_path_size,
                    min_ratio=self.min_ratio,
                ).fit(X_tr, y_tr, g_tr)

                y_pred = est.predict(X_te, g_te)
                fold_errors[a].append(np.mean((y_te - y_pred) ** 2))

        self.cv_results_ = {a: float(np.mean(errs)) for a, errs in fold_errors.items()}
        self.alpha_ = min(self.cv_results_, key=self.cv_results_.__getitem__)

        # ------------------------------------------------------------------
        # Refit on all data with the selected alpha
        # ------------------------------------------------------------------
        self.best_estimator_ = PretrainedLasso(
            alpha=self.alpha_,
            fit_intercept=self.fit_intercept,
            lmda_path_size=self.lmda_path_size,
            min_ratio=self.min_ratio,
        ).fit(X, y, groups, group_labels=group_labels, feature_names=feature_names)

        # Mirror fitted attributes from best estimator
        self.overall_model_ = self.best_estimator_.overall_model_
        self.overall_lmda_idx_ = self.best_estimator_.overall_lmda_idx_
        self.overall_coef_ = self.best_estimator_.overall_coef_
        self.overall_intercept_ = self.best_estimator_.overall_intercept_
        self.pretrain_models_ = self.best_estimator_.pretrain_models_
        self.individual_models_ = self.best_estimator_.individual_models_
        self.groups_ = self.best_estimator_.groups_
        self.feature_names_in_ = self.best_estimator_.feature_names_in_

        return self

    def predict(self, X, groups, **kwargs):
        check_is_fitted(self)
        return self.best_estimator_.predict(X=X, groups=groups, **kwargs)

    def score(self, X, y, groups):
        """Return R² on the given data."""
        return r2_score(y, self.predict(X, groups))

    def __repr__(self):
        alphas = self.alphas if self.alphas is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        params = (
            f"alphas={alphas}, "
            f"cv={self.cv}, "
            f"fit_intercept={self.fit_intercept}"
        )
        r = f"PretrainedLassoCV({params})"

        if not hasattr(self, "best_estimator_"):
            return r + "\n  [not fitted]"

        cv_summary = ", ".join(
            f"{a:.2f}: {v:.4f}" if not np.isnan(v) else f"{a:.2f}: —"
            for a, v in self.cv_results_.items()
        )
        return (
            f"{r}\n"
            f"  n_features   : {self.n_features_in_}\n"
            f"  n_groups     : {len(self.groups_)}\n"
            f"  alpha_       : {self.alpha_}  (selected)\n"
            f"  alphalist    : {list(self.alphalist_)}\n"
            f"  cv MSE       : {cv_summary}\n"
            f"  best model   :\n    {repr(self.best_estimator_).replace(chr(10), chr(10) + '    ')}"
        )

    def get_coef(self, model="all", **kwargs):
        check_is_fitted(self)
        return self.best_estimator_.get_coef(model=model, **kwargs)
