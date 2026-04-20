"""Support (active feature set) extraction for PretrainedLasso models."""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._ptlasso import _coef_at


def _nonzero(fit, state, lmda_idx):
    """0-based feature indices that are nonzero at ``lmda_idx``.

    For multinomial, a feature is considered active if its coefficient is
    nonzero for any class.
    """
    coef = _coef_at(state, lmda_idx)
    if fit.n_classes_ is not None:
        coef = coef.reshape(fit.n_features_in_, fit.n_classes_, order="F")
        return np.where(np.any(coef != 0, axis=1))[0]
    return np.where(coef != 0)[0]


def _nonzero_overall(fit, lmda_idx):
    """0-based X-feature indices that are nonzero in the overall model.

    The overall model is trained on ``[onehot | X]`` when group intercepts
    are enabled (``fit._n_onehot_ > 0``).  The first ``_n_onehot_`` columns
    of the coefficient vector correspond to the group-dummy variables and must
    be skipped so that returned indices refer to the original X features.
    """
    n_skip = getattr(fit, "_n_onehot_", 0)
    coef = _coef_at(fit.overall_model_, lmda_idx)
    if fit.n_classes_ is not None:
        p_aug = fit.n_features_in_ + n_skip
        coef_mat = coef.reshape(p_aug, fit.n_classes_, order="F")
        coef_x = coef_mat[n_skip:, :]
        return np.where(np.any(coef_x != 0, axis=1))[0]
    return np.where(coef[n_skip:] != 0)[0]


def _resolve(fit, indices):
    """Return feature names for indices if available, else return the indices."""
    names = getattr(fit, "feature_names_in_", None)
    return names[indices] if names is not None else indices


def _union_support(per_group, base=None):
    """Union of per-group supports (list of arrays), optionally including a base set."""
    arrays = per_group if base is None else per_group + [base]
    return np.sort(np.unique(np.concatenate(arrays))) if arrays else np.array([], dtype=int)


def _majority_support(per_group, base, n_groups):
    """Features selected by more than half the groups, unioned with base."""
    all_features = _union_support(per_group, base)
    counts = np.array([sum(f in s for s in per_group) for f in all_features])
    majority = all_features[counts > n_groups / 2]
    return np.sort(np.unique(np.concatenate([majority, base])))


def _combine(per_group, base, common_only, n_groups):
    """Combine per-group supports with a base set."""
    if not per_group:
        return np.sort(base)
    if common_only:
        return _majority_support(per_group, base, n_groups)
    return _union_support(per_group, base)


def get_overall_support(fit, lmda_idx=None):
    """Nonzero features from the overall model.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
    lmda_idx : int or None
        Index into the overall model's lambda path.
        Defaults to ``overall_lmda_idx_`` (the CV-selected lambda).

    Returns
    -------
    support : ndarray of int or str
    """
    check_is_fitted(fit)
    idx = lmda_idx if lmda_idx is not None else fit.overall_lmda_idx_
    return _resolve(fit, _nonzero_overall(fit, idx))


def get_pretrain_support(fit, lmda_idx=None, groups=None, include_overall=True, common_only=False):
    """Nonzero features from the per-group pretrained models.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
    lmda_idx : int or None
        Lambda index for the pretrained group models.  When ``None`` (default),
        each group uses its own CV-selected index (``pretrain_lmda_idx_``),
        matching the lambda used by ``predict``.
    groups : array-like or None
        Subset of group labels to consider.  Default is all groups.
    include_overall : bool, default=True
        Union the per-group support with the overall model support.
        Ignored when ``alpha=1`` (R convention).
    common_only : bool, default=False
        If True, return only features selected by more than half the groups.

    Returns
    -------
    support : ndarray of int or str
    """
    check_is_fitted(fit)
    groups = fit.groups_ if groups is None else np.asarray(groups)

    base = (
        _nonzero_overall(fit, fit.overall_lmda_idx_)
        if include_overall and fit.alpha < 1
        else np.array([], dtype=int)
    )
    # Use each group's CV-selected lambda when lmda_idx is not specified.
    per_group_idxs = getattr(fit, "pretrain_lmda_idx_", {})
    per_group = [
        _nonzero(
            fit,
            fit.pretrain_models_[g],
            lmda_idx if lmda_idx is not None else per_group_idxs.get(g, -1),
        )
        for g in groups
    ]

    return _resolve(fit, _combine(per_group, base, common_only, len(groups)))


def get_individual_support(fit, lmda_idx=None, groups=None, common_only=False):
    """Nonzero features from the per-group individual (no-pretraining) models.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
    lmda_idx : int or None
        Lambda index for the individual group models.  When ``None`` (default),
        each group uses its own CV-selected index (``individual_lmda_idx_``),
        matching the lambda used by ``predict``.
    groups : array-like or None
        Subset of group labels to consider.  Default is all groups.
    common_only : bool, default=False
        If True, return only features selected by more than half the groups.

    Returns
    -------
    support : ndarray of int or str
    """
    check_is_fitted(fit)
    groups = fit.groups_ if groups is None else np.asarray(groups)

    per_group_idxs = getattr(fit, "individual_lmda_idx_", {})
    per_group = [
        _nonzero(
            fit,
            fit.individual_models_[g],
            lmda_idx if lmda_idx is not None else per_group_idxs.get(g, -1),
        )
        for g in groups
    ]

    return _resolve(fit, _combine(per_group, np.array([], dtype=int), common_only, len(groups)))


def get_pretrain_support_split(fit, lmda_idx=None, groups=None):
    """Split pretrain support into stage-1 ("common") and stage-2 ("individual") parts.

    Mirrors the R package's ``suppre.common`` / ``suppre.individual`` convention:

    - **common**     : features selected by the overall model (stage 1).
    - **individual** : features selected by the per-group models (stage 2)
                       that are *not* in the common support.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
    lmda_idx : int or None
        Lambda index for the group models.  When ``None`` (default), each
        group uses its own CV-selected index (``pretrain_lmda_idx_``).
    groups : array-like or None
        Subset of group labels.  Default is all groups.

    Returns
    -------
    common : ndarray of int or str
    individual : ndarray of int or str
    """
    check_is_fitted(fit)
    groups = fit.groups_ if groups is None else np.asarray(groups)

    per_group_idxs = getattr(fit, "pretrain_lmda_idx_", {})
    overall_idx = set(_nonzero_overall(fit, fit.overall_lmda_idx_).tolist())
    stage2_idx = set()
    for g in groups:
        idx = lmda_idx if lmda_idx is not None else per_group_idxs.get(g, -1)
        stage2_idx |= set(_nonzero(fit, fit.pretrain_models_[g], idx).tolist())

    common_idx = np.sort(np.array(list(overall_idx), dtype=int))
    individual_idx = np.sort(np.array(list(stage2_idx - overall_idx), dtype=int))

    return _resolve(fit, common_idx), _resolve(fit, individual_idx)
