"""Support (active feature set) extraction for PretrainedLasso models."""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._ptlasso import _coef_at


def _nonzero(state, lmda_idx):
    """Return 0-based indices of nonzero features at the given lambda index."""
    coef = _coef_at(state, lmda_idx)
    return np.where(coef != 0)[0]


def _resolve(fit, indices):
    """Return feature names for indices if fit has feature_names_in_, else indices."""
    names = getattr(fit, "feature_names_in_", None)
    if names is not None:
        return names[indices]
    return indices


def get_overall_support(fit, lmda_idx=None):
    """Nonzero feature indices (or names) from the overall model.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
    lmda_idx : int or None
        Index into the overall model's lambda path. Defaults to -1
        (the CV-selected lambda, which is the last one fitted).

    Returns
    -------
    support : ndarray of int or str
        Sorted feature indices, or feature names if ``feature_names`` was
        passed to ``fit()``.
    """
    check_is_fitted(fit)
    idx = lmda_idx if lmda_idx is not None else -1
    return _resolve(fit, _nonzero(fit.overall_model_, idx))


def get_pretrain_support(
    fit, lmda_idx=None, groups=None, include_overall=True, common_only=False
):
    """Nonzero feature indices (or names) from the per-group pretrained models.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
    lmda_idx : int or None
        Lambda index for the pretrained group models. Defaults to -1.
    groups : array-like or None
        Subset of group labels to consider. Default is all groups.
    include_overall : bool, default=True
        If True, union the per-group support with the overall model support.
        Ignored when ``alpha=1`` (R convention: overall support would double-count).
    common_only : bool, default=False
        If True, return only features selected by more than half of the groups.

    Returns
    -------
    support : ndarray of int or str
        Sorted feature indices, or feature names if ``feature_names`` was
        passed to ``fit()``.
    """
    check_is_fitted(fit)
    groups = fit.groups_ if groups is None else np.asarray(groups)

    # Mirror R logic: include overall support only when alpha < 1.
    # When alpha=1 the group models capture full residuals from the overall
    # model — including overall support would double-count those features.
    base_idx = (
        _nonzero(fit.overall_model_, -1)
        if include_overall and fit.alpha < 1
        else np.array([], dtype=int)
    )

    per_group = []
    for g in groups:
        state = fit.pretrain_models_[g]
        idx = lmda_idx if lmda_idx is not None else -1
        per_group.append(_nonzero(state, idx))

    indices = _combine_support(per_group, base_idx, common_only, n_groups=len(groups))
    return _resolve(fit, indices)


def get_individual_support(fit, lmda_idx=None, groups=None, common_only=False):
    """Nonzero feature indices (or names) from the per-group individual models.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
    lmda_idx : int or None
        Lambda index for the individual group models. Defaults to -1.
    groups : array-like or None
        Subset of group labels to consider. Default is all groups.
    common_only : bool, default=False
        If True, return only features selected by more than half of the groups.

    Returns
    -------
    support : ndarray of int or str
        Sorted feature indices, or feature names if ``feature_names`` was
        passed to ``fit()``.
    """
    check_is_fitted(fit)
    groups = fit.groups_ if groups is None else np.asarray(groups)

    per_group = []
    for g in groups:
        state = fit.individual_models_[g]
        idx = lmda_idx if lmda_idx is not None else -1
        per_group.append(_nonzero(state, idx))

    indices = _combine_support(
        per_group, np.array([], dtype=int), common_only, n_groups=len(groups)
    )
    return _resolve(fit, indices)


def _combine_support(per_group, base, common_only, n_groups):
    """Union or majority-vote the per-group supports, then union with base."""
    if not per_group:
        return np.sort(base)

    all_features = np.sort(np.unique(np.concatenate(per_group + [base])))

    if not common_only:
        return all_features

    # Keep only features chosen by more than half the groups
    # (base features are always included regardless)
    counts = np.array([sum(f in supp for supp in per_group) for f in all_features])
    majority = all_features[counts > n_groups / 2]
    return np.sort(np.unique(np.concatenate([majority, base])))
