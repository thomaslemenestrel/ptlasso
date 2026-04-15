"""Support (active feature set) extraction for PretrainedLasso models."""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._ptlasso import _coef_at


def _nonzero(state, lmda_idx, n_features=None, n_classes=None):
    """0-based feature indices that are nonzero at ``lmda_idx``.

    For multinomial, a feature is considered active if its coefficient is
    nonzero for *any* class.
    """
    coef = _coef_at(state, lmda_idx)
    if n_features is not None and n_classes is not None and len(coef) == n_features * n_classes:
        coef_mat = coef.reshape(n_features, n_classes, order="F")
        return np.where(np.any(coef_mat != 0, axis=1))[0]
    return np.where(coef != 0)[0]


def _nonzero_fit(fit, state, lmda_idx):
    """_nonzero with shape info taken from a fitted estimator."""
    return _nonzero(
        state,
        lmda_idx,
        n_features=fit.n_features_in_,
        n_classes=fit.n_classes_,
    )


def _resolve(fit, indices):
    """Return feature names for indices if available, else indices."""
    names = getattr(fit, "feature_names_in_", None)
    if names is not None:
        return names[indices]
    return indices


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
    return _resolve(fit, _nonzero_fit(fit, fit.overall_model_, idx))


def get_pretrain_support(fit, lmda_idx=None, groups=None, include_overall=True, common_only=False):
    """Nonzero features from the per-group pretrained models.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
    lmda_idx : int or None
        Lambda index for the pretrained group models. Defaults to -1.
    groups : array-like or None
        Subset of group labels to consider. Default is all groups.
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

    base_idx = (
        _nonzero_fit(fit, fit.overall_model_, fit.overall_lmda_idx_)
        if include_overall and fit.alpha < 1
        else np.array([], dtype=int)
    )

    per_group = [
        _nonzero_fit(fit, fit.pretrain_models_[g], lmda_idx if lmda_idx is not None else -1)
        for g in groups
    ]

    return _resolve(fit, _combine_support(per_group, base_idx, common_only, len(groups)))


def get_individual_support(fit, lmda_idx=None, groups=None, common_only=False):
    """Nonzero features from the per-group individual (no-pretraining) models.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
    lmda_idx : int or None
        Lambda index for the individual group models. Defaults to -1.
    groups : array-like or None
        Subset of group labels to consider. Default is all groups.
    common_only : bool, default=False
        If True, return only features selected by more than half the groups.

    Returns
    -------
    support : ndarray of int or str
    """
    check_is_fitted(fit)
    groups = fit.groups_ if groups is None else np.asarray(groups)

    per_group = [
        _nonzero_fit(fit, fit.individual_models_[g], lmda_idx if lmda_idx is not None else -1)
        for g in groups
    ]

    return _resolve(
        fit, _combine_support(per_group, np.array([], dtype=int), common_only, len(groups))
    )


def get_pretrain_support_split(fit, lmda_idx=None, groups=None):
    """Split pretrain support into stage-1 ("common") and stage-2 ("individual") parts.

    Mirrors the R package's ``suppre.common`` / ``suppre.individual`` distinction:

    - **common**     : features selected by the *overall* model (stage 1).
    - **individual** : features selected by the per-group models (stage 2) that
                       are *not* in the common support.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
    lmda_idx : int or None
        Lambda index for the group models. Defaults to -1.
    groups : array-like or None
        Subset of group labels. Default is all groups.

    Returns
    -------
    common : ndarray of int or str
        Features from the overall model's support.
    individual : ndarray of int or str
        Features from stage-2 group models not already in ``common``.
    """
    check_is_fitted(fit)
    groups = fit.groups_ if groups is None else np.asarray(groups)

    overall_idx = set(_nonzero_fit(fit, fit.overall_model_, fit.overall_lmda_idx_).tolist())

    stage2_idx = set()
    for g in groups:
        idx = lmda_idx if lmda_idx is not None else -1
        stage2_idx |= set(_nonzero_fit(fit, fit.pretrain_models_[g], idx).tolist())

    common_idx = np.sort(np.array(list(overall_idx), dtype=int))
    individual_idx = np.sort(np.array(list(stage2_idx - overall_idx), dtype=int))

    return _resolve(fit, common_idx), _resolve(fit, individual_idx)


def _combine_support(per_group, base, common_only, n_groups):
    """Union or majority-vote the per-group supports, then union with base."""
    if not per_group:
        return np.sort(base)

    all_features = np.sort(np.unique(np.concatenate(per_group + [base])))

    if not common_only:
        return all_features

    counts = np.array([sum(f in supp for supp in per_group) for f in all_features])
    majority = all_features[counts > n_groups / 2]
    return np.sort(np.unique(np.concatenate([majority, base])))
