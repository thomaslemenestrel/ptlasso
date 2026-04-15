"""Data simulation utilities for ptlasso examples and testing.

Mirrors R's ``makedata()``, ``gaussian.example.data()`` and ``binomial.example.data()``.
"""

import numpy as np


def make_data(
    k,
    class_sizes,
    s_common,
    s_indiv,
    beta_common,
    beta_indiv,
    intercepts=None,
    sigma=1.0,
    family="gaussian",
    seed=None,
):
    """Generate synthetic grouped data for ptlasso.

    Feature layout
    --------------
    - Columns 0 … s_common-1             : shared features (active in all groups)
    - Columns s_common … s_common+s_indiv[0]-1 : features specific to group 0
    - ... and so on for each group
    - Remaining columns (if any)          : pure noise

    Parameters
    ----------
    k : int
        Number of groups.
    class_sizes : array-like of length k
        Number of observations per group.
    s_common : int
        Number of shared (common) features.
    s_indiv : int or array-like of length k
        Number of group-specific features per group.
    beta_common : float or array-like
        Coefficients for common features.  Scalar → same value for all s_common
        features in every group.  1-D array of length s_common → per-feature
        coefficient (same across groups).  List of k arrays → per-group,
        per-feature coefficients.
    beta_indiv : float or array-like
        Coefficients for group-specific features, same shapes as beta_common.
    intercepts : array-like of length k or None
        Per-group intercepts.  Default is 0.
    sigma : float, default=1.0
        Gaussian noise std (only used for ``family="gaussian"``).
    family : {"gaussian", "binomial"}, default="gaussian"
    seed : int or None
        Random seed passed to ``numpy.random.default_rng``.

    Returns
    -------
    dict with keys
        ``X``      : ndarray (n, p)
        ``y``      : ndarray (n,)
        ``groups`` : ndarray (n,) of int
    """
    rng = np.random.default_rng(seed)

    class_sizes = np.asarray(class_sizes, dtype=int)
    s_indiv = np.broadcast_to(s_indiv, (k,)).copy()
    n = int(class_sizes.sum())
    p = s_common + int(s_indiv.sum())

    X = rng.standard_normal((n, p))
    y = np.empty(n)
    groups = np.empty(n, dtype=int)

    row_start = 0
    feat_start = s_common

    if intercepts is None:
        intercepts = np.zeros(k)

    for kk in range(k):
        row_end = row_start + class_sizes[kk]
        feat_end = feat_start + s_indiv[kk]

        beta = np.zeros(p)
        beta[:s_common] = _expand_coef(beta_common, kk, s_common)
        beta[feat_start:feat_end] = _expand_coef(beta_indiv, kk, s_indiv[kk])

        mu = X[row_start:row_end] @ beta + intercepts[kk]

        if family == "gaussian":
            y[row_start:row_end] = mu + sigma * rng.standard_normal(class_sizes[kk])
        elif family == "binomial":
            prob = 1.0 / (1.0 + np.exp(-mu))
            y[row_start:row_end] = rng.binomial(1, prob).astype(float)
        else:
            raise ValueError(f"family must be 'gaussian' or 'binomial', got '{family}'")

        groups[row_start:row_end] = kk
        row_start = row_end
        feat_start = feat_end

    return {"X": X, "y": y, "groups": groups}


def gaussian_example_data(
    k=2,
    class_sizes=None,
    s_common=5,
    s_indiv=5,
    beta_common=1.0,
    beta_indiv=0.5,
    sigma=1.0,
    seed=None,
):
    """Convenience wrapper for Gaussian grouped data.

    Parameters
    ----------
    k : int, default=2
    class_sizes : array-like or None
        Defaults to 50 observations per group.
    s_common : int, default=5
    s_indiv : int or array-like, default=5
    beta_common : float, default=1.0
    beta_indiv : float, default=0.5
    sigma : float, default=1.0
    seed : int or None

    Returns
    -------
    dict with keys ``X``, ``y``, ``groups``
    """
    if class_sizes is None:
        class_sizes = [50] * k
    return make_data(
        k=k,
        class_sizes=class_sizes,
        s_common=s_common,
        s_indiv=s_indiv,
        beta_common=beta_common,
        beta_indiv=beta_indiv,
        sigma=sigma,
        family="gaussian",
        seed=seed,
    )


def binomial_example_data(
    k=2,
    class_sizes=None,
    s_common=5,
    s_indiv=5,
    beta_common=0.5,
    beta_indiv=0.3,
    seed=None,
):
    """Convenience wrapper for binary grouped data.

    Parameters
    ----------
    k : int, default=2
    class_sizes : array-like or None
        Defaults to 100 observations per group.
    s_common : int, default=5
    s_indiv : int or array-like, default=5
    beta_common : float, default=0.5
    beta_indiv : float, default=0.3
    seed : int or None

    Returns
    -------
    dict with keys ``X``, ``y``, ``groups``
    """
    if class_sizes is None:
        class_sizes = [100] * k
    return make_data(
        k=k,
        class_sizes=class_sizes,
        s_common=s_common,
        s_indiv=s_indiv,
        beta_common=beta_common,
        beta_indiv=beta_indiv,
        family="binomial",
        seed=seed,
    )


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------


def _expand_coef(coef, group_idx, size):
    """Return a 1-D array of length ``size`` for group ``group_idx``."""
    if isinstance(coef, list):
        c = coef[group_idx]
        return np.broadcast_to(c, (size,)).copy() if np.isscalar(c) else np.asarray(c)
    if np.isscalar(coef):
        return np.full(size, float(coef))
    arr = np.asarray(coef)
    if arr.ndim == 1:
        return arr  # same for every group
    return arr[group_idx]
