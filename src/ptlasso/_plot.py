"""Publication-quality plotting utilities for PretrainedLasso models."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._constants import COLORS, FIGURE_WIDTHS
from ._support import get_overall_support, get_pretrain_support, get_individual_support


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _betas_dense(state):
    """Return the (L, p) beta matrix from an adelie state."""
    b = state.betas
    return b.toarray() if hasattr(b, "toarray") else np.asarray(b)


def _cv_ylabel(family):
    return "CV MSE" if family == "gaussian" else "CV log-loss"


def _active_features(fit):
    """Integer indices of every feature that is nonzero in any sub-model."""
    coef = fit.overall_coef_
    # overall_coef_ is (p,) for gaussian/binomial, (p, K) for multinomial
    overall = set(np.where(np.reshape(coef, (fit.n_features_in_, -1)).any(axis=1))[0])
    group_active = set()
    for g in fit.groups_:
        group_active |= set(np.where(_betas_dense(fit.pretrain_models_[g])[-1] != 0)[0])
        group_active |= set(np.where(_betas_dense(fit.individual_models_[g])[-1] != 0)[0])
    return sorted(overall | group_active)


def _feature_color_map(fit):
    """Map each globally active feature index to a tab20 colour."""
    active = _active_features(fit)
    n = max(len(active), 1)
    cmap = matplotlib.colormaps["tab20"]
    return {j: cmap(i / n) for i, j in enumerate(active)}


def _label_map(fit):
    """Return {index: name} if feature names are available, else None."""
    if fit.feature_names_in_ is None:
        return None
    return {i: fit.feature_names_in_[i] for i in range(fit.n_features_in_)}


# ------------------------------------------------------------------
# plot_cv
# ------------------------------------------------------------------


def plot_cv(fit, ax=None, plot_alphahat=True, column="single", save=None):
    """Plot the cross-validation curve for a :class:`PretrainedLassoCV`.

    Draws the mean CV loss ±1 SE band over alpha, with horizontal reference
    lines for the individual and overall baselines.

    Parameters
    ----------
    fit : PretrainedLassoCV
        A fitted CV estimator.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.  A new figure is created when ``None``.
    plot_alphahat : bool, default=True
        Whether to draw a vertical line at the selected ``alpha_``.
    column : {"single", "double"}, default="single"
        Target figure width — ``"single"`` ≈ 3.5 in, ``"double"`` ≈ 7 in.
    save : str or None, default=None
        File path to save the figure (300 dpi).  No file is written when ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    check_is_fitted(fit)

    w = FIGURE_WIDTHS.get(column, 3.5)
    if ax is None:
        fig, ax = plt.subplots(figsize=(w, w * 0.8))
    else:
        fig = ax.get_figure()

    alphas = list(fit.alphalist_)
    cv_mean = np.array([fit.cv_results_[a] for a in alphas])
    cv_se = np.array([fit.cv_results_se_[a] for a in alphas])

    # ±1 SE band + pretrain curve
    ax.fill_between(
        alphas, cv_mean - cv_se, cv_mean + cv_se, color=COLORS["pretrain"], alpha=0.15, linewidth=0
    )
    ax.plot(
        alphas,
        cv_mean,
        color=COLORS["pretrain"],
        marker="o",
        markersize=5,
        linewidth=2,
        label="Pretrain",
        zorder=3,
    )

    # Individual and overall baselines
    ax.axhline(
        fit.cv_results_individual_, color=COLORS["individual"], linestyle="--", linewidth=1.8
    )
    ax.axhline(fit.cv_results_overall_, color=COLORS["overall"], linestyle="--", linewidth=1.8)

    # Selected alpha
    if plot_alphahat:
        ax.axvline(
            fit.alpha_,
            color="#555555",
            linestyle=":",
            linewidth=1.5,
            label=f"$\\hat{{\\alpha}}={fit.alpha_}$",
        )

    # Support sizes at right edge
    n_pre = len(get_pretrain_support(fit))
    n_ind = len(get_individual_support(fit))
    n_ov = len(get_overall_support(fit))
    xmax = max(alphas)
    ax.text(
        xmax + 0.01,
        fit.cv_results_individual_,
        f"Individual  $|\\hat{{S}}|={n_ind}$",
        va="center",
        fontsize=8,
        color=COLORS["individual"],
        clip_on=False,
    )
    ax.text(
        xmax + 0.01,
        fit.cv_results_overall_,
        f"Overall  $|\\hat{{S}}|={n_ov}$",
        va="center",
        fontsize=8,
        color=COLORS["overall"],
        clip_on=False,
    )

    # Top axis: support size at selected alpha
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([fit.alpha_])
    ax2.set_xticklabels([f"$|\\hat{{S}}|={n_pre}$"], fontsize=8, color=COLORS["pretrain"])
    ax2.tick_params(length=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    _despine(ax)
    ax.set_xlabel("$\\alpha$", fontsize=11)
    ax.set_ylabel(_cv_ylabel(fit.family), fontsize=11)
    ax.set_title("Cross-validation over $\\alpha$", fontsize=12, pad=14)
    ax.set_xticks(alphas)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig, ax


# ------------------------------------------------------------------
# plot_paths
# ------------------------------------------------------------------


def _draw_paths(ax, state, title, color, feature_colors, support, labels, xlim):
    """Draw regularisation paths for one sub-model panel."""
    betas = _betas_dense(state)  # (L, p)
    lmdas = np.log(np.asarray(state.lmdas))
    active = np.where(np.any(betas != 0, axis=0))[0]
    final = set(support)

    # Grey paths for features not in final support
    for j in active:
        if j not in final:
            ax.plot(lmdas, betas[:, j], color="#cccccc", linewidth=0.6, alpha=0.5, zorder=1)

    # Coloured paths + endpoint labels for final support
    x_end = lmdas[-1]
    placed = []  # y-positions of labels already placed (for de-overlapping)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-9

    for j in sorted(final, key=lambda j: abs(betas[-1, j]), reverse=True):
        c = feature_colors.get(j, "#2c7bb6")
        ax.plot(lmdas, betas[:, j], color=c, linewidth=1.8, alpha=0.92, zorder=2)

        y_end = float(betas[-1, j])
        name = str(labels[j]) if labels else str(j)
        if all(abs(y_end - y) > 0.04 * y_range for y in placed):
            ax.text(
                x_end + 0.05 * (xlim[1] - xlim[0]),
                y_end,
                name,
                fontsize=6.5,
                va="center",
                color=c,
                clip_on=False,
            )
            placed.append(y_end)

    ax.axvline(x_end, color="#888888", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axhline(0, color="#888888", linewidth=0.4, alpha=0.3)
    _despine(ax)
    ax.set_xlim(xlim)
    ax.set_xlabel("log $\\lambda$", fontsize=8)
    ax.set_ylabel("Coefficient", fontsize=8)
    ax.set_title(
        f"{title}  $(|\\hat{{S}}|={len(final)})$", fontsize=9, pad=5, color=color, fontweight="bold"
    )
    ax.tick_params(labelsize=7)


def plot_paths(fit, column="double", save=None):
    """Plot regularisation paths for all sub-models in a :class:`PretrainedLasso`.

    Produces a 3-row grid: overall model (full width), per-group pretrained
    models, and per-group individual models.  Features in the final support are
    coloured; inactive features are shown in grey.

    Parameters
    ----------
    fit : PretrainedLasso or PretrainedLassoCV
        A fitted estimator.
    column : {"single", "double"}, default="double"
        Target figure width — ``"single"`` ≈ 3.5 in, ``"double"`` ≈ 7 in.
    save : str or None, default=None
        File path to save the figure (300 dpi).  No file is written when ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    check_is_fitted(fit)

    k = len(fit.groups_)
    w = FIGURE_WIDTHS.get(column, 7.0)
    labels = _label_map(fit)
    feature_colors = _feature_color_map(fit)

    all_lmdas = np.log(np.asarray(fit.overall_model_.lmdas))
    xlim = (all_lmdas[0], all_lmdas[-1] + 0.18 * (all_lmdas[-1] - all_lmdas[0]))
    overall_sup = set(
        np.where(np.reshape(fit.overall_coef_, (fit.n_features_in_, -1)).any(axis=1))[0]
    )

    fig = plt.figure(figsize=(w, w * 0.42 * 3))
    gs = gridspec.GridSpec(3, k, figure=fig, hspace=0.7, wspace=0.45)

    # Row 0: overall model (spans all columns)
    _draw_paths(
        fig.add_subplot(gs[0, :]),
        fit.overall_model_,
        "Overall",
        COLORS["overall"],
        feature_colors,
        overall_sup,
        labels,
        xlim,
    )

    # Rows 1 & 2: per-group pretrain and individual
    for col, g in enumerate(fit.groups_):
        lbl = fit._label(g)
        pre_sup = set(np.where(_betas_dense(fit.pretrain_models_[g])[-1] != 0)[0])
        ind_sup = set(np.where(_betas_dense(fit.individual_models_[g])[-1] != 0)[0])

        _draw_paths(
            fig.add_subplot(gs[1, col]),
            fit.pretrain_models_[g],
            lbl,
            COLORS["pretrain"],
            feature_colors,
            pre_sup,
            labels,
            xlim,
        )
        _draw_paths(
            fig.add_subplot(gs[2, col]),
            fit.individual_models_[g],
            lbl,
            COLORS["individual"],
            feature_colors,
            ind_sup,
            labels,
            xlim,
        )

    # Row labels in left margin
    for row, (text, color) in enumerate(
        [
            ("Overall", COLORS["overall"]),
            ("Pretrain", COLORS["pretrain"]),
            ("Individual", COLORS["individual"]),
        ]
    ):
        fig.text(
            0.01,
            1 - (row + 0.5) / 3,
            text,
            va="center",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=color,
            rotation=90,
            transform=fig.transFigure,
        )

    fig.suptitle("Regularisation paths", fontsize=13, y=1.01)
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig
