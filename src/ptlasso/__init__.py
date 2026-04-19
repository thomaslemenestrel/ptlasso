"""ptlasso: Pretrained Lasso for grouped sparse linear models.

Implements the two-step Pretrained Lasso estimator of Craig, Pilanci,
Le Menestrel et al. (2025, JRSSB).  A global Lasso is fitted on all
samples first; its linear predictor is then used as an offset when
fitting per-group Lasso models, allowing each group to borrow strength
from the overall signal while still learning group-specific coefficients.

Main classes
------------
PretrainedLasso
    Two-step estimator with a fixed pretraining strength ``alpha``.
PretrainedLassoCV
    Same estimator with cross-validation over ``alpha``.

Support utilities
-----------------
get_overall_support, get_pretrain_support, get_individual_support,
get_pretrain_support_split

Plotting
--------
plot_cv, plot_paths

Simulation helpers
------------------
make_data, gaussian_example_data, binomial_example_data
"""

from ._ptlasso import PretrainedLasso, PretrainedLassoCV
from ._support import (
    get_individual_support,
    get_overall_support,
    get_pretrain_support,
    get_pretrain_support_split,
)
from ._plot import plot_cv, plot_paths
from ._simulate import make_data, gaussian_example_data, binomial_example_data
from ._constants import COLORS, FIGURE_WIDTHS

try:
    from importlib.metadata import version as _version, PackageNotFoundError
    __version__ = _version("ptlasso")
except PackageNotFoundError:
    __version__ = "unknown"
__all__ = [
    "PretrainedLasso",
    "PretrainedLassoCV",
    "get_overall_support",
    "get_pretrain_support",
    "get_pretrain_support_split",
    "get_individual_support",
    "plot_cv",
    "plot_paths",
    "make_data",
    "gaussian_example_data",
    "binomial_example_data",
    "COLORS",
    "FIGURE_WIDTHS",
]
