"""Package-wide constants for ptlasso."""

# ------------------------------------------------------------------
# Model families and lambda selection modes
# ------------------------------------------------------------------

FAMILIES = ("gaussian", "binomial", "multinomial")

LMDA_MODES = ("lambda.1se", "lambda.min")

# ------------------------------------------------------------------
# Colour palette — matches the R package
# ------------------------------------------------------------------

COLORS = {
    "overall":    "#E9C46A",
    "pretrain":   "#2A9D8F",
    "individual": "#E76F51",
}

# ------------------------------------------------------------------
# Journal figure widths in inches
# ------------------------------------------------------------------

FIGURE_WIDTHS = {
    "single": 3.5,
    "double": 7.0,
    "full":   7.0,
}
