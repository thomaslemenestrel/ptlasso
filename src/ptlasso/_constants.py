"""Package-wide constants for ptlasso."""

# ------------------------------------------------------------------
# Model families and lambda selection modes
# ------------------------------------------------------------------

FAMILIES = ("gaussian", "binomial", "multinomial")

LMDA_MODES = ("lambda.1se", "lambda.min")

PREDICT_MODELS = ("pretrain", "individual", "overall")

PREDICT_TYPES = ("response", "link", "class")

COEF_MODELS = ("all", "overall", "pretrain", "individual")

ALPHATYPES = ("best", "varying")

DEFAULT_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ------------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------------

COLORS = {
    "overall": "#E9C46A",
    "pretrain": "#2A9D8F",
    "individual": "#E76F51",
}

# ------------------------------------------------------------------
# Figure widths in inches
# ------------------------------------------------------------------

FIGURE_WIDTHS = {
    "single": 3.5,
    "double": 7.0,
}
