"""Package-wide constants for ptlasso."""

# ------------------------------------------------------------------
# Model families and lambda selection modes
# ------------------------------------------------------------------

FAMILIES = ("gaussian", "binomial", "multinomial")

LMDA_MODES = ("lambda.1se", "lambda.min")

PREDICT_MODELS = ("pretrain", "individual", "overall")

COEF_MODELS = ("all", "overall", "pretrain", "individual")

ALPHATYPES = ("best", "varying")

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
