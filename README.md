# ptlasso

Python implementation of the **Pretrained Lasso** — a two-step procedure for fitting sparse linear models when samples belong to distinct groups, leveraging shared structure across groups via pretraining.

Based on:
> Craig, E., Pilanci, M., Le Menestrel, T., Narasimhan, B., Rivas, M. A., Gullaksen, S. E., ... & Tibshirani, R. (2025). Pretraining and the lasso. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, qkaf050.

---

## The idea

Standard group-specific Lasso models are fit independently per group, ignoring shared signal. The Pretrained Lasso fits in two steps:

**Step 1 — Overall model.** Fit a Lasso on all samples to capture shared structure:

$$\hat{\beta}^{\text{overall}} = \arg\min_\beta \frac{1}{2n}\|y - X\beta\|^2 + \lambda\|\beta\|_1$$

**Step 2 — Group models.** For each group $k$, fit a Lasso with an offset equal to $\alpha$ times the overall model's linear predictor:

$$\hat{\beta}^{(k)} = \arg\min_\beta \frac{1}{2n_k}\|y^{(k)} - \underbrace{\alpha \cdot X^{(k)}\hat{\beta}^{\text{overall}}}_{\text{offset}} - X^{(k)}\beta\|^2 + \lambda_k\|\beta\|_1$$

The parameter $\alpha \in [0, 1]$ controls the pretraining strength:
- $\alpha = 0$: pure group-specific models, no pretraining
- $\alpha = 1$: group models explain residuals from the overall model
- $\alpha \in (0, 1)$: group models are anchored to the overall fit

Final prediction for group $k$: $\hat{y}^{(k)} = \alpha \cdot X^{(k)}\hat{\beta}^{\text{overall}} + X^{(k)}\hat{\beta}^{(k)}$

Supports **gaussian**, **binomial**, and **multinomial** families.

---

## Installation

```bash
pip install ptlasso
```

Requires Python ≥ 3.9 and [adelie](https://github.com/JamesYang007/adelie) for the underlying Lasso solver, which supports fitting with offsets (unlike scikit-learn).

---

## Quick start

```python
import numpy as np
from ptlasso import PretrainedLasso, PretrainedLassoCV

rng = np.random.default_rng(42)
n, p, k = 300, 100, 3

X      = rng.standard_normal((n, p))
groups = rng.integers(0, k, size=n)
beta   = np.zeros(p)
beta[:5] = [2, -1.5, 1, -0.8, 0.5]
y      = X @ beta + 0.5 * rng.standard_normal(n)

# Fixed alpha
model = PretrainedLasso(alpha=0.5)
model.fit(X, y, groups)
print(model)
# PretrainedLasso(alpha=0.5, family='gaussian', overall_lambda='lambda.1se', ...)
#   family       : gaussian
#   n_features   : 100
#   n_groups     : 3
#   overall |Ŝ|  : |Ŝ| = 5 / 100  [0, 1, 2, 3, 4]
#   pretrain |Ŝ| : 0: |Ŝ|=5, 1: |Ŝ|=4, 2: |Ŝ|=6

y_pred = model.predict(X, groups)
print("R²:", model.score(X, y, groups))

# Cross-validate over alpha
cv = PretrainedLassoCV(alphas=[0.0, 0.25, 0.5, 0.75, 1.0])
cv.fit(X, y, groups)
print("Best alpha:", cv.alpha_)
```

---

## Families

```python
# Binary classification
model = PretrainedLasso(alpha=0.5, family="binomial")
model.fit(X, y_binary, groups)
probs = model.predict(X, groups)          # shape (n,), P(y=1)

# Multi-class classification (integer labels 0..K-1)
model = PretrainedLasso(alpha=0.5, family="multinomial")
model.fit(X, y_multiclass, groups)
probs = model.predict(X, groups)          # shape (n, K)
```

---

## Feature names and group labels

Both `fit()` methods accept human-readable names. pandas DataFrames are supported natively — column names are picked up automatically.

```python
import pandas as pd

X_df         = pd.DataFrame(X, columns=[f"gene_{i}" for i in range(p)])
group_labels = {0: "control", 1: "treated_A", 2: "treated_B"}

model = PretrainedLasso(alpha=0.5)
model.fit(X_df, y, groups, group_labels=group_labels)
# overall |Ŝ|  : |Ŝ| = 5 / 100  [gene_0, gene_1, gene_2, gene_3, gene_4]
# pretrain |Ŝ| : control: |Ŝ|=5, treated_A: |Ŝ|=4, treated_B: |Ŝ|=6
```

---

## Inspecting the support

```python
from ptlasso import (
    get_overall_support,
    get_pretrain_support,
    get_pretrain_support_split,
    get_individual_support,
)

get_overall_support(model)                      # features from the overall model
get_pretrain_support(model)                     # union across pretrained group models
get_pretrain_support(model, common_only=True)   # features selected by >50% of groups
get_pretrain_support(model, groups=[0, 1])      # restrict to specific groups
get_individual_support(model)                   # features from no-pretraining baselines

common, indiv = get_pretrain_support_split(model)
# common : features from the overall model (stage 1)
# indiv  : additional features picked up by group models (stage 2)
```

---

## Evaluating all sub-models at once

```python
result = model.evaluate(X_test, y_test, groups_test)
# {"pretrain":   {"predictions": ..., "score": ...},
#  "individual": {"predictions": ..., "score": ...},
#  "overall":    {"predictions": ..., "score": ...}}
```

---

## Retrieving coefficients

```python
coefs = model.get_coef()                   # all sub-models
coefs["overall"]                           # {"coef": ndarray, "intercept": ndarray}
coefs["pretrain"]["control"]               # {"coef": ndarray, "intercept": ndarray}
coefs["individual"]["treated_A"]

model.get_coef(model="pretrain")           # just pretrain sub-dict
```

---

## CV details

```python
cv = PretrainedLassoCV(
    alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
    cv=5,
    alphahat_choice="overall",   # or "mean" (unweighted mean of per-group CV errors)
    family="gaussian",
    overall_lambda="lambda.1se", # or "lambda.min"
    foldid=my_foldid,            # optional: custom integer fold assignments
)
cv.fit(X, y, groups)

cv.alpha_                        # globally best alpha
cv.varying_alphahat_             # {group: best_alpha} per group
cv.cv_results_                   # {alpha: mean CV loss}
cv.cv_results_se_                # {alpha: SE of CV loss}
cv.cv_results_per_group_         # {alpha: {group: mean CV loss}}
cv.cv_results_mean_              # {alpha: unweighted mean of per-group losses}
cv.cv_results_wtd_mean_          # {alpha: size-weighted mean of per-group losses}
cv.cv_results_individual_        # CV loss for individual (no-pretraining) baseline
cv.cv_results_overall_           # CV loss for overall model baseline
cv.best_estimator_               # PretrainedLasso fitted with alpha_
cv.all_estimators_               # {alpha: PretrainedLasso} for varying-alpha prediction

# Predict using each group's own best alpha
cv.predict(X, groups, alphatype="varying")
cv.evaluate(X, y, groups, alphatype="varying")
```

---

## Plotting

```python
from ptlasso import plot_cv, plot_paths

plot_cv(cv)           # CV loss curve over alpha with ±1 SE band
plot_paths(model)     # regularisation paths for all sub-models
```

---

## API reference

### `PretrainedLasso`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.5` | Pretraining strength $\in [0, 1]$ |
| `family` | `"gaussian"` | `"gaussian"`, `"binomial"`, or `"multinomial"` |
| `overall_lambda` | `"lambda.1se"` | Lambda rule for stage-1 offset: `"lambda.1se"` or `"lambda.min"` |
| `fit_intercept` | `True` | Fit an intercept in all sub-models |
| `lmda_path_size` | `100` | Number of $\lambda$ values in the regularisation path |
| `min_ratio` | `0.01` | Ratio of smallest to largest $\lambda$ |
| `verbose` | `False` | Show adelie progress bar |

**Methods:**
- `fit(X, y, groups, group_labels=None, feature_names=None)`
- `predict(X, groups, model="pretrain", lmda_idx=None)` — `model` ∈ `{"pretrain", "individual", "overall"}`
- `score(X, y, groups)` — R² or accuracy
- `evaluate(X, y, groups)` — predict + score for all three sub-models
- `get_coef(model="all", lmda_idx=None)`

### `PretrainedLassoCV`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alphas` | `[0, 0.25, 0.5, 0.75, 1.0]` | Candidate $\alpha$ values |
| `cv` | `5` | Number of CV folds |
| `alphahat_choice` | `"overall"` | `"overall"` or `"mean"` (unweighted per-group mean) |
| `family` | `"gaussian"` | Same as `PretrainedLasso` |
| `overall_lambda` | `"lambda.1se"` | Same as `PretrainedLasso` |
| `fit_intercept` | `True` | |
| `lmda_path_size` | `100` | |
| `min_ratio` | `0.01` | |
| `verbose` | `False` | |
| `foldid` | `None` | Integer array of fold assignments (overrides `cv`) |

Same `fit` / `predict` / `score` / `evaluate` / `get_coef` interface as `PretrainedLasso`, plus:

| Fitted attribute | Description |
|-----------------|-------------|
| `alpha_` | Best $\alpha$ selected by CV |
| `varying_alphahat_` | `{group: alpha}` — per-group best $\alpha$ |
| `cv_results_` | `{alpha: mean CV loss}` |
| `cv_results_se_` | `{alpha: SE of CV loss}` |
| `cv_results_per_group_` | `{alpha: {group: mean CV loss}}` |
| `cv_results_mean_` | `{alpha: unweighted mean of per-group losses}` |
| `cv_results_wtd_mean_` | `{alpha: size-weighted mean of per-group losses}` |
| `cv_results_individual_` | CV loss for individual baseline |
| `cv_results_overall_` | CV loss for overall baseline |
| `best_estimator_` | `PretrainedLasso` fitted with `alpha_` |
| `all_estimators_` | `{alpha: PretrainedLasso}` for each unique varying alpha |

`predict` also accepts `alphatype="varying"` to route each group through its own best alpha.

---

## Citation

```bibtex
@article{craig2025pretraining,
  title   = {Pretraining and the lasso},
  author  = {Craig, Erin and Pilanci, Mert and Le Menestrel, Thomas and Narasimhan, Balasubramanian and Rivas, Manuel A. and Gullaksen, Stein-Erik and Tibshirani, Robert},
  journal = {Journal of the Royal Statistical Society Series B: Statistical Methodology},
  pages   = {qkaf050},
  year    = {2025}
}
```

---

## License

MIT
