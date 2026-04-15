# ptlasso

Python implementation of the **Pretrained Lasso** — a two-step procedure for fitting sparse linear models when samples belong to distinct groups, leveraging shared structure across groups via pretraining.

Based on:
> *Pretraining and the Lasso*, Erin Craig & Rob Tibshirani, Journal of the Royal Statistical Society Series B: Statistical Methodology.

---

## The idea

Standard group-specific Lasso models are fit independently per group, ignoring shared signal. The Pretrained Lasso fits in two steps:

**Step 1 — Overall model.** Fit a Lasso on all samples to capture shared structure:

$$\hat{\beta}^{\text{overall}} = \arg\min_\beta \frac{1}{2n}\|y - X\beta\|^2 + \lambda\|\beta\|_1$$

**Step 2 — Group models.** For each group $k$, fit a Lasso with an offset equal to $\alpha$ times the overall model's predictions:

$$\hat{\beta}^{(k)} = \arg\min_\beta \frac{1}{2n_k}\|y^{(k)} - \underbrace{\alpha \cdot X^{(k)}\hat{\beta}^{\text{overall}}}_{\text{offset}} - X^{(k)}\beta\|^2 + \lambda_k\|\beta\|_1$$

The parameter $\alpha \in [0, 1]$ controls the pretraining strength:
- $\alpha = 0$: pure group-specific models, no pretraining
- $\alpha = 1$: group models explain residuals from the overall model
- $\alpha \in (0, 1)$: a blend — group models are anchored to the overall fit

Final prediction for group $k$: $\hat{y}^{(k)} = \alpha \cdot X^{(k)}\hat{\beta}^{\text{overall}} + X^{(k)}\hat{\beta}^{(k)}$

---

## Installation

```bash
pip install ptlasso
```

Requires Python ≥ 3.9 and [adelie](https://github.com/JamesYang007/adelie) for the underlying Lasso solver (supports fitting with offsets, which scikit-learn cannot do).

---

## Quick start

```python
import numpy as np
from ptlasso import PretrainedLasso, PretrainedLassoCV

rng = np.random.default_rng(42)
n, p, k = 300, 100, 3

X = rng.standard_normal((n, p))
groups = rng.integers(0, k, size=n)           # group label per sample
true_coef = np.zeros(p)
true_coef[:5] = [2, -1.5, 1, -0.8, 0.5]
y = X @ true_coef + 0.5 * rng.standard_normal(n)

# Fit with a fixed alpha
model = PretrainedLasso(alpha=0.5)
model.fit(X, y, groups)

print(model)
# PretrainedLasso(alpha=0.5, ...)
#   n_features   : 100
#   n_groups     : 3
#   overall nz   : 5 / 100  [0, 1, 2, 3, 4]
#   pretrain nz  : 0: 4, 1: 3, 2: 5

y_pred = model.predict(X, groups)
print("R²:", model.score(X, y, groups))

# Cross-validate over alpha
cv_model = PretrainedLassoCV(alphas=[0.0, 0.25, 0.5, 0.75, 1.0])
cv_model.fit(X, y, groups)
print("Best alpha:", cv_model.alpha_)
```

---

## Feature names and group labels

Both `fit()` methods accept human-readable names. pandas DataFrames are supported natively — column names are picked up automatically.

```python
import pandas as pd

feature_names = [f"gene_{i}" for i in range(p)]
X_df = pd.DataFrame(X, columns=feature_names)

group_labels = {0: "control", 1: "treated_A", 2: "treated_B"}

model = PretrainedLasso(alpha=0.5)
model.fit(X_df, y, groups, group_labels=group_labels)

print(model)
# PretrainedLasso(alpha=0.5, ...)
#   overall nz   : 5 / 100  [gene_0, gene_1, gene_2, gene_3, gene_4]
#   pretrain nz  : control: 4, treated_A: 3, treated_B: 5
```

---

## Inspecting the support

```python
from ptlasso import get_overall_support, get_pretrain_support, get_individual_support

# Features selected by the overall model
get_overall_support(model)
# array(['gene_0', 'gene_1', 'gene_2', 'gene_3', 'gene_4'], dtype='<U6')

# Union of features selected by pretrained group models
# (includes overall support when alpha < 1, matching R behaviour)
get_pretrain_support(model)

# Features selected by more than half the groups
get_pretrain_support(model, common_only=True)

# Restrict to specific groups
get_pretrain_support(model, groups=[0, 1])

# Features from the no-pretraining (alpha=0) baselines
get_individual_support(model)
```

---

## Retrieving coefficients

```python
# All coefficients as a nested dict
coefs = model.get_coef(model="all")
coefs["overall"]          # {"coef": ndarray, "intercept": float}
coefs["pretrain"]         # {"control": {...}, "treated_A": {...}, ...}
coefs["individual"]       # {"control": {...}, "treated_A": {...}, ...}

# Single sub-model
model.get_coef(model="pretrain")
model.get_coef(model="overall")
```

---

## API reference

### `PretrainedLasso`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.5` | Pretraining strength $\in [0, 1]$ |
| `fit_intercept` | `True` | Fit intercept in all sub-models |
| `lmda_path_size` | `100` | Number of $\lambda$ values in the regularization path |
| `min_ratio` | `0.01` | Ratio of smallest to largest $\lambda$ |

**`fit(X, y, groups, group_labels=None, feature_names=None)`**  
**`predict(X, groups, model="pretrain")`** — `model` ∈ `{"pretrain", "individual", "overall"}`  
**`score(X, y, groups)`** — returns $R^2$  
**`get_coef(model="all")`**

### `PretrainedLassoCV`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alphas` | `[0, 0.25, 0.5, 0.75, 1.0]` | Candidate $\alpha$ values |
| `cv` | `5` | Number of cross-validation folds |
| `fit_intercept` | `True` | |
| `lmda_path_size` | `100` | |
| `min_ratio` | `0.01` | |

Same `fit` / `predict` / `score` / `get_coef` interface as `PretrainedLasso`, plus:

| Fitted attribute | Description |
|-----------------|-------------|
| `alpha_` | Best $\alpha$ selected by CV |
| `cv_results_` | Mean MSE per candidate $\alpha$ |
| `best_estimator_` | `PretrainedLasso` fitted with `alpha_` |

---

## Citation

```bibtex
@article{craig2024pretraining,
  title   = {Pretraining and the Lasso},
  author  = {Craig, Erin and Tibshirani, Robert},
  journal = {Journal of the Royal Statistical Society Series B: Statistical Methodology},
  year    = {2024}
}
```

---

## License

MIT
