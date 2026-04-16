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

Requires Python ≥ 3.9 and [adelie](https://github.com/JamesYang007/adelie) for the underlying Lasso solver.

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

```python
import pandas as pd

X_df         = pd.DataFrame(X, columns=[f"gene_{i}" for i in range(p)])
group_labels = {0: "control", 1: "treated_A", 2: "treated_B"}

model = PretrainedLasso(alpha=0.5)
model.fit(X_df, y, groups, group_labels=group_labels)
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

model.get_coef(model="pretrain")           # just pretrain sub-dict
```

---

## Plotting

```python
from ptlasso import plot_cv, plot_paths

plot_cv(cv)           # CV loss curve over alpha with ±1 SE band
plot_paths(model)     # regularisation paths for all sub-models
```

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
