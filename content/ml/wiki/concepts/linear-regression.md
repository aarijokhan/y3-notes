---
type: concept
name: linear regression
description: A regression model that predicts a continuous output as a linear combination of (possibly transformed) input features, fit via squared-error minimisation
sources:
  - raw/week-07/Wk_7_Lec_1-1.pdf
  - raw/week-07/Wk_7_Lec_2-1.pdf
  - raw/week-07/ML_Exercise_Sheet_7a_solution.pdf
  - raw/week-07/ML_Exercise_Sheet_7b_solution.pdf
status: draft
updated: 2026-04-27
---

*A regression model that assumes the target $y$ is a linear function of (possibly transformed) input features plus noise: $y = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}) + \varepsilon$. The standard fit is [[ordinary-least-squares|ordinary least squares]] — minimise the sum of squared residuals — which has a closed-form solution via the normal equation.*

## The Model

Linear regression posits that the target $y$ depends linearly on a fixed set of basis functions of the input:

$$\hat{y}(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M} w_j \phi_j(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x})$$

where:

- $\boldsymbol{\phi}(\mathbf{x}) = (\phi_0(\mathbf{x}), \phi_1(\mathbf{x}), \ldots, \phi_M(\mathbf{x}))^\top$ is a vector of **basis functions**, with $\phi_0(\mathbf{x}) = 1$ as a dummy for the intercept.
- $\mathbf{w} = (w_0, w_1, \ldots, w_M)^\top$ is the weight vector to be learned.
- $\hat{y}$ is the predicted value; the actual observed $y$ is assumed to differ by some noise $\varepsilon$.

The simplest case ($\phi_j(\mathbf{x}) = x_j$) gives plain $\hat{y} = w_0 + w_1 x_1 + \cdots + w_d x_d$ — a hyperplane in input space.

## Why It's Called "Linear"

The name refers to **linearity in the parameters $\mathbf{w}$**, not linearity in $\mathbf{x}$. The following are all linear regression models, even though they bend dramatically in input space:

$$y = w_0 + w_1 x_1^2 + w_2 x_2^2 + \cdots + w_D x_D^2$$
$$y = w_0 + w_1 e^{x_1} + w_2 e^{x_2}$$
$$y = w_0 + w_1 \exp\left(-\tfrac{(x - \mu_1)^2}{2 s^2}\right) + w_2 \exp\left(-\tfrac{(x - \mu_2)^2}{2 s^2}\right)$$

What makes them "linear" is that $\partial \hat{y} / \partial w_j$ is a function of $\mathbf{x}$ alone — not of $\mathbf{w}$. The prediction is a *linear combination* of pre-computed basis values; the unknowns $\mathbf{w}$ enter linearly.

This matters because [[ordinary-least-squares|OLS]] only relies on linearity in $\mathbf{w}$ to give a closed-form solution. The same recipe handles polynomial regression, RBF regression, and arbitrary fixed-basis models.

## Common Basis Functions

| Basis | Form | When |
|---|---|---|
| Polynomial | $\phi_j(x) = x^j$ | Smooth low-degree relationships |
| Gaussian / RBF | $\phi_j(x) = \exp(-\tfrac{(x - \mu_j)^2}{2 s^2})$ | Local bumps; bell-shaped responses |
| Sigmoidal | $\phi_j(x) = (1 + e^{-(x - \mu_j)/s})^{-1}$ | Smooth saturation effects |
| tanh | $\phi_j(x) = \tanh((x - \mu_j)/s)$ | Symmetric saturation |

For multi-input data, basis functions are typically applied per dimension or to selected combinations.

## Fitting the Model

Two equivalent recipes for finding $\mathbf{w}$:

1. **[[ordinary-least-squares|Ordinary Least Squares (OLS)]]** — minimise the sum of squared residuals. Yields the **normal equation**:
   $$\mathbf{w}_{\text{OLS}} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}$$
   where $\boldsymbol{\Phi}$ is the [[design-matrix|design matrix]].

2. **Maximum Likelihood Estimation under Gaussian noise** — assume $y = \hat{y}(\mathbf{x}, \mathbf{w}) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. The [[maximum-likelihood-estimation-ml|MLE]] for $\mathbf{w}$ is **identical** to the OLS solution. See [[ordinary-least-squares#MLE Equivalence|OLS § MLE equivalence]].

The probabilistic view (recipe 2) supplies the missing justification for *why* we minimise squared error: it's the optimal estimator under additive Gaussian noise.

## Predictions and Evaluation

Once $\mathbf{w}$ is fit, predict $\hat{y}(\mathbf{x}_{\text{new}}) = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_{\text{new}})$.

Common evaluation metrics:

| Metric | Formula | Notes |
|---|---|---|
| MSE | $\tfrac{1}{N} \sum (y_i - \hat{y}_i)^2$ | What OLS minimises (up to scale) |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as $y$; interpretable |
| MAE | $\tfrac{1}{N} \sum |y_i - \hat{y}_i|$ | Robust to outliers |
| $R^2$ | $1 - \tfrac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Fraction of variance explained; 1 = perfect |

## Choosing the Polynomial Degree

A higher-degree polynomial can fit any training set arbitrarily well (with $M = N$, you can interpolate every point exactly). But this overfits — the curve thrashes wildly between training points and fails on test data.

This is where **regularisation** and **validation** enter the picture (later weeks). The intuition: pick the degree that fits the data well *without* contorting itself into the noise.

## Strengths and Limitations

**Strengths:**

- **Closed-form solution** via the normal equation. No iterations, no learning rate.
- **Convex objective** — unique global optimum.
- **Interpretable coefficients** — each $w_j$ is the (partial) effect of feature $j$ on the output.
- **Probabilistic backing** — MLE under Gaussian noise gives the same answer.
- **Extends to non-linear shapes** via basis expansion without leaving the linear-regression machinery.

**Limitations:**

- **Sensitive to outliers** — squared loss penalises large residuals quadratically; one extreme point can dominate the fit. Use MAE or robust regression if outliers are an issue.
- **Numerical issues** — $(\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1}$ becomes ill-conditioned when columns of $\boldsymbol{\Phi}$ are near-collinear, or when $M > N$. Then OLS fails outright; gradient descent or regularisation is needed.
- **Overfitting with too many basis functions** — high-degree polynomials or many RBFs without regularisation will memorise noise.
- **Linear-in-parameters is still a constraint** — if the true relationship is genuinely non-linear-in-parameters (e.g., $y = e^{w_1 x}$), linear regression with any basis can only approximate, never recover it.

## Connections

- [[ordinary-least-squares]] — the criterion and closed-form solution.
- [[design-matrix]] — the $N \times (M+1)$ matrix whose rows are basis-function evaluations of each training input.
- [[non-linear-transformation]] — the basis-expansion trick, shared with SVMs.
- [[gaussian-distribution]] — the noise model that justifies squared loss.
- [[maximum-likelihood-estimation-ml|maximum likelihood estimation]] — the principle that links OLS to a probabilistic model.
- [[gradient-descent-ml|gradient descent]] — alternative fitting method when $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ is too large/singular.
- [[logistic-regression]] — the classification analogue: linear in parameters, but with sigmoid + Bernoulli noise instead of identity + Gaussian noise.

## Active Recall

> [!question]- Why is the model $y = w_0 + w_1 x^2$ called "linear regression" even though it traces a parabola?
> Linearity refers to the parameters $\mathbf{w}$, not the input $x$. We can write it as $y = \mathbf{w}^\top \boldsymbol{\phi}(x)$ with $\boldsymbol{\phi}(x) = (1, x^2)^\top$. The derivative $\partial y / \partial w_j$ is a function of $x$ alone, not of $\mathbf{w}$ — that's the linear-in-parameters property. OLS works because of *this* linearity, regardless of what $\boldsymbol{\phi}$ does in input space.

> [!question]- A dataset has 14 examples and 3 features (plus intercept). What are the dimensions of $\boldsymbol{\Phi}$, $\mathbf{y}$, and $\mathbf{w}$ in the normal equation?
> $\boldsymbol{\Phi}$ is $14 \times 4$ (one row per example, one column per parameter including the intercept). $\mathbf{y}$ is $14 \times 1$. $\mathbf{w}$ is $4 \times 1$.

> [!question]- Given training data $(x, y)$: $(1, 0.5), (0, 0), (6, 3), (4, 2)$. Linear regression $y = w_0 + w_1 x$ fits perfectly. What are $w_0$ and $w_1$?
> The relationship is $y = 0.5 x$, so $w_0 = 0$ and $w_1 = 0.5$. (Verify: $0.5 \cdot 1 = 0.5$, $0.5 \cdot 0 = 0$, $0.5 \cdot 6 = 3$, $0.5 \cdot 4 = 2$. ✓)
