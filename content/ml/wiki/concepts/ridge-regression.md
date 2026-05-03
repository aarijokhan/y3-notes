---
type: concept
name: ridge regression
description: Linear regression with an L2 penalty on the weights — the MAP estimate of Bayesian linear regression under a Gaussian prior, and a fix for ill-conditioned $\Phi^\top \Phi$
sources:
  - raw/week-08/Wk_8_Lec_1-1.pdf
status: draft
updated: 2026-04-28
---

*Linear regression with an additional L2 penalty on the weight magnitudes. The objective $\sum (y_i - \hat{y}_i)^2 + \lambda \|\mathbf{w}\|^2$ has a closed-form solution that's just like OLS but with $\lambda \mathbf{I}$ added to $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$. From the Bayesian view, ridge regression is **MAP** [[bayesian-linear-regression|Bayesian linear regression]] under a zero-mean Gaussian prior on $\mathbf{w}$. The L2 term is the negative log-prior — regularisation has a probabilistic interpretation.*

## The Objective

$$\boxed{\mathbf{w}_{\text{ridge}} = \arg\min_{\mathbf{w}} \;\; \sum_{i=1}^N (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 + \lambda \|\mathbf{w}\|^2}$$

The first term is the [[ordinary-least-squares|OLS]] residual sum of squares. The second is the **L2 regulariser** with coefficient $\lambda \geq 0$.

- $\lambda = 0$: pure OLS.
- Small $\lambda$: light regularisation — weights are pulled mildly towards zero.
- Large $\lambda$: heavy regularisation — weights forced small, fit allowed to drift.
- $\lambda \to \infty$: $\mathbf{w} \to \mathbf{0}$, ignoring the data.

## Closed-Form Solution

Setting $\nabla = 0$ on the objective:

$$2 \boldsymbol{\Phi}^\top (\boldsymbol{\Phi} \mathbf{w} - \mathbf{y}) + 2 \lambda \mathbf{w} = 0$$

Rearranging:

$$\boxed{\mathbf{w}_{\text{ridge}} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \mathbf{I})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}}$$

Compare to the OLS normal equation: $\mathbf{w}_{\text{OLS}} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}$. The only change is adding $\lambda \mathbf{I}$ before inverting.

## Why It Fixes OLS Pathologies

OLS fails when $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ is singular (when $M > N$, or when columns are collinear). Adding $\lambda \mathbf{I}$ shifts every eigenvalue up by $\lambda$, making the matrix invertible:

- **Rank-deficient $\boldsymbol{\Phi}$ ($M > N$).** OLS has no unique solution. Ridge has a unique one for any $\lambda > 0$.
- **Collinear features.** $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ is nearly singular; OLS weights swing wildly with small data changes. Ridge stabilises by trading bias for variance.
- **Numerical conditioning.** Even when $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ is technically invertible, ill-conditioning amplifies floating-point errors. Ridge improves the condition number.

## The Bayesian View — Where Does $\lambda$ Come From?

[[bayesian-linear-regression|Bayesian linear regression]] places a Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1} \mathbf{I})$ and a Gaussian likelihood with noise precision $\beta$. The log-posterior is:

$$\ln p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) = -\frac{\beta}{2} \sum_i (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 - \frac{\alpha}{2} \|\mathbf{w}\|^2 + \text{const}$$

Maximising (the **MAP estimate**) is the same as minimising:

$$\frac{1}{2} \sum_i (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 + \frac{\alpha}{2 \beta} \|\mathbf{w}\|^2$$

This is *exactly* ridge regression with $\lambda = \alpha / \beta$.

> [!info]+ Regularisation is a prior in disguise
> L2 regularisation is the negative log of a zero-mean Gaussian prior on $\mathbf{w}$. The "regularisation strength" $\lambda$ is the ratio of prior precision to noise precision: high prior precision ($\alpha$ large) or low noise precision ($\beta$ small) → strong regularisation. The Bayesian view turns "we should penalise large weights" from a heuristic into a statement about prior belief.

Other regularisers correspond to other priors:

| Regulariser | Prior | Effect |
|---|---|---|
| $\lambda \|\mathbf{w}\|^2$ (L2) | Gaussian | Shrinks all weights smoothly toward 0 |
| $\lambda \|\mathbf{w}\|_1$ (L1, lasso) | Laplace | Drives some weights to *exactly* 0 (sparsity) |
| $\lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|^2$ (elastic net) | Mixture | Sparsity with stability |

## Effect on Generalisation

Ridge trades **bias** for **variance**:

- **Bias up.** Shrinking weights toward zero biases the fit — even the optimal $\mathbf{w}$ under regularisation isn't quite the true weight vector.
- **Variance down.** The fit is less sensitive to noise in the training data — small data changes produce small weight changes.

For high-capacity models (high-degree polynomials, many basis functions), this is usually a net win on test performance: the variance reduction outweighs the bias increase.

A canonical illustration: degree-8 polynomial on 10 noisy points.

- $\lambda = 0$ (OLS): the fit interpolates every training point but oscillates wildly between them. Tiny training error, large test error.
- $\lambda > 0$ (ridge): the fit is smoother. Slightly larger training error, much smaller test error.

## Choosing $\lambda$

Cross-validation. For a grid of candidate $\lambda$'s:
1. Split training data into $K$ folds.
2. For each fold, train on the others and validate on the held-out fold.
3. Pick the $\lambda$ that minimises mean validation error.

The Bayesian view suggests an alternative: **empirical Bayes**, where $\alpha$ (and thus $\lambda$) is chosen by maximising the *evidence* $p(\mathbf{y} \mid \mathbf{X})$. This avoids cross-validation but requires more setup.

## Properties

- **Convex.** The objective is strictly convex (assuming $\lambda > 0$) → unique global optimum.
- **Closed-form.** No iteration required (for moderate $M$).
- **Stabilises against multicollinearity.** Even severely correlated features give well-defined weights.
- **Doesn't produce sparse solutions.** Weights shrink toward zero but rarely hit exactly zero. Use lasso (L1) if sparsity is desired.

## What Could Go Wrong

- **Standardisation matters.** L2 penalises raw weight magnitudes, so feature scaling affects the result. Standardise inputs before fitting.
- **The intercept $w_0$ is usually not penalised.** Penalising it would shrink predictions toward zero rather than toward $\bar{y}$. Most implementations split it out.
- **Wrong $\lambda$.** Too small → no regularisation, OLS pathologies return. Too large → excessive shrinkage, model underfits.

## Connections

- [[ordinary-least-squares]] — recovered as $\lambda \to 0$.
- [[bayesian-linear-regression]] — ridge is the MAP estimate; the prior is the L2 penalty.
- [[bayes-law]] — the rule that derives ridge from a Gaussian prior.
- [[linear-regression]] — the underlying model.
- [[generalization-bound]] — regularisation is one of the levers that controls model complexity (and thus the bound).

## Active Recall

> [!question]- Why does adding $\lambda \mathbf{I}$ to $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ "fix" ill-conditioned matrices?
> Because every eigenvalue of $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ shifts up by $\lambda$. A near-zero eigenvalue (the source of ill-conditioning) becomes $\lambda$, which is well away from zero. The condition number — ratio of largest to smallest eigenvalue — improves accordingly. With $\lambda > 0$, the inverse always exists; with $\lambda$ large, the inverse is well-conditioned.

> [!question]- What's the Bayesian interpretation of the L2 penalty?
> It's the negative log-density of a zero-mean isotropic Gaussian prior on $\mathbf{w}$, scaled by the prior precision $\alpha$. Maximising the posterior (MAP) is the same as minimising "negative log-likelihood + negative log-prior" — and "negative log-prior" of a Gaussian is exactly $\frac{\alpha}{2} \|\mathbf{w}\|^2$.

> [!question]- A high-degree polynomial fits the training data perfectly but generalises poorly. How does ridge regression help?
> It trades bias for variance. The L2 penalty discourages the polynomial coefficients from taking the extreme values needed to interpolate every training point. The fit becomes smoother — slightly worse on training data, much better on test data. Mathematically, ridge shrinks the eigenvector components of $\mathbf{w}$ along directions of small data variance (where OLS would overfit) more than along directions of large data variance.
