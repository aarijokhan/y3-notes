---
type: concept
name: ordinary least squares (OLS)
description: The closed-form fitting criterion for linear regression — minimise the sum of squared residuals, yielding $w = (\Phi^\top \Phi)^{-1} \Phi^\top y$ via the normal equation
sources:
  - raw/week-07/Wk_7_Lec_1-1.pdf
  - raw/week-07/Wk_7_Lec_2-1.pdf
status: draft
updated: 2026-04-27
---

*The fitting criterion for [[linear-regression|linear regression]]: choose the weights $\mathbf{w}$ that minimise the sum of squared residuals between predictions and observed targets. Yields a closed-form solution via the **normal equation** $\mathbf{w}^* = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}$, computable in one matrix inversion — no iterations.*

## The Objective

Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ and a linear model $\hat{y}_i = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i)$, define the **residual** for example $i$ as:

$$r_i = y_i - \hat{y}_i = y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i)$$

OLS picks the $\mathbf{w}$ that minimises the sum of squared residuals:

$$\boxed{\mathbf{w}_{\text{OLS}} = \arg\min_{\mathbf{w}} R(\mathbf{w}), \qquad R(\mathbf{w}) = \sum_{i=1}^N (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2}$$

The objective is convex and quadratic in $\mathbf{w}$, so its unique minimum is found by setting $\nabla R = 0$.

## Deriving the Normal Equation

Stack the basis-function evaluations into the [[design-matrix|design matrix]] $\boldsymbol{\Phi}$ (an $N \times (M+1)$ matrix whose row $i$ is $\boldsymbol{\phi}(\mathbf{x}_i)^\top$). Then:

$$R(\mathbf{w}) = \|\mathbf{y} - \boldsymbol{\Phi} \mathbf{w}\|^2$$

Differentiate w.r.t. $\mathbf{w}$ and set to zero:

$$\nabla_{\mathbf{w}} R = -2 \boldsymbol{\Phi}^\top (\mathbf{y} - \boldsymbol{\Phi} \mathbf{w}) = 0$$

Rearranging gives the **normal equation**:

$$\boldsymbol{\Phi}^\top \boldsymbol{\Phi} \, \mathbf{w} = \boldsymbol{\Phi}^\top \mathbf{y}$$

Solving (assuming $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ is invertible):

$$\boxed{\mathbf{w}_{\text{OLS}} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}}$$

The matrix $\boldsymbol{\Phi}^{\dagger} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top$ is the **Moore–Penrose pseudoinverse** of $\boldsymbol{\Phi}$. So OLS is just $\mathbf{w}^* = \boldsymbol{\Phi}^{\dagger} \mathbf{y}$.

## Worked Derivation (Degree-1 Polynomial)

Take $\hat{y} = w_0 + w_1 x$. The objective is:

$$R(w_0, w_1) = \sum_i (y_i - w_0 - w_1 x_i)^2$$

Setting $\partial R / \partial w_0 = 0$ and $\partial R / \partial w_1 = 0$ gives a $2 \times 2$ linear system:

$$\sum_i y_i = w_0 N + w_1 \sum_i x_i$$
$$\sum_i x_i y_i = w_0 \sum_i x_i + w_1 \sum_i x_i^2$$

In matrix form:

$$\begin{pmatrix} \sum_i y_i \\ \sum_i x_i y_i \end{pmatrix} = \begin{pmatrix} N & \sum_i x_i \\ \sum_i x_i & \sum_i x_i^2 \end{pmatrix} \begin{pmatrix} w_0 \\ w_1 \end{pmatrix}$$

Inverting the $2 \times 2$ matrix gives the closed-form solution. The general matrix-form derivation uses exactly the same logic, just with $\boldsymbol{\Phi}$ in place of the $[\mathbf{1}, \mathbf{x}]$ matrix.

## MLE Equivalence — The Probabilistic Justification

OLS picks "minimise squared residuals" without saying *why* squared, as opposed to absolute or any other power. The justification comes from a probabilistic model: assume

$$y_i = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$$

Then $y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i), \sigma^2)$. The likelihood of the dataset is:

$$\mathcal{L}(\mathbf{w}, \sigma^2) = \prod_{i=1}^N \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2}{2 \sigma^2}\right)$$

Taking the log:

$$\ln \mathcal{L} = -\frac{N}{2} \ln(2\pi \sigma^2) - \frac{1}{2 \sigma^2} \sum_{i=1}^N (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2$$

The only $\mathbf{w}$-dependent term is the sum of squared residuals — and it appears with a *negative* coefficient. So:

$$\arg\max_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = \arg\min_{\mathbf{w}} \sum_i (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 = \mathbf{w}_{\text{OLS}}$$

> [!info]+ MLE = OLS under additive Gaussian noise
> Under the assumption of i.i.d. Gaussian noise, the maximum-likelihood weights are *exactly* the OLS weights. This is the missing justification for "why squared error" — squared error isn't arbitrary, it's the negative log-likelihood of a Gaussian noise model (up to constants).

The MLE for $\sigma^2$ falls out of the same derivation: $\hat{\sigma}^2_{\text{MLE}} = \text{RSS} / N$, where $\text{RSS} = \|\mathbf{y} - \boldsymbol{\Phi} \mathbf{w}_{\text{OLS}}\|^2$.

> [!info]+ OLS makes no distributional assumption — but its justification does
> The OLS objective itself is just "minimise squared residuals" — no statistics required. The probabilistic justification (Gaussian noise → squared loss is optimal) is layered on top. So you can use OLS without believing in Gaussian noise; you just lose the optimality argument.

## What Could Go Wrong

- **Ill-conditioned $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$.** If columns of $\boldsymbol{\Phi}$ are near-collinear (e.g., redundant features, polynomial basis evaluated on a narrow range), the matrix is nearly singular. Inverting it is numerically unstable; small data changes swing $\mathbf{w}$ wildly. Use ridge regression or pseudoinverse via SVD.
- **$M > N$.** When the number of basis functions exceeds the number of examples, $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ is rank-deficient and not invertible. Either reduce $M$, regularise, or use a method that doesn't require inversion.
- **Outliers.** Squared loss heavily penalises large residuals. A single mislabelled point can shift the entire fit. Robust alternatives: MAE, Huber loss, RANSAC.
- **Wrong noise model.** If the true noise is heteroscedastic (variance depends on $\mathbf{x}$) or heavy-tailed, the MLE justification breaks down. OLS still minimises squared residuals, but is no longer the optimal estimator.

## Computational Cost

- Forming $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$: $O(N M^2)$.
- Inverting (or solving via Cholesky): $O(M^3)$.
- Total: $O(N M^2 + M^3)$.

For modest $M$ (≤ a few thousand), OLS is extremely fast — one shot, no learning rate. For large $M$ or $N$, [[gradient-descent-ml|gradient descent]] or stochastic methods scale better, at the cost of iteration.

## When OLS Beats Iterative Methods

- $M$ is small enough that $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ fits in memory and inverts cheaply.
- The data is well-conditioned (no severe collinearity).
- You want exact, deterministic weights without learning-rate tuning.

When *not* to use OLS:

- $M$ is huge (e.g., kernel methods with $N = M$) — the matrix doesn't fit.
- The design matrix is ill-conditioned — use ridge regression or SVD-based pseudoinverse.
- You need online updates — gradient descent on incoming data is more natural.

## Connections

- [[linear-regression]] — the model OLS fits.
- [[design-matrix]] — the matrix $\boldsymbol{\Phi}$ that appears in the normal equation.
- [[maximum-likelihood-estimation-ml|maximum likelihood estimation]] — the principle that OLS implements under a Gaussian noise assumption.
- [[gaussian-distribution]] — the assumed noise distribution that makes MLE = OLS.
- [[gradient-descent-ml|gradient descent]] — iterative alternative when the closed form is impractical.
- [[hessian-matrix]] — for quadratic objectives, $H = 2 \boldsymbol{\Phi}^\top \boldsymbol{\Phi}$, so [[newton-raphson-method|Newton's method]] converges in *one step* (and that step is exactly the normal equation).

## Active Recall

> [!question]- Write the OLS solution in a single equation. What does $\boldsymbol{\Phi}^\dagger$ stand for?
> $\mathbf{w}_{\text{OLS}} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y} = \boldsymbol{\Phi}^\dagger \mathbf{y}$, where $\boldsymbol{\Phi}^\dagger$ is the **Moore–Penrose pseudoinverse** of $\boldsymbol{\Phi}$.

> [!question]- Why is "minimise squared error" a reasonable choice of objective?
> Under the assumption that observations are corrupted by additive i.i.d. Gaussian noise, the MLE for the regression weights is *exactly* the squared-error minimiser. So squared loss is the negative log-likelihood (up to constants) of a Gaussian noise model — minimising it is principled, not arbitrary.

> [!question]- True or false: OLS requires an assumption about the noise distribution.
> False. OLS is purely an optimisation criterion ("minimise sum of squared residuals") — no distributional assumption. The *MLE-equivalence justification* requires Gaussian noise, but you can still apply OLS without believing it.

> [!question]- When does the normal equation fail?
> When $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ is singular or ill-conditioned. This happens with collinear features, redundant basis functions, or when there are more features than examples ($M > N$). Mitigations: drop redundant features, regularise (ridge), or use the SVD-based pseudoinverse.
