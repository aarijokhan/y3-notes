---
type: concept
name: Gaussian distribution
description: Univariate and multivariate normal distributions, their MLE estimators, and why the Gaussian shows up everywhere
sources:
  - raw/week-07/Wk_7_Lec_2-1.pdf
status: draft
updated: 2026-04-27
---

*The bell-shaped distribution parameterised by a mean $\mu$ and variance $\sigma^2$ (or covariance $\boldsymbol{\Sigma}$ in higher dimensions). Ubiquitous in ML because the Central Limit Theorem promises that *any* sum of independent random variables tends towards Gaussian — making it a defensible default for noise. The MLE for its parameters is the sample mean and sample covariance.*

## Univariate Gaussian

The probability density function:

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)$$

with $\mathbb{E}[X] = \mu$ (location) and $\mathrm{Var}[X] = \sigma^2$ (spread). Sometimes the **precision** $\beta = 1/\sigma^2$ is used instead — convenient for derivations because it removes the inverse.

The "68–95–99.7 rule": about 68% of mass within $\pm \sigma$ of the mean, 95% within $\pm 2\sigma$, 99.7% within $\pm 3\sigma$.

## Multivariate Gaussian

For an $N$-dimensional vector $\mathbf{x}$:

$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{N/2} |\det(\boldsymbol{\Sigma})|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

- $\boldsymbol{\mu} \in \mathbb{R}^N$ is the mean vector.
- $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$ is the covariance matrix — symmetric, positive semidefinite, with diagonal entries $\sigma_{ii}^2 = \mathrm{Var}[X_i]$ and off-diagonal entries $\sigma_{ij} = \mathrm{Cov}[X_i, X_j]$.

The contour lines of constant density are ellipsoids; their axes are the eigenvectors of $\boldsymbol{\Sigma}$, scaled by $\sqrt{\lambda_i}$.

**Special cases:**
- $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$ — **isotropic**: equal variance in every direction, circular contours.
- $\boldsymbol{\Sigma}$ diagonal — components are uncorrelated; for Gaussians, this also implies independence.
- $\boldsymbol{\Sigma}$ general — correlated components; ellipsoidal contours.

## Independence vs Uncorrelatedness

For Gaussians (and only Gaussians among standard distributions), **uncorrelated $\Leftrightarrow$ independent**. A diagonal covariance matrix factorises the joint density into a product of univariate marginals, so the variables are independent.

This is a *Gaussian-specific* property. In general, uncorrelatedness only rules out *linear* dependence — non-Gaussian distributions can have zero correlation but strong non-linear dependence.

A non-trivial covariance matrix means a multivariate Gaussian **cannot** be written as a product of two univariate Gaussians. But you can always *diagonalise* (by linear transformation, via PCA-like rotation) to decorrelate the variables — at which point they become independent.

## Why Gaussians Are Everywhere

The **Central Limit Theorem** says: the sum (or average) of many independent random variables, regardless of their individual distributions, tends to a Gaussian as the number of variables grows. This is why Gaussians are the default noise model — measurement error, biological variation, market fluctuations are all approximately sums of many small independent contributions.

Consequence: even when you have no specific reason to assume Gaussian noise, it's often a reasonable first approximation, and it leads to clean math.

## MLE for the Gaussian

Given $N$ i.i.d. samples $\mathbf{x}_1, \ldots, \mathbf{x}_N$ from $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, the [[maximum-likelihood-estimation-ml|maximum-likelihood]] estimators are the **sample mean** and **sample covariance**:

$$\boxed{\hat{\boldsymbol{\mu}}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i, \qquad \hat{\boldsymbol{\Sigma}}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^\top}$$

> [!info]+ Derivation sketch
> The log-likelihood is:
> $$\ln p(\mathbf{X} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{N D}{2} \ln(2\pi) - \frac{N}{2} \ln |\boldsymbol{\Sigma}| - \frac{1}{2} \sum_{i=1}^N (\mathbf{x}_i - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})$$
> Setting $\nabla_{\boldsymbol{\mu}} = 0$ gives the sample mean directly. Setting $\nabla_{\boldsymbol{\Sigma}} = 0$ (using matrix-derivative identities) gives the sample covariance. See lecture notes for the full derivation.

> [!warning] MLE underestimates $\sigma^2$
> The MLE estimate $\hat{\sigma}^2 = \frac{1}{N} \sum (x_i - \hat{\mu})^2$ is **biased** — it systematically underestimates the true variance. The unbiased estimator divides by $N - 1$ instead of $N$ (Bessel's correction). For large $N$ the difference is negligible; for small $N$ it matters. This is one of the standard "MLE has known biases" examples.

## The Role in Linear Regression

Linear regression's [[ordinary-least-squares|OLS]] solution is justified by assuming the residuals are i.i.d. Gaussian:

$$y_i = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$$

Under this assumption, the conditional distribution of $y$ is itself Gaussian:

$$p(y \mid \mathbf{x}, \mathbf{w}, \sigma^2) = \mathcal{N}(y \mid \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}), \sigma^2)$$

and the MLE for $\mathbf{w}$ matches the OLS estimator exactly. Squared-error loss is the negative log-likelihood of a Gaussian noise model — that's the principled reason we use it.

## Active Recall

> [!question]- For a Gaussian, are uncorrelatedness and independence the same thing?
> Yes — for Gaussians specifically. A diagonal covariance matrix factorises the joint density into a product of univariate Gaussians, so uncorrelated implies independent. This is *not* true for general distributions, where uncorrelatedness only rules out linear dependence.

> [!question]- Why does the Central Limit Theorem matter for ML?
> It justifies the Gaussian as a default noise model: measurement errors and similar quantities are typically sums of many small independent contributions, which converge to Gaussian regardless of the individual contributions' distributions. This is why squared-error loss (the Gaussian-noise MLE objective) is so widely used.

> [!question]- Can a multivariate Gaussian with non-diagonal covariance always be written as a product of univariate Gaussians?
> No. A non-trivial covariance matrix couples the variables; the joint density doesn't factorise. However, you can always *diagonalise* the covariance matrix by linear transformation (rotating into the eigenbasis), making the rotated variables independent.

## Connections

- [[maximum-likelihood-estimation-ml|maximum likelihood estimation]] — the principle that picks $\hat{\boldsymbol{\mu}}$ and $\hat{\boldsymbol{\Sigma}}$ given data.
- [[ordinary-least-squares]] — the regression analog: under Gaussian noise, MLE for $\mathbf{w}$ equals OLS.
- [[gaussian-kernel]] — the same exp-of-squared-distance functional form, repurposed as a kernel.
- [[bayes-law]] — Gaussians are conjugate to themselves (Gaussian prior + Gaussian likelihood → Gaussian posterior), making them computationally tractable in Bayesian inference.
- [[cross-entropy-loss]] — the Bernoulli-noise analog of squared-error: same recipe (negative log-likelihood), different distribution.
