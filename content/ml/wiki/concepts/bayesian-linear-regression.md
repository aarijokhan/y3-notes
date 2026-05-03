---
type: concept
name: Bayesian linear regression
description: Linear regression with a prior over the weights, yielding a Gaussian posterior whose mean is the MAP estimate (equivalent to ridge regression) and whose spread quantifies model uncertainty
sources:
  - raw/week-08/Wk_8_Lec_1-1.pdf
  - raw/week-08/ML_Exercise_Sheet_8a_solution.pdf
status: draft
updated: 2026-04-28
---

*Linear regression treated probabilistically: place a Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1} \mathbf{I})$ on the weights, combine with the Gaussian likelihood, and recover a closed-form Gaussian **posterior** $p(\mathbf{w} \mid \mathbf{y}, \mathbf{X})$. The posterior mean is the MAP estimate (equivalent to [[ridge-regression|ridge regression]]); the posterior covariance quantifies how confident we are in those weights. Predictions become a distribution rather than a point.*

## Motivation — The Underdetermined Case

[[ordinary-least-squares|OLS]] works when there are more observations than unknowns. With one input/output pair $(x_1, y_1)$ and two unknowns $(w_0, w_1)$, the equation $y_1 = w_0 + w_1 x_1$ has *infinite* solutions — every line through that single point fits perfectly. OLS doesn't help; the system is underdetermined.

The Bayesian fix: declare an a-priori belief about which $\mathbf{w}$'s are plausible. A natural choice is "small weights are more likely than large ones" — formalised as a Gaussian prior centred at zero. Now the posterior over $\mathbf{w}$ is a *distribution*, peaked where prior + data agree.

## The Bayesian Setup

**Likelihood** (same as standard linear regression):

$$p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, \beta) = \prod_{i=1}^N \mathcal{N}(y_i \mid \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i), \beta^{-1})$$

where $\beta = 1/\sigma^2$ is the noise precision (assumed known here).

**Prior** — Gaussian, centred at zero, with isotropic precision $\alpha$:

$$p(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{0}, \alpha^{-1} \mathbf{I})$$

Larger $\alpha$ → tighter prior → strong preference for small weights. Smaller $\alpha$ → weaker prior → behaviour closer to MLE/OLS.

**Posterior** by [[bayes-law|Bayes' law]]:

$$p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) \cdot p(\mathbf{w})$$

## Conjugacy — Why the Math Closes

A Gaussian likelihood combined with a Gaussian prior gives a **Gaussian posterior**. This is the canonical example of a **conjugate prior** — when the posterior stays in the same distributional family as the prior, the math closes in finite form. No integrals to estimate, no MCMC required.

Taking the log of $p(\mathbf{w} \mid \mathbf{y}, \mathbf{X})$:

$$\ln p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) = -\frac{\beta}{2} \sum_{i=1}^N (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 - \frac{\alpha}{2} \mathbf{w}^\top \mathbf{w} + \text{const}$$

This is quadratic in $\mathbf{w}$, so completing the square gives a Gaussian posterior:

$$\boxed{p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) = \mathcal{N}(\mathbf{w} \mid \mathbf{m}_{\text{post}}, \mathbf{S}_{\text{post}})}$$

with:

$$\mathbf{m}_{\text{post}} = \beta \mathbf{S}_{\text{post}} \boldsymbol{\Phi}^\top \mathbf{y}, \qquad \mathbf{S}_{\text{post}}^{-1} = \alpha \mathbf{I} + \beta \boldsymbol{\Phi}^\top \boldsymbol{\Phi}$$

where $\boldsymbol{\Phi}$ is the [[design-matrix|design matrix]].

## What the Posterior Tells You

- **The mean $\mathbf{m}_{\text{post}}$ is the MAP estimate** — the most probable single $\mathbf{w}$ under the posterior. It coincides with the [[ridge-regression|ridge regression]] solution: it's OLS with the regularisation term $\frac{\alpha}{\beta} \|\mathbf{w}\|^2$ added to the loss.
- **The covariance $\mathbf{S}_{\text{post}}$ quantifies uncertainty.** A tight posterior means the data has narrowed down $\mathbf{w}$ confidently; a wide posterior means many $\mathbf{w}$'s are still plausible.
- **As $N$ grows, the posterior tightens.** With $N=1$, the posterior is barely narrower than the prior. By $N=1000$, it's a tiny ellipse around the true weights.

## Predictive Distribution

A non-Bayesian model gives a single prediction $\hat{y}$ for a new $\mathbf{x}$. A Bayesian model gives a *distribution* over predictions, integrating over all plausible $\mathbf{w}$'s:

$$p(y \mid \mathbf{x}, \mathbf{y}, \mathbf{X}) = \int p(y \mid \mathbf{x}, \mathbf{w}) \, p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) \, d\mathbf{w}$$

For Gaussian likelihood + Gaussian posterior, this integral is closed-form Gaussian:

$$p(y \mid \mathbf{x}, \mathbf{y}, \mathbf{X}) = \mathcal{N}(y \mid \mathbf{m}_{\text{post}}^\top \boldsymbol{\phi}(\mathbf{x}), \;\; \beta^{-1} + \boldsymbol{\phi}(\mathbf{x})^\top \mathbf{S}_{\text{post}} \boldsymbol{\phi}(\mathbf{x}))$$

The variance has two contributions: $\beta^{-1}$ (irreducible noise) and $\boldsymbol{\phi}^\top \mathbf{S}_{\text{post}} \boldsymbol{\phi}$ (model uncertainty, vanishing as data accumulates).

> [!info]+ Why this matters in practice
> Bayesian regression doesn't just give "$\hat{y} = 89.3$" — it gives "$y \sim \mathcal{N}(89.3, 0.04)$." For decision-making (medical risk, financial pricing), the variance is as important as the mean. A confident wrong answer is worse than a hedged correct one.

## MAP and Ridge Regression

Maximising the log-posterior with respect to $\mathbf{w}$:

$$\arg\max_{\mathbf{w}} \ln p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) = \arg\min_{\mathbf{w}} \left\{ \frac{1}{2} \sum_i (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 + \frac{\alpha}{2 \beta} \mathbf{w}^\top \mathbf{w} \right\}$$

The first term is the OLS objective. The second is an L2 penalty on $\mathbf{w}$, with regularisation coefficient $\lambda = \alpha/\beta$. **This is exactly [[ridge-regression|ridge regression]]**.

The Bayesian view explains *why* L2 regularisation works: it's the negative log-prior of a zero-mean Gaussian belief. Larger $\alpha$ (tighter prior) → larger $\lambda$ (more shrinkage) → flatter, simpler models. Smaller $\alpha$ → looser prior → behaviour closer to MLE/OLS.

| L_p regulariser | Equivalent prior |
|---|---|
| L2 ($\|\mathbf{w}\|^2$) | Gaussian: $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1} \mathbf{I})$ |
| L1 ($\|\mathbf{w}\|_1$) | Laplace: $w_j \sim \text{Laplace}(0, b)$ |

## When It Beats MLE

A canonical example: degree-8 polynomial regression on $N=10$ points.

- **MLE** fits all 10 points (almost) exactly, but oscillates wildly between them — terrible test performance.
- **MAP** with a Gaussian prior produces a smoother fit. The prior says "small weights are more likely," which prevents the polynomial coefficients from blowing up to interpolate every training point.

Same data, same model class — different criterion, dramatically different generalisation.

## Practical Notes

- **Hyperparameters $\alpha, \beta$.** Treated as fixed constants in the basic setup. In practice, choose by cross-validation, by maximising the *evidence* $p(\mathbf{y} \mid \mathbf{X})$ (empirical Bayes), or by placing hyperpriors and integrating those out too.
- **Computation.** Forming $\mathbf{S}_{\text{post}}$ requires inverting an $(M+1) \times (M+1)$ matrix — same cost as OLS. Posterior mean reduces to one matrix-vector product.
- **Online updates.** The posterior from $N$ points becomes the prior for the $(N+1)$th — you can update one point at a time without re-processing the dataset.

## What Could Go Wrong

- **Wrong noise model.** If $\beta$ (noise precision) is misspecified, the posterior is wrong. Either learn $\beta$ from data or place a prior on it (Gamma is conjugate to Gaussian-precision).
- **Wrong prior shape.** A zero-centred Gaussian assumes weights "should be small." If the truth is sparse (few non-zero weights), use a Laplace prior — recovers L1 / lasso.
- **High dimensions.** $\mathbf{S}_{\text{post}}^{-1}$ is $(M+1) \times (M+1)$; inverting becomes expensive past a few thousand parameters.

## Connections

- [[linear-regression]] — the underlying model whose weights we're inferring.
- [[ordinary-least-squares]] — the MLE-equivalent point estimate; recovered as $\alpha \to 0$ (flat prior).
- [[ridge-regression]] — the MAP point estimate; recovered by maximising the posterior.
- [[bayes-law]] — the rule combining prior and likelihood.
- [[gaussian-distribution]] — the conjugate self-pairing that makes everything closed-form.
- [[maximum-likelihood-estimation-ml|maximum likelihood estimation]] — the no-prior limit; equivalent to OLS under Gaussian noise.

## Active Recall

> [!question]- Why is Gaussian-likelihood + Gaussian-prior a "conjugate" pair?
> Because the resulting posterior is also Gaussian — same family as the prior. Conjugacy means the math stays in closed form: no integrals to estimate numerically, no MCMC required. For Bayesian linear regression with Gaussian noise, this is what makes the entire inference tractable analytically.

> [!question]- What's the relationship between the Bayesian posterior mean and ridge regression?
> They're identical. Maximising the log-posterior is the same as minimising the OLS loss + $(\alpha/\beta) \|\mathbf{w}\|^2$. So the MAP estimate of Bayesian linear regression with a zero-mean Gaussian prior is exactly the ridge regression solution. The Bayesian view supplies the "why" for L2 regularisation: it's the negative log of a Gaussian prior belief.

> [!question]- What does the posterior covariance $\mathbf{S}_{\text{post}}$ tell you that a point estimate (OLS / ridge) doesn't?
> Model uncertainty — how confident we are in those weights. A wide posterior means many $\mathbf{w}$'s are still plausible (typical with little data); a narrow posterior means data has pinned down the weights tightly. For predictions, this propagates to a *predictive variance* that grows in regions of input space far from training data, and shrinks where you have lots of evidence. Point-estimate methods give a single $\hat{y}$ with no uncertainty.
