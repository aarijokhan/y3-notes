---
type: concept
name: Bayes' law
description: Posterior ∝ likelihood × prior — the rule for inverting conditional probabilities, and the foundation of Bayesian inference
sources:
  - raw/week-07/Wk_7_Lec_2-1.pdf
status: draft
updated: 2026-04-27
---

*A rule for inverting conditional probabilities: $P(\theta \mid X) = P(X \mid \theta) P(\theta) / P(X)$. The Bayesian view treats parameters $\theta$ as random variables with a prior distribution; observed data updates this prior to a posterior. Stands in contrast to the frequentist view, which treats $\theta$ as a fixed unknown to be point-estimated.*

## The Rule

For events:

$$P(\theta \mid X) = \frac{P(X \mid \theta) P(\theta)}{P(X)}$$

For continuous parameters with prior density $p(\theta)$ and likelihood $p(X \mid \theta)$:

$$p(\theta \mid X) = \frac{p(X \mid \theta) p(\theta)}{p(X)}, \qquad p(X) = \int p(X \mid \theta) p(\theta) \, d\theta$$

The denominator $p(X)$ — the **marginal likelihood** or "evidence" — is a normalising constant that ensures the posterior integrates to 1. It rarely needs to be computed explicitly because of the proportionality:

$$\boxed{p(\theta \mid X) \propto \underbrace{p(X \mid \theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}}$$

Or in slogan form: **posterior ∝ likelihood × prior**.

## What Each Term Means

- **Prior** $p(\theta)$ — what we believed about the parameters *before* seeing the data. Encodes domain knowledge, regularisation, or genuine ignorance (uniform, weak, or non-informative priors).
- **Likelihood** $p(X \mid \theta)$ — how plausible the observed data is under each candidate $\theta$. Same object that [[maximum-likelihood-estimation-ml|MLE]] maximises.
- **Posterior** $p(\theta \mid X)$ — what we believe *after* seeing the data. Combines prior knowledge with observational evidence.
- **Evidence** $p(X)$ — the probability of the data marginalised over all $\theta$. Useful for model comparison; usually ignored when computing the posterior shape.

## Frequentist vs Bayesian

The two paradigms differ on what $\theta$ *is*:

| Frequentist | Bayesian |
|---|---|
| $\theta$ is a fixed unknown constant | $\theta$ is a random variable with a distribution |
| Data is random; $\theta$ is not | Data is observed; $\theta$ is the unknown |
| Confidence in $\theta$ comes from imagining repeating the experiment | Confidence in $\theta$ is a probability distribution |
| Tool: MLE, hypothesis tests, confidence intervals | Tool: Bayes' law, credible intervals, posterior predictive |

A Bayesian works with the *one* dataset they actually observed and quantifies uncertainty in $\theta$ as a probability distribution. A frequentist asks how the estimator $\hat{\theta}$ would behave across many hypothetical datasets and uses cross-validation, bootstrapping, or sampling distributions.

The frameworks aren't always at odds — many results overlap, and Bayesian inference with a flat prior often recovers the MLE. But the *interpretations* differ.

## Worked Example — Disease Test

A disease affects 1% of the population. A test is 99% accurate (99% sensitivity *and* 99% specificity). You test positive. What's the probability you have the disease?

Apply Bayes:

$$P(D \mid +) = \frac{P(+ \mid D) P(D)}{P(+)}$$

The denominator expands by total probability:

$$P(+) = P(+ \mid D) P(D) + P(+ \mid \neg D) P(\neg D) = 0.99 \cdot 0.01 + 0.01 \cdot 0.99 = 0.0198$$

So:

$$P(D \mid +) = \frac{0.99 \cdot 0.01}{0.0198} = 0.5$$

Despite the 99%-accurate test, a positive result only means a 50% chance of having the disease. The intuition: positives from the 99%-of-the-population healthy group nearly match positives from the 1%-of-the-population sick group, because the prior $P(D)$ is so small.

This is the classic illustration of why **base rates matter** — and why MLE-style "ignore the prior" reasoning misleads in low-prevalence settings.

## Example — Bayesian Linear Regression (preview)

For [[linear-regression|linear regression]], the Bayesian recipe is:

1. Place a prior over weights, e.g., $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1} \mathbf{I})$ — small weights are a priori more likely.
2. The likelihood, assuming Gaussian noise, is $p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = \prod_i \mathcal{N}(y_i \mid \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i), \beta^{-1})$.
3. Apply Bayes: $p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) p(\mathbf{w})$.
4. The posterior is itself Gaussian (by conjugacy); its mean is the MAP/Bayes estimate of $\mathbf{w}$.

The MAP estimate ends up being equivalent to **ridge regression** — the prior acts as L2 regularisation. This is one of the cleanest "regularisation = Bayesian prior" connections in ML.

(Bayesian linear regression itself is covered in later weeks; this is foreshadowing.)

## When Priors Matter

- **Small datasets** — the likelihood is weak, so the prior dominates the posterior. Choose carefully.
- **Domain knowledge** — if you know weights should be small, sparse, or positive, encode that as a prior.
- **Regularisation** — many regularisation schemes have Bayesian interpretations (L2 ↔ Gaussian prior, L1 ↔ Laplace prior).
- **Avoiding overfitting** — even uninformative priors can prevent MLE pathologies (e.g., infinite weights in separable logistic regression).

When *not* to lean on priors:

- **Lots of data** — the likelihood overwhelms the prior; results are nearly identical to MLE.
- **No defensible prior** — choosing a "non-informative" prior is itself a non-trivial decision (Jeffreys prior, reference prior, etc.).

## Connections

- [[maximum-likelihood-estimation-ml|maximum likelihood estimation]] — recovers MAP under a flat prior (or in the large-data limit). MLE is the "no prior" special case.
- [[gaussian-distribution]] — the most common conjugate prior pair: Gaussian prior + Gaussian likelihood → Gaussian posterior, all in closed form.
- [[ordinary-least-squares]] — its MLE-equivalence under Gaussian noise can be extended to ridge regression by adding a Gaussian prior on $\mathbf{w}$.

## Active Recall

> [!question]- State Bayes' law in the "posterior ∝ likelihood × prior" form. What's been dropped?
> $p(\theta \mid X) \propto p(X \mid \theta) p(\theta)$. The denominator $p(X)$ — the marginal likelihood — has been absorbed into the proportionality. Since it doesn't depend on $\theta$, it doesn't change the shape of the posterior; it's only needed when normalising or comparing across models.

> [!question]- A test is 99% accurate. The disease is rare (1% prevalence). You test positive. Why isn't your probability of having the disease close to 99%?
> Because base rates matter. The 99% accuracy means few false positives *per healthy person*, but there are 99× more healthy people than sick people. So the *count* of false positives among healthy folks is comparable to the count of true positives among sick folks. After applying Bayes' law: $P(D \mid +) = 0.5$. The prior $P(D) = 0.01$ pulls the posterior down hard.

> [!question]- What's the relationship between MLE and Bayesian inference?
> MLE picks the single $\theta$ that maximises the likelihood, ignoring the prior. Bayesian inference returns the full posterior distribution over $\theta$, weighted by the prior. With a flat prior (or as data grows), the MAP (posterior mode) converges to the MLE. So MLE is the "no-prior" or "infinite-data" limit of Bayesian inference.
