---
type: week
week: 8
title: "From Bayesian Priors to Generalization Bounds"
dates: 2025-11-17 to 2025-11-23
sources:
  - raw/week-08/Wk_8_Lec_1-1.pdf
  - raw/week-08/Wk_8_Lec_2-1.pdf
  - raw/week-08/Monday_ November 17_ 2025 at 4_19_15 PM_Captions_English (United States).txt
  - raw/week-08/Tuesday_ November 18_ 2025 at 10_04_07 AM_Captions_English (United States).txt
  - raw/week-08/Tuesday_ November 18_ 2025 at 11_03_15 AM_Captions_English (United States).txt
  - raw/week-08/Tutorial_Wk_8-1.pdf
  - raw/week-08/ML_Exercise_Sheet_8a.pdf
  - raw/week-08/ML_Exercise_Sheet_8a_solution.pdf
  - raw/week-08/ML_Exercise_Sheet_8b.pdf
  - raw/week-08/ML_Exercise_Sheet_8b_solution.pdf
concepts:
  - "[[bayesian-linear-regression]]"
  - "[[ridge-regression]]"
  - "[[hoeffding-inequality]]"
  - "[[generalization-bound]]"
status: draft
updated: 2026-04-28
---

> [!question]+ THE CRUX: Last week ended with two open threads. (1) Linear regression's MLE/OLS is a *point estimate* — what if we want uncertainty over $\mathbf{w}$, or a principled justification for regularisation? (2) We've been assuming throughout that "training error close to test error" is reasonable — but on what grounds? When does learning *provably* generalise, and how much data do we need?

*The two halves of week 8 each answer one. **(1) Bayesian linear regression** places a Gaussian prior on $\mathbf{w}$ and updates to a Gaussian posterior via Bayes' law. The posterior mean is the MAP estimate — exactly **ridge regression** — so L2 regularisation is the negative log of a Gaussian prior. Regularisation isn't a heuristic; it's a probabilistic statement about what weights are plausible. **(2) Hoeffding's inequality** gives the foundational concentration bound: for a fixed hypothesis $h$, training error and test error differ by more than $\epsilon$ with probability at most $2 e^{-2 \epsilon^2 N}$. Combined with a union bound over a hypothesis set of size $M$, this becomes the **generalisation bound**: $E_{\text{out}} \leq E_{\text{in}} + \sqrt{(1/2N) \log(2M/\delta)}$ with confidence $1 - \delta$. Two opposing forces — large $M$ helps fit, small $M$ helps generalise — formalise the bias-variance trade-off.*

---

## Part 1: Bayesian Regression

### The Underdetermined Case

Linear regression with $\hat{y} = w_0 + w_1 x$ requires at least two $(x, y)$ pairs to identify the slope and intercept. With **one** data point, there are infinite solutions — every line through that point fits perfectly. OLS doesn't help; the system is underdetermined.

The Bayesian fix: **declare a prior** on which $\mathbf{w}$'s are plausible *before seeing the data*. A natural choice is "small weights are more likely than large ones," formalised as a Gaussian prior centred at zero. Now the posterior is a *distribution* over all admissible lines through the point — not a single answer, but a quantified set of possibilities.

This is more than a hack for tiny datasets. It generalises: even when OLS would work, the Bayesian view supplies model uncertainty, principled regularisation, and a graceful way to update beliefs as data arrives.

### The Setup

**Likelihood** (same as standard linear regression, with noise precision $\beta = 1/\sigma^2$):

$$p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, \beta) = \prod_{i=1}^N \mathcal{N}(y_i \mid \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i), \beta^{-1})$$

**Prior** — zero-mean Gaussian with isotropic precision $\alpha$:

$$p(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{0}, \alpha^{-1} \mathbf{I})$$

**Posterior** by [[bayes-law|Bayes' law]]:

$$p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) \cdot p(\mathbf{w})$$

### Why the Math Closes — Conjugacy

A Gaussian likelihood combined with a Gaussian prior gives a Gaussian posterior. This is the canonical example of **conjugacy** — the posterior stays in the same family as the prior, so no integrals to estimate, no MCMC. Everything in closed form.

Taking the log:

$$\ln p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) = -\frac{\beta}{2} \sum_i (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 - \frac{\alpha}{2} \mathbf{w}^\top \mathbf{w} + \text{const}$$

Quadratic in $\mathbf{w}$ → completing the square gives:

$$\boxed{p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) = \mathcal{N}(\mathbf{w} \mid \mathbf{m}_{\text{post}}, \mathbf{S}_{\text{post}})}$$

with:

$$\mathbf{m}_{\text{post}} = \beta \mathbf{S}_{\text{post}} \boldsymbol{\Phi}^\top \mathbf{y}, \qquad \mathbf{S}_{\text{post}}^{-1} = \alpha \mathbf{I} + \beta \boldsymbol{\Phi}^\top \boldsymbol{\Phi}$$

### MAP = Ridge Regression

Maximising the log-posterior (the **MAP estimate**) is the same as minimising:

$$\frac{1}{2} \sum_i (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 + \frac{\alpha}{2 \beta} \mathbf{w}^\top \mathbf{w}$$

The first term is the OLS loss. The second is an **L2 penalty** on $\mathbf{w}$ with regularisation coefficient $\lambda = \alpha/\beta$. **This is exactly [[ridge-regression|ridge regression]].**

> [!info]+ Regularisation is a prior in disguise
> The L2 penalty $\frac{\lambda}{2} \|\mathbf{w}\|^2$ is the negative log of a zero-mean Gaussian prior. Maximising the posterior = minimising "negative log-likelihood + negative log-prior" = minimising "OLS loss + L2 penalty." The Bayesian view explains *why* L2 regularisation works: it's a probabilistic statement that small weights are a-priori more plausible than large ones.
>
> Other regularisers correspond to other priors: L1 (lasso) ↔ Laplace prior; mixtures correspond to elastic net; structured priors give group lasso, etc.

### Posterior Behaviour as Data Grows

A sequence of plots (lecture slides 16–17) traces what happens to the posterior over $(w_0, w_1)$ as $N$ grows from 1 to 1000:

- **$N = 1$**: posterior is broad — barely tighter than the prior.
- **$N = 10$**: a clear elongated ellipse appears, confirming the slope-intercept correlation.
- **$N = 100$**: posterior shrinks to a small region.
- **$N = 1000$**: a tiny spot — almost a point estimate.

Equivalently: sample 10 lines from the posterior at each $N$ and plot them. With small $N$, the lines fan out in many directions; with large $N$, they're nearly indistinguishable. **Model uncertainty reduces with data.**

### Predictive Distribution

A non-Bayesian model gives a single $\hat{y}$ for each new $\mathbf{x}$. Bayesian regression integrates over all plausible $\mathbf{w}$'s:

$$p(y \mid \mathbf{x}, \mathbf{y}, \mathbf{X}) = \int p(y \mid \mathbf{x}, \mathbf{w}) \, p(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) \, d\mathbf{w}$$

The result is Gaussian: mean $\mathbf{m}_{\text{post}}^\top \boldsymbol{\phi}(\mathbf{x})$, variance $\beta^{-1} + \boldsymbol{\phi}(\mathbf{x})^\top \mathbf{S}_{\text{post}} \boldsymbol{\phi}(\mathbf{x})$. The variance has two parts: irreducible noise ($\beta^{-1}$) and model uncertainty (the second term, vanishing as data accumulates).

### When MAP Beats MLE

A degree-8 polynomial fit to $N = 10$ noisy points:

- **MLE** interpolates training points exactly, oscillates wildly between them, and fails on test data.
- **MAP** with a Gaussian prior is smoother. Slightly larger training error, dramatically smaller test error.

Same model class, same data — different criterion, qualitatively different behaviour. The prior prevents the polynomial coefficients from blowing up to chase noise.

## Part 2: Is Learning Feasible?

### The Question

The whole machine-learning enterprise rests on a leap: from observed training data, infer something about *unseen* test data. Is this leap justified? Or are we just memorising?

**Deterministic answer: NO.** From a finite sample, you cannot say anything *certain* about points outside it. Anyone can construct a function that agrees with you on training points and disagrees everywhere else.

**Probabilistic answer: YES.** If training and test are i.i.d. from the same distribution, then with high probability the training-set behaviour resembles the population behaviour. Quantifying *how* high — and as a function of *what* — is the job of [[hoeffding-inequality|Hoeffding's inequality]].

### Hoeffding's Inequality

For i.i.d. random variables $z_1, \ldots, z_N$ with $0 \leq z_i \leq 1$:

$$\mathbb{P}\left(\left|\frac{1}{N} \sum_{i=1}^N z_i - \mathbb{E}[z]\right| > \epsilon\right) \leq 2 e^{-2 \epsilon^2 N}$$

Two messages:

1. **You can bound the unknown using what you observe.** The sample mean (you know) approximates the true mean (you don't), with quantified probability of error.
2. **Universal.** The right-hand side depends on neither the underlying distribution $p(z)$ nor anything else — works for *any* bounded distribution.

For ML, set $z_i = \mathbb{1}[h(\mathbf{x}_i) \neq f(\mathbf{x}_i)]$ — the misclassification indicator. Then $\frac{1}{N} \sum z_i = E_{\text{in}}(h)$ (training error) and $\mathbb{E}[z] = E_{\text{out}}(h)$ (test error). Hoeffding becomes:

$$\mathbb{P}\left(|E_{\text{in}}(h) - E_{\text{out}}(h)| > \epsilon\right) \leq 2 e^{-2 \epsilon^2 N}$$

> [!info]+ The basic feasibility result
> "Training error is probably close to test error, when training set is large enough." That's what Hoeffding gives — and it's the foundation of every formal generalisation argument in ML.

### Accuracy and Confidence — PAC

Setting $\delta = 2 e^{-2 \epsilon^2 N}$:

$$\mathbb{P}\left(|E_{\text{in}}(h) - E_{\text{out}}(h)| \leq \epsilon\right) \geq 1 - \delta$$

- $\epsilon$: **accuracy** — how close training and test errors are.
- $1 - \delta$: **confidence** — probability the claim holds.

A target function is **Probably Approximately Correct (PAC)** learnable if for any $\epsilon, \delta > 0$ there exists an algorithm and a sample size $N(\epsilon, \delta)$ achieving the above. PAC is the formal sense in which ML "works": not perfect correctness, but *probably approximately correct*, given enough data.

### The One-Hypothesis Catch

Hoeffding requires $h$ to be **fixed before** observing the data. But algorithms *choose* $g$ from a hypothesis set $\mathcal{H}$ based on the training data — so the per-example errors $\mathbb{1}[g(\mathbf{x}_i) \neq f(\mathbf{x}_i)]$ aren't i.i.d. with respect to a *fixed* $g$.

Fix: apply Hoeffding to *each* candidate $h_m \in \mathcal{H}$, then take a **union bound**:

$$\mathbb{P}\left(|E_{\text{in}}(g) - E_{\text{out}}(g)| > \epsilon\right) \leq \sum_{m=1}^M \mathbb{P}\left(|E_{\text{in}}(h_m) - E_{\text{out}}(h_m)| > \epsilon\right) \leq 2 M e^{-2 \epsilon^2 N}$$

The factor $M = |\mathcal{H}|$ is the price for letting the algorithm *choose* among $M$ hypotheses.

### The Generalization Bound

Solving $2 M e^{-2 \epsilon^2 N} = \delta$ for $\epsilon$ and rearranging:

$$\boxed{E_{\text{out}}(g) \leq E_{\text{in}}(g) + \sqrt{\frac{1}{2 N} \log \frac{2 M}{\delta}}}$$

with probability at least $1 - \delta$. This is the **[[generalization-bound|generalization bound]]**.

Reading the variables:

| Variable | Effect | What you control |
|---|---|---|
| $N$ — training-set size | More $N$ → smaller gap | Get more data |
| $M$ — hypothesis-set size | Larger $M$ → larger gap | Choose simpler models |
| $\delta$ — confidence tolerance | Smaller $\delta$ → larger gap | Trade-off |
| $E_{\text{in}}$ — training error | Smaller → tighter | Train better |

The square root means halving the gap requires *quadrupling* $N$. Improving $\delta$ or adding hypotheses are logarithmic — cheap.

### The Two Central Questions

The bound splits learning into:

1. **Can $E_{\text{out}}$ be close to $E_{\text{in}}$?** — generalisation. Favours **small $M$**.
2. **Can $E_{\text{in}}$ be small enough?** — fitting. Favours **large $M$** (more options).

These pull in opposite directions:

| $M$ | Generalisation | Fitting |
|---|---|---|
| Small | ✓ small gap | ✗ poor fit |
| Large | ✗ large gap | ✓ good fit |

Picking the right $M$ balances them. This is the formal source of the **bias-variance / model-complexity trade-off** — too simple → underfit (high bias), too complex → overfit (high variance).

### Worked Example — How Much Data?

Target: $\epsilon = 0.1$, $\delta = 0.05$, $M = 100$ candidates.

$$N \geq \frac{1}{2 \epsilon^2} \log \frac{2 M}{\delta} = \frac{1}{0.02} \log \frac{200}{0.05} = 50 \cdot \log(4000) \approx 50 \cdot 8.29 \approx 415$$

So 415 examples suffice for 10% accuracy with 95% confidence over 100 hypotheses.

### The $M = \infty$ Problem

Linear regression has $|\mathcal{H}| = \infty$ — there's a continuum of weight vectors. Naively, $\sqrt{\log M} = \infty$ and the bound is vacuous. Same for SVMs, neural networks, and almost everything practical.

**Fix: replace $M$ with a finite "effective" complexity.** Many hypotheses in $\mathcal{H}$ produce the same labelling on a fixed training set — the bad events overlap massively. Counting *distinct labellings* (rather than distinct $\mathbf{w}$'s) gives a finite quantity even when $|\mathcal{H}|$ is infinite.

Define a **dichotomy** as a hypothesis "as seen by" $N$ specific inputs — the labelling pattern $(h(\mathbf{x}_1), \ldots, h(\mathbf{x}_N))$ it produces. The number of distinct dichotomies is at most $2^N$ for binary classification (finite for any finite $N$), and for "structured" hypothesis sets grows polynomially in $N$ rather than exponentially.

For lines in $\mathbb{R}^2$:

| $N$ inputs | Max dichotomies | $2^N$ |
|---|---|---|
| 1 | 2 | 2 |
| 2 | 4 | 4 |
| 3 | 8 (or 6 if collinear) | 8 |
| 4 | **14** | 16 |

At $N = 4$, lines in $\mathbb{R}^2$ can no longer realise every possible labelling — *some* labellings are impossible regardless of where you place the points. This is the structural restriction that makes infinite-$\mathcal{H}$ learning feasible.

The replacement complexity is the **growth function** $m_{\mathcal{H}}(N)$, formalised in upcoming weeks via **VC dimension**.

---

## Concepts Introduced This Week

- [[bayesian-linear-regression]] — full Bayesian treatment of linear regression: prior, conjugate Gaussian likelihood, closed-form Gaussian posterior, predictive distribution.
- [[ridge-regression]] — L2-regularised linear regression; the MAP estimate of Bayesian linear regression under a Gaussian prior; the practical fix for ill-conditioned $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$.
- [[hoeffding-inequality]] — the master concentration bound: sample mean and true mean are close with probability $\geq 1 - 2 e^{-2 \epsilon^2 N}$. Foundation of PAC learning.
- [[generalization-bound]] — Hoeffding + union bound: $E_{\text{out}} \leq E_{\text{in}} + \sqrt{(1/2N) \log(2M/\delta)}$ with confidence $1 - \delta$. The formal source of the bias-variance trade-off.

## Connections

- **Builds on** [[week-07]]: regularisation (ridge) is the Bayesian completion of linear regression — the prior supplies the regularisation. The two halves of the module's "regression" arc are now joined: MLE/OLS for point estimates, MAP/ridge and full Bayesian for uncertainty.
- **Builds on** [[bayes-law]]: the abstract formula "posterior ∝ likelihood × prior" gets its first concrete payoff. Conjugate Gaussian-Gaussian pairing keeps everything tractable.
- **Builds on** [[gaussian-distribution]]: the Gaussian's role expands — it's no longer just the noise model, it's also the prior, and (consequently) the posterior. Conjugacy means the family is closed under inference.
- **Sets up** later weeks: VC dimension (formal way to replace $M$ with a finite quantity); cross-validation (practical way to pick model complexity); deep learning generalisation (where Hoeffding-style bounds are vacuous and tighter notions are needed).

## Open Questions

- **How do we replace $M$ with something finite for infinite hypothesis sets?** VC dimension — next weeks. The lecture's "dichotomy" preview is the setup.
- **How do we choose $\alpha, \beta$ in Bayesian regression?** Cross-validation, empirical Bayes (maximise the evidence), or hierarchical priors with hyperpriors integrated out.
- **Are Hoeffding-style bounds useful in practice for deep networks?** Generally no — $M$ is astronomical and the bound is vacuous. Tighter notions (Rademacher complexity, PAC-Bayes, margin-based bounds) are active research topics.
- **What if priors and noise aren't Gaussian?** Conjugacy is lost; closed-form inference fails. Then we use approximate inference: MCMC, variational Bayes, Laplace approximations.
