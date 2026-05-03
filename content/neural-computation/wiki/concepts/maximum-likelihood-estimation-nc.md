---
type: concept
sources:
  - raw/week-01/w01-l03-transcript.txt
  - raw/week-01/w01-slides.pdf
status: stable
updated: 2026-04-20
---

*A principled way to choose parameters: pick the values that make the observed data most probable.*

## Notation

Before diving in, a few symbols that appear throughout:

| Symbol | Meaning |
|---|---|
| $\theta$ | A **parameter** — any unknown value we want to estimate (e.g. the true temperature, or a weight in a neural network) |
| $\hat{\theta}$ | Our **estimate** (or candidate value) for $\theta$ |
| $\theta^*$ or $\hat{\theta}_{\text{MLE}}$ | The **optimal estimate** — the specific value of $\hat{\theta}$ that maximises the likelihood (or equivalently minimises the loss) |
| $L(\hat{\theta})$ | A **[[loss-function]]** — maps $\hat{\theta}$ to a number measuring how bad that estimate is. Lower is better. |
| $\arg\min$ / $\arg\max$ | The value of the argument that achieves the minimum / maximum (not the min/max value itself) |

## Definition

**Maximum likelihood estimation (MLE)** finds the parameter $\hat{\theta}$ that maximises the likelihood of the observed data:

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\hat{\theta}} \; P(\text{data} \mid \hat{\theta})$$

MLE asks: "out of all possible parameter values, which one would have been most likely to produce the data we actually saw?"

## Likelihood vs probability

The expression $P(\text{data} \mid \hat{\theta})$ can be read in two directions, and the distinction matters:

- **Probability** (fixed $\theta$, varying data): "Given that the true temperature is 20°C, what is the probability of observing a reading of 23°C?" Here we know the parameter and ask about possible outcomes.
- **Likelihood** (fixed data, varying $\theta$): "Given that we observed 23°C, how likely is it that the true temperature is 20°C? What about 21°C? 22°C?" Here we know the data and ask which parameter value best explains it.

The formula is the same — $P(\text{data} \mid \theta)$ — but the interpretation flips depending on what we treat as fixed and what we vary. In MLE, the data is fixed (we already observed it) and we sweep over parameter values, so we call it the *likelihood* of $\theta$, not the probability of the data.

## The normal distribution

MLE requires a probabilistic model of how data is generated. A common and well-justified assumption is that measurements follow a **normal (Gaussian) distribution**:

$$P(x \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

where $\mu$ is the mean (the true value we're estimating) and $\sigma$ is the standard deviation (how spread out the noise is).

**What the Gaussian says about the data:**

- The distribution is **symmetric** around $\mu$ — overestimates and underestimates are equally likely
- It is **bell-shaped**: values near $\mu$ are most probable, and probability drops off exponentially as you move away
- $\sigma$ controls the **width** of the bell — small $\sigma$ means measurements are tightly clustered, large $\sigma$ means they're spread out
- About 68% of observations fall within $\pm 1\sigma$ of $\mu$, and about 95% within $\pm 2\sigma$

**Why assume a Gaussian?**

- **It's everywhere in nature.** Heights, weights, measurement errors, sensor noise — many real-world quantities approximately follow it. If you measure the same thing repeatedly, the readings will typically scatter in a bell curve around the true value.
- **Central Limit Theorem.** When an observation is the result of many small, independent random effects, their sum tends toward a Gaussian *regardless* of what the individual effects look like. This is why the Gaussian shows up so often — most noise is the accumulation of many tiny disturbances.
- **Mathematical convenience.** The Gaussian has properties that make optimisation clean: its log is a simple quadratic (the $\exp$ and $\ln$ cancel), which means the MLE derivation lands on a smooth, differentiable loss function (SSE) rather than something harder to work with.
- **Reasonable default for noise.** Even when the true noise distribution is unknown, the Gaussian is often a good first approximation. It's symmetric (no systematic bias), unimodal (one peak), and its tails decay quickly (extreme outliers are rare). For many practical problems, this is close enough.

## Deriving squared error from MLE

This is the key result: starting from a Gaussian noise assumption, MLE leads directly to minimising the sum of squared errors.

**Setup.** We have $n$ observations $x_1, \dots, x_n$, each independently drawn from $\mathcal{N}(\hat{x}, \sigma^2)$ where $\hat{x}$ is the unknown true value.

**Step 1 — Write the likelihood.** Since observations are independent, the joint probability is the product of individual probabilities:

$$P(x_1, \dots, x_n \mid \hat{x}) = \prod_{i=1}^{n} P(x_i \mid \hat{x}) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(x_i - \hat{x})^2}{2\sigma^2}\right)$$

**Step 2 — Take the log.** We work with the **log-likelihood** $\ell(\hat{x}) = \ln L(\hat{x})$ instead of the raw likelihood. This is valid because $\ln$ is monotonically increasing ($a > b \Rightarrow \ln a > \ln b$), so the $\hat{x}$ that maximises the likelihood also maximises the log-likelihood. But *why* do we want to take the log in the first place? Three reasons:

1. **Turns products into sums — makes differentiation tractable.** The likelihood is a product of $n$ PDFs. To find its maximum we'd need to differentiate using the product rule across all $n$ terms — algebraically brutal for large $n$. The log converts $\prod \to \sum$, and differentiating a sum is straightforward (just differentiate each term independently).

2. **Eliminates the Gaussian's exponential.** Each Gaussian PDF contains $\exp(\dots)$. The log and exp cancel: $\ln(\exp(z)) = z$, leaving a simple polynomial $(x_i - \hat{x})^2$. Without the log, we'd be optimising a product of exponentials — far harder.

3. **Avoids numerical underflow.** Individual probabilities can be extremely small (e.g. $10^{-50}$). Multiplying many such numbers causes a computer to round the result to zero, making optimisation impossible. Taking the log first converts these to manageable negative numbers (e.g. $\ln(10^{-50}) \approx -115$), and adding them preserves numerical precision.

Applying the log:

$$\ell(\hat{x}) = \ln P = \sum_{i=1}^{n} \left[\ln\!\left(\frac{1}{\sqrt{2\pi}\,\sigma}\right) - \frac{(x_i - \hat{x})^2}{2\sigma^2}\right]$$

**Step 3 — Drop constants (preserves optimum via translation invariance).** The first term inside the sum ($\log\frac{1}{\sqrt{2\pi}\sigma}$) and the denominator $2\sigma^2$ do not depend on $\hat{x}$. Adding or subtracting a constant shifts the entire loss curve up or down uniformly — every candidate $\hat{x}$ is affected equally — so the location of the maximum does not move. Multiplying by a positive constant ($1/2\sigma^2$) stretches the curve vertically but likewise leaves the peak in the same place. After dropping these:

$$\log P \propto -\sum_{i=1}^{n} (x_i - \hat{x})^2$$

**Step 4 — Flip sign (preserves optimum via negation).** Multiplying by $-1$ flips the curve upside-down: every peak becomes a valley and vice versa. So the $\hat{x}$ that *maximises* $-\sum(x_i - \hat{x})^2$ is the same $\hat{x}$ that *minimises* $\sum(x_i - \hat{x})^2$:

$$\hat{x}_{\text{MLE}} = \arg\min_{\hat{x}} \sum_{i=1}^{n} (x_i - \hat{x})^2$$

This is the **sum of squared errors (SSE)** — the same [[loss-function]] we use to train models.

### Three optimum-preserving operations (summary)

| Operation | Why it preserves the optimum |
|---|---|
| Apply $\log$ | Monotonically increasing — doesn't reorder values, so the peak stays in the same place |
| Drop additive constants / multiply by positive scalar | Shifts or stretches the curve uniformly — all candidates move together, so the peak doesn't shift |
| Negate ($\times -1$) | Flips max $\leftrightarrow$ min — converts $\arg\max$ to $\arg\min$ at the same location |

## Why this matters

The key insight: **we did not arbitrarily choose squared error as a loss function — it falls out naturally from our probabilistic assumptions.** Specifically, if you believe:

1. Each observation is generated independently
2. The noise around the true value follows a normal distribution

then the mathematically correct thing to do is minimise the sum of squared errors. MLE *derives* the loss function rather than asserting it.

**Why this matters for neural networks.** MLE is the bridge between probability and training. When training a neural network:

1. We assume our data comes from some underlying probability distribution
2. We choose an appropriate distribution for the problem (Gaussian for regression, Bernoulli for binary classification, categorical for multi-class, etc.)
3. Maximum likelihood **automatically gives us the right loss function** — we don't guess or hand-pick it
4. Gaussian noise assumption $\to$ **Mean Squared Error (MSE)**
5. Categorical distribution assumption $\to$ **Cross-Entropy Loss** (covered in later weeks)

The entire training pipeline — forward pass, loss computation, backpropagation — rests on this probabilistic foundation. Every time you see a loss function in this module, ask: "what distribution assumption does this correspond to?"

## Related

- [[loss-function]] — MLE provides a probabilistic justification for squared error loss
- [[perceptron]] — the model whose parameters we find via loss minimisation

## Active Recall

> [!question]- Walk through the MLE derivation: why does maximising a Gaussian likelihood reduce to minimising the sum of squared errors?
> Start with the product of Gaussian PDFs for independent observations. Take the log to convert the product to a sum. The Gaussian PDF has the form $\exp(-(x_i - \hat{x})^2 / 2\sigma^2)$, so the log-likelihood becomes a sum of $-(x_i - \hat{x})^2$ terms (plus constants). Dropping constants and flipping the sign to convert max to min gives $\arg\min \sum (x_i - \hat{x})^2$ — the SSE.

> [!question]- Why is it valid to apply the logarithm to the likelihood before optimising? Could we use any function?
> The log is monotonically increasing: if $f(a) > f(b)$, then $\log f(a) > \log f(b)$. This means the value that maximises $f$ also maximises $\log f$. We cannot use any function — only strictly monotonically increasing functions preserve the location of the maximum.

> [!question]- Under what assumptions does minimising MSE give the maximum likelihood estimate? When might those assumptions fail?
> MSE gives the MLE when (1) observations are independent and (2) noise follows a normal distribution. These assumptions can fail with heavy-tailed noise (outliers), correlated observations, or non-symmetric error distributions. In such cases, other loss functions (like MAE or Huber loss) may be more appropriate.

> [!question]- In the derivation, we "drop constants" that don't depend on $\hat{x}$. Why specifically are $\frac{1}{\sqrt{2\pi}\sigma}$ and $2\sigma^2$ irrelevant to the optimisation?
> We are optimising over $\hat{x}$, not $\sigma$. The normalisation constant $\frac{1}{\sqrt{2\pi}\sigma}$ and the denominator $2\sigma^2$ are the same for every candidate $\hat{x}$, so they shift or scale the log-likelihood uniformly. The location of the maximum — the value of $\hat{x}$ that achieves it — is unchanged.

> [!question]- Why do we need to flip the sign in step 4 of the derivation?
> The log-likelihood is $-\sum(x_i - \hat{x})^2$ (negative). MLE *maximises* this, but in machine learning we conventionally *minimise* loss functions. Maximising $-f$ is equivalent to minimising $f$, so we negate and switch from $\arg\max$ to $\arg\min$.

> [!question]- What is the difference between probability and likelihood? They use the same formula — so what changes?
> Both use $P(\text{data} \mid \theta)$, but the role of what is fixed and what varies flips. Probability fixes $\theta$ and asks "what data might we see?" Likelihood fixes the observed data and asks "which $\theta$ best explains it?" In MLE we have already observed the data, so we treat it as fixed and sweep over $\theta$ — that makes it a likelihood problem, not a probability problem.
