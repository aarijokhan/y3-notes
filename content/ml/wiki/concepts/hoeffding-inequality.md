---
type: concept
name: Hoeffding's inequality
description: A probabilistic bound stating that the sample average is unlikely to differ from the true mean by more than ε, with probability decaying exponentially in N — the foundation for PAC-style generalization claims
sources:
  - raw/week-08/Wk_8_Lec_1-1.pdf
  - raw/week-08/ML_Exercise_Sheet_8b_solution.pdf
status: draft
updated: 2026-04-28
---

*A probabilistic bound: the sample mean $\frac{1}{N} \sum z_i$ of $N$ i.i.d. random variables (each in $[0,1]$) deviates from the true mean by more than $\epsilon$ with probability at most $2 e^{-2 \epsilon^2 N}$. In ML terms: training error and test error are close with high probability, *for any fixed hypothesis*. The exponential decay in $N$ is what makes learning feasible at all.*

## The Inequality

For i.i.d. random variables $z_1, \ldots, z_N$ with $0 \leq z_i \leq 1$:

$$\boxed{\mathbb{P}\left(\left|\frac{1}{N} \sum_{i=1}^N z_i - \mathbb{E}_{z \sim p(z)}[z]\right| > \epsilon\right) \leq 2 e^{-2 \epsilon^2 N}}$$

The sample mean concentrates around the true mean. The probability of being "$\epsilon$-far" decays *exponentially* in $N$ (the sample size), not polynomially.

Two messages:

1. **You can bound the unknown using what you observe.** The sample mean (which you compute) is an estimate of the true mean (which you don't know). Hoeffding tells you how much they can differ.
2. **The bound is universal.** The right-hand side is independent of the underlying distribution $p(z)$ — works for *any* bounded distribution, no further assumptions.

## Application to Machine Learning

Set $z_i = \mathbb{1}[h(\mathbf{x}_i) \neq f(\mathbf{x}_i)]$ — the 0/1 misclassification indicator for example $i$ under a fixed hypothesis $h$. Then:

- $\frac{1}{N} \sum_i z_i = E_{\text{in}}(h)$ — the **in-sample (training) error**.
- $\mathbb{E}_z[z] = E_{\text{out}}(h)$ — the **out-of-sample (true) error**.

Plugging into Hoeffding:

$$\mathbb{P}\left(\left|E_{\text{in}}(h) - E_{\text{out}}(h)\right| > \epsilon\right) \leq 2 e^{-2 \epsilon^2 N}$$

> [!info]+ The basic feasibility statement
> "Training error is probably close to test error, when the training set is large enough." That's all Hoeffding says — but it's the foundation of every formal generalization argument in ML.

## Accuracy and Confidence

Rewriting with $\delta = 2 e^{-2 \epsilon^2 N}$:

$$\mathbb{P}\left(\left|E_{\text{in}}(h) - E_{\text{out}}(h)\right| \leq \epsilon\right) \geq 1 - \delta$$

- $\epsilon$ is the **accuracy**: how close $E_{\text{in}}$ and $E_{\text{out}}$ are.
- $1 - \delta$ is the **confidence**: probability that the bound holds.
- Solving for $\epsilon$: $\epsilon = \sqrt{\frac{1}{2N} \log \frac{2}{\delta}}$.

So with probability $\geq 1 - \delta$:

$$E_{\text{in}}(h) - \epsilon \leq E_{\text{out}}(h) \leq E_{\text{in}}(h) + \epsilon$$

## PAC Learnability

A target function is **PAC-learnable** (Probably Approximately Correct) if for any tolerances $\epsilon, \delta > 0$, there exists an algorithm and a sample size $N(\epsilon, \delta)$ such that:

$$\mathbb{P}\left[|E_{\text{in}}(h) - E_{\text{out}}(h)| \leq \epsilon\right] \geq 1 - \delta$$

In words: with as much accuracy as you want and as much confidence as you want, you can find $N$ large enough that your training error approximates your test error to that tolerance.

PAC is the formal sense in which ML "works": not "we will be perfectly correct" but "we will be probably approximately correct, for any $\epsilon$ and $\delta$ we care about, given enough data."

## The Catch — One Hypothesis vs the Final Hypothesis

Hoeffding requires the hypothesis $h$ to be **fixed before looking at the data**. But in ML, we *choose* our final hypothesis $g$ from a hypothesis set $\mathcal{H} = \{h_1, \ldots, h_M\}$ based on the training data. This breaks the i.i.d. assumption that Hoeffding needs.

Fix: apply Hoeffding to *each* candidate hypothesis and take a union bound. If we choose $g$ from $M$ hypotheses:

$$\mathbb{P}\left(|E_{\text{in}}(g) - E_{\text{out}}(g)| > \epsilon\right) \leq 2 M e^{-2 \epsilon^2 N}$$

The $M$ factor is the price for letting the algorithm pick $g$ from a set rather than committing to one in advance. See [[generalization-bound]] for what this means in practice.

## Worked Numerical Example

A bin contains marbles with true mean $\mu = 0.4$. We sample $N = 10$ marbles. What's the bound on $\mathbb{P}(\nu \leq 0.1)$ where $\nu$ is the sample mean?

$\nu \leq 0.1$ implies $|\nu - \mu| \geq 0.3$, so $\epsilon = 0.3$.

$$\mathbb{P}(|\nu - \mu| > 0.3) \leq 2 e^{-2 (0.3)^2 \cdot 10} = 2 e^{-1.8} \approx 0.33$$

So the bound is about 33%.

## How Much Data Do We Need?

Given target accuracy $\epsilon$, target confidence $1 - \delta$, and hypothesis set size $M$:

$$2 M e^{-2 \epsilon^2 N} \leq \delta \;\;\Longleftrightarrow\;\; N \geq \frac{1}{2 \epsilon^2} \log \frac{2 M}{\delta}$$

Worked example: $\epsilon = 0.1$, $\delta = 0.05$, $M = 100$:

$$N \geq \frac{1}{2 (0.1)^2} \log \frac{200}{0.05} = 50 \log(4000) \approx 50 \cdot 8.29 \approx 415$$

So 415 examples suffice for 10% accuracy with 95% confidence over 100 hypotheses.

Notice the scaling:
- $1/\epsilon^2$ — halving accuracy quadruples required $N$.
- $\log(1/\delta)$ — improving confidence is *cheap* (logarithmic).
- $\log M$ — adding hypotheses is also cheap (logarithmic).

## Tightness

Hoeffding is **not tight** — it's a worst-case upper bound. For specific distributions, sharper bounds exist (Bernstein, Bennett). In practice the *shape* matters more than the constant: exponential decay in $N$ is what guarantees learning is feasible at scale.

## What Could Go Wrong

- **The i.i.d. assumption.** If training and test data come from different distributions, Hoeffding doesn't apply. This is "covariate shift" or "distribution shift."
- **Variables outside $[0,1]$.** Hoeffding for general bounded variables exists; the constant changes. For unbounded distributions (heavy tails), use Bernstein-type bounds with variance terms.
- **Choosing $h$ after looking at the data.** This is the whole reason we need the union bound and the $M$ factor — see the next section.

## Connections

- [[generalization-bound]] — Hoeffding + union bound gives the master generalization bound.
- [[gaussian-distribution]] — for sums of bounded variables, Hoeffding is the non-Gaussian-CLT version: it gives a bound that doesn't require Gaussianity in the limit.
- [[bayes-law]] — different framework for uncertainty (probability distributions over parameters); Hoeffding is the frequentist counterpart.

## Active Recall

> [!question]- Why does Hoeffding give exponential decay in $N$ rather than polynomial?
> Hoeffding is a *concentration* inequality — it bounds how much an *average* of $N$ independent variables fluctuates. Averaging amplifies cancellation (positive and negative deviations balance), and the probability of large deviations decays exponentially because each independent draw multiplies the chance of "everything going wrong together." Polynomial decay would correspond to a much weaker concentration; exponential is what makes finite-data learning provably feasible.

> [!question]- What's wrong with applying Hoeffding directly to the final hypothesis $g$ that an algorithm chooses?
> Hoeffding requires $h$ to be fixed *before* observing the data. The final hypothesis $g$ is chosen *based on* the training data, so the per-example errors $\mathbb{1}[g(\mathbf{x}_i) \neq f(\mathbf{x}_i)]$ are not i.i.d. with respect to a fixed $g$. The fix is the union bound: apply Hoeffding to each candidate $h \in \mathcal{H}$ and sum, paying a factor of $M = |\mathcal{H}|$.

> [!question]- For $\epsilon = 0.05$, $\delta = 0.01$, $M = 1000$, roughly how many training examples do we need?
> $N \geq \frac{1}{2 \epsilon^2} \log \frac{2M}{\delta} = \frac{1}{0.005} \log \frac{2000}{0.01} = 200 \cdot \log(200{,}000) \approx 200 \cdot 12.2 = 2440$. So roughly 2500 examples.
