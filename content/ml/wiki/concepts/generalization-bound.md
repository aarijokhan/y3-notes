---
type: concept
name: generalization bound
description: A high-probability upper bound on test error in terms of training error, derived from Hoeffding + union bound, with a model-complexity term that grows with the size of the hypothesis set
sources:
  - raw/week-08/Wk_8_Lec_1-1.pdf
  - raw/week-08/Wk_8_Lec_2-1.pdf
status: draft
updated: 2026-04-28
---

*A high-probability upper bound on the test error $E_{\text{out}}$ in terms of the training error $E_{\text{in}}$ and a model-complexity term. With probability at least $1 - \delta$ over the training set, $E_{\text{out}}(g) \leq E_{\text{in}}(g) + \sqrt{\frac{1}{2N} \log \frac{2M}{\delta}}$, where $M$ is the hypothesis-set size. The first time formal generalization theory says anything useful — and the source of the bias-variance / complexity trade-off.*

## The Bound

Combining [[hoeffding-inequality|Hoeffding's inequality]] with the union bound over a hypothesis set $\mathcal{H} = \{h_1, \ldots, h_M\}$, then rearranging:

$$\boxed{E_{\text{out}}(g) \leq E_{\text{in}}(g) + \sqrt{\frac{1}{2N} \log \frac{2M}{\delta}}}$$

with probability at least $1 - \delta$ over the random draw of the training set. There's also a matching lower bound:

$$E_{\text{out}}(g) \geq E_{\text{in}}(g) - \sqrt{\frac{1}{2N} \log \frac{2M}{\delta}}$$

so $|E_{\text{out}} - E_{\text{in}}|$ is bounded on both sides.

The right-hand side has two pieces:

1. **Training error $E_{\text{in}}(g)$** — what you measure on training data. Want it small.
2. **Generalization gap $\epsilon = \sqrt{\frac{1}{2N} \log \frac{2M}{\delta}}$** — how much test error can exceed training error. Want it small.

## Reading the Bound

Each variable in the bound has a clear role:

| Variable | Effect on the bound | What you control |
|---|---|---|
| $N$ — training-set size | More $N$ → smaller $\epsilon$ | Get more data |
| $M$ — hypothesis-set size | Larger $M$ → larger $\epsilon$ | Choose simpler models |
| $\delta$ — confidence tolerance | Smaller $\delta$ (more confidence) → larger $\epsilon$ | Trade-off you set |
| $E_{\text{in}}$ — training error | Smaller $E_{\text{in}}$ → tighter bound | Train better |

The square root means halving $\epsilon$ requires *quadrupling* $N$. Improving $\delta$ is logarithmic — cheap. Adding hypotheses ($M$) is also logarithmic.

## The Two Central Questions

The bound splits the learning problem into two:

1. **Can $E_{\text{out}}(g)$ be close to $E_{\text{in}}(g)$?** — generalisation question. Favours **small $M$** (simple models with few hypotheses).
2. **Can $E_{\text{in}}(g)$ be small?** — fitting question. Favours **large $M$** (complex models with many hypotheses).

These pull in opposite directions:

| $M$ | Question 1 (gap) | Question 2 (fit) |
|---|---|---|
| Small | ✓ Small $\epsilon$ | ✗ Hypothesis set too poor to fit |
| Large | ✗ Large $\epsilon$ | ✓ Many options, can fit well |

The right $M$ balances them. This is the formal source of the **bias-variance** / model-complexity trade-off — too simple → high bias, too complex → high variance.

## The $M = \infty$ Problem

The bound is useless when $M = \infty$, which happens for any continuously parameterised hypothesis set — including linear regression, all SVMs, all neural networks. Naively, the bound says nothing about generalisation for these models.

**Fix: replace $M$ with a "complexity" measure that's finite even when $|\mathcal{H}| = \infty$.** The intuition: many hypotheses in $\mathcal{H}$ are nearly identical from the perspective of a fixed training set. When two hypotheses agree on every training point, treating them as separate hypotheses in the union bound is wasteful — the corresponding "bad events" overlap massively.

The recipe (preview, formalised in later weeks):

1. Define a **dichotomy** as a hypothesis "as seen by" a specific set of $N$ inputs — i.e., the labelling pattern $(h(\mathbf{x}_1), \ldots, h(\mathbf{x}_N))$ it produces.
2. The number of distinct dichotomies $\mathcal{H}(\mathbf{x}_1, \ldots, \mathbf{x}_N)$ is at most $2^N$ for binary classification — finite for any finite $N$.
3. For "well-behaved" $\mathcal{H}$ (e.g., lines in $\mathbb{R}^2$), this count is much smaller than $2^N$, and grows polynomially in $N$ rather than exponentially.
4. Replace $M$ with this growth function $m_{\mathcal{H}}(N)$.

For lines in $\mathbb{R}^2$, the counts are:

| $N$ | Max dichotomies (effective M) | $2^N$ |
|---|---|---|
| 1 | 2 | 2 |
| 2 | 4 | 4 |
| 3 | 8 (or 6 if collinear) | 8 |
| 4 | 14 (not 16) | 16 |
| 5 | 22 | 32 |

At $N = 4$, "lines in $\mathbb{R}^2$" can no longer realise every possible labelling. This restriction is what saves us — the *VC dimension*, which formalises this idea, is the topic of upcoming weeks.

## What Bounds Imply

- **Worst-case guarantee.** $E_{\text{out}} \leq E_{\text{in}} + \epsilon$ tells you the *worst* test error you might see. If you can manage that, you're safe.
- **Non-vacuous bounds are hard.** For deep networks (huge $M$), the simple Hoeffding bound is essentially $\infty$ — useless. Practitioners rely on cross-validation rather than theoretical bounds for small-sample estimates of $E_{\text{out}}$.
- **The bound is not tight.** Real models often generalise far better than the bound predicts. The bound is a *sufficient* condition, not a description of empirical reality.

## Worked Example

For $\epsilon = 0.1$, $\delta = 0.05$, $M = 100$:

$$N \geq \frac{1}{2 \epsilon^2} \log \frac{2M}{\delta} = \frac{1}{0.02} \log \frac{200}{0.05} \approx 50 \cdot 8.29 \approx 415$$

So with 415 examples, we have 95% confidence ($\delta = 0.05$) that $E_{\text{out}}$ is within 10% of $E_{\text{in}}$ for any hypothesis chosen from a 100-element set.

## What Could Go Wrong

- **Distribution shift.** The bound assumes training and test data are i.i.d. from the same distribution. Real-world deployments usually involve some shift.
- **Vacuous bounds for complex models.** $M$ for deep networks is astronomical; the bound says nothing useful. Tighter notions (Rademacher complexity, PAC-Bayes, VC dimension) are needed.
- **Interpreting $\epsilon$ as a probability.** $\epsilon$ is an error margin, not a probability. The probability is $1 - \delta$.
- **Multiple comparisons.** If you tune hyperparameters by cross-validation, you're effectively choosing among more hypotheses — increase $M$ accordingly when interpreting the bound.

## Connections

- [[hoeffding-inequality]] — the per-hypothesis bound that, after a union bound, becomes this generalisation bound.
- [[ridge-regression]] — one way to control effective $M$ (regularisation tightens the hypothesis set).
- [[non-linear-transformation]] — basis expansion increases effective $M$, raising the gap term.
- [[support-vector-machine]] — margin-based bounds replace $M$ with margin-related quantities; SVMs work even though $|\mathcal{H}| = \infty$.

## Active Recall

> [!question]- Why does the bound contain $\log M$ rather than $M$ itself?
> Because the union bound multiplies probabilities by $M$: $\mathbb{P}(\bigcup_i \mathcal{B}_i) \leq M \mathbb{P}(\mathcal{B}_1)$. The base probability is $2 e^{-2 \epsilon^2 N}$. Setting $2 M e^{-2 \epsilon^2 N} = \delta$ and solving for $\epsilon$ gives $\epsilon = \sqrt{(1/2N) \log(2M/\delta)}$. The $\log$ appears because we're inverting an exponential.

> [!question]- A model has $E_{\text{in}} = 0.05$, $N = 1000$, $M = 50$, $\delta = 0.05$. What's the upper bound on $E_{\text{out}}$?
> $\epsilon = \sqrt{(1/2000) \log(100/0.05)} = \sqrt{(1/2000) \log(2000)} = \sqrt{0.0005 \cdot 7.6} = \sqrt{0.0038} \approx 0.062$. So $E_{\text{out}} \leq 0.05 + 0.062 = 0.112$ with 95% confidence. Test error could be up to ~11% even though training error is 5%.

> [!question]- Explain why the bound is useless for $M = \infty$, and what's the standard fix.
> When $M$ is infinite, $\sqrt{\log M}$ is infinite, so the bound says $E_{\text{out}}$ could be arbitrarily larger than $E_{\text{in}}$ — meaningless. The fix is to replace $M$ with the *number of distinct dichotomies* — labellings that $\mathcal{H}$ can produce on a specific training set of $N$ points. This count is at most $2^N$ (finite for any finite $N$), and for "structured" hypothesis sets (like lines in $\mathbb{R}^2$) grows polynomially in $N$ rather than exponentially. The formal version of this is the VC dimension.
