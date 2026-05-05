---
type: concept
name: bias-variance decomposition
description: An average-case decomposition of the expected squared-error generalisation error into bias (how far the average hypothesis is from the truth) and variance (how much the hypothesis fluctuates across training sets). Complements VC analysis with a cleaner conceptual picture of overfitting vs underfitting.
sources:
  - raw/week-09/Wk_9_Lec_2-1.pdf
  - raw/week-09/Tuesday_ November 25_ 2025 at 9_42_04 AM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*For squared-error regression, the expected out-of-sample error decomposes as $\mathbb{E}_{\mathcal{D}}[E_{\text{out}}(g^{(\mathcal{D})})] = \text{bias} + \text{var} + \sigma^2$, where bias measures how far the average hypothesis $\bar{g}$ deviates from the truth $f$, variance measures how much individual hypotheses fluctuate around $\bar{g}$, and $\sigma^2$ is irreducible noise. The model class $\mathcal{H}$ controls bias and variance in opposite directions — the source of the underfitting/overfitting trade-off.*

## The Decomposition

Let $g^{(\mathcal{D})}$ be the hypothesis fit to a particular training set $\mathcal{D}$ and the **average hypothesis** be

$$\bar{g}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}[g^{(\mathcal{D})}(\mathbf{x})]$$

— what you'd get if you averaged the trained models across infinitely many independently-drawn training sets. (Operationally: $\bar{g}(\mathbf{x}) \approx \frac{1}{K} \sum_k g^{(\mathcal{D}_k)}(\mathbf{x})$ for $K$ datasets.)

Starting from the squared-error out-of-sample error and adding/subtracting $\bar{g}(\mathbf{x})$:

$$
\begin{aligned}
\mathbb{E}_{\mathcal{D}}\left[(g^{(\mathcal{D})}(\mathbf{x}) - f(\mathbf{x}))^2\right]
&= \mathbb{E}_{\mathcal{D}}\left[(g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x}) + \bar{g}(\mathbf{x}) - f(\mathbf{x}))^2\right] \\
&= \underbrace{\mathbb{E}_{\mathcal{D}}\left[(g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x}))^2\right]}_{\text{var}(\mathbf{x})} + \underbrace{(\bar{g}(\mathbf{x}) - f(\mathbf{x}))^2}_{\text{bias}(\mathbf{x})}.
\end{aligned}
$$

The cross term vanishes: $\mathbb{E}_{\mathcal{D}}[g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x})] = 0$ by definition of $\bar{g}$, and $\bar{g}(\mathbf{x}) - f(\mathbf{x})$ doesn't depend on $\mathcal{D}$.

Taking the expectation over $\mathbf{x}$ as well:

$$\boxed{\mathbb{E}_{\mathcal{D}}[E_{\text{out}}(g^{(\mathcal{D})})] = \text{bias} + \text{var}}$$

with

$$\text{bias} = \mathbb{E}_{\mathbf{x}}\left[(\bar{g}(\mathbf{x}) - f(\mathbf{x}))^2\right], \qquad \text{var} = \mathbb{E}_{\mathbf{x}}\left[\mathbb{E}_{\mathcal{D}}[(g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x}))^2]\right].$$

## Interpretation

**Bias** asks: *how far from the truth is the typical hypothesis we'd learn?* Low bias means $\mathcal{H}$ is rich enough that some hypothesis in $\mathcal{H}$ can closely approximate $f$, and the average of trained hypotheses is close to that approximation. High bias means $\mathcal{H}$ is structurally too simple — even the best hypothesis is far from $f$.

**Variance** asks: *how much do individual hypotheses fluctuate from one training set to another?* Low variance means the algorithm is stable: similar datasets produce similar hypotheses. High variance means the algorithm is fitting random structure in each dataset and producing wildly different hypotheses.

The dart-board picture summarises the four regimes:

| | Low variance | High variance |
|---|---|---|
| **Low bias** | Tight cluster on bullseye (ideal) | Scattered around bullseye (right model class but unstable) |
| **High bias** | Tight cluster off bullseye (consistent but wrong) | Scattered off bullseye (worst case) |

## With Noise

If the target itself is noisy — $y = f(\mathbf{x}) + \epsilon(\mathbf{x})$ with $\mathbb{E}[\epsilon] = 0$, $\text{Var}[\epsilon] = \sigma^2$ — the same algebra produces an extra term:

$$\mathbb{E}_{\mathcal{D}, \epsilon}[E_{\text{out}}(g^{(\mathcal{D})})] = \text{bias} + \text{var} + \sigma^2.$$

The third term is **irreducible noise**: even if you knew $f$ exactly, predictions $\hat{y}(\mathbf{x}) = f(\mathbf{x})$ would still be wrong by $\sigma^2$ on average, because the labels themselves are noisy. No amount of data, model complexity, or algorithmic cleverness reduces it.

## The Trade-Off

Bias and variance pull in opposite directions as $\mathcal{H}$ changes:

- **More complex $\mathcal{H}$**: better chance of approximating $f$ closely → **bias down**. But more parameters fit more random training noise → **variance up**.
- **Simpler $\mathcal{H}$**: fewer ways to fit noise → **variance down**. But the best hypothesis in $\mathcal{H}$ may be a poor approximation to $f$ → **bias up**.

A vivid example: target $f(x) = \sin(\pi x)$, $N = 2$ samples, two hypothesis sets:

- $\mathcal{H}_0 = \{h(x) = b\}$ (constant lines): bias $= 0.50$, var $= 0.25$, total $= 0.75$.
- $\mathcal{H}_1 = \{h(x) = ax + b\}$ (sloped lines): bias $= 0.21$, var $= 1.69$, total $= 1.90$.

The simpler model wins despite having higher bias — its variance is so much lower with only 2 samples that the total is smaller. With $N = 5$ the picture flips: bias unchanged, but variance for $\mathcal{H}_1$ drops to $0.21$, and $\mathcal{H}_1$ now wins ($0.42$ vs $0.60$).

The lesson: **what wins depends on $N$.** A complex model that overfits with little data may be the right choice with abundant data.

## Bias-Variance vs VC

Both decompositions live underneath the [[generalization-bound|generalisation bound]] but differ in:

| | VC analysis | Bias–variance analysis |
|---|---|---|
| Loss | 0–1 (binary error) | Squared error |
| Over what | Worst-case over $\mathcal{D}$ | Expectation over $\mathcal{D}$ |
| Bound | $E_{\text{out}} \leq E_{\text{in}} + \Omega(d_{\text{VC}})$ | $\mathbb{E}_{\mathcal{D}}[E_{\text{out}}] = \text{bias} + \text{var} + \sigma^2$ |
| Distribution-free? | Yes | No (needs $\mathbb{E}_{\mathcal{D}}$) |

VC gives a uniform bound that holds with high probability; bias–variance gives a clean decomposition of the *typical* generalisation error and a conceptual picture of overfitting. The two complement rather than compete: VC dimension says *whether* learning generalises, bias–variance says *what kind of error* is left.

## Practical Note

You can't compute bias and variance from a single training set — they require expectation over $\mathcal{D}$, which means access to many datasets from the same distribution. So bias–variance is mostly a **conceptual tool** for designing algorithms (telling you whether more data, more capacity, or stronger regularisation is the right move), not a quantity you measure directly.

Diagnostics:

- **High bias** (underfitting): training and test error both high and similar.
- **High variance** (overfitting): training error low, test error much higher.
- **Both**: both errors high and very different.

Remedies:

| Diagnosis | Move |
|---|---|
| High bias | More features, more layers, less regularisation |
| High variance | More data, simpler model, more regularisation, ensembling |
| Irreducible noise floor | Accept it — the noise is the noise |

## Related

- [[generalization-bound]] — the worst-case complement to this average-case analysis.
- [[vc-dimension]] — controls how bias and variance can simultaneously be small.
- [[ridge-regression]] — directly trades bias up for variance down via L2 penalty.
- [[learning-curve]] — visualises how bias and variance evolve with $N$.

## Active Recall

> [!question]- Derive the bias–variance decomposition by inserting $\bar{g}(\mathbf{x})$ into $(g^{(\mathcal{D})}(\mathbf{x}) - f(\mathbf{x}))^2$ and explain why the cross term vanishes.
> Write $g^{(\mathcal{D})}(\mathbf{x}) - f(\mathbf{x}) = (g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x})) + (\bar{g}(\mathbf{x}) - f(\mathbf{x}))$. Squaring: the diagonal terms give variance and bias respectively. The cross term is $2 \mathbb{E}_{\mathcal{D}}[(g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x}))(\bar{g}(\mathbf{x}) - f(\mathbf{x}))] = 2(\bar{g}(\mathbf{x}) - f(\mathbf{x})) \mathbb{E}_{\mathcal{D}}[g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x})]$, which is zero because $\mathbb{E}_{\mathcal{D}}[g^{(\mathcal{D})}(\mathbf{x})] = \bar{g}(\mathbf{x})$ by definition. The factor $\bar{g}(\mathbf{x}) - f(\mathbf{x})$ pulls outside because it doesn't depend on $\mathcal{D}$.

> [!question]- For target $f(x) = \sin(\pi x)$ with $N = 2$ samples, the constant-line hypothesis $\mathcal{H}_0$ gives bias 0.5, variance 0.25; the sloped-line hypothesis $\mathcal{H}_1$ gives bias 0.21, variance 1.69. Which has lower expected $E_{\text{out}}$ and what's the lesson?
> $E_{\text{out}}(\mathcal{H}_0) = 0.5 + 0.25 = 0.75$; $E_{\text{out}}(\mathcal{H}_1) = 0.21 + 1.69 = 1.90$. The constant-line wins despite having higher bias — its variance is so much smaller that the total error is lower. **The lesson: with few samples, prefer simpler models even if they have higher bias.** As $N$ grows, the variance of $\mathcal{H}_1$ shrinks (its slope estimate stabilises) and the more flexible model eventually wins. Optimal model complexity depends on $N$.

> [!question]- An algorithm has bias 0.05 and variance 0.30 on noisy data with $\sigma^2 = 0.10$. What's the expected $E_{\text{out}}$, and which term should you focus on reducing first?
> $\mathbb{E}_{\mathcal{D}}[E_{\text{out}}] = 0.05 + 0.30 + 0.10 = 0.45$. Variance is the dominant reducible term, so focus on variance reduction: more data, simpler model, regularisation, or averaging multiple models (ensembling). Bias is small enough that simplifying further would only hurt. Noise is a floor — there's no way to push below 0.10 without different (less noisy) data.

> [!question]- Why can you not, in general, measure bias and variance from a single training set, and what does this imply about how the decomposition is used in practice?
> Bias and variance are defined as expectations over the dataset distribution $\mathcal{D}$ — they require averaging the trained hypothesis $g^{(\mathcal{D})}$ over many independent draws of $\mathcal{D}$. With one fixed dataset, you have one $g$ and no way to estimate its expected behaviour. In practice this makes bias–variance a *conceptual* tool: you reason about whether your error pattern looks like high bias (under-flexible model, both errors high) or high variance (over-flexible, training error much lower than test) and choose the corresponding remedy. The actual bias and variance numbers stay implicit.
