---
type: concept
sources:
  - raw/Wk_1_Lec_2-1.pdf
  - raw/week-01/Tuesday_ September 30_ 2025 at 10_01_17 AM_Captions_English (United States).txt
status: draft
updated: 2026-04-20
---

*A smooth, S-shaped function that maps any real number to the interval $(0, 1)$, serving as the bridge between unbounded linear scores and valid probabilities.*

## Definition

The sigmoid (logistic) function is defined as:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Equivalently: $\sigma(z) = \frac{e^z}{1 + e^z}$.

![[JPEG_image-48C3-AF5E-04-0.jpeg]]

## Key Properties

| Property | Statement |
|---|---|
| Range | $\sigma: \mathbb{R} \to (0, 1)$ |
| Midpoint | $\sigma(0) = 0.5$ |
| Symmetry | $1 - \sigma(z) = \sigma(-z)$ |
| Monotonicity | Strictly increasing for all $z$ |
| Limits | $\lim_{z \to +\infty} \sigma(z) = 1$, $\lim_{z \to -\infty} \sigma(z) = 0$ |
| Derivative | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ |

The **symmetry property** is particularly useful. In [[logistic-regression]], it means:

$$P(y = 0 \mid \mathbf{x}) = 1 - \sigma(\mathbf{w}^\top \mathbf{x}) = \sigma(-\mathbf{w}^\top \mathbf{x})$$

so both class probabilities can be expressed through the same function.

## Role in Logistic Regression

In [[logistic-regression]], the linear combination $\mathbf{w}^\top \mathbf{x}$ can be any real number, but we need a probability in $[0, 1]$. The sigmoid provides exactly this mapping:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x})$$

The sigmoid is the **inverse of the logit function**: if $\text{logit}(p) = \ln \frac{p}{1-p} = z$, then $p = \sigma(z)$.

## Shape and Behaviour

The S-shape means the function is steepest around $z = 0$ (where it equals $0.5$) and flattens out at the extremes — saturating toward 0 and 1. Practically:

- $|z| > 5$: $\sigma(z)$ is effectively 0 or 1.
- $|z| < 1$: $\sigma(z)$ is roughly linear, centred on $0.5$.

This saturation is important: points far from the [[decision-boundary-ml|decision boundary]] (large $|\mathbf{w}^\top \mathbf{x}|$) get near-certain probabilities, while points near the boundary get probabilities close to $0.5$.

## Related

- [[logistic-regression]] — primary user of the sigmoid in this module
- [[decision-boundary-ml|decision boundary]] — the locus where $\sigma(\mathbf{w}^\top \mathbf{x}) = 0.5$

## Active Recall

> [!question]- Prove that $1 - \sigma(z) = \sigma(-z)$.
> $1 - \sigma(z) = 1 - \frac{1}{1+e^{-z}} = \frac{1+e^{-z} - 1}{1+e^{-z}} = \frac{e^{-z}}{1+e^{-z}}$. Multiply numerator and denominator by $e^z$: $= \frac{1}{e^z + 1} = \frac{1}{1 + e^{-(-z)}} = \sigma(-z)$.

> [!question]- If $\mathbf{w}^\top \mathbf{x} = 0$, what probability does the sigmoid assign, and what does this mean for classification?
> $\sigma(0) = 1/(1+e^0) = 1/2 = 0.5$. The model is maximally uncertain — it assigns equal probability to both classes. This point lies exactly on the [[decision-boundary-ml|decision boundary]].

> [!question]- Why does the sigmoid saturate (flatten) for large $|z|$, and what is the practical consequence for classification confidence?
> As $z \to +\infty$, $e^{-z} \to 0$, so $\sigma(z) \to 1$; as $z \to -\infty$, $e^{-z} \to \infty$, so $\sigma(z) \to 0$. The function approaches but never reaches 0 or 1. Practically, points with large $|\mathbf{w}^\top \mathbf{x}|$ — far from the decision boundary — get near-certain class probabilities, while points near the boundary (small $|\mathbf{w}^\top \mathbf{x}|$) stay close to $0.5$.
