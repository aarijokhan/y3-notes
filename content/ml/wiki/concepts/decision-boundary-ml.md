---
type: concept
sources:
  - raw/Wk_1_Lec_2-1.pdf
  - raw/week-01/Tuesday_ September 30_ 2025 at 10_01_17 AM_Captions_English (United States).txt
status: draft
updated: 2026-04-20
---

*The surface in input space that separates regions assigned to different classes — for linear classifiers, a hyperplane defined by $\mathbf{w}^\top \mathbf{x} = 0$.*

## Definition

In a binary classifier, the **decision boundary** is the set of points $\mathbf{x}$ where the classifier transitions from predicting one class to the other. For [[logistic-regression]] and other linear classifiers, this boundary is the hyperplane:

$$\mathbf{w}^\top \mathbf{x} = w_0 x_0 + w_1 x_1 + \cdots + w_d x_d = 0$$

where $x_0 = 1$ (dummy variable for the bias).

## Geometry

The dimensionality of the boundary is always one less than the input space:

| Input dimensions $d$ | Boundary |
|---|---|
| 2 | Line |
| 3 | Plane |
| $d$ | $(d{-}1)$-dimensional hyperplane |

The weight vector $\mathbf{w}$ (excluding $w_0$) is the **normal** to the hyperplane — it points toward the class-1 side. The bias $w_0$ shifts the hyperplane away from the origin.

## Distance and Confidence

In [[logistic-regression]], the signed quantity $\mathbf{w}^\top \mathbf{x}$ measures how far $\mathbf{x}$ is from the boundary (up to scaling by $\|\mathbf{w}\|$):

- $\mathbf{w}^\top \mathbf{x} \gg 0$: far into the class-1 region, $P(y=1) \approx 1$ — high confidence.
- $\mathbf{w}^\top \mathbf{x} \ll 0$: far into the class-0 region, $P(y=1) \approx 0$ — high confidence.
- $\mathbf{w}^\top \mathbf{x} \approx 0$: near the boundary, $P(y=1) \approx 0.5$ — maximum uncertainty.

The [[sigmoid-function-ml|sigmoid function]] translates this signed distance into a smooth probability.

## Linearity and Its Limits

Logistic regression produces a **linear** boundary regardless of the data. If the true classes are not linearly separable (e.g., one class surrounds the other), a linear boundary cannot correctly classify all points. Two strategies for handling this appear later in the module:

1. **Non-linear feature transformations** — map inputs to a higher-dimensional space where the classes become linearly separable.
2. **Kernel methods / SVMs** — implicitly work in a transformed space without computing the transformation directly.

## Margin

Logistic regression finds *a* separating hyperplane but does not optimize the **margin** — the minimum distance from the boundary to the nearest training point. A larger margin means the classifier is more robust to small perturbations in the data. Support Vector Machines (covered in weeks 3–5) explicitly maximize the margin.

## Related

- [[logistic-regression]] — produces a linear decision boundary
- [[sigmoid-function-ml|sigmoid function]] — converts distance from boundary to probability
- [[generalization]] — margin and boundary placement affect generalization

## Active Recall

> [!question]- For a logistic regression model with $\mathbf{w}^\top = (1, -2, 3)$, what is the equation of the decision boundary in terms of $x_0, x_1, x_2$? Sketch (verbally) what it looks like in $(x_1, x_2)$ space.
> The boundary is $1 \cdot x_0 + (-2) \cdot x_1 + 3 \cdot x_2 = 0$. Since $x_0 = 1$, this simplifies to $-2x_1 + 3x_2 + 1 = 0$, or $x_2 = \frac{2x_1 - 1}{3}$. In the $(x_1, x_2)$ plane, this is a straight line with slope $2/3$ and $x_2$-intercept $-1/3$.

> [!question]- Why does $\mathbf{w}^\top \mathbf{x} \approx 0$ correspond to maximum classification uncertainty in logistic regression?
> When $\mathbf{w}^\top \mathbf{x} = 0$, the sigmoid outputs $\sigma(0) = 0.5$, meaning both classes are equally likely. The point sits exactly on the decision boundary. As $|\mathbf{w}^\top \mathbf{x}|$ grows, the sigmoid saturates toward 0 or 1, and uncertainty decreases.

> [!question]- Logistic regression can only produce linear decision boundaries. Describe a data layout where this fails, and name two strategies for overcoming the limitation.
> If class 1 forms a cluster in the centre of the input space and class 0 surrounds it (e.g., concentric circles), no single line or hyperplane can separate them. Two strategies: (1) apply a non-linear feature transformation to map the data into a higher-dimensional space where a linear boundary suffices; (2) use kernel methods (e.g., SVMs with RBF kernel) to implicitly work in the transformed space.
