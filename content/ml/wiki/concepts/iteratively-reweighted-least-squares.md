---
type: concept
sources:
  - raw/week-02/Wk_2_Lec_2-1.pdf
  - raw/week-02/Tuesday_ October 7_ 2025 at 10_01_35 AM_Captions_English (United States).txt
status: draft
updated: 2026-04-26
---

*The [[newton-raphson-method|Newton-Raphson]] method applied to the [[cross-entropy-loss]] for [[logistic-regression]]. Each iteration solves a weighted least-squares problem whose weights depend on the model's current uncertainty, hence "iteratively reweighted."*

## The Update Rule

For [[logistic-regression]] with [[cross-entropy-loss]], the gradient and [[hessian-matrix|Hessian]] are:

$$\nabla E(\mathbf{w}) = \sum_{i=1}^{N} (p_1(\mathbf{x}^{(i)}, \mathbf{w}) - y^{(i)}) \mathbf{x}^{(i)}$$

$$H_E(\mathbf{w}) = \sum_{i=1}^{N} p_1(\mathbf{x}^{(i)}, \mathbf{w})(1 - p_1(\mathbf{x}^{(i)}, \mathbf{w})) \mathbf{x}^{(i)} \mathbf{x}^{(i)\top}$$

The Newton-Raphson update is:

$$\mathbf{w} \leftarrow \mathbf{w} - H_E^{-1}(\mathbf{w}) \, \nabla E(\mathbf{w})$$

Both $\nabla E$ and $H_E$ depend on $\mathbf{w}$ (through $p_1$), so they must be recomputed at each iteration — making the procedure iterative.

## Why "Reweighted Least Squares"?

The Hessian above can be written compactly. Let $R(\mathbf{w})$ be the diagonal matrix with $R_{ii} = p_1(\mathbf{x}^{(i)}, \mathbf{w})(1 - p_1(\mathbf{x}^{(i)}, \mathbf{w}))$, and let $\mathbf{X}$ be the $N \times (d+1)$ design matrix with rows $\mathbf{x}^{(i)\top}$. Then:

$$H_E(\mathbf{w}) = \mathbf{X}^\top R(\mathbf{w}) \mathbf{X}$$

This has the same form as the normal-equations matrix in **weighted least squares**, where each example $i$ is weighted by $R_{ii}$. The Newton-Raphson update can be rearranged into a weighted least-squares problem at each iteration, with weights $R(\mathbf{w})$ that change as $\mathbf{w}$ changes — hence "iteratively reweighted least squares."

The weights have a clean meaning: $p_1(1 - p_1)$ is the variance of the predicted Bernoulli for example $i$ — i.e., how *uncertain* the model is about that example.

| Predicted $p_1$ | Weight $p_1(1-p_1)$ | Meaning |
|---|---|---|
| $0.5$ | $0.25$ (max) | Model is at chance — example highly informative |
| $0.1$ or $0.9$ | $0.09$ | Reasonably confident |
| $0.01$ or $0.99$ | $0.0099$ | Very confident — example contributes little |
| $\to 0$ or $\to 1$ | $\to 0$ | Saturated — no influence on update |

So at every iteration, IRLS pays the most attention to examples near the decision boundary (where the model is unsure) and largely ignores examples it is already confident about. As the model fits the data better, the set of "hard" examples evolves, and the weights shift accordingly.

## Why It Beats Gradient Descent

Gradient descent struggles with the differential curvature of the loss surface — eigenvalues of the Hessian differ across directions, so a single learning rate can't be both aggressive enough to make progress in shallow directions and cautious enough to avoid overshooting in steep directions.

IRLS sidesteps this entirely: the inverse Hessian rescales the gradient direction-by-direction. Steps are automatically:

- **Shrunk** in directions with high curvature (steep walls of the loss surface).
- **Stretched** in directions with low curvature (flat valleys).

The result is far fewer iterations, especially when features are correlated or have very different scales.

## Why It Works for Logistic Regression

Two properties make IRLS especially well-suited to logistic regression:

1. **The cross-entropy loss is strictly [[convex-function|convex]]** in $\mathbf{w}$. The Hessian is positive definite (well, semi-definite — it can be singular if a feature is collinear with another), so the Newton step always points downhill. No worry about non-convexity-induced uphill steps.
2. **The quadratic approximation of cross-entropy is reasonably accurate.** The deviations from quadratic are not too large. If they were, a learning rate could be reintroduced for safety: $\mathbf{w} \leftarrow \mathbf{w} - \eta H_E^{-1} \nabla E$.

Convergence is typically extremely fast — a handful of iterations rather than the hundreds or thousands gradient descent might need.

## Cost

Each IRLS iteration requires:

- **$O(N d^2)$** to assemble $H_E = \mathbf{X}^\top R \mathbf{X}$.
- **$O(d^3)$** to invert the $(d+1) \times (d+1)$ Hessian.

For modest $d$ (up to perhaps a few thousand features), this is fast. For high-dimensional problems (text classification with millions of features, image features), the $O(d^3)$ per iteration becomes prohibitive and gradient descent (or stochastic variants) wins despite needing more iterations.

## Related

- [[newton-raphson-method]] — IRLS is the application of Newton-Raphson to logistic regression
- [[logistic-regression]] — the model IRLS fits
- [[cross-entropy-loss]] — the loss IRLS minimises
- [[hessian-matrix]] — the matrix IRLS inverts at every iteration
- [[gradient-descent-ml|gradient descent]] — cheaper alternative when $d$ is large

## Active Recall

> [!question]- Why does the Hessian for logistic regression weight each training example by $p_1(1 - p_1)$? What does that imply about which examples drive the update at each iteration?
> $p_1(1 - p_1)$ is the variance of the predicted Bernoulli — it measures the model's *uncertainty* about that example. Examples near the decision boundary have $p_1 \approx 0.5$ and weight close to $0.25$ (the maximum). Examples the model is highly confident about ($p_1$ near $0$ or $1$) have weight near zero. So IRLS effectively focuses on the "hard" examples — those near the boundary — and ignores the easy ones, until the boundary moves and a new set of examples becomes hard.

> [!question]- How does IRLS avoid the differential-curvature problem that hobbles gradient descent?
> Gradient descent uses a single scalar learning rate $\eta$ that scales every dimension of the update equally. When the loss has very different curvatures along different axes (eigenvalues of $H$ differ wildly), no single $\eta$ works — too aggressive in steep directions, too timid in shallow ones. IRLS multiplies the gradient by $H^{-1}$, which in the eigenbasis of $H$ divides each component by the corresponding eigenvalue. Steep directions get shrunk; shallow directions get stretched; the result is a Newton step that points more or less straight at the minimum.

> [!question]- IRLS converges quadratically near the optimum and typically needs far fewer iterations than gradient descent. Why isn't IRLS used everywhere?
> Cost per iteration. Each IRLS step requires assembling and inverting the $(d+1) \times (d+1)$ Hessian — $O(d^3)$ work and $O(d^2)$ memory. For logistic regression with hundreds or thousands of features, that's fine. For models with millions of parameters (deep networks, high-dimensional sparse features), the per-iteration cost dwarfs the savings from fewer iterations, and gradient descent (cheap per step, many steps) wins on wall-clock time.
