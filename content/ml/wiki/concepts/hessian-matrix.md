---
type: concept
sources:
  - raw/week-02/Wk_2_Lec_2-1.pdf
  - raw/week-02/Tuesday_ October 7_ 2025 at 10_01_35 AM_Captions_English (United States).txt
status: draft
updated: 2026-04-26
---

*The matrix of second-order partial derivatives of a multivariate scalar function. Captures local curvature in every direction and the interactions between variables — the multivariate analogue of $f''(x)$.*

## Definition

For a twice-differentiable scalar function $f: \mathbb{R}^d \to \mathbb{R}$, the **Hessian matrix** at $\mathbf{x}$ is:

$$H_f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_0^2} & \frac{\partial^2 f}{\partial x_0 \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_0 \partial x_d} \\
\frac{\partial^2 f}{\partial x_1 \partial x_0} & \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_d} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_d \partial x_0} & \frac{\partial^2 f}{\partial x_d \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_d^2}
\end{bmatrix}$$

The diagonal entries $\partial^2 f / \partial x_i^2$ measure pure curvature along each axis. The off-diagonal entries $\partial^2 f / \partial x_i \partial x_j$ measure interactions: how the slope along axis $i$ changes as you move along axis $j$.

When $f$ is twice continuously differentiable, **mixed partials commute** ($\partial^2 f / \partial x_i \partial x_j = \partial^2 f / \partial x_j \partial x_i$), making $H$ symmetric.

## What the Hessian Tells You

The Hessian is a complete description of the local quadratic behaviour of $f$. The degree-2 [[taylor-polynomial|Taylor expansion]] around $\mathbf{x}_0$ is:

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top (\mathbf{x} - \mathbf{x}_0) + \tfrac{1}{2} (\mathbf{x} - \mathbf{x}_0)^\top H_f(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0)$$

The Hessian replaces $f''$ in the univariate version. Its eigenvalues encode curvature in the principal directions:

| Hessian property | Curvature | Critical point classification |
|---|---|---|
| All eigenvalues $> 0$ (positive definite) | Convex bowl | Local minimum |
| All eigenvalues $< 0$ (negative definite) | Concave dome | Local maximum |
| Mixed signs (indefinite) | Saddle | Saddle point |
| Some zero (semi-definite) | Flat directions | Inconclusive (degenerate) |

A function is [[convex-function|convex]] iff its Hessian is positive semi-definite everywhere; strictly convex iff positive definite.

## Use in Optimisation

The Hessian appears wherever curvature matters. In [[newton-raphson-method|Newton-Raphson]], the multivariate update is:

$$\mathbf{w} \leftarrow \mathbf{w} - H_E^{-1}(\mathbf{w}) \, \nabla E(\mathbf{w})$$

The inverse Hessian rescales the gradient direction to account for differential curvature: components of the gradient in *high-curvature* directions are shrunk (small steps in steep directions), while components in *low-curvature* directions are amplified (large steps in shallow directions). This is what makes Newton-Raphson converge in fewer iterations than [[gradient-descent-ml|gradient descent]] on ill-conditioned losses.

For [[logistic-regression]] with [[cross-entropy-loss]]:

$$H_E(\mathbf{w}) = \sum_{i=1}^{N} p_1(\mathbf{x}^{(i)}, \mathbf{w})(1 - p_1(\mathbf{x}^{(i)}, \mathbf{w})) \mathbf{x}^{(i)} \mathbf{x}^{(i)\top}$$

Each training example contributes a rank-1 outer product, weighted by $p_1(1 - p_1)$ — the variance of the predicted Bernoulli for that example. Examples the model is *uncertain* about ($p_1 \approx 0.5$, weight $\approx 0.25$) dominate; examples it is highly confident about ($p_1$ near 0 or 1, weight $\approx 0$) contribute almost nothing.

## Cost

Computing the Hessian and inverting it are expensive:

- **Storage**: $O(d^2)$ for $d$ parameters.
- **Inversion**: $O(d^3)$ per iteration in general.

For a model with $d = 10^6$ parameters (a small neural network), a $10^{12}$-entry matrix is infeasible to even store, let alone invert. This is why second-order methods stay in the realm of small-to-medium problems (logistic regression, GLMs) and first-order methods dominate deep learning.

Approximations like **L-BFGS** (limited-memory Broyden–Fletcher–Goldfarb–Shanno) keep some second-order benefit by maintaining a low-rank approximation to $H^{-1}$ rather than the full inverse Hessian.

## Related

- [[newton-raphson-method]] — uses the Hessian for second-order updates
- [[iteratively-reweighted-least-squares]] — Newton-Raphson applied to logistic regression
- [[taylor-polynomial]] — the Hessian appears in the degree-2 multivariate Taylor expansion
- [[convex-function]] — convexity is characterised by positive (semi-)definiteness of $H$

## Active Recall

> [!question]- For $f(\mathbf{w}) = w_1^2 + 4 w_2^2$, write the Hessian. What does it tell you about the loss surface?
> $\partial^2 f / \partial w_1^2 = 2$, $\partial^2 f / \partial w_2^2 = 8$, mixed partials are zero. So $H = \text{diag}(2, 8)$. The Hessian is constant (the loss is a perfect quadratic), positive definite (both eigenvalues $> 0$, so the function is strictly convex with a unique minimum), and *not* a multiple of the identity — curvature differs by a factor of 4 between axes. This is the canonical "differential curvature" example that breaks plain gradient descent.

> [!question]- Why does the Hessian inverse appear in the Newton-Raphson update rule rather than the Hessian itself?
> Setting the gradient of the degree-2 Taylor approximation to zero yields $\nabla E(\mathbf{w}_0) + H_E(\mathbf{w}_0) (\mathbf{w} - \mathbf{w}_0) = 0$. Solving for $\mathbf{w}$ requires "dividing" both sides by $H_E$ — i.e., multiplying by $H_E^{-1}$. Conceptually, the gradient says "move this much in this direction," and the inverse Hessian rescales that into the actual displacement that takes us to the parabola's minimum.

> [!question]- For logistic regression, why does the Hessian's per-example weighting $p_1(1 - p_1)$ make sense intuitively?
> $p_1(1-p_1)$ is the variance of the predicted Bernoulli — it measures how *uncertain* the model is about that example. When $p_1 \approx 0.5$, the model is unsure and the example is informative for learning the boundary; the weight is at its maximum of $0.25$. When $p_1 \approx 0$ or $1$, the model is confident and the example contributes negligibly to the curvature — moving the boundary slightly won't change the prediction. Newton-Raphson naturally focuses curvature information on examples near the decision boundary, which is where adjustments matter.
