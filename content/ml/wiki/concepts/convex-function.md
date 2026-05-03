---
type: concept
sources:
  - raw/week-02/Wk_2_Lec_1-1.pdf
  - raw/week-02/Monday_ October 6_ 2025 at 4_14_08 PM_Captions_English (United States).txt
status: draft
updated: 2026-04-26
---

*A function is convex if any line segment between two points on its graph lies above (or on) the graph. Convexity is the property that makes optimisation tractable — every local minimum is a global minimum.*

## Definition

A function $f: \mathbb{R}^d \to \mathbb{R}$ is **convex** if, for all $\mathbf{x}, \mathbf{y}$ in its domain and $\lambda \in [0, 1]$:

$$f(\lambda \mathbf{x} + (1 - \lambda) \mathbf{y}) \leq \lambda f(\mathbf{x}) + (1 - \lambda) f(\mathbf{y})$$

Geometrically: the chord connecting any two points on the graph never dips below the graph.

![[Screenshot_2025-10-19_at_3.49.49_pm.png]] **Strict convexity** replaces $\leq$ with $<$ (for $\lambda \in (0, 1)$ and $\mathbf{x} \neq \mathbf{y}$), ruling out flat regions.

A **concave** function is just the negative of a convex one (chord lies *below* the graph). Maximising a concave function is the same problem as minimising a convex one.

## Why Convexity Matters

The key consequence: **for a convex function, every local minimum is a global minimum.** If gradient descent converges to a critical point ($\nabla f = \mathbf{0}$), that point is the global optimum — no worry about being trapped in a worse local minimum.

For *strictly* convex functions, the global minimum is also unique (no flat plateaus or multiple equally-good optima).

## Tests for Convexity

**For univariate, twice-differentiable $f$:** $f$ is convex iff $f''(x) \geq 0$ everywhere. Strictly convex iff $f''(x) > 0$ everywhere.

**For multivariate, twice-differentiable $f$:** $f$ is convex iff its [[hessian-matrix|Hessian]] $H$ is positive semi-definite at every point ($\mathbf{v}^\top H \mathbf{v} \geq 0$ for all $\mathbf{v}$). Strictly convex iff $H$ is positive definite (strict inequality for $\mathbf{v} \neq \mathbf{0}$).

**Subgradient test (for non-differentiable convex functions):** any *subgradient* — a linear function that touches the graph at a point and lies below it everywhere — exists at every point in the interior of the domain. If $\mathbf{0}$ is a subgradient at $\mathbf{x}^0$, then $\mathbf{x}^0$ is a global minimum.

## Examples

| Function | Convex? | Notes |
|---|---|---|
| $f(x) = x^2$ | Yes (strictly) | $f'' = 2 > 0$ |
| $f(x) = e^x$ | Yes (strictly) | $f'' = e^x > 0$ |
| $f(x) = -\ln x$ | Yes (strictly) on $(0, \infty)$ | $f'' = 1/x^2 > 0$ |
| $f(x) = \|x\|$ | Yes | Convex but not differentiable at 0; subgradient exists |
| $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ | Yes (strictly) | Bowl |
| $f(x) = \sin x$ | No | Curvature flips sign |

## Convexity in Machine Learning

The optimisation problems we want to solve are convex if both the loss and any regularisers are convex (and combined with non-negative weights). Convex problems include:

- **Linear regression** with squared error.
- **[[logistic-regression|Logistic regression]]** with [[cross-entropy-loss|cross-entropy loss]] — strictly convex in $\mathbf{w}$.
- **Support Vector Machines** with hinge loss + L2 regularisation.
- **Lasso** (linear regression + L1 regularisation).

Non-convex problems include neural networks (the network's output is a non-convex function of its weights), most clustering objectives (K-means is non-convex), and many latent-variable models.

![[Screenshot_2025-10-19_at_3.50.12_pm.png]]

![[Screenshot_2025-10-19_at_3.52.33_pm.png]]

When the problem is convex, [[gradient-descent-ml|gradient descent]] and [[newton-raphson-method|Newton-Raphson]] are guaranteed to find the global optimum (under mild conditions). When it is not, optimisation becomes a heuristic search — gradient descent finds *some* local minimum, and we hope it's a good one.

## Operations That Preserve Convexity

- Sum of convex functions is convex.
- Maximum of convex functions is convex (e.g., $\max(0, x)$ — the ReLU).
- Composition with an affine function: $f(A\mathbf{x} + \mathbf{b})$ is convex if $f$ is.
- Non-negative scaling: $\alpha f$ is convex for $\alpha \geq 0$.

These rules let you build complex convex losses from simple convex pieces.

## Related

- [[cross-entropy-loss]] — strictly convex in $\mathbf{w}$, hence has a unique global minimum
- [[gradient-descent-ml|gradient descent]] — succeeds reliably on convex losses
- [[newton-raphson-method]] — guaranteed to descend on (locally) convex regions
- [[hessian-matrix]] — positive semi-definite Hessian characterises convexity

## Active Recall

> [!question]- Why does convexity make optimisation easy?
> For a convex function, every local minimum is a global minimum. So any iterative method that descends ([[gradient-descent-ml|gradient descent]], [[newton-raphson-method|Newton-Raphson]]) and converges to a critical point is guaranteed to land on the global optimum. There is no risk of getting trapped in a worse local minimum, and no need for expensive techniques like random restarts or simulated annealing.

> [!question]- How can you tell if a twice-differentiable multivariate function is convex?
> Check whether its [[hessian-matrix|Hessian]] is positive semi-definite at every point in the domain — i.e., $\mathbf{v}^\top H(\mathbf{x}) \mathbf{v} \geq 0$ for all $\mathbf{v}$ and all $\mathbf{x}$. For strict convexity, replace $\geq$ with $>$ (for $\mathbf{v} \neq \mathbf{0}$). Equivalently: all eigenvalues of $H$ are non-negative (positive for strict convexity).

> [!question]- Is $f(x) = \|x\|$ (absolute value) convex? Differentiable? How would gradient-based optimisation handle it?
> Yes, it's convex (the chord between any two points lies above or on the graph). But it's not differentiable at $x = 0$. Standard gradient descent fails there because there's no unique gradient. The fix is the *subgradient*: any line that touches the graph at $x = 0$ and lies below it (any slope between $-1$ and $+1$) qualifies as a subgradient, and subgradient descent uses one to determine the update direction. At $x = 0$, the subgradient $0$ exists, signalling that we're at the global minimum.
