---
type: concept
sources:
  - raw/week-02/Wk_2_Lec_2-1.pdf
  - raw/week-02/Tuesday_ October 7_ 2025 at 10_01_35 AM_Captions_English (United States).txt
status: draft
updated: 2026-04-26
---

*A second-order iterative optimisation method: at each step, replace the loss with its degree-2 Taylor approximation and jump to that approximation's minimum. Uses curvature to set step size automatically — no learning rate to tune.*

## The Idea

[[gradient-descent-ml|gradient descent]] uses the slope (first derivative) and an externally-supplied step size. Its weakness: it has no idea how the slope is changing, so a single learning rate has to handle differential curvature poorly.

Newton-Raphson uses both the slope and the *curvature* (second derivative). At each iterate, it builds a quadratic approximation of the loss using a degree-2 [[taylor-polynomial]], then jumps directly to the minimum of that quadratic.

## Univariate Update

The degree-2 Taylor approximation of $E(w)$ around the current iterate $w_0$:

$$T_2(w) = E(w_0) + (w - w_0) E'(w_0) + \tfrac{(w - w_0)^2}{2} E''(w_0)$$

This is a parabola. Setting its derivative to zero:

$$E'(w_0) + (w - w_0) E''(w_0) = 0 \implies w = w_0 - \frac{E'(w_0)}{E''(w_0)}$$

So the update rule is:

$$w \leftarrow w - \frac{E'(w)}{E''(w)}$$

Reading the formula:
- **Direction**: the sign of $E'(w)$ — same as gradient descent — so we move opposite the gradient (assuming we're in a convex region with $E'' > 0$).
- **Step size**: $1 / E''(w)$ — automatically inversely proportional to curvature. High curvature → small step; low curvature → large step. **No learning rate.**

## Multivariate Update

For multi-dimensional $\mathbf{w}$, the second derivative becomes the [[hessian-matrix|Hessian]]:

$$\mathbf{w} \leftarrow \mathbf{w} - H_E^{-1}(\mathbf{w}) \, \nabla E(\mathbf{w})$$

The inverse Hessian rescales the gradient component-by-component (in the eigenbasis of $H$): it shrinks updates in steep (high-curvature) directions and stretches updates in shallow (low-curvature) directions, exactly cancelling the differential-curvature pathology that plagues gradient descent.

## Why "Iterative"?

If the loss were *exactly* quadratic, Newton-Raphson would reach the minimum in **a single step** — the parabola we approximate with is the function itself.

For real losses (like [[cross-entropy-loss]]), the quadratic is only a *local* approximation. After one Newton step, we recompute the gradient and Hessian at the new point and rebuild the quadratic there, then step again. The procedure converges, typically in far fewer iterations than gradient descent, because each iteration uses a much better local model.

## Convergence Behaviour

Near a (well-conditioned) minimum, Newton-Raphson exhibits **quadratic convergence**: the number of correct digits roughly doubles each iteration. Gradient descent typically achieves only linear convergence (correct digits grow linearly).

Caveats:

- **Non-convex regions**: if the Hessian is not positive definite, the "Newton step" can point *uphill*. Modifications (damped Newton, trust regions, adding a multiple of the identity to $H$) handle this.
- **Far from the optimum**: the quadratic approximation may be poor, and a Newton step can overshoot. A learning rate can be reintroduced as a safety knob: $\mathbf{w} \leftarrow \mathbf{w} - \eta H^{-1} \nabla E$ with $\eta \in (0, 1)$.
- **Cost**: $O(d^3)$ per iteration to invert $H$, and $O(d^2)$ memory. For high-dimensional models this is prohibitive.

## When to Use Newton-Raphson

| Setting | Newton-Raphson? |
|---|---|
| Logistic regression, small to moderate $d$ | Yes — fast convergence, no learning-rate tuning |
| GLMs (Poisson, Gamma, etc.) | Yes — same idea |
| Deep neural networks | Rarely — Hessian too expensive, often non-convex |
| Convex problems with $d \lesssim 10^4$ | Yes — feasible and reliable |
| Problems where Hessian has nice structure | Yes — exploit structure for efficiency |

For logistic regression, applying Newton-Raphson gives the algorithm called [[iteratively-reweighted-least-squares|Iteratively Reweighted Least Squares]].

## Related

- [[iteratively-reweighted-least-squares]] — Newton-Raphson applied to logistic regression
- [[taylor-polynomial]] — the degree-2 approximation Newton-Raphson is built on
- [[hessian-matrix]] — the multivariate second-derivative used in the update
- [[gradient-descent-ml|gradient descent]] — first-order alternative; cheaper per step but slower to converge

## Active Recall

> [!question]- Derive the Newton-Raphson update rule from the degree-2 Taylor polynomial of $E(w)$ around $w_0$.
> Write $T_2(w) = E(w_0) + (w - w_0) E'(w_0) + \tfrac{(w - w_0)^2}{2} E''(w_0)$. To find its minimum, differentiate with respect to $w$: $T_2'(w) = E'(w_0) + (w - w_0) E''(w_0)$. Set to zero: $E'(w_0) + (w - w_0) E''(w_0) = 0 \Rightarrow w - w_0 = -E'(w_0)/E''(w_0) \Rightarrow w = w_0 - E'(w_0)/E''(w_0)$. Replacing $w_0$ with the current iterate gives the update rule $w \leftarrow w - E'(w)/E''(w)$.

> [!question]- Why does Newton-Raphson not need a learning rate, while gradient descent does?
> Gradient descent uses only the slope, which gives a direction but not a magnitude — there's no information available to decide how *far* to move. Newton-Raphson uses the curvature (second derivative or Hessian), which sets the step size automatically: a parabola with curvature $E''$ has its minimum at distance $E'/E''$ from the current point. The second derivative supplies the missing scale.

> [!question]- Newton-Raphson converges quadratically — much faster than gradient descent. Why don't deep learning practitioners use it?
> Two reasons. First, the Hessian for a deep network with millions of parameters has a $10^{12}$-entry matrix that is infeasible to store, let alone invert at $O(d^3)$ per iteration. Second, deep network losses are non-convex, so the Hessian is often not positive definite — the Newton step can point *uphill*. First-order methods (gradient descent, Adam) trade per-iteration progress for tractable per-iteration cost and avoid the non-convexity pitfall. Approximations like L-BFGS retrieve some Newton-Raphson benefit at lower cost for medium-scale problems.
