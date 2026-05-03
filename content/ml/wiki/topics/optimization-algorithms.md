---
type: topic
name: Optimization Algorithms in This Module
description: Synthesis of the three optimizers covered so far — gradient descent, Newton-Raphson/IRLS, and SMO — and when each one is the right tool
status: draft
updated: 2026-04-27
---

*Three optimization algorithms appear in weeks 1–5, each chosen to fit the structure of a different learning problem. Gradient descent is the default for unconstrained smooth losses. Newton-Raphson (and its statistical incarnation IRLS) accelerates convergence using local curvature, but at a per-step cost. SMO solves a constrained QP that the other two can't handle directly. The choice is dictated by the loss surface, the constraints, and the dimensionality.*

## The Three Algorithms at a Glance

| Algorithm | Update rule | Uses | Constraints? | Per-step cost | Convergence |
|---|---|---|---|---|---|
| [[gradient-descent\|Gradient Descent]] | $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla E(\mathbf{w})$ | Gradient only | Unconstrained (penalty for soft) | $O(d)$ per gradient eval | Linear; tuning $\eta$ is awkward |
| [[newton-raphson-method\|Newton-Raphson]] / [[iteratively-reweighted-least-squares\|IRLS]] | $\mathbf{w} \leftarrow \mathbf{w} - H_E^{-1}(\mathbf{w}) \nabla E(\mathbf{w})$ | Gradient + Hessian | Unconstrained | $O(d^3)$ for Hessian inverse | Quadratic near optimum |
| [[sequential-minimal-optimization\|SMO]] | Pair-wise analytic update on $a^{(i)}, a^{(j)}$ | Kernel evaluations + KKT check | Box + linear equality | $O(N)$ per error update | Heuristic; depends on pair selection |

## Why Three Different Algorithms?

The module's three classifiers each present a *different* optimization problem:

| Classifier | Loss / objective | Optimizer | Why |
|---|---|---|---|
| [[logistic-regression]] | $-\log \mathcal{L}(\mathbf{w})$ — convex, smooth, unconstrained | GD or Newton/IRLS | Smooth + unconstrained → GD works; logistic loss has cheap Hessian → Newton wins |
| [[support-vector-machine]] (hard) | $\tfrac{1}{2}\|\mathbf{w}\|^2$ s.t. $y h \geq 1$ | Dualise → QP → SMO | Inequality constraints make GD inapplicable directly |
| [[soft-margin-svm]] | $\tfrac{1}{2}\|\mathbf{w}\|^2 + C \sum \xi$ s.t. $y h \geq 1 - \xi$ | Dualise → box-constrained QP → SMO | Same as above; box constraint makes generic QP slow at scale |

The pattern: **convex + smooth + unconstrained → GD/Newton; convex + constrained → dualise then SMO**.

## Gradient Descent — The Default

GD takes one principled step per iteration: move opposite the gradient, scaled by the learning rate $\eta$.

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla E(\mathbf{w})$$

**Strengths.**
- One hyperparameter ($\eta$).
- Works on any differentiable loss.
- Memory: $O(d)$. Scales to millions of parameters.

**Weaknesses.**
- Linear convergence — many iterations near the minimum where gradients shrink.
- Step size is uniform across directions. On an elliptical loss like $E = w_1^2 + 4 w_2^2$, the same $\eta$ that's safe in $w_2$ (large curvature) is too small in $w_1$ — and vice versa. The descent path zig-zags.
- Sensitive to feature scale and to the curvature of the loss surface.

**Where it appears in this module.**
Logistic regression with cross-entropy loss. The loss is convex, smooth, and unconstrained — exactly the setting GD handles cleanly.

## Newton-Raphson / IRLS — Curvature-Aware

Newton's update rescales the gradient by the inverse Hessian, so the step size is chosen *per direction* based on local curvature:

$$\mathbf{w} \leftarrow \mathbf{w} - H_E^{-1}(\mathbf{w}) \nabla E(\mathbf{w})$$

The intuition: a quadratic approximation of $E$ around the current $\mathbf{w}$ has its minimum at exactly this updated point. If $E$ *is* quadratic, Newton finds the optimum in **one step**. For non-quadratic but convex losses, you iterate until the quadratic approximation locks in.

**IRLS** is the same algorithm specialised to logistic-regression-style losses: the Hessian factorises into a re-weighted least-squares form, and each Newton step is computed by solving a weighted normal equation rather than inverting a $d \times d$ matrix from scratch.

**Strengths.**
- Quadratic convergence near the optimum — in practice, a handful of iterations.
- No learning rate to tune.
- Curvature-aware: handles ill-conditioned losses where GD zig-zags.

**Weaknesses.**
- $O(d^3)$ per iteration to invert (or solve via) the Hessian. Becomes infeasible past ~$10^4$ parameters.
- Requires the Hessian to be positive definite (true for convex losses; can fail for non-convex).
- Memory: $O(d^2)$ for the Hessian itself.

**Where it appears in this module.**
Logistic regression as IRLS — the standard fitting algorithm in `glm` / `statsmodels`. The "weighted least squares" view comes from substituting the logistic Hessian into Newton's update; the weights track per-example variance $p_i (1 - p_i)$.

## Sequential Minimal Optimization — For Constrained QPs

SMO is a *decomposition* method, not a generic optimizer. It exists because the SVM dual is a constrained QP:

$$\arg\max_{\mathbf{a}} \tilde{L}(\mathbf{a}) \quad \text{s.t. } 0 \leq a^{(n)} \leq C, \;\; \sum_n a^{(n)} y^{(n)} = 0$$

Generic QP solvers run in $O(N^3)$ time and $O(N^2)$ memory — both blow up past ~$10^4$ examples. SMO's trick: at each step, fix all but two multipliers $a^{(i)}, a^{(j)}$, and solve the resulting two-variable subproblem **analytically**.

**Why two and not one.** The equality constraint $\sum_n a^{(n)} y^{(n)} = 0$ pins any single multiplier in place — you can't change it without violating the constraint. Two is the minimum number that can move while preserving feasibility.

**Strengths.**
- No matrix factorisation. Each step is an analytic formula.
- Memory: $O(N)$ for the multipliers and cached errors. Doesn't store the full Gram matrix.
- Naturally maintains feasibility throughout.

**Weaknesses.**
- Pair-selection heuristics matter enormously. Lazy heuristics (random-random) are 10×–100× slower than the max-step heuristic.
- Convergence is heuristic, not provably fast. Stop on KKT-within-tolerance.
- Specialised to SVM-shaped duals — not a drop-in replacement for GD.

**Where it appears in this module.**
Hard- and soft-margin SVMs. The kernel trick lives inside SMO via $k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ in the analytic update — kernels and SMO are joined at the hip.

## Choosing Between Them

```
Convex, smooth, unconstrained?
├── Small d (≤ 10⁴) and Hessian cheap → Newton/IRLS
└── Large d → GD (or stochastic GD)

Convex with inequality + equality constraints?
└── Dualise → SVM-shaped QP → SMO
```

The constraint structure usually decides for you. The dimensionality breaks ties between GD and Newton.

## What's Common Across All Three

All three algorithms share the same outer skeleton: **start somewhere feasible, iterate until a stopping criterion is met.** The differences are in:

1. **What information they use.** GD uses the gradient. Newton uses gradient + Hessian. SMO uses kernel evaluations and the KKT residual.
2. **How they handle constraints.** GD ignores them (or absorbs them via penalty). Newton ignores them. SMO is constraint-native.
3. **Stopping criterion.** GD/Newton: gradient norm small. SMO: KKT conditions satisfied within tolerance.

The "find the optimum of a convex function" problem looks deceptively uniform on the surface — but the structure of the constraint set, the curvature of the loss, and the cost of computing derivatives all push you toward different algorithms.

## Open Questions for Later Weeks

- **Stochastic gradient descent** (SGD) — for losses defined as sums over millions of examples, mini-batches replace full-gradient evaluations. Will appear when the module covers neural networks.
- **Coordinate descent** for primal SVM with linear kernels — `LIBLINEAR`-style solvers beat SMO when the kernel is the identity.
- **Interior-point methods** as an alternative to SMO for small-to-medium QPs.
- **Convergence proofs** — we've taken on faith that SMO converges. The formal argument (Platt 1998) uses convexity of $\tilde{L}$ + careful pair selection.

## Connections

- [[gradient-descent-ml|gradient descent]] — the workhorse for unconstrained convex problems.
- [[newton-raphson-method]] — second-order method; inverts the Hessian.
- [[iteratively-reweighted-least-squares]] — Newton specialised to logistic losses.
- [[sequential-minimal-optimization]] — pair-wise analytic optimizer for SVM duals.
- [[hessian-matrix]] — what curvature-aware methods actually use.
- [[taylor-polynomial]] — the second-order Taylor expansion *is* Newton's update.
- [[quadratic-programming]] — the mathematical genus to which the SVM dual belongs.
- [[kkt-conditions]] — the optimality test that SMO checks at termination.
