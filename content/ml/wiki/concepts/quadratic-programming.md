---
type: concept
sources:
  - raw/week-03/Wk_3_Lec_2-1.pdf
  - raw/week-03/Tuesday_ October 14_ 2025 at 10_48_34 AM_Captions_English (United States).txt
status: draft
updated: 2026-04-26
---

*A class of optimisation problems with a quadratic objective and linear (in)equality constraints. Convex QPs have unique global optima and efficient off-the-shelf solvers, which is why many ML problems work hard to be expressible in QP form — most notably the [[support-vector-machine|SVM]] primal.*

## Definition

A **quadratic program** has the form:

$$\min_{\mathbf{x}} \tfrac{1}{2} \mathbf{x}^\top Q \mathbf{x} + \mathbf{c}^\top \mathbf{x}$$

subject to:

$$A \mathbf{x} \leq \mathbf{b} \quad \text{(linear inequalities)}$$
$$E \mathbf{x} = \mathbf{d} \quad \text{(linear equalities, optional)}$$

where $Q \in \mathbb{R}^{n \times n}$ is symmetric, $\mathbf{c} \in \mathbb{R}^n$, $A \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$, and similarly for $E, \mathbf{d}$.

The QP is **convex** iff $Q$ is positive semi-definite. Convex QPs are the well-behaved case: unique global optimum (assuming the feasible set is non-empty), no local minima to worry about, and a rich library of polynomial-time solvers.

## Why QPs Matter

Three properties make convex QPs especially useful in ML:

1. **Global optima.** Convexity guarantees that any local min is a global min. There is no "did we get stuck somewhere bad?" concern.
2. **Efficient solvers.** Interior-point methods, active-set methods, and other algorithms solve QPs reliably in polynomial time. Implementations in CVXPY, CVXOPT, MOSEK, Gurobi, etc. accept the problem in standard form and return the optimum.
3. **Sparse active sets.** At the optimum, only some of the inequality constraints are *active* (binding with equality). For SVMs, the active constraints are exactly the support vectors — a small fraction of the training set.

## The SVM Primal Is a QP

The hard-margin [[support-vector-machine|SVM]] formulation is:

$$\arg\min_{\mathbf{w}, b} \tfrac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to } y^{(n)} (\mathbf{w}^\top \mathbf{x}^{(n)} + b) \geq 1 \;\; \forall n$$

To match the standard QP form, let the unknown be $\mathbf{z} = (\mathbf{w}^\top, b)^\top \in \mathbb{R}^{d+1}$. Then:

- Objective: $\tfrac{1}{2}\|\mathbf{w}\|^2 = \tfrac{1}{2} \mathbf{z}^\top Q \mathbf{z}$ where $Q$ is the identity in the $\mathbf{w}$ block and zero in the $b$ entry.
- Constraints: $y^{(n)} (\mathbf{w}^\top \mathbf{x}^{(n)} + b) \geq 1$ rewrites as $-y^{(n)} \mathbf{x}^{(n)\top} \mathbf{w} - y^{(n)} b \leq -1$ for each $n$ — a linear inequality in $\mathbf{z}$.

$Q$ is positive semi-definite (the $\mathbf{w}$ block is identity, the $b$ entry is zero), the constraints are linear, so the SVM primal is a convex QP.

## Convexity at a Glance

For $\tfrac{1}{2} \mathbf{x}^\top Q \mathbf{x} + \mathbf{c}^\top \mathbf{x}$:

| $Q$ | Objective | Optimisation difficulty |
|---|---|---|
| Positive definite | Strictly convex | Easy — unique global min |
| Positive semi-definite | Convex (possibly with flat directions) | Easy — global min, may not be unique |
| Indefinite (mixed eigenvalues) | Saddle-shaped | NP-hard in general |
| Negative definite | Strictly concave | Easy — unique global *max* (so flip signs to use a min solver) |

The SVM primal lives squarely in the "easy" zone. By contrast, generic non-convex quadratic programs (e.g., with $Q$ indefinite) are NP-hard, and ML problems that *can* be expressed as them — such as discrete combinatorial relaxations — usually need approximation algorithms.

## Active Constraints and Support Vectors

At the optimum of a constrained problem, each inequality constraint $g_i(\mathbf{x}) \leq 0$ is either:

- **Active** (binding): $g_i(\mathbf{x}^*) = 0$. The constraint is "pulling" on the solution.
- **Inactive** (slack): $g_i(\mathbf{x}^*) < 0$. The constraint isn't a barrier; the solution would be the same without it.

For SVMs, the constraint $y^{(n)} h(\mathbf{x}^{(n)}) - 1 \geq 0$ is active when $y^{(n)} h(\mathbf{x}^{(n)}) = 1$ — i.e., when $\mathbf{x}^{(n)}$ is a support vector. The number of active constraints equals the number of support vectors, which is typically much smaller than $N$. This sparsity is what makes the SVM solution structurally informative.

The Karush-Kuhn-Tucker (KKT) conditions characterise optimality of constrained QPs and are the formal vehicle for stating "support vectors are exactly the points with active constraints." Treated in week 4 alongside the SVM dual.

## Other QPs in ML

QPs appear in several other places besides SVMs:

- **Lasso** (linear regression with $\ell_1$ regularisation) is a QP after introducing auxiliary variables.
- **Portfolio optimisation** in quantitative finance.
- **Constrained least-squares** problems with bounds or equality constraints.
- **Trust-region subproblems** inside many nonlinear optimisers.
- **Quadratic discriminant analysis** (some variants).

The pattern is general: any time you have a smooth quadratic objective and constraints describable by half-spaces, you have a QP.

## Related

- [[support-vector-machine]] — primary user of QP in this module
- [[convex-function]] — convexity of the objective is what makes the QP tractable
- [[newton-raphson-method]] — Newton-Raphson on a quadratic objective converges in one step (related but different setting: unconstrained)

## Active Recall

> [!question]- For a quadratic program $\min \tfrac{1}{2} \mathbf{x}^\top Q \mathbf{x} + \mathbf{c}^\top \mathbf{x}$ subject to $A\mathbf{x} \leq \mathbf{b}$, what condition on $Q$ guarantees a unique global minimum (assuming feasibility)?
> $Q$ must be positive *definite* (all eigenvalues strictly positive). Positive *semi*-definite (some eigenvalues zero) gives convexity and so a global min, but the optimum can lie on a flat ridge — the value is unique but the location may not be. The SVM primal has $Q$ that is positive semi-definite (zero in the $b$ direction), so the optimum value of $\|\mathbf{w}\|$ is unique, though strictly speaking the proof of uniqueness of $(\mathbf{w}^*, b^*)$ relies on the constraints fixing $b$.

> [!question]- The hard-margin SVM primal becomes infeasible when the data is not linearly separable — there is no $(\mathbf{w}, b)$ satisfying all the constraints. Why doesn't this break the QP formulation, and what's the standard fix?
> The QP machinery doesn't break — it correctly reports infeasibility, which is a meaningful result. The fix is **soft-margin SVM**: introduce non-negative slack variables $\xi^{(n)} \geq 0$, relax the constraint to $y^{(n)} h(\mathbf{x}^{(n)}) \geq 1 - \xi^{(n)}$, and add a penalty $C \sum_n \xi^{(n)}$ to the objective. The result is still a convex QP — now in $(\mathbf{w}, b, \boldsymbol{\xi})$ — that always has a feasible solution. $C$ trades margin width against misclassification cost.

> [!question]- Why are off-the-shelf QP solvers preferred over rolling your own gradient descent for the SVM problem?
> Three reasons. First, the SVM problem has *constraints*, and gradient descent has no built-in mechanism for handling them — you'd need projection or penalty methods that introduce their own approximations and tuning. Second, QP solvers exploit problem structure (sparsity, convexity, KKT conditions) to converge much faster than gradient descent, often in tens rather than thousands of iterations. Third, they handle numerical conditioning issues that would require manual care in a custom implementation. For ML researchers, "this reduces to a QP" is essentially a closed problem — pass it to CVXPY and move on.
