---
type: concept
name: Karush–Kuhn–Tucker (KKT) Conditions
description: Necessary and sufficient optimality conditions for convex constrained optimisation; generalise "gradient = 0" to the constrained setting
sources:
  - raw/week-04/Wk_4_Lec_1-1.pdf
status: draft
updated: 2026-04-26
---

*For unconstrained convex optimisation, $\nabla F(\mathbf{x}) = 0$ is necessary and sufficient. Add inequality constraints, and a single gradient condition isn't enough — KKT extends optimality to four conditions that the primal and dual solutions must jointly satisfy.*

## The Conditions

For the problem $\min_\mathbf{x} F(\mathbf{x})$ subject to $f_i(\mathbf{x}) \leq 0$ (with convex $F$ and $f_i$), a primal solution $\mathbf{x}^*$ and dual multipliers $\mathbf{a}^*$ are jointly optimal iff:

**1. Stationarity.** The Lagrangian's gradient w.r.t. $\mathbf{x}$ vanishes:

$$\nabla_\mathbf{x} L(\mathbf{x}^*, \mathbf{a}^*) = \nabla F(\mathbf{x}^*) + \sum_i a_i^* \nabla f_i(\mathbf{x}^*) = 0$$

**2. Complementary slackness.** For every $i$:

$$a_i^* \, f_i(\mathbf{x}^*) = 0$$

Either $a_i^* = 0$ *or* $f_i(\mathbf{x}^*) = 0$ (or both). Constraints with $a_i^* > 0$ are *active* (binding with equality at the optimum); inactive constraints have $a_i^* = 0$.

**3. Primal feasibility.** $f_i(\mathbf{x}^*) \leq 0$ for all $i$.

**4. Dual feasibility.** $a_i^* \geq 0$ for all $i$.

## Why Complementary Slackness Matters

This is the structural condition that makes SVMs sparse. In SVM the constraint is $1 - y^{(n)} h(\mathbf{x}^{(n)}) \leq 0$ (i.e., the point is correctly classified beyond the margin). KKT says, for each training point, **either**:

- $a^{(n)} = 0$ — the point sits *strictly outside* the margin and contributes nothing to the dual sum, **or**
- $1 - y^{(n)} h(\mathbf{x}^{(n)}) = 0$ — the point is exactly *on* the margin: a **support vector**.

So at the optimum, only support vectors have $a^{(n)} > 0$. Every other training point is structurally invisible to the prediction function. This is the formal reason SVMs depend on a small subset of the data.

## Stationarity in Practice (SVM Worked Example)

For the SVM Lagrangian $L(\mathbf{w}, b, \mathbf{a}) = \tfrac{1}{2}\|\mathbf{w}\|^2 + \sum_n a^{(n)} (1 - y^{(n)}(\mathbf{w}^\top \phi(\mathbf{x}^{(n)}) + b))$:

- $\partial L / \partial \mathbf{w} = 0$ gives $\mathbf{w} = \sum_n a^{(n)} y^{(n)} \phi(\mathbf{x}^{(n)})$.
- $\partial L / \partial b = 0$ gives $\sum_n a^{(n)} y^{(n)} = 0$.

Substituting these back into $L$ eliminates $\mathbf{w}$ and $b$, producing the SVM dual problem in $\mathbf{a}$ alone. KKT stationarity is what does the elimination.

## Active Recall

> [!question]- Why is complementary slackness the structural reason SVMs are sparse, while logistic regression isn't?
> Logistic regression has no inequality constraints — every training point contributes a non-zero gradient term at every iteration, so every point shapes the final $\mathbf{w}$. SVM's margin constraint $1 - y^{(n)} h(\mathbf{x}^{(n)}) \leq 0$ is an inequality, and KKT forces $a^{(n)} (1 - y^{(n)} h(\mathbf{x}^{(n)})) = 0$. Most training points sit strictly inside their constraint ($f_i < 0$), forcing $a^{(n)} = 0$ — they drop out of the dual sum entirely. Sparsity is not an algorithmic accident; it's a structural consequence of having inequality constraints.

> [!question]- KKT requires both stationarity *and* complementary slackness. Why isn't stationarity enough on its own, like in unconstrained optimisation?
> Stationarity alone identifies critical points of the Lagrangian — but the Lagrangian is parameterised by the multipliers $\mathbf{a}$. Without complementary slackness, you could pick *any* $\mathbf{a} \geq 0$, find the corresponding stationary $\mathbf{x}$, and call it a solution — but most such pairs aren't primal-dual optimal. Complementary slackness is the link that ties $\mathbf{a}$ to the constraint geometry: $a_i$ can only be non-zero when constraint $i$ is *active*. Together with feasibility, the four conditions pin down the unique primal-dual pair.

## Related

- [[lagrangian]] — KKT conditions emerge from the Lagrangian framework
- [[support-vector-machine]] — KKT is what produces support-vector sparsity
- [[convex-function]] — KKT is necessary and sufficient only for convex problems; for non-convex it's only necessary
