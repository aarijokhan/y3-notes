---
type: concept
name: Lagrangian and Lagrange Duality
description: Lagrange multipliers convert constrained optimisation into unconstrained — minimax/maxmin formulations and the conditions under which the dual equals the primal
sources:
  - raw/week-04/Wk_4_Lec_1-1.pdf
  - raw/week-04/4a-lagrange-dual-exercises.pdf
  - raw/week-04/4a-lagrange-dual-answers.pdf
status: draft
updated: 2026-04-26
---

*A trick for handling inequality constraints: instead of *enforcing* $f_i(\mathbf{x}) \leq 0$ from outside the optimisation, fold a penalty $a_i f_i(\mathbf{x})$ into the objective, with $a_i \geq 0$ a Lagrange multiplier. The resulting **Lagrangian** $L(\mathbf{x}, \mathbf{a}) = F(\mathbf{x}) + \sum_i a_i f_i(\mathbf{x})$ can be optimised by alternating min over $\mathbf{x}$ and max over $\mathbf{a}$ — and under the right conditions, swapping the order ("strong duality") gives an equivalent but often easier problem.*

## The Setup: Constrained Convex Optimisation

We have a primal problem:

$$\min_\mathbf{x} F(\mathbf{x}) \quad \text{subject to } f_i(\mathbf{x}) \leq 0, \;\; i \in \{1, \dots, N\}$$

We'd like to use unconstrained calculus ($\nabla F = 0$), but the constraints get in the way. They forbid certain regions of $\mathbf{x}$-space, and the unconstrained minimum may sit in a forbidden region.

## Lagrange Relaxation

Form the **Lagrangian**:

$$L(\mathbf{x}, \mathbf{a}) = F(\mathbf{x}) + \sum_{i=1}^N a_i f_i(\mathbf{x}), \qquad a_i \geq 0$$

The $a_i$ are **Lagrange multipliers**. They re-express the constraints as a penalty:

- If a constraint is *violated* ($f_i(\mathbf{x}) > 0$), the term $a_i f_i(\mathbf{x})$ is positive — it inflates the objective, pushing the optimiser away from infeasible $\mathbf{x}$.
- If a constraint is *satisfied* ($f_i(\mathbf{x}) \leq 0$), the term is $\leq 0$ — at best it helps the objective, at worst it does nothing.

## Minimax Primal Formulation

A naive Lagrangian relaxation has a problem: with fixed multipliers, the penalty for violation may be too small. The fix is to *maximise* over $\mathbf{a}$, then minimise over $\mathbf{x}$:

$$\min_\mathbf{x} \max_{\mathbf{a} \geq 0} L(\mathbf{x}, \mathbf{a})$$

Reading the inner max:

- If any constraint is *violated* ($f_i(\mathbf{x}) > 0$), the inner max can drive $a_i \to \infty$, sending $L \to \infty$. The outer min refuses to land there.
- If all constraints are *satisfied*, every $a_i f_i(\mathbf{x}) \leq 0$, so the inner max is achieved at $a_i = 0$ (or wherever $f_i = 0$), giving $L(\mathbf{x}, \mathbf{a}) = F(\mathbf{x})$.

So the minimax exactly reproduces the constrained primal — no information is lost.

## The Dual Formulation

The minimax requires solving a constrained max problem inside an unconstrained min — still awkward. The **dual** swaps the order:

$$\max_{\mathbf{a} \geq 0} \min_\mathbf{x} L(\mathbf{x}, \mathbf{a})$$

Now the inner min is *unconstrained* (no inequality constraints on $\mathbf{x}$ alone), so we can attack it with $\nabla_\mathbf{x} L = 0$. This often yields a closed-form expression for $\mathbf{x}^*$ in terms of $\mathbf{a}$, which we substitute back to get a problem in $\mathbf{a}$ alone.

### Weak Duality

Always true:

$$\max_{\mathbf{a}} \min_{\mathbf{x}} L \;\leq\; \min_\mathbf{x} \max_{\mathbf{a}} L$$

The dual gives a *lower bound* on the primal optimum. The gap between them is the **duality gap**.

### Strong Duality

When the gap is zero — primal and dual have the same optimal value — we say **strong duality** holds. Two sufficient conditions:

1. **$F$ and all $f_i$ are convex** ($L$ is convex in $\mathbf{x}$ for fixed $\mathbf{a}$, concave in $\mathbf{a}$ for fixed $\mathbf{x}$).
2. **Slater's condition**: there exists at least one strictly feasible $\mathbf{x}$ with $f_i(\mathbf{x}) < 0$ for all $i$.

Both hold for SVM and most ML problems, so we can freely move between primal and dual representations.

> [!tip] TIP — Saddle point picture
> When strong duality holds, the optimum is a **saddle point** of $L$: a minimum along the $\mathbf{x}$ axis and a maximum along the $\mathbf{a}$ axis. Whether you walk down then up (minimax) or up then down (maxmin), you arrive at the same point.

## Why Bother Going to the Dual?

Three reasons that show up in SVM:

1. **The inner min has closed form.** Setting $\nabla_\mathbf{x} L = 0$ and solving eliminates $\mathbf{x}$, leaving a problem purely in $\mathbf{a}$.
2. **The dual may have fewer variables.** SVM primal has $D + 1$ unknowns ($\mathbf{w}, b$ where $D = \dim(\phi)$); SVM dual has $N$ unknowns (one $a^{(n)}$ per training point). When $D \gg N$ (high-dimensional embedding, modest training set), the dual is dramatically smaller.
3. **The dual depends on $\mathbf{x}$ only through inner products $\phi(\mathbf{x}^{(i)})^\top \phi(\mathbf{x}^{(j)})$.** This opens the door to the [[kernel-trick]] — replace inner products with a kernel function and never compute $\phi$ at all.

## Worked Example: SVM Primal → Lagrangian → Minimax → Dual

The SVM constraint $y^{(n)}(\mathbf{w}^\top \phi(\mathbf{x}^{(n)}) + b) \geq 1$ is rewritten as $1 - y^{(n)}(\mathbf{w}^\top \phi(\mathbf{x}^{(n)}) + b) \leq 0$, which slots straight into the Lagrangian template:

![[svm-primal-to-dual-derivation.png]]

Setting up the minimax (and equivalently the dual maxmin) gives the saddle-point form whose inner min has closed-form solutions for $\mathbf{w}^*$ and the constraint $\sum_n a^{(n)} y^{(n)} = 0$:

![[svm-minimax-dual-formulation.png]]

## Active Recall

> [!question]- Why does the inner $\max_{\mathbf{a}}$ in the minimax formulation reproduce the original constraint exactly?
> Because the multipliers $a_i \geq 0$ are unbounded from above. If a constraint is violated ($f_i > 0$), the max wants to drive $a_i \to \infty$ to make the penalty arbitrarily large — the outer min then refuses to land on any infeasible $\mathbf{x}$. If a constraint is satisfied ($f_i \leq 0$), the max wants $a_i = 0$ to avoid making $L$ smaller than $F$. Either way, the minimax recovers the original primal: $L = F$ on the feasible set, $L = \infty$ outside it.

> [!question]- Strong duality fails for non-convex problems. What can the dual still tell us?
> Weak duality holds unconditionally — the dual is always a lower bound on the primal. Even when there's a duality gap, the dual gives a *certificate* of optimality (the primal can't do better than the dual value). For non-convex problems where the primal is intractable, solving the dual can still yield useful bounds on how good a heuristic primal solution is.

## Related

- [[kkt-conditions]] — the necessary and sufficient optimality conditions that primal/dual solutions must jointly satisfy
- [[support-vector-machine]] — the canonical ML use case
- [[kernel-trick]] — what becomes possible after going to the dual
- [[quadratic-programming]] — the structural form SVM lives in (quadratic objective, linear constraints)
