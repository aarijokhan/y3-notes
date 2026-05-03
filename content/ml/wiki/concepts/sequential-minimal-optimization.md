---
type: concept
name: Sequential Minimal Optimization (SMO)
description: Decomposition algorithm for solving the SVM dual QP by repeatedly optimising pairs of Lagrange multipliers analytically while keeping the rest fixed
sources:
  - raw/week-05/Wk_5_Lec_2-1.pdf
  - raw/week-05/5b-SMO-answers.pdf
status: draft
updated: 2026-04-27
---

*A decomposition method for the [[soft-margin-svm|soft-margin SVM]] dual: instead of solving the full $N$-variable QP at once, repeatedly pick a **pair** of Lagrange multipliers $(a^{(i)}, a^{(j)})$, solve the two-variable subproblem analytically, and iterate until KKT conditions are met within tolerance. Sidesteps the $O(N^2)$ memory and $O(N^3)$ time cost of off-the-shelf QP.*

## Why Decomposition

The soft-margin SVM dual

$$\arg\max_{\mathbf{a}} \;\; \sum_n a^{(n)} - \tfrac{1}{2} \sum_{n,m} a^{(n)} a^{(m)} y^{(n)} y^{(m)} k(\mathbf{x}^{(n)}, \mathbf{x}^{(m)})$$

subject to $0 \leq a^{(n)} \leq C$ and $\sum_n a^{(n)} y^{(n)} = 0$ is a convex [[quadratic-programming|QP]] with $N$ variables. Generic QP solvers store the Gram matrix ($N \times N$, often dense) and run in $O(N^3)$ — infeasible past tens of thousands of examples.

SMO observes that we can restrict attention to a small subset of variables at a time, fix the rest, and solve the resulting subproblem cheaply. With **two** variables it has a closed-form solution.

## Why Two, Not One

The equality constraint $\sum_n a^{(n)} y^{(n)} = 0$ couples all multipliers. If you change a *single* $a^{(m)}$ while keeping everything else fixed, the constraint forces:

$$a^{(m)} y^{(m)} = -\sum_{n \neq m} a^{(n)} y^{(n)} = \text{constant}$$

So $a^{(m)}$ is pinned. Updating one multiplier alone is impossible without violating feasibility. **Two** is the minimum number that can change while preserving $\sum_n a^{(n)} y^{(n)} = 0$ — a change in $a^{(i)}$ can be absorbed by a compensating change in $a^{(j)}$.

## The Algorithm

```
Initialise a^(n) = 0 for all n.       # feasible: satisfies both constraints
Repeat until convergence:
    Select pair (i, j) heuristically (see below).
    Compute candidate a^(j,new) by analytic update.
    Clip a^(j,new) to feasible interval [L, H].
    Set a^(i,new) so that ζ = a^(i) y^(i) + a^(j) y^(j) is preserved.
```

**Initialisation.** $a^{(n)} = 0$ for all $n$ trivially satisfies $0 \leq a^{(n)} \leq C$ and $\sum_n a^{(n)} y^{(n)} = 0$.

**Convergence.** Stop when no training example violates the KKT conditions within a specified tolerance $e$.

## The Two-Variable Update

Fix all $a^{(n)}$ for $n \neq i, j$. The equality constraint becomes:

$$a^{(i)} y^{(i)} + a^{(j)} y^{(j)} = \zeta \quad \text{(constant)}$$

so $a^{(i)}$ is determined by $a^{(j)}$. Substitute into the dual objective and differentiate w.r.t. $a^{(j)}$. The closed-form update is:

$$\boxed{a^{(j, \text{new})} = a^{(j)} + \frac{y^{(j)}(E^{(i)} - E^{(j)})}{k(\mathbf{x}^{(i)}, \mathbf{x}^{(i)}) + k(\mathbf{x}^{(j)}, \mathbf{x}^{(j)}) - 2 k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})}}$$

where $E^{(n)} = f(\mathbf{x}^{(n)}) - y^{(n)}$ is the prediction error on example $n$ (using the current $a$-values).

**Intuition.** The denominator is $\|\phi(\mathbf{x}^{(i)}) - \phi(\mathbf{x}^{(j)})\|^2$ — the squared distance between the two examples in feature space. The numerator scales with the disagreement between the two errors. Examples that are **far apart** and have **very different errors** produce the largest moves; nearby points or points already in agreement produce small ones.

## Clipping to the Box

The unclipped update may push $a^{(j, \text{new})}$ outside $[0, C]$, or push $a^{(i)}$ (computed from $a^{(j)}$) outside the box. Define:

- If $y^{(i)} = y^{(j)}$ (the line $a^{(i)} + a^{(j)} = \zeta/y^{(i)}$):
  $$L = \max(0, a^{(j)} + a^{(i)} - C), \quad H = \min(C, a^{(j)} + a^{(i)})$$
- If $y^{(i)} \neq y^{(j)}$ (the line $a^{(j)} - a^{(i)} = \zeta/y^{(j)}$):
  $$L = \max(0, a^{(j)} - a^{(i)}), \quad H = \min(C, a^{(j)} - a^{(i)} + C)$$

Clip the candidate:

$$a^{(j, \text{new\&clipped})} = \begin{cases} H & \text{if } a^{(j, \text{new})} \geq H \\ a^{(j, \text{new})} & \text{if } L < a^{(j, \text{new})} < H \\ L & \text{if } a^{(j, \text{new})} \leq L \end{cases}$$

Then recover $a^{(i, \text{new})}$ from the equality $a^{(i, \text{new})} y^{(i)} + a^{(j, \text{new\&clipped})} y^{(j)} = \zeta$.

## How to Pick the Pair

Selection is heuristic — convergence is guaranteed regardless, but speed depends heavily on the choice.

**Selecting $a^{(i)}$ (the first multiplier).** Alternate between two strategies:

1. Pick randomly among examples that violate the KKT conditions (within tolerance $e$).
2. Pick randomly among non-bound multipliers (those with $0 < a^{(i)} < C$ — i.e., margin support vectors) that violate KKT.

Strategy 2 keeps the search focused on the active set; strategy 1 occasionally checks bound multipliers in case they should leave the bound.

**Selecting $a^{(j)}$ (the second multiplier).** Try in order until a positive improvement in $\tilde{L}$ is observed:

1. **Maximum step heuristic.** Pick the $j$ that maximises $|E^{(i)} - E^{(j)}|$ — this approximates the largest update to $a^{(j)}$ (since the numerator of the SMO update is $y^{(j)}(E^{(i)} - E^{(j)})$).
2. Iterate over non-bound multipliers ($0 < a^{(j)} < C$) until improvement.
3. Iterate over the entire training set until improvement.
4. If still no improvement, replace $a^{(i)}$ and try again.

The errors $E^{(n)}$ are typically cached and updated incrementally so the heuristic is cheap.

## Convergence

The KKT conditions for soft-margin SVM, in terms of $a^{(n)}$:

| Condition | Meaning |
|---|---|
| $a^{(n)} = 0 \iff y^{(n)} h(\mathbf{x}^{(n)}) \geq 1$ | Non-SV: outside margin |
| $0 < a^{(n)} < C \iff y^{(n)} h(\mathbf{x}^{(n)}) = 1$ | Margin SV: on the margin exactly |
| $a^{(n)} = C \iff y^{(n)} h(\mathbf{x}^{(n)}) \leq 1$ | Bound SV: on, inside, or violating |

SMO checks each example against these (within tolerance $e$). When no example violates, the algorithm has found the optimum.

## Worked Example

Three points: $\mathbf{x}^{(1)} = (0.1, 0.1), y^{(1)} = +1$; $\mathbf{x}^{(2)} = (0.1, 0.2), y^{(2)} = +1$; $\mathbf{x}^{(3)} = (0.2, 0.2), y^{(3)} = -1$. Current $a^{(1)} = 0.4, a^{(2)} = 0, a^{(3)} = 0.4$, $C = 1$, linear kernel ($\phi(\mathbf{x}) = \mathbf{x}$). Pretend $h(\mathbf{x}^{(1)}) = 0.3$ and $h(\mathbf{x}^{(2)}) = -0.3$. Update the pair $(j = 1, i = 2)$.

Errors: $E^{(2)} = h(\mathbf{x}^{(2)}) - y^{(2)} = -0.3 - 1 = -1.3$ and $E^{(1)} = h(\mathbf{x}^{(1)}) - y^{(1)} = 0.3 - 1 = -0.7$.

Kernels: $k_{11} = 0.02$, $k_{22} = 0.05$, $k_{12} = 0.03$.

Denominator: $0.05 + 0.02 - 2(0.03) = 0.01$.

$$a^{(1, \text{new})} = 0.4 + \frac{1 \cdot (-1.3 - (-0.7))}{0.01} = 0.4 + \frac{-0.6}{0.01} = -59.6$$

Clip. Since $y^{(1)} = y^{(2)}$, $L = \max(0, 0.4 + 0 - 1) = 0$. The candidate $-59.6 < L$, so set $a^{(1, \text{new})} = 0$.

Recover $a^{(2)}$: $\zeta = 0.4 \cdot 1 + 0 \cdot 1 = 0.4$, so $a^{(2, \text{new})} = (0.4 - 0 \cdot 1) / 1 = 0.4$.

The pair $(0.4, 0)$ becomes $(0, 0.4)$ — multiplier mass shifted from example 1 to example 2.

## What Could Go Wrong

- **Numerical instability when $\mathbf{x}^{(i)} \approx \mathbf{x}^{(j)}$.** The denominator $k_{ii} + k_{jj} - 2 k_{ij}$ approaches zero, blowing up the update. Handle by skipping the pair.
- **Bad pair selection.** The algorithm always converges, but lazy heuristics (e.g., random both) can be orders of magnitude slower than the maximum-step heuristic.
- **Tolerance too tight.** Setting $e$ too small means pairs keep flipping with no real progress. The standard tolerance is around $10^{-3}$.

## Connections

- **Solves** [[soft-margin-svm]] — and hard-margin (set $C \to \infty$, drop the upper bound).
- **Uses** [[kkt-conditions]] — the optimality and stopping test.
- **Why two-variable subproblems are special** — the equality constraint $\sum a y = 0$ rules out one-variable updates; two is the minimum that preserves feasibility *and* admits a closed-form analytic update. Larger subsets would need numerical sub-solvers.
- **Alternatives** — interior-point methods, gradient projection, coordinate descent on the primal hinge-loss form. SMO dominates for kernel SVMs because of the analytic update; for linear SVMs, primal solvers like LIBLINEAR are typically faster.
