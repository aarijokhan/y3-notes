---
type: concept
name: soft-margin SVM
description: SVM variant that introduces slack variables ξ and a regularisation hyperparameter C, allowing some training examples to violate the margin constraint
sources:
  - raw/week-05/Wk_5_Lec_1-1.pdf
  - raw/week-05/5a-soft-margin-svm-answers.pdf
status: draft
updated: 2026-04-27
---

*An extension of the hard-margin [[support-vector-machine|SVM]] that permits training points to sit inside the margin or even on the wrong side of the decision boundary, paying a per-example penalty $\xi^{(n)} \geq 0$ for each violation. A hyperparameter $C > 0$ controls the trade-off between margin width and total violation.*

## Why Hard-Margin Isn't Enough

The hard-margin SVM solves:

$$\arg\min_{\mathbf{w}, b} \tfrac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t. } y^{(n)}(\mathbf{w}^\top \phi(\mathbf{x}^{(n)}) + b) \geq 1 \;\; \forall n$$

Two failure modes:

1. **Non-separable data.** If no hyperplane in $\phi$-space separates the classes, the constraint set is empty and the problem has no feasible solution.
2. **Overfitting on barely-separable data.** Even when the data *is* separable, a single noisy or mislabelled point near the boundary forces the hyperplane into a contorted position with a tiny margin. With kernels (especially Gaussian), $\phi$-space is high- or infinite-dimensional, and *some* separating boundary almost always exists — but it's the wrong one.

The fix: stop demanding perfect separation. Allow each example a controlled amount of margin violation.

## The Primal Formulation

Introduce a [[slack-variables|slack variable]] $\xi^{(n)} \geq 0$ for each training example, relaxing the margin constraint:

$$y^{(n)}(\mathbf{w}^\top \phi(\mathbf{x}^{(n)}) + b) \geq 1 - \xi^{(n)}$$

Penalise total slack in the objective:

$$\boxed{\arg\min_{\mathbf{w}, b, \boldsymbol{\xi}} \;\; \tfrac{1}{2}\|\mathbf{w}\|^2 + C \sum_{n=1}^N \xi^{(n)} \quad \text{s.t. } y^{(n)} h(\mathbf{x}^{(n)}) \geq 1 - \xi^{(n)}, \;\; \xi^{(n)} \geq 0}$$

Three things changed from hard-margin:

1. The constraint right-hand side is now $1 - \xi^{(n)}$ instead of $1$ — points can violate the margin by an amount $\xi^{(n)}$.
2. The objective gains $C \sum_n \xi^{(n)}$ — total violation is penalised.
3. The optimisation now has $N$ extra variables ($\boldsymbol{\xi}$).

The margin is now simply $1/\|\mathbf{w}\|$: there is no longer a "closest training point" definition because some points might be inside the margin.

## The Hyperparameter $C$

$C$ trades off margin width against violation tolerance:

| Regime | Behaviour | Risk |
|---|---|---|
| $C \to \infty$ | Slacks heavily penalised; recovers hard-margin behaviour | Overfit, especially with high-capacity kernels |
| Large $C$ | Few/small violations; narrow margin | Overfit, may not generalise |
| Small $C$ | Many violations allowed; wide margin | Underfit, may misclassify too aggressively |
| $C \to 0$ | Slack is free; margin maximised at all costs | Boundary collapses (any classification works) |

![[soft-margin-c-effect.png]]

Set $C$ via cross-validation. The `sklearn` `SVC` default is $C = 1.0$.

## The Dual Formulation

Repeat the [[lagrangian|Lagrangian]] / KKT machinery from week 4, this time with two sets of multipliers — $a^{(n)} \geq 0$ for the margin constraint and $\beta^{(n)} \geq 0$ for $\xi^{(n)} \geq 0$:

$$L = \tfrac{1}{2}\|\mathbf{w}\|^2 + C \sum_n \xi^{(n)} + \sum_n a^{(n)}(1 - \xi^{(n)} - y^{(n)} h(\mathbf{x}^{(n)})) - \sum_n \beta^{(n)} \xi^{(n)}$$

Setting partials to zero:

- $\partial L / \partial \mathbf{w} = 0 \implies \mathbf{w}^* = \sum_n a^{(n)} y^{(n)} \phi(\mathbf{x}^{(n)})$ — same as hard-margin.
- $\partial L / \partial b = 0 \implies \sum_n a^{(n)} y^{(n)} = 0$ — same as hard-margin.
- $\partial L / \partial \xi^{(n)} = 0 \implies C - a^{(n)} - \beta^{(n)} = 0 \implies \beta^{(n)} = C - a^{(n)}$ — **new**.

Combining $\beta^{(n)} \geq 0$ with $a^{(n)} \geq 0$ gives the **box constraint**:

$$0 \leq a^{(n)} \leq C$$

Substituting back, $\xi^{(n)}$ and $\beta^{(n)}$ vanish entirely from the objective — and we get exactly the same dual as hard-margin, but with $a^{(n)}$ now upper-bounded by $C$:

$$\boxed{\arg\max_{\mathbf{a}} \;\; \tilde{L}(\mathbf{a}) = \sum_n a^{(n)} - \tfrac{1}{2} \sum_{n,m} a^{(n)} a^{(m)} y^{(n)} y^{(m)} k(\mathbf{x}^{(n)}, \mathbf{x}^{(m)})}$$

subject to $0 \leq a^{(n)} \leq C$ and $\sum_n a^{(n)} y^{(n)} = 0$.

The kernel trick still applies — only inner products appear. The dual is *almost* identical to the hard-margin one. The single difference, $a^{(n)} \leq C$, captures the entire effect of the slack relaxation.

## KKT Conditions and Support Vector Categories

The complementary slackness conditions for soft-margin SVM are:

$$a^{(n)}(1 - \xi^{(n)} - y^{(n)} h(\mathbf{x}^{(n)})) = 0$$
$$(C - a^{(n)}) \xi^{(n)} = 0 \quad \text{(equivalently } \beta^{(n)} \xi^{(n)} = 0\text{)}$$

These partition training points into three categories based on the value of $a^{(n)}$:

| $a^{(n)}$ | $\xi^{(n)}$ | Type | Geometric position |
|---|---|---|---|
| $a^{(n)} = 0$ | $\xi^{(n)} = 0$ | Not a support vector | Strictly outside margin envelope, satisfies $y h \geq 1$ |
| $0 < a^{(n)} < C$ | $\xi^{(n)} = 0$ | **Margin support vector** | Exactly on the margin: $y h = 1$ |
| $a^{(n)} = C$ | $\xi^{(n)} \geq 0$ | **Bound (clipped) support vector** | On margin ($\xi = 0$), inside margin ($0 < \xi < 1$), on decision boundary ($\xi = 1$), or misclassified ($\xi > 1$) |

> [!info]+ Why $0 < a < C$ pins the point exactly on the margin
> If $0 < a^{(n)} < C$ then $\beta^{(n)} = C - a^{(n)} > 0$, so the second complementary-slackness condition forces $\xi^{(n)} = 0$. And $a^{(n)} > 0$ forces the first condition's bracket to vanish: $1 - 0 - y h = 0$, i.e., $y h = 1$. The point sits exactly on the margin.

The support-vector population in soft-margin SVM is **larger** than in hard-margin: every margin-violating example contributes ($a^{(n)} = C$), in addition to the on-margin ones. Sparsity weakens but is preserved — examples comfortably outside the margin still have $a^{(n)} = 0$.

## Predictions

Identical in form to the hard-margin dual:

$$h(\mathbf{x}) = \sum_{n \in S} a^{(n)} y^{(n)} k(\mathbf{x}, \mathbf{x}^{(n)}) + b$$

For $b$, average over the **margin support vectors only** (those with $0 < a^{(n)} < C$, where $y h = 1$ exactly):

$$b = \frac{1}{|M|} \sum_{n \in M} \left( y^{(n)} - \sum_{m \in S} a^{(m)} y^{(m)} k(\mathbf{x}^{(n)}, \mathbf{x}^{(m)}) \right)$$

where $M = \{n : 0 < a^{(n)} < C\}$. Don't use bound support vectors ($a^{(n)} = C$) — their $y h = 1 - \xi^{(n)} \neq 1$ in general, so they'd give wrong values for $b$.

## What Could Go Wrong

- **Wrong $C$.** Too small and the model underfits (everything gets absorbed into slack); too large and we're back to overfitting via hard-margin behaviour. Always cross-validate.
- **No margin support vectors.** If every support vector is at $a^{(n)} = C$, the formula above for $b$ has no inputs. In practice this is rare but possible; pick any of them and accept the resulting $b$ (or add small regularisation).
- **Unscaled features.** Same as hard-margin — kernels depend on $\mathbf{x}^\top \mathbf{z}$ or $\|\mathbf{x} - \mathbf{z}\|$. Standardise before training.

## How It's Solved

The soft-margin dual is a convex QP with $N$ variables and a linear equality constraint plus box constraints — solvable in principle by off-the-shelf QP solvers, but $O(N^2)$ memory and $O(N^3)$ time make this prohibitive for large $N$. The standard approach is [[sequential-minimal-optimization|Sequential Minimal Optimization (SMO)]], which decomposes into two-variable subproblems each solvable analytically.

## Connections

- **Builds on** [[support-vector-machine]] — relaxes the hard-margin constraint without changing the dual structure.
- **Uses** [[slack-variables]] — the per-example penalty mechanism.
- **Solved by** [[sequential-minimal-optimization]] — analytic two-multiplier updates.
- **Complements** [[kernel-trick]] — kernels handle non-linearity; slack handles non-separability. In practice you nearly always combine them: RBF kernel + soft margin is the default `SVC`.
