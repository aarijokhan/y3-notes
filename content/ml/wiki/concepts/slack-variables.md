---
type: concept
name: slack variables
description: Per-example penalty variables ξ that quantify how badly each training point violates the SVM margin constraint, allowing soft-margin classification
sources:
  - raw/week-05/Wk_5_Lec_1-1.pdf
  - raw/week-05/5a-soft-margin-svm-answers.pdf
status: draft
updated: 2026-04-27
---

*A non-negative scalar $\xi^{(n)} \geq 0$ attached to each training example, measuring how far that example slips inside or across the SVM margin envelope. Slack variables convert the hard "everyone outside the margin" constraint into a budgeted penalty — and turn an infeasible separation problem into a tractable one.*

## The Constraint Modification

The hard-margin [[support-vector-machine|SVM]] requires every example to satisfy:

$$y^{(n)} h(\mathbf{x}^{(n)}) \geq 1$$

If the data is not linearly separable (in $\phi$-space), this constraint set is *empty* and the optimisation has no feasible point. Slack relaxes the right-hand side by a non-negative amount:

$$y^{(n)} h(\mathbf{x}^{(n)}) \geq 1 - \xi^{(n)}, \qquad \xi^{(n)} \geq 0$$

Each $\xi^{(n)}$ is a per-example quota for "how much we're willing to bend the margin rule for this point."

## Geometric Interpretation

![[soft-margin-slack-geometric.png]]

The value of $\xi^{(n)}$ encodes the position of point $n$ relative to the margin and decision boundary:

| $\xi^{(n)}$ | Geometric meaning |
|---|---|
| $\xi^{(n)} = 0$ | Correctly classified, on or outside the margin (constraint $y h \geq 1$ holds) |
| $\xi^{(n)} \in (0, 1)$ | Correctly classified, but *inside* the margin envelope |
| $\xi^{(n)} = 1$ | Sits *exactly on* the decision boundary ($h(\mathbf{x}^{(n)}) = 0$) |
| $\xi^{(n)} > 1$ | Misclassified — on the *wrong side* of the decision boundary |

A useful identity: $\xi^{(n)} = \max(0, 1 - y^{(n)} h(\mathbf{x}^{(n)}))$ — the **hinge loss** of example $n$. If the example already satisfies the margin, $\xi^{(n)} = 0$; otherwise, $\xi^{(n)}$ measures the shortfall.

## Where Slack Enters the Objective

The slack-augmented [[soft-margin-svm|soft-margin SVM]] objective is:

$$\arg\min_{\mathbf{w}, b, \boldsymbol{\xi}} \;\; \tfrac{1}{2}\|\mathbf{w}\|^2 + C \sum_{n=1}^N \xi^{(n)}$$

The sum $\sum_n \xi^{(n)}$ is the **total margin violation** across the training set, weighted by the hyperparameter $C$. Note: we minimise the *sum of values*, not the *count* of non-zero slacks. A misclassified example with $\xi = 1.5$ costs more than two margin-grazing examples with $\xi = 0.4$ each.

> [!warning]+ Slack on every example, even those that don't need it
> Every training point gets its own $\xi^{(n)}$ in the formulation — including ones that are happily outside the margin. At the optimum those have $\xi^{(n)} = 0$, but they are still variables in the problem. The slack is never *added* selectively to violators; it is allocated to all and pinned to zero where the data permits.

## Why Slack Always Hits the Constraint With Equality

At the optimum, whenever $\xi^{(n)} > 0$ for any $C > 0$, the margin constraint is satisfied with **equality**:

$$y^{(n)} h(\mathbf{x}^{(n)}) = 1 - \xi^{(n)}$$

This follows from KKT complementary slackness — see [[soft-margin-svm]] for the derivation. The practical upshot: $\xi^{(n)}$ is not just a budget upper bound on the violation, it equals the violation exactly.

## Why Slack Doesn't Run Away

If slack carried no penalty, the optimiser would dial every $\xi^{(n)}$ to infinity, making all constraints trivially satisfiable and shrinking $\|\mathbf{w}\|$ to zero. The hyperparameter $C$ prevents this:

- **Large $C$**: each unit of slack is expensive → the optimiser tolerates fewer/smaller margin violations → narrow margin, may overfit
- **Small $C$**: slack is cheap → many points allowed inside or across the margin → wide margin, may underfit

See [[soft-margin-svm]] for the full $C$-tradeoff discussion.

## Active-Recall Questions

> [!question]- What does $\xi^{(n)} = 0.4$ tell you about example $n$?
> The example is correctly classified but lies *inside* the margin envelope. The margin constraint is satisfied with equality at $y h = 0.6$, so the point sits 40% of the way from the margin towards the decision boundary, on the correct side.

> [!question]- A misclassified example always has larger slack than a correctly-classified one inside the margin. True or false?
> True. Misclassified means $y h < 0$, which forces $\xi = 1 - y h > 1$. Correctly-classified-but-inside-margin means $0 < y h < 1$, which gives $\xi = 1 - y h \in (0, 1)$. So misclassified $\xi > 1 >$ inside-margin $\xi$.
>
> However, between two misclassified examples, the one *deeper* into the wrong territory (more negative $y h$) has the larger slack. A barely-misclassified point can have $\xi$ only slightly above 1.

> [!question]- Why minimise $\sum_n \xi^{(n)}$ rather than $|\{n : \xi^{(n)} > 0\}|$?
> The count is non-convex and combinatorial — it would make the problem NP-hard. The sum is a convex relaxation: it preserves the ranking (more violation → worse objective) while keeping the QP tractable. As a side effect, the sum penalises *severity* not just *occurrence*: one badly-misclassified point hurts more than several margin-grazing ones.

## Connections

- **Component of** [[soft-margin-svm]] — slack variables are the structural mechanism that makes SVM work on non-separable data.
- **Hinge loss** — $\xi^{(n)} = \max(0, 1 - y^{(n)} h(\mathbf{x}^{(n)}))$ is the per-example hinge loss; soft-margin SVM is equivalent to minimising mean hinge loss + L2 regularisation.
- **Why slack ≠ misclassification indicator** — examples with $0 < \xi < 1$ are correctly classified; only $\xi > 1$ implies misclassification. The "slack count" overcounts errors.
