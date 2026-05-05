---
week: 5
topic: "Allowing Mistakes: Soft Margins and How to Actually Solve the QP"
deck: MachineLearning::Week-05
---

TARGET DECK
MachineLearning::Week-05

## Soft-Margin SVM

> [!question]- Why do we need a **soft-margin** SVM?
> The hard-margin constraint $y^{(n)} h(\mathbf{x}^{(n)}) \geq 1$ for all $n$ requires *every* point to be correctly classified beyond the margin. If even one point can't satisfy that — because classes overlap, or noise has shifted a point to the wrong side — the constraint set is **empty** and there's no solution. Real data usually has overlap or label noise, so hard-margin is too brittle.

> [!question]- Write the **soft-margin SVM primal** problem.
> $$\arg\min_{\mathbf{w}, b, \boldsymbol{\xi}} \;\; \tfrac{1}{2} \|\mathbf{w}\|^2 + C \sum_{n=1}^N \xi^{(n)}$$
> $$\text{subject to} \quad y^{(n)} (\mathbf{w}^\top \mathbf{x}^{(n)} + b) \geq 1 - \xi^{(n)}, \quad \xi^{(n)} \geq 0 \quad \forall n$$
> Each training example gets a personal violation budget $\xi^{(n)} \geq 0$. The hyperparameter $C > 0$ trades margin width against violation tolerance.

> [!question]- What is a **slack variable** $\xi^{(n)}$, and how does its value encode a point's position?
> $\xi^{(n)} \geq 0$ is the per-example margin-violation budget. At the optimum, $\xi^{(n)} = \max(0, 1 - y^{(n)} h(\mathbf{x}^{(n)}))$ — exactly the **hinge loss**. Position by value:
> - $\xi = 0$: outside or on the margin envelope
> - $\xi \in (0, 1)$: inside the margin, correctly classified
> - $\xi = 1$: exactly on the decision boundary
> - $\xi > 1$: misclassified — wrong side of the boundary

> [!question]- What does the hyperparameter $C$ control in soft-margin SVM, and what happens at the extremes?
> $C > 0$ sets the price of slack:
> - **Large $C$**: slack is expensive → narrow margin, behaves like hard-margin (many SVs, risk of overfitting).
> - **Small $C$**: slack is cheap → wide margin, many margin-violators tolerated (risk of underfitting; in extreme case, model collapses to "always predict majority class").
>
> $C$ is the bias-variance trade-off knob, typically chosen by cross-validation.

> [!question]- What is the **soft-margin SVM dual**, and how does it differ from the hard-margin dual?
> $$\arg\max_\mathbf{a} \;\; \sum_n a^{(n)} - \tfrac{1}{2} \sum_{n,m} a^{(n)} a^{(m)} y^{(n)} y^{(m)} k(\mathbf{x}^{(n)}, \mathbf{x}^{(m)})$$
> subject to $\sum_n a^{(n)} y^{(n)} = 0$ and $\boxed{0 \leq a^{(n)} \leq C}$.
>
> The only difference from hard-margin: the **box constraint** $a^{(n)} \leq C$. Slack and its multipliers vanish in the dual derivation; the entire effect of slack collapses to one upper bound.

> [!question]- What are the **three categories of training point** in a soft-margin SVM, indexed by $a^{(n)}$ and $\xi^{(n)}$?
> | $a^{(n)}$ | $\xi^{(n)}$ | Type | Position |
> |---|---|---|---|
> | $0$ | $0$ | Not a SV | Strictly outside margin |
> | $\in (0, C)$ | $0$ | **Margin SV** | Exactly on margin: $y h = 1$ |
> | $C$ | $\geq 0$ | **Bound SV** | On margin / inside / misclassified |
>
> Only **margin SVs** can be used to compute $b$ — bound SVs have $y h \neq 1$ in general.

> [!question]- Why is sparsity weakened (but preserved) in soft-margin SVM compared to hard-margin?
> Hard-margin: only points exactly on the margin have $a^{(n)} > 0$ — typically a tiny fraction. Soft-margin: every margin violator *also* has $a^{(n)} = C > 0$, expanding the active set. But points comfortably outside the margin still have $a^{(n)} = 0$ and contribute nothing to predictions. So sparsity weakens (more support vectors) but doesn't vanish.

## Sequential Minimal Optimization

> [!question]- What is **Sequential Minimal Optimization** (SMO), and why update **two** multipliers at a time rather than one?
> SMO solves the SVM dual by picking a small subset of multipliers, optimising over them analytically, and iterating. The minimum subset size is **two** because of the equality constraint $\sum_n a^{(n)} y^{(n)} = 0$:
> $$a^{(m)} y^{(m)} = -\sum_{n \neq m} a^{(n)} y^{(n)} = \text{constant}$$
> Picking just *one* multiplier with everything else fixed pins it in place — no movement possible. Two multipliers can move while preserving the constraint.

> [!question]- What is the **analytic update** in SMO for the second multiplier $a^{(j)}$?
> $$a^{(j, \text{new})} = a^{(j)} + \frac{y^{(j)}(E^{(i)} - E^{(j)})}{k(\mathbf{x}^{(i)}, \mathbf{x}^{(i)}) + k(\mathbf{x}^{(j)}, \mathbf{x}^{(j)}) - 2 k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})}$$
> where $E^{(n)} = f(\mathbf{x}^{(n)}) - y^{(n)}$ is the prediction error.
> The denominator is $\|\phi(\mathbf{x}^{(i)}) - \phi(\mathbf{x}^{(j)})\|^2$ — the squared distance between the two examples in feature space. The numerator scales with error disagreement.

> [!question]- After computing the analytic SMO update, why must the result be **clipped**, and what is the clipping interval?
> Because the candidate update may push $a^{(j)}$ out of $[0, C]$ or push the paired $a^{(i)}$ out of its box. The feasible interval $[L, H]$ depends on whether the labels match:
> - **Same labels** ($y^{(i)} = y^{(j)}$): $L = \max(0, a^{(j)} + a^{(i)} - C), \;\; H = \min(C, a^{(j)} + a^{(i)})$
> - **Different labels** ($y^{(i)} \neq y^{(j)}$): $L = \max(0, a^{(j)} - a^{(i)}), \;\; H = \min(C, a^{(j)} - a^{(i)} + C)$
>
> Then clip: $a^{(j, \text{new\&clipped})} = \mathrm{clip}(a^{(j, \text{new})}, L, H)$, and recover $a^{(i)}$ from the equality constraint.

> [!question]- What is the **maximum-step heuristic** for choosing the second multiplier in SMO?
> After picking $a^{(i)}$, choose $j$ to **maximise $|E^{(i)} - E^{(j)}|$** — the largest disagreement between cached prediction errors. This approximates the largest update size and accelerates convergence. SMO converges for any pair-selection rule, but heuristics matter hugely — random-random pair selection can be 10×–100× slower.

> [!question]- Restate the SVM **KKT optimality conditions** in terms of $a^{(n)}$ alone.
> | Condition | Geometric meaning |
> |---|---|
> | $a^{(n)} = 0 \iff y^{(n)} h(\mathbf{x}^{(n)}) \geq 1$ | Outside margin; not a SV |
> | $0 < a^{(n)} < C \iff y^{(n)} h(\mathbf{x}^{(n)}) = 1$ | On margin; margin SV |
> | $a^{(n)} = C \iff y^{(n)} h(\mathbf{x}^{(n)}) \leq 1$ | On / inside / violating margin; bound SV |
>
> When every example satisfies its KKT condition within tolerance, the dual is at its optimum.

## Practical Comparison

> [!question]- Why is the soft-margin SVM dual still a convex quadratic program despite the added slack?
> Because in the dual, slack disappears entirely — only the box constraint $0 \leq a^{(n)} \leq C$ remains. The dual objective is concave (negative-definite quadratic form in $\mathbf{a}$ via the kernel Gram matrix), and the constraints are linear. Standard convex QP. Strong duality still holds because the primal is convex.

> [!question]- Why use only **margin support vectors** (and not bound SVs) to compute the bias $b$?
> A margin SV has $0 < a^{(n)} < C$, which forces $\xi^{(n)} = 0$ via complementary slackness, which pins the point exactly on the margin: $y^{(n)} h(\mathbf{x}^{(n)}) = 1$. Solving for $b$ gives the correct value. **Bound SVs** ($a^{(n)} = C$) have $\xi^{(n)} \geq 0$, so $y^{(n)} h(\mathbf{x}^{(n)}) = 1 - \xi^{(n)}$ generally $\neq 1$ — substituting gives the wrong $b$. Average over all margin SVs for numerical stability.
