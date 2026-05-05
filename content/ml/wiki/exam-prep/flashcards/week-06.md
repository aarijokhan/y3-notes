---
week: 6
topic: "Revision: Synthesis Across Weeks 1–5"
deck: MachineLearning::Week-06
---

TARGET DECK
MachineLearning::Week-06

## Cross-Cutting Synthesis

> [!question]- What do **logistic regression** and **hard-margin SVM** have in common, and how do they differ?
> *Common*: same hypothesis class — a linear function of $\mathbf{x}$ (or $\phi(\mathbf{x})$): $h(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$. Both produce linear decision boundaries.
> *Different*: the **fitting criterion**.
> - **Logistic regression**: minimise cross-entropy (probabilistic / MLE-derived); every training point contributes to the gradient.
> - **Hard-margin SVM**: maximise margin (geometric); only support vectors determine the boundary.
>
> Same hypothesis, different objectives — leads to different solutions even on the same data.

> [!question]- Compare the three optimization algorithms covered: **gradient descent**, **Newton-Raphson/IRLS**, and **SMO**. When do you use each?
> | Algorithm | Used for | Per-iteration cost | Iterations |
> |---|---|---|---|
> | **Gradient descent** | Logistic regression, neural networks | $O(Nd)$ | Many; needs $\eta$ |
> | **Newton-Raphson / IRLS** | Logistic regression (small $d$) | $O(d^3)$ Hessian inverse | Few; no $\eta$ |
> | **SMO** | SVM dual (soft- or hard-margin) | $O(N)$ analytic pair update | Many but cheap |
>
> GD scales with parameters; IRLS uses curvature but doesn't scale; SMO exploits the dual's structure (equality + box constraints).

> [!question]- For an SVM with $N = 1{,}000$ examples and a Gaussian kernel, why is the **dual** preferred over the **primal**?
> The Gaussian kernel implies an **infinite-dimensional** $\phi$ — the primal would have infinitely many variables. The dual has only $N = 1{,}000$ variables (one per training point), and the kernel trick computes $k(\mathbf{x}, \mathbf{z})$ in $O(d)$ regardless of $\dim \phi$. The dual depends only on inner products, so $\phi$ is never explicitly formed.

> [!question]- Why does the **kernel trick** work, and what does **Mercer's condition** guarantee?
> The kernel trick replaces $\phi(\mathbf{x})^\top \phi(\mathbf{z})$ everywhere with $k(\mathbf{x}, \mathbf{z})$ — sidestepping any explicit construction of $\phi$. Mercer's condition (symmetric + positive-semidefinite Gram matrix) guarantees that *some* embedding $\phi$ exists such that $k = \phi^\top \phi$, even if you can't write it down. PSD also keeps the SVM dual a convex QP.

> [!question]- How does **basis expansion** $\phi$ change the optimisation problem for logistic regression and SVM?
> *Procedurally*: substitute $\mathbf{x} \to \phi(\mathbf{x})$ everywhere — the loss/objective, gradient, Hessian, and update rules are otherwise unchanged. *Functionally*: the model is still linear in $\mathbf{w}$, so convexity is preserved. The decision boundary becomes non-linear in the original $\mathbf{x}$-space (e.g. a circle for $\phi = (\|\mathbf{x}\|^2, x_1, x_2)$).

> [!question]- What's the role of **complementary slackness** in producing SVM sparsity?
> KKT complementary slackness: $a^{(n)} f_n(\mathbf{w}^\ast) = 0$ for every constraint $f_n$. For SVM, $f_n = 1 - y^{(n)} h(\mathbf{x}^{(n)}) \leq 0$. So either:
> - $a^{(n)} = 0$ — point strictly outside the margin, contributes nothing.
> - $f_n = 0$ — point exactly on the margin (a support vector); $a^{(n)} > 0$ allowed.
>
> Only support vectors enter $\mathbf{w}^\ast = \sum_n a^{(n)} y^{(n)} \phi(\mathbf{x}^{(n)})$. **Sparsity is structural**, not heuristic.

> [!question]- What does **strong duality** require for the SVM, and what does it buy us?
> Strong duality requires:
> - Convex primal (yes — quadratic objective, linear constraints)
> - Slater's condition: a strictly feasible point exists (yes for separable data, or for soft-margin)
>
> When it holds, $\max_\mathbf{a} \min_\mathbf{x} L = \min_\mathbf{x} \max_\mathbf{a} L$ — solving the dual is *exactly* equivalent to solving the primal. We get all the benefits (kernel trick, smaller variable count) without any loss of optimality.

> [!question]- What's the relationship between the **hard-margin** and **soft-margin** SVM dual?
> Identical except for one constraint:
> - Hard-margin: $a^{(n)} \geq 0$
> - Soft-margin: $\boxed{0 \leq a^{(n)} \leq C}$ — the **box constraint**
>
> Slack variables and their multipliers vanish in the dual derivation; the entire effect of slack collapses to the upper bound $a^{(n)} \leq C$. The kernel trick, KKT conditions, and prediction formulas are otherwise unchanged.

## Conceptual Foundations

> [!question]- What is **i.i.d.** and why do all generalisation arguments rely on it?
> Independent and identically distributed: training and test examples are independent draws from the *same* fixed joint $P(\mathbf{x}, y)$. Generalisation guarantees say things like "training error close to test error with high probability" — they require the test data to follow the same distribution as the training data. Distribution shift breaks every such guarantee.

> [!question]- When is **logistic regression's loss strictly convex**, and why does that matter?
> The cross-entropy loss $E(\mathbf{w}) = -\sum_i [y^{(i)} \ln \sigma(\mathbf{w}^\top \mathbf{x}^{(i)}) + (1 - y^{(i)}) \ln(1 - \sigma(\mathbf{w}^\top \mathbf{x}^{(i)}))]$ is strictly convex in $\mathbf{w}$ (assuming the design matrix has full column rank). It matters because **every local minimum is a global minimum** — gradient descent converging to a stationary point converges to *the* optimum. No worries about getting stuck in suboptimal valleys.

> [!question]- The polynomial kernel $k(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^\top \mathbf{z})^p$ corresponds to what implicit feature space?
> All monomials of degree $\leq p$ in the components of $\mathbf{x}$. For $\mathbf{x} \in \mathbb{R}^d$, the implicit feature space has $\binom{d+p}{d} = O(d^p)$ dimensions. The kernel computes in $O(d)$ regardless of how large that becomes — e.g., $d = 100, p = 5$ implies $\sim 10^7$ feature dimensions but only 100 multiplications per kernel evaluation.

> [!question]- Why does **maximising the margin** give better generalisation than just any separating hyperplane?
> Geometrically, a wide margin means small perturbations to a training point (or any new test point) won't flip its class — the boundary is robust. Theoretically (formalised in week 9), restricting to fat hyperplanes of margin $\geq \rho$ shrinks the effective VC dimension to $\lceil R^2/\rho^2 \rceil + 1$ — independent of input dimension. This explains why high-dimensional kernel SVMs generalise even with infinite-dimensional $\phi$.
