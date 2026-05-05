---
week: 4
topic: "Going Dual: Lagrangians, Kernels, and the Trick That Makes Them Tractable"
deck: MachineLearning::Week-04
---

TARGET DECK
MachineLearning::Week-04

## Lagrangian Duality

> [!question]- What is the **Lagrangian** of a constrained optimisation problem?
> For minimising $F(\mathbf{x})$ subject to $f_i(\mathbf{x}) \leq 0$ for $i = 1, \ldots, N$:
> $$L(\mathbf{x}, \mathbf{a}) = F(\mathbf{x}) + \sum_{i=1}^N a_i \, f_i(\mathbf{x}), \qquad a_i \geq 0$$
> Each multiplier $a_i \geq 0$ folds an inequality constraint into the objective. When a constraint is violated, $a_i f_i > 0$ inflates $L$, pushing the optimiser away.

> [!question]- What is the difference between the **primal** and **dual** problems?
> - **Primal**: $\min_\mathbf{x} \max_{\mathbf{a} \geq 0} L(\mathbf{x}, \mathbf{a})$ — outer min, inner max.
> - **Dual**: $\max_{\mathbf{a} \geq 0} \min_\mathbf{x} L(\mathbf{x}, \mathbf{a})$ — outer max, inner min.
>
> Swapping order makes the inner optimisation *unconstrained*, so we can solve $\nabla_\mathbf{x} L = 0$ in closed form, substitute back, and get a problem purely in $\mathbf{a}$.

> [!question]- What are **weak duality** and **strong duality**?
> - **Weak duality** (always holds): $\max_\mathbf{a} \min_\mathbf{x} L \;\leq\; \min_\mathbf{x} \max_\mathbf{a} L$. The dual gives a lower bound on the primal.
> - **Strong duality** (holds when the problem is convex and Slater's condition is satisfied — both true for SVM): equality. Dual and primal have the same optimum.
>
> For SVMs, strong duality means we can solve the dual instead of the primal without losing anything.

> [!question]- What are the four **KKT conditions** for a primal-dual optimum?
> A pair $(\mathbf{x}^\ast, \mathbf{a}^\ast)$ is jointly optimal iff:
> - **Stationarity**: $\nabla_\mathbf{x} L(\mathbf{x}^\ast, \mathbf{a}^\ast) = 0$
> - **Complementary slackness**: $a_i^\ast f_i(\mathbf{x}^\ast) = 0$ for all $i$ — either the multiplier is zero or the constraint is active
> - **Primal feasibility**: $f_i(\mathbf{x}^\ast) \leq 0$
> - **Dual feasibility**: $a_i^\ast \geq 0$
>
> Complementary slackness is what produces SVM's sparsity.

## SVM Dual

> [!question]- Write the **SVM dual** problem.
> $$\arg\max_\mathbf{a} \;\; \sum_{n=1}^N a^{(n)} - \tfrac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a^{(n)} a^{(m)} y^{(n)} y^{(m)} \, \phi(\mathbf{x}^{(n)})^\top \phi(\mathbf{x}^{(m)})$$
> subject to $a^{(n)} \geq 0$ and $\sum_n a^{(n)} y^{(n)} = 0$.
> Three things to notice: (1) variables are now $a^{(n)}$ — one per training example, independent of $\dim \phi$; (2) data appears only through inner products; (3) concave in $\mathbf{a}$, still a QP.

> [!question]- How does the dual recover $\mathbf{w}^\ast$ from the multipliers $a^{(n)}$?
> Setting $\partial L / \partial \mathbf{w} = 0$ gives:
> $$\mathbf{w}^\ast = \sum_{n=1}^N a^{(n)} y^{(n)} \phi(\mathbf{x}^{(n)})$$
> The optimal weights are a linear combination of the training feature vectors, weighted by their multipliers and labels. By complementary slackness, only support vectors have $a^{(n)} > 0$ — so $\mathbf{w}^\ast$ depends only on the support vectors.

> [!question]- Why does **KKT complementary slackness** force SVM solutions to be sparse?
> The condition $a^{(n)}(1 - y^{(n)} h(\mathbf{x}^{(n)})) = 0$ requires, for each $n$, *either* $a^{(n)} = 0$ *or* $y^{(n)} h(\mathbf{x}^{(n)}) = 1$ (point on margin). So:
> - Points strictly outside the margin: $a^{(n)} = 0$, contribute nothing.
> - Points on the margin: $a^{(n)} > 0$ — these are the support vectors.
>
> Sparsity is a *structural* consequence of the KKT conditions, not a heuristic.

> [!question]- How do you compute SVM predictions in dual form?
> $$h(\mathbf{x}) = \sum_{n \in S} a^{(n)} y^{(n)} \, \phi(\mathbf{x}^{(n)})^\top \phi(\mathbf{x}) + b$$
> where $S$ is the index set of support vectors. The bias $b$ is recovered from any support vector ($y^{(n)} h(\mathbf{x}^{(n)}) = 1$); average over all SVs for stability. **Predictions depend only on inner products** — the opening for the kernel trick.

## Kernel Trick

> [!question]- What is the **kernel trick**?
> Replace every inner product $\phi(\mathbf{x})^\top \phi(\mathbf{z})$ in the SVM dual with a **kernel function** $k(\mathbf{x}, \mathbf{z})$ that returns the same value computed entirely in the original input space:
> $$k(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top \phi(\mathbf{z})$$
> Now $\phi$ is **implicit** — the algorithm needs only $k$, often computable in $O(d)$ even when $\dim \phi$ is huge or infinite. Lets us use very high (even infinite) dimensional embeddings for free.

> [!question]- What is **Mercer's condition**, and what does it guarantee?
> A function $k(\mathbf{x}, \mathbf{z})$ is a valid kernel — i.e., corresponds to an inner product in *some* Hilbert space — iff for every finite set of points $\{\mathbf{x}^{(i)}\}$:
> - **Symmetry**: $k(\mathbf{x}, \mathbf{z}) = k(\mathbf{z}, \mathbf{x})$
> - **Positive semidefiniteness**: the Gram matrix $K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ has $\mathbf{z}^\top K \mathbf{z} \geq 0$ for all $\mathbf{z}$
>
> Mercer's theorem guarantees the existence of an embedding $\phi$ such that $k = \phi^\top \phi$ without requiring you to construct $\phi$ explicitly. Algorithmically, PSD keeps the SVM dual a convex QP.

> [!question]- What is the formula for the **polynomial kernel**, and what does it implicitly compute?
> $$k(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^\top \mathbf{z})^p$$
> Corresponds to an explicit embedding of all monomials of degree $\leq p$. For $\mathbf{x}, \mathbf{z} \in \mathbb{R}^d$ with $p$ small, the implicit feature space has $\binom{d+p}{d} = O(d^p)$ dimensions — but the kernel is computed in $O(d)$ regardless. Massive speed-up.

> [!question]- What is the formula for the **Gaussian (RBF) kernel**, and why is its implicit feature space infinite-dimensional?
> $$k(\mathbf{x}, \mathbf{z}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{z}\|^2}{2 \sigma^2}\right)$$
> Expanding the exponential as a Taylor series produces all monomial degrees:
> $$e^{\mathbf{x}^\top \mathbf{z}} = \sum_{j=0}^\infty \frac{(\mathbf{x}^\top \mathbf{z})^j}{j!}$$
> The implicit $\phi$ contains monomials of *every* degree — infinite-dimensional. We could never form $\phi$ explicitly, but $k$ is computed in $O(d)$.

> [!question]- A polynomial-of-degree-2 kernel for 2D inputs has the explicit feature map $\phi(\mathbf{x}) = (1, \sqrt{2} x_1, \sqrt{2} x_2, x_1^2, x_2^2, \sqrt{2} x_1 x_2)$. What is $\phi(\mathbf{x})^\top \phi(\mathbf{z})$ as a single closed-form expression?
> Computing the inner product term by term:
> $$\phi(\mathbf{x})^\top \phi(\mathbf{z}) = 1 + 2 x_1 z_1 + 2 x_2 z_2 + x_1^2 z_1^2 + x_2^2 z_2^2 + 2 x_1 x_2 z_1 z_2 = (1 + \mathbf{x}^\top \mathbf{z})^2$$
> Six multiplications collapse into one inner product plus one square. This is exactly the polynomial kernel of degree 2.

> [!question]- Given valid kernels $k_1, k_2$, list four operations that produce another valid kernel.
> Any of:
> - $c \cdot k_1$ for $c \geq 0$
> - $f(\mathbf{x}) \cdot k_1(\mathbf{x}, \mathbf{z}) \cdot f(\mathbf{z})$ for any function $f$
> - $q(k_1)$ for a polynomial $q$ with non-negative coefficients
> - $\exp(k_1)$
> - $k_1 + k_2$
> - $k_1 \cdot k_2$
>
> These composition rules let you build complex kernels (Gaussian, polynomial-with-bias) without verifying Mercer's condition from scratch.

## Comparison

> [!question]- For an SVM with $N = 1{,}000$ training examples and a polynomial-of-degree-3 kernel for 100-dimensional inputs, how many variables does the primal vs the dual have?
> - **Primal**: $\dim \phi + 1 = \binom{103}{3} + 1 \approx 176{,}851$ — one per feature dimension.
> - **Dual**: $N = 1{,}000$ — one per training example.
>
> When $\dim \phi \gg N$, the dual is dramatically smaller. Plus, the dual depends only on inner products, so the kernel trick avoids ever forming $\phi$ explicitly.
