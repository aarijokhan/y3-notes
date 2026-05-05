---
week: 3
topic: "Beyond the Line: Basis Expansion and Maximum Margin"
deck: MachineLearning::Week-03
---

TARGET DECK
MachineLearning::Week-03

## Non-linear Transformation

> [!question]- What is a **non-linear transformation** (basis expansion), and what does it allow a linear model to do?
> A pre-processing map $\phi: \mathbb{R}^d \to \mathbb{R}^D$ (typically with $D > d$) that lifts each input into a higher-dimensional feature space. A linear model $h(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x})$ in the lifted space corresponds to a **non-linear** decision boundary in the original space — circles, polynomials, anything determined by $\phi$. The model is still linear in $\mathbf{w}$, so optimisation machinery and convexity are preserved.

> [!question]- For input $\mathbf{x} = (1, x_1, x_2)^\top$, what is the polynomial basis expansion of degree 2?
> $$\phi(\mathbf{x}) = (1, x_1, x_2, x_1^2, x_2^2, x_1 x_2)^\top$$
> All monomials of total degree $\leq 2$. The cross term $x_1 x_2$ is essential — without it, the model can only express boundaries that are sums of separate single-variable functions of $x_1$ and $x_2$ (no rotated ellipses, for example).

> [!question]- Why does basis expansion *not* break the convexity of logistic regression's loss?
> Convexity of the cross-entropy loss is a property of $\mathbf{w}$, not of $\mathbf{x}$. After basis expansion, the loss becomes:
> $$E(\mathbf{w}) = -\sum_i [y^{(i)} \ln \sigma(\mathbf{w}^\top \phi(\mathbf{x}^{(i)})) + \ldots]$$
> still linear in $\mathbf{w}$, still wrapping the same convex sigmoid composition. The new inputs $\phi(\mathbf{x}^{(i)})$ are constants from the optimiser's perspective. **Linearity in $\mathbf{w}$ is preserved; nothing in optimisation changes.**

> [!question]- What two costs do you pay for using a high-degree polynomial basis expansion?
> - **Dimensionality**: the number of monomials of degree $\leq p$ in $d$ variables grows as $\binom{d+p}{p} \approx d^p$. With 100 inputs and $p = 3$, $\phi(\mathbf{x})$ has more than 170{,}000 components — heavy on memory, computation, and *especially* sample complexity.
> - **Overfitting**: more capacity means the model can fit training noise. A degree-4 polynomial can perfectly classify any reasonable training set while generalising worse than a linear classifier.

## Margin and SVM

> [!question]- What is the **margin** of a separating hyperplane?
> The perpendicular distance from the decision boundary to the *closest* training example:
> $$\text{margin} = \min_n \frac{|\mathbf{w}^\top \mathbf{x}^{(n)} + b|}{\|\mathbf{w}\|}$$
> A wider margin means more buffer between the boundary and the data — small perturbations are less likely to flip a label. The quantity SVMs maximise.

> [!question]- What is a **support vector**, and why are non-support-vectors irrelevant to the SVM solution?
> A **support vector** is a training point sitting exactly on the margin envelope ($y^{(n)} h(\mathbf{x}^{(n)}) = 1$). These are the points closest to the decision boundary in each class. If you deleted every non-support-vector and re-trained, you'd get the *same* hyperplane — only the support vectors actively constrain the optimum; everyone else has slack and contributes nothing.

> [!question]- Write the **canonical hard-margin SVM primal** problem.
> $$\arg\min_{\mathbf{w}, b} \;\; \tfrac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to} \quad y^{(n)}(\mathbf{w}^\top \mathbf{x}^{(n)} + b) \geq 1 \;\; \forall n$$
> The objective minimises $\|\mathbf{w}\|^2$ (which maximises margin $1/\|\mathbf{w}\|$); each constraint requires every training point to lie on the correct side of the margin envelope. A convex quadratic program.

> [!question]- Why does maximising margin reduce to *minimising* $\|\mathbf{w}\|$?
> Apparent paradox — bigger weights should give bigger margin, right? No. After **canonical rescaling** (where the closest training point has $y^{(n)} h(\mathbf{x}^{(n)}) = 1$), its perpendicular distance to the boundary is exactly $1/\|\mathbf{w}\|$. To make that distance large, $\|\mathbf{w}\|$ must be small. Larger weights mean a *steeper* linear function, which can only achieve $h = \pm 1$ closer to the boundary — corresponding to a *smaller* margin.

> [!question]- What is **quadratic programming** (QP), and why is the SVM primal a QP?
> A QP minimises a convex quadratic objective subject to linear inequality (and possibly equality) constraints. The SVM primal $\min \tfrac{1}{2} \mathbf{w}^\top \mathbf{w}$ s.t. $y^{(n)}(\mathbf{w}^\top \mathbf{x}^{(n)} + b) \geq 1$ matches: quadratic objective (in $\mathbf{w}$), linear inequality constraints. Off-the-shelf QP solvers handle this in polynomial time, and convexity guarantees a unique global optimum.

> [!question]- What is the **canonical representation** of a separating hyperplane?
> The scaling of $(\mathbf{w}, b)$ where the closest training point satisfies $y^{(n)}(\mathbf{w}^\top \mathbf{x}^{(n)} + b) = 1$. Used because the geometric hyperplane is invariant to multiplying $(\mathbf{w}, b)$ by any $\kappa > 0$ — the canonical scaling fixes that ambiguity by demanding the closest point hits the value $\pm 1$. Under this normalisation the margin is $1/\|\mathbf{w}\|$ and the constraint becomes $y^{(n)} h(\mathbf{x}^{(n)}) \geq 1$ for all $n$.

## Comparison

> [!question]- A degree-4 polynomial classifier perfectly classifies all training data; a linear classifier misclassifies a few. Which is likely to generalise better, and why?
> Often the **linear** classifier. Training accuracy is not the goal — generalisation is. The degree-4 polynomial has enough capacity to memorise training noise, drawing wildly contorted boundaries to capture every point; those contortions are unlikely to reflect the true underlying structure. Capacity must be matched to data — a principle formalised later by VC dimension and bias–variance analysis.

> [!question]- What's the structural difference between SVM and logistic regression in how they use training data?
> - **Logistic regression**: every training point contributes to the gradient at every iteration; gradient $\nabla E = \sum_i (p_1^{(i)} - y^{(i)}) \mathbf{x}^{(i)}$ touches all $N$ examples.
> - **SVM**: only support vectors (typically a tiny fraction) determine the hyperplane; non-SVs could be deleted without changing the optimum.
>
> SVMs achieve a kind of **data efficiency** by structurally identifying which examples actually matter. Logistic regression has no such notion of "irrelevant" training points.

> [!question]- After basis expansion, what is the SVM primal in $\phi$-space?
> $$\arg\min_{\mathbf{w}, b} \;\; \tfrac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y^{(n)}(\mathbf{w}^\top \phi(\mathbf{x}^{(n)}) + b) \geq 1 \;\; \forall n$$
> Same form, but $\mathbf{w}$ now lives in $\phi$-space. The decision boundary is non-linear in the original space (e.g., a circle from the embedding $\phi(\mathbf{x}) = (\|\mathbf{x}\|^2, x_1, x_2)$). The QP machinery is unchanged — the practical concern is that $\dim \phi$ may be huge, motivating the dual formulation and kernels in week 4.
