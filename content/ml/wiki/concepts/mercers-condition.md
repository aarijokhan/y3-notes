---
type: concept
name: Mercer's Condition
description: The criterion a candidate function must satisfy to be a valid kernel — the Gram matrix must be symmetric and positive semidefinite
sources:
  - raw/week-04/Wk_4_Lec_2-1.pdf
status: draft
updated: 2026-04-26
---

*A function $k(\mathbf{x}, \mathbf{z})$ is a valid kernel iff it corresponds to an inner product in some feature space $\phi$. Mercer's condition is the test: for any finite set of points, the **Gram matrix** $K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ must be symmetric and positive semidefinite.*

## Statement

Given any finite set of points $\{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(M)}\}$, build the $M \times M$ Gram (kernel) matrix:

$$K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$$

Mercer's condition requires that for every such finite set:

1. **Symmetric**: $k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = k(\mathbf{x}^{(j)}, \mathbf{x}^{(i)})$.
2. **Positive semidefinite**: $\mathbf{z}^\top K \mathbf{z} \geq 0$ for all $\mathbf{z} \in \mathbb{R}^M$.

If both hold, $k$ corresponds to the inner product $\phi(\mathbf{x})^\top \phi(\mathbf{z})$ for some feature map $\phi$ (which Mercer's theorem constructs but doesn't require us to compute).

## Why These Conditions

Inner products *must* be symmetric ($\langle u, v\rangle = \langle v, u\rangle$) and produce non-negative self-products ($\langle v, v\rangle \geq 0$). Mercer's condition demands the kernel inherit those properties at every scale — for any sample of points, the implied geometry is consistent with being an inner product in a Hilbert space.

Algorithmically, positive semidefiniteness is what keeps the SVM dual a *convex* QP. If $K$ has a negative eigenvalue, the dual objective is no longer concave and the optimisation can wander to nonsense.

## Kernel Composition Rules

Verifying Mercer's condition from scratch is often hard. Instead, build kernels by composition: starting from validated kernels $k_1, k_2$, the following are also valid:

| Rule | Description |
|---|---|
| $k = c \, k_1$ | Scaling by a constant $c \geq 0$ |
| $k = f(\mathbf{x}) \, k_1 \, f(\mathbf{z})$ | Multiplying by any function on each side |
| $k = q(k_1)$ | Polynomial of $k_1$ with non-negative coefficients |
| $k = e^{k_1}$ | Exponential of a kernel |
| $k = k_1 + k_2$ | Sum of kernels |
| $k = k_1 \cdot k_2$ | Product of kernels |

The Gaussian kernel can be derived as a chain of these rules starting from the linear kernel $k_1 = \mathbf{x}^\top \mathbf{z}$ (see [[gaussian-kernel]] for the proof).

## Active Recall

> [!question]- A friend proposes using $k(\mathbf{x}, \mathbf{z}) = -\|\mathbf{x} - \mathbf{z}\|^2$ as a kernel because it captures "similarity" (closer points get larger values, since the values are less negative). Is it a valid kernel?
> No. Take any two distinct points $\mathbf{x}^{(1)} \neq \mathbf{x}^{(2)}$. The Gram matrix is $K = \begin{pmatrix} 0 & -d^2 \\ -d^2 & 0 \end{pmatrix}$ where $d = \|\mathbf{x}^{(1)} - \mathbf{x}^{(2)}\|$. This has eigenvalues $\pm d^2$ — the negative eigenvalue means $K$ is not positive semidefinite, so Mercer's condition fails. The function is "similarity-shaped" but doesn't correspond to any inner product geometry.

> [!question]- Why does verifying Mercer's condition matter — couldn't we just plug any function into the SVM dual and see what happens?
> Two failure modes. **Mathematically**, a non-Mercer function may produce a Gram matrix with negative eigenvalues; the dual objective is then non-concave, the optimisation problem is non-convex, and there are no guarantees of finding the global optimum. **Geometrically**, the resulting decision function doesn't correspond to a hyperplane in any feature space, so the margin interpretation evaporates and predictions can be wildly wrong even on the training set. The "trick" only works when the kernel really does compute an inner product.

## Related

- [[kernel-trick]] — Mercer's condition is the gate that determines which functions can be used as kernels
- [[gaussian-kernel]] — proof of validity uses the composition rules
- [[polynomial-kernel]] — derived from a polynomial of the linear kernel (rule 3)
