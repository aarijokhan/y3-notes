---
type: concept
name: Polynomial Kernel
description: $k(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^\top \mathbf{z})^p$ — corresponds to a finite embedding of all monomials up to degree $p$
sources:
  - raw/week-04/Wk_4_Lec_2-1.pdf
status: draft
updated: 2026-04-26
---

*$k(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^\top \mathbf{z})^p$ — the kernel whose feature map is all monomials in the inputs of total degree $\leq p$, with appropriate weights. Useful when you suspect the data has polynomial structure, especially in computer vision where polynomial features model image-feature interactions.*

## Definition and Embedding

$$k(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^\top \mathbf{z})^p$$

For $p = 2$ and $\mathbf{x} \in \mathbb{R}^2$, the implicit feature map is:

$$\phi(\mathbf{x}) = (1, \sqrt{2} x_1, \sqrt{2} x_2, x_1^2, x_2^2, \sqrt{2} x_1 x_2)^\top$$

— a 6-dimensional vector that contains every monomial of degree $\leq 2$, weighted to make the inner product factor cleanly. The kernel computes that inner product as $(1 + \mathbf{x}^\top \mathbf{z})^2$, replacing six multiplications with two operations in the input space.

For general $p$ and $D$-dimensional inputs, the embedding has $\binom{D + p}{p}$ dimensions — combinatorial in $D$ and $p$. The kernel evaluation, by contrast, is always one inner product plus one exponentiation.

## Validity (Why It's a Kernel)

Apply the [[mercers-condition|composition rules]]: the linear kernel $k_1(\mathbf{x}, \mathbf{z}) = \mathbf{x}^\top \mathbf{z}$ is valid. Adding the constant $1$ (a degenerate kernel: $\phi = 1$ always) keeps it valid. Raising a valid kernel to a non-negative integer power keeps it valid (rule: polynomial with non-negative coefficients applied to the kernel). So $(1 + \mathbf{x}^\top \mathbf{z})^p$ is valid for any $p \in \mathbb{Z}_{\geq 0}$.

## When to Use

- **Polynomial structure suspected**: image features (where products of pixel intensities encode texture), interactions between numeric variables.
- **Moderate degree**: $p \in \{2, 3, 4\}$ common. Higher degrees overfit aggressively in high-dimensional input.
- **As a sanity-check baseline** before reaching for a Gaussian kernel.

The trade-off versus the [[gaussian-kernel|Gaussian kernel]]:

| | Polynomial kernel | Gaussian kernel |
|---|---|---|
| Embedding dim | Finite, $O(D^p)$ | Infinite |
| Hyperparameters | Degree $p$ | Bandwidth $\sigma$ |
| Boundary shape | Polynomial in $\mathbf{x}$ | Any smooth shape |
| Built-in regularisation | Yes (degree caps complexity) | No (need explicit regularisation) |
| Sensitive to feature scale | Yes | Yes (more so) |

## Active Recall

> [!question]- For 100-dimensional input, the polynomial kernel of degree 5 has roughly $10^7$ implicit feature dimensions. How is this different in cost from running an SVM in that explicit feature space?
> In the *primal*, you'd need to materialise every $\phi(\mathbf{x}^{(n)})$ — $N \times 10^7$ numbers — and solve a QP with $10^7$ variables. Memory and time both blow up. With the kernel, you only ever evaluate $k(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^\top \mathbf{z})^5$ — one 100-dim inner product plus one exponentiation. The dual QP has $N$ variables (one per training point), independent of polynomial degree. The kernel trick converts what would be a $10^7$-dimensional optimisation into an $N$-dimensional one.

## Related

- [[kernel-trick]] — what makes the polynomial embedding tractable
- [[non-linear-transformation]] — explicit polynomial basis expansion is the primal version
- [[gaussian-kernel]] — alternative non-linear kernel; richer but unbounded in expressiveness
- [[mercers-condition]] — validity proof relies on the composition rules
