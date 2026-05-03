---
type: concept
name: design matrix
description: The $N \times (M+1)$ matrix Φ whose rows are basis-function evaluations of training inputs — the data side of the OLS normal equation
sources:
  - raw/week-07/Wk_7_Lec_1-1.pdf
status: draft
updated: 2026-04-27
---

*An $N \times (M+1)$ matrix $\boldsymbol{\Phi}$ whose row $i$ is the basis-function vector $\boldsymbol{\phi}(\mathbf{x}_i)^\top$ evaluated at training input $\mathbf{x}_i$. Compactly encodes "all training inputs through all basis functions" so that $\hat{\mathbf{y}} = \boldsymbol{\Phi} \mathbf{w}$ is a single matrix-vector product, and the [[ordinary-least-squares|OLS]] solution becomes $(\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}$.*

## Construction

Given training inputs $\{\mathbf{x}_i\}_{i=1}^N$ and basis functions $\{\phi_j\}_{j=0}^M$:

$$\boldsymbol{\Phi} = \begin{pmatrix} \phi_0(\mathbf{x}_1) & \phi_1(\mathbf{x}_1) & \cdots & \phi_M(\mathbf{x}_1) \\ \phi_0(\mathbf{x}_2) & \phi_1(\mathbf{x}_2) & \cdots & \phi_M(\mathbf{x}_2) \\ \vdots & \vdots & \ddots & \vdots \\ \phi_0(\mathbf{x}_N) & \phi_1(\mathbf{x}_N) & \cdots & \phi_M(\mathbf{x}_N) \end{pmatrix}$$

By convention $\phi_0(\mathbf{x}) = 1$, so the first column is all ones — pairing with the intercept weight $w_0$.

**Dimensions.** $\boldsymbol{\Phi}$ is $N \times (M+1)$:
- $N$ = number of training examples (rows)
- $M + 1$ = number of basis functions including the intercept (columns)

## Why It's Useful

The model $\hat{y}_i = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i)$ for all $i$ becomes a single matrix product:

$$\hat{\mathbf{y}} = \boldsymbol{\Phi} \mathbf{w}$$

The OLS objective $\sum_i (y_i - \hat{y}_i)^2$ becomes $\|\mathbf{y} - \boldsymbol{\Phi} \mathbf{w}\|^2$, which differentiates to the [[ordinary-least-squares#Deriving the Normal Equation|normal equation]] in one line.

Whatever the basis functions are — polynomial, Gaussian, sigmoidal, custom — the math is the same. The design matrix decouples *what features* you use from *how you fit*.

## Examples

**Polynomial basis (degree $M$, single input $x$):**

$$\boldsymbol{\Phi} = \begin{pmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^M \\ 1 & x_2 & x_2^2 & \cdots & x_2^M \\ \vdots & & & & \vdots \\ 1 & x_N & x_N^2 & \cdots & x_N^M \end{pmatrix}$$

This is a **Vandermonde matrix** — a recurring object in interpolation theory.

**Gaussian RBF basis ($M$ centres at $\mu_1, \ldots, \mu_M$, width $s$):**

$$\boldsymbol{\Phi}_{i, j} = \begin{cases} 1 & \text{if } j = 0 \\ \exp\left(-\frac{(x_i - \mu_j)^2}{2 s^2}\right) & \text{otherwise} \end{cases}$$

**Multi-input ($\mathbf{x} \in \mathbb{R}^d$, plain linear basis):**

$$\boldsymbol{\Phi} = \begin{pmatrix} 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_d^{(1)} \\ 1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_d^{(2)} \\ \vdots & & & & \vdots \\ 1 & x_1^{(N)} & x_2^{(N)} & \cdots & x_d^{(N)} \end{pmatrix}$$

## Practical Notes

- **The first column is always ones.** This is the dummy basis function $\phi_0 \equiv 1$ that pairs with the intercept $w_0$. Forgetting it is a common bug — fits are forced through the origin.
- **Standardise inputs first.** Polynomial and Gaussian bases on raw inputs can have wildly different column magnitudes, making $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ ill-conditioned.
- **Tall vs wide.** $\boldsymbol{\Phi}$ is "tall" when $N > M$ (more examples than parameters) — the standard setting where OLS works. "Wide" ($M > N$) makes $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ rank-deficient and OLS degenerates.

## Connections

- [[ordinary-least-squares]] — uses $\boldsymbol{\Phi}$ in the normal equation $\mathbf{w} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}$.
- [[linear-regression]] — the model whose predictions are $\hat{\mathbf{y}} = \boldsymbol{\Phi} \mathbf{w}$.
- [[non-linear-transformation]] — the basis-expansion idea; the design matrix is the "$\phi$-space training set".
