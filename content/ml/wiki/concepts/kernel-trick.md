---
type: concept
name: Kernel Trick
description: Compute inner products in a high-dimensional feature space using only operations in the original space — never form the explicit embedding
sources:
  - raw/week-04/Wk_4_Lec_2-1.pdf
  - raw/week-04/4b-predictions-dual-kernels-exercises.pdf
  - raw/week-04/4b-predictions-dual-kernels-answers.pdf
status: draft
updated: 2026-04-26
---

*A "trick" because we get the benefit of working in a high-dimensional embedding $\phi(\mathbf{x})$ — sometimes infinite-dimensional — without ever computing it. Whenever an algorithm depends on data only through inner products $\phi(\mathbf{x})^\top \phi(\mathbf{z})$, we can replace those inner products with a **kernel function** $k(\mathbf{x}, \mathbf{z})$ that returns the same value computed entirely in the original space.*

![[kernel-trick-input-feature-space.png]]

## What is a Kernel Function?

A kernel function takes two points in the original input space and returns a real number:

$$k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}, \qquad k(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top \phi(\mathbf{z})$$

It is the *inner product* of the two points after they've been mapped into some feature space $\phi$. Conceptually, $k$ measures **similarity** — how aligned the two vectors are in the embedded space.

The trick: we don't have to compute $\phi$ to evaluate $k$. For many useful $\phi$, there's a closed-form expression for $k$ that uses only the original-space coordinates.

## Worked Example: Polynomial Kernel of Degree 2

Take $\mathbf{x} = (x_1, x_2)^\top$ and the embedding:

$$\phi(\mathbf{x}) = (1, \sqrt{2} x_1, \sqrt{2} x_2, x_1^2, x_2^2, \sqrt{2} x_1 x_2)^\top$$

(The $\sqrt{2}$ factors are chosen to simplify the algebra below.) Computing the inner product $\phi(\mathbf{x})^\top \phi(\mathbf{z})$ explicitly:

$$\phi(\mathbf{x})^\top \phi(\mathbf{z}) = 1 + 2 x_1 z_1 + 2 x_2 z_2 + x_1^2 z_1^2 + x_2^2 z_2^2 + 2 x_1 x_2 z_1 z_2$$

This factors as $(1 + x_1 z_1 + x_2 z_2)^2 = (1 + \mathbf{x}^\top \mathbf{z})^2$. So:

$$k(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^\top \mathbf{z})^2$$

Two operations in the original space (one inner product, one square) replace six multiplications in the 6-dimensional feature space. As the polynomial degree grows, the saving compounds.

## Why the Dual Lets Us Use Kernels

Recall the SVM dual:

$$\arg\max_{\mathbf{a}} \;\; \sum_n a^{(n)} - \tfrac{1}{2} \sum_n \sum_m a^{(n)} a^{(m)} y^{(n)} y^{(m)} \, \phi(\mathbf{x}^{(n)})^\top \phi(\mathbf{x}^{(m)})$$

The data appears *only* through pairwise inner products $\phi(\mathbf{x}^{(n)})^\top \phi(\mathbf{x}^{(m)})$. Replacing each with $k(\mathbf{x}^{(n)}, \mathbf{x}^{(m)})$ makes the entire optimisation independent of $\phi$:

$$\arg\max_{\mathbf{a}} \;\; \sum_n a^{(n)} - \tfrac{1}{2} \sum_n \sum_m a^{(n)} a^{(m)} y^{(n)} y^{(m)} \, k(\mathbf{x}^{(n)}, \mathbf{x}^{(m)})$$

Same story for prediction:

$$h(\mathbf{x}) = \sum_{n \in S} a^{(n)} y^{(n)} k(\mathbf{x}, \mathbf{x}^{(n)}) + b$$

where $S$ is the set of support-vector indices. Train and predict, all without forming $\phi$ once.

## Kernel Functions Without an Explicit $\phi$

The deeper insight: **we don't need to design $\phi$ first**. We can write down a similarity function $k(\mathbf{x}, \mathbf{z})$ directly, as long as it corresponds to *some* valid inner product in *some* feature space. The condition that decides which functions qualify is **Mercer's condition** (see [[mercers-condition]]).

This unlocks two things:

1. **Infinite-dimensional embeddings**: the [[gaussian-kernel|Gaussian / RBF kernel]] $k(\mathbf{x}, \mathbf{z}) = e^{-\|\mathbf{x} - \mathbf{z}\|^2 / 2\sigma^2}$ corresponds to an *infinite-dimensional* $\phi$, which we'd never be able to compute explicitly.
2. **Non-numeric data**: kernels can be defined directly on strings, trees, graphs, sets — anything for which a similarity function exists. The "feature space" $\phi$ is implicit, never realised.

## When the Kernel Trick Pays Off

In the *primal*, computation per training example is $O(\dim \phi)$. In the *dual* with a kernel, it's $O(\dim \mathbf{x})$ per kernel evaluation, times $N^2$ pairs. Dual is preferred when:

- $\dim \phi \gg N$ (high-dimensional or infinite embedding, modest training set), or
- $\phi$ is implicit (no closed form), or
- the input space is non-numeric.

When $\dim \phi$ is small relative to $N$ (e.g., low-degree polynomial, large dataset), the primal can actually be cheaper.

## Active Recall

> [!question]- The Gaussian kernel corresponds to an infinite-dimensional embedding. Why is this not a computational disaster?
> Because we never compute the embedding. The kernel $k(\mathbf{x}, \mathbf{z}) = e^{-\|\mathbf{x} - \mathbf{z}\|^2 / 2\sigma^2}$ is evaluated entirely in the original $\mathbf{x}$-space — one subtraction, one squared norm, one exponential. The infinite dimensionality is structural (the Taylor series of $e^{\mathbf{x}^\top \mathbf{z}}$ contains arbitrarily high-order monomials), but invisible to the algorithm. We get the *expressive power* of an infinite feature space at the cost of evaluating a single 2-argument function.

> [!question]- Can any function I think captures "similarity" be used as a kernel?
> No. The function must correspond to an inner product in *some* embedding — equivalently, the Gram matrix $K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ must be symmetric and positive semidefinite for any finite set of inputs ([[mercers-condition|Mercer's condition]]). Otherwise the dual SVM optimisation may be ill-posed (non-convex), produce nonsense predictions, or break the geometric guarantees. In practice we either construct $k$ from validated building blocks (linear, polynomial, Gaussian) using composition rules, or verify Mercer's condition directly.

> [!question]- The dual SVM's prediction is $h(\mathbf{x}) = \sum_{n \in S} a^{(n)} y^{(n)} k(\mathbf{x}, \mathbf{x}^{(n)}) + b$. Why does the prediction time depend on the number of *support vectors*, not the number of *training examples*?
> Because of complementary slackness: at the optimum, $a^{(n)} = 0$ for all non-support-vectors. Their contribution to the sum vanishes regardless of $k(\mathbf{x}, \mathbf{x}^{(n)})$. Storing only the support vectors (with their $a^{(n)}$ and $y^{(n)}$) is sufficient to make any future prediction. This is a key efficiency: a typical SVM trained on $N$ points might use only $\sim \sqrt{N}$ or fewer support vectors, making prediction dramatically faster than re-evaluating against the full training set.

## Related

- [[lagrangian]] — duality is what produces a kernel-friendly optimisation
- [[mercers-condition]] — when a candidate function is a valid kernel
- [[gaussian-kernel]] — most popular kernel; infinite-dimensional embedding
- [[polynomial-kernel]] — generalises the worked example above
- [[support-vector-machine]] — the canonical algorithm where the trick lives
