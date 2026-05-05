---
type: concept
name: lasso regression
description: Linear regression with an L1 penalty on the weights. Drives some coefficients to exactly zero, producing sparse solutions and acting as automatic feature selection. The L1 analogue of ridge — Laplace prior in the Bayesian view.
sources:
  - raw/week-10/Wk_10_Lec_2-1.pdf
  - raw/week-10/Thursday_ December 4_ 2025 at 4_05_35 PM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*The **Least Absolute Shrinkage and Selection Operator** — linear regression with an L1 penalty $\lambda \|\mathbf{w}\|_1 = \lambda \sum_i |w_i|$ on the weights. Unlike [[ridge-regression|ridge]], lasso drives some coefficients to *exactly* zero, producing sparse solutions. Useful when the underlying model is sparse (most features irrelevant) and as an automatic feature-selection tool.*

## The Objective

$$\boxed{\mathbf{w}_{\text{lasso}} = \arg\min_{\mathbf{w}} \;\; \frac{1}{2} \sum_{i=1}^N (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 + \lambda \sum_{q=1}^Q |w_q|}$$

Written in matrix form:

$$\mathbf{w}_{\text{lasso}} = \arg\min_{\mathbf{w}} \;\; \frac{1}{2} \|\mathbf{y} - \mathbf{Z} \mathbf{w}\|^2 + \lambda \|\mathbf{w}\|_1.$$

Compared to [[ridge-regression|ridge]], the only difference is the regulariser: $\|\mathbf{w}\|_1$ instead of $\|\mathbf{w}\|_2^2$. Equivalent constrained form: $\min E_{\text{in}}(\mathbf{w})$ subject to $\|\mathbf{w}\|_1 \leq C$.

## Why L1 Produces Sparsity

The constraint sets have qualitatively different shapes:

- **L2 ball** $\{\mathbf{w} : \|\mathbf{w}\|_2 \leq \sqrt{C}\}$: a sphere — smooth, no corners.
- **L1 ball** $\{\mathbf{w} : \|\mathbf{w}\|_1 \leq C\}$: a "diamond" (cross-polytope) — corners and edges *exactly on the coordinate axes*.

The optimisation finds the point in the constraint set closest (in $E_{\text{in}}$ contour terms) to the unregularised optimum $\mathbf{w}_{\text{lin}}$. Geometrically, the contours of $E_{\text{in}}$ — ellipses around $\mathbf{w}_{\text{lin}}$ — first touch the constraint set at:

- For L2: typically a generic point on the sphere — every coordinate non-zero.
- For L1: typically a *corner* of the diamond — some coordinates *exactly zero*.

The corners of the L1 ball are sparse vectors (axis-aligned points), and the geometry makes the optimum land on them generically. This is the source of lasso's sparsity: the corners of the L1 constraint are the sparse solutions, and they're geometrically attractive to the optimum.

## Properties

| Property | L2 (ridge) | L1 (lasso) |
|---|---|---|
| Convex | ✓ | ✓ |
| Differentiable everywhere | ✓ | ✗ (kink at $w_q = 0$) |
| Closed-form solution | ✓ | ✗ (no closed form) |
| Solver | Linear algebra | Quadratic programming, coordinate descent, ISTA, FISTA |
| Sparse solution | ✗ (weights small but non-zero) | ✓ (some weights exactly zero) |
| Stable under correlated features | ✓ | ✗ (picks one of a correlated group) |

The non-differentiability of $|w_q|$ at zero is the source of both lasso's sparsity (gradient has a "discontinuity at zero" that pins weights there) and its computational difficulty (no nice closed form).

## When Lasso Wins

Lasso is the right choice when the **true model is sparse**: most features are irrelevant, and you want the algorithm to select which ones matter.

**Example.** You fit a 50th-order polynomial when the true target is 3rd-order. Lasso recovers the three relevant coefficients and sets the other 47 to zero. Ridge keeps all 50 small but non-zero, hiding the structural sparsity in the noise.

When the true model is dense (most features matter, but with small effects), ridge typically wins: lasso's pickier selection discards information that ridge retains.

The lecture's concrete demonstration: 20-dimensional ground-truth model with 5 non-zero coefficients fit to a 20-degree polynomial with a small $N$. Vanilla regression gives wildly oscillating coefficients (overfitting). Ridge dampens them all but keeps every coefficient non-zero. Lasso correctly identifies the 5 non-zero coefficients and zeroes the rest.

## Bayesian Interpretation

Where ridge corresponds to a Gaussian prior on $\mathbf{w}$, lasso corresponds to a **Laplace** (double-exponential) prior:

$$p(w_q) \propto \exp(-|w_q|/b).$$

The Laplace distribution has heavier tails *and* a sharper peak at zero than a Gaussian of comparable variance. The "sharper peak" is what manifests as sparsity in MAP estimation — the prior prefers exactly-zero weights to slightly-non-zero ones, where Gaussian prefers slight non-zero to exactly zero.

| Regulariser | Prior |
|---|---|
| L2 (ridge) | Gaussian — smooth shrinkage |
| L1 (lasso) | Laplace — sparsity |
| L1 + L2 (elastic net) | Mixture — sparsity with stability under correlation |

## Choosing $\lambda$

Same procedure as ridge: cross-validate on a grid of $\lambda$ values and pick the one minimising held-out error. Commonly, the lasso path (the trajectory of $\mathbf{w}_{\text{lasso}}$ as $\lambda$ varies) is computed once and the cross-validation pick is read off — algorithms like LARS produce the entire path efficiently.

A practical rule: pick the largest $\lambda$ within one standard error of the minimum cross-validation error. The "1-SE rule" prefers slightly more regularisation than strictly optimal — paying a tiny statistical cost for a more parsimonious, more interpretable model.

## What Could Go Wrong

- **Correlated features.** Lasso picks roughly one feature per correlated group and zeros the rest, even if all are equally relevant. The choice between correlated features is unstable across training sets. Elastic net (L1 + L2) addresses this by reintroducing some L2 stability.
- **Dense ground truth.** When most coefficients are non-zero, lasso's aggressive zeroing throws away signal. Ridge is the better default.
- **Standardisation.** Same as ridge: L1 penalises raw weight magnitudes, so input scaling matters. Standardise features first.
- **Selection-induced bias.** Selected features have biased coefficient estimates (they were chosen because the data made them look strong). Post-selection inference requires care.

## Related

- [[ridge-regression]] — L2 analogue; smooth shrinkage instead of sparsity.
- [[regularization-ml|regularization]] — the general framework; lasso is its L1 instance.
- [[bayesian-linear-regression]] — Bayesian view with prior $\propto e^{-\lambda \|\mathbf{w}\|_1}$.
- [[overfitting-ml|overfitting]] — the disease lasso treats.

## Active Recall

> [!question]- Geometrically, why does L1 regularisation drive some coefficients to exactly zero while L2 regularisation does not?
> The L1 constraint set $\{\mathbf{w} : \|\mathbf{w}\|_1 \leq C\}$ is a diamond (cross-polytope) with corners on the coordinate axes — at sparse vectors. The L2 constraint set is a smooth sphere with no corners. The optimum is the point where the contours of $E_{\text{in}}$ first touch the constraint set; for the L1 diamond, this is generically a corner (sparse), and for the L2 sphere, generically a smooth point (dense). The corners of the L1 ball are precisely the sparse vectors, and they're geometrically attractive — the optimum lands on them by default rather than as a special case.

> [!question]- A 20-dimensional regression problem has a true model with only 5 non-zero coefficients. You fit (i) ridge, (ii) lasso, both with cross-validated $\lambda$. Predict qualitatively how the coefficient profiles will differ.
> *Ridge*: all 20 coefficients shrunk towards zero but few exactly zero. The 5 true coefficients will be larger than the 15 spurious ones, but the spurious ones won't vanish — they'll be small non-zero values reflecting the L2 penalty's smooth shrinkage. *Lasso*: the 5 true coefficients will be retained (perhaps slightly biased downward), and the 15 spurious ones will be set to *exactly* zero. Lasso recovers the true sparsity pattern; ridge does not. This is why lasso wins on sparse ground truth — it can express the structural fact "this feature doesn't matter" exactly, where ridge can only approximate it.

> [!question]- Why does lasso fail to have a closed-form solution despite being convex?
> Because the L1 norm $\|\mathbf{w}\|_1 = \sum |w_q|$ is not differentiable at $w_q = 0$. The gradient of the objective involves $\text{sign}(w_q)$, which is discontinuous at zero. Setting the gradient to zero doesn't give a linear system — instead, the optimality conditions involve subgradients (case analysis depending on whether each $w_q$ is positive, negative, or zero at the optimum). Solvers like coordinate descent, ISTA, and FISTA handle this with iterative updates that include a "soft-thresholding" step explicitly accounting for the kink at zero. Convexity ensures convergence to the global optimum; non-differentiability just rules out the tidy closed form.

> [!question]- Lasso corresponds to a Laplace prior in the Bayesian view. What does the shape of the Laplace distribution explain about lasso's behaviour, compared to ridge's Gaussian prior?
> The Laplace distribution $p(w) \propto e^{-|w|/b}$ has a *sharp peak* at zero — its density at exactly zero is much larger relative to nearby values than the Gaussian's. This sharpness expresses a strong prior preference for exactly-zero weights. In MAP estimation, the optimum sits at the prior's peak when data is weak, and the Laplace's peak is exactly at zero — so lasso pins coefficients to zero unless data evidence overcomes the peak. Ridge's Gaussian prior is smooth at zero, so its peak is "spread out" — MAP shrinks weights but rarely lands exactly on zero. The geometric corner-on-axis story for the L1 ball and the probabilistic sharp-peak-at-zero story for the Laplace prior are two views of the same fact.
