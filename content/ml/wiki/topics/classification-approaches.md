---
type: topic
name: Classification Approaches in This Module
description: Synthesis comparing logistic regression, hard-margin SVM, and soft-margin SVM — what each models, what each assumes, and when each wins
status: draft
updated: 2026-04-27
---

*Three binary classifiers appear in weeks 1–5: logistic regression, hard-margin SVM, and soft-margin SVM. They share a hypothesis form — a hyperplane in (possibly transformed) feature space — but differ in what they treat as "good": calibrated probabilities, maximum margin, or maximum margin with controlled violation. The choice depends on whether you need probabilities, whether your data is separable, and whether your features live in a kernelisable space.*

## The Three Classifiers at a Glance

| | [[logistic-regression\|Logistic Regression]] | [[support-vector-machine\|Hard-Margin SVM]] | [[soft-margin-svm\|Soft-Margin SVM]] |
|---|---|---|---|
| **Output** | Probability $p_1 = \sigma(\mathbf{w}^\top \mathbf{x})$ | Class label only | Class label only |
| **Hypothesis** | $h(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x})$ | $h(\mathbf{x}) = \mathrm{sign}(\mathbf{w}^\top \phi(\mathbf{x}) + b)$ | Same as hard, with slack |
| **Training criterion** | Maximum likelihood (cross-entropy) | Maximum margin | Margin – $C \cdot$(slack) |
| **Loss** | $-\sum y \log p + (1-y) \log(1-p)$ | None — pure constraint | Hinge: $\max(0, 1 - y h)$ |
| **Probabilistic?** | Yes, calibrated | No | No |
| **Sparse in training data?** | No — all examples contribute via gradient | Yes — only support vectors | Yes — non-bound + bound SVs |
| **Handles non-separable?** | Yes (probabilistic, no hard constraint) | **No** — infeasible | Yes (slack absorbs violations) |
| **Kernelisable?** | Awkward — primal-only by default | Yes (dual form) | Yes (dual form) |
| **Optimizer** | GD or [[iteratively-reweighted-least-squares\|IRLS]] | [[sequential-minimal-optimization\|SMO]] on the dual | SMO with box constraint |
| **Hyperparameters** | None (regularisation optional) | None | $C$ (slack penalty) |

## What Each One Models

**Logistic regression** models the conditional probability $P(y \mid \mathbf{x})$ directly. The *decision* (which class?) is downstream of a real-valued probability output. It is a [[discriminative-vs-generative-models|discriminative]] probabilistic classifier.

**Hard-margin SVM** models the *decision boundary* — the hyperplane itself — and chooses among separating boundaries the one with the largest [[margin]]. There is no probabilistic interpretation; the prediction is the sign of $\mathbf{w}^\top \phi(\mathbf{x}) + b$. The choice of boundary is geometric, not statistical.

**Soft-margin SVM** is the same geometric idea, relaxed: still maximises margin, but allows training points to violate the margin (or even be misclassified) at a controlled cost. Adds the hyperparameter $C$ that prices each unit of [[slack-variables|slack]].

The fundamental split: **logistic regression is statistical** (fitting a probability model via MLE); **SVMs are geometric** (placing a maximum-margin hyperplane via constrained optimisation).

## The Same Hypothesis, Different Criteria

All three predict via a linear function in a (possibly transformed) feature space:

$$h(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x}) + b$$

Logistic regression squashes this with the sigmoid; SVMs threshold its sign. The hypothesis sets are *the same family* — what differs is **the criterion that picks $\mathbf{w}, b$**:

- **Logistic regression** picks the $\mathbf{w}$ that maximises the likelihood of the labels under the assumed Bernoulli model. Equivalently, minimises cross-entropy. Every training example contributes to the objective; well-classified examples just contribute less.
- **Hard-margin SVM** picks the $\mathbf{w}$ that maximises the margin subject to perfect separation. Most training examples are inactive — only the [[margin|margin]]-touching support vectors influence the boundary.
- **Soft-margin SVM** picks the $\mathbf{w}$ that balances margin width against total slack. Both margin support vectors *and* margin-violators are active; well-classified, comfortably-outside examples are inactive.

## When Each One Wins

**Use logistic regression when:**
- You need calibrated probabilities (medical risk, fraud scoring, threshold tuning).
- The data may not be separable; you want a graceful answer rather than infeasibility.
- You want a fully parametric model that's deployable on embedded systems (just a dot product + sigmoid; no support vectors to ship).
- Feature interpretability matters — each weight has a direct odds-ratio meaning.

**Use hard-margin SVM when:**
- The data is genuinely linearly separable (or after a kernel) and you want a maximum-margin boundary.
- In practice this is rare — almost everyone uses soft-margin. Hard-margin appears as a stepping stone to introduce the dual + kernel trick.

**Use soft-margin SVM when:**
- You want a geometrically robust boundary with kernel-based non-linearity (RBF, polynomial, custom).
- Probabilities are not required.
- The training set is moderate (≤ ~$10^5$): SMO is fast at this scale; large-$N$ regimes favour primal solvers.
- You're willing to cross-validate $C$.

## The Hinge Loss vs Cross-Entropy Loss

The cleanest way to see the difference is through the loss functions. Both are functions of the **margin** $z = y \cdot h(\mathbf{x})$:

| Loss | Formula | Behaviour |
|---|---|---|
| Cross-entropy (LogReg) | $\log(1 + e^{-z})$ | Smooth, differentiable everywhere; positive even for confident correct predictions |
| Hinge (Soft-margin SVM) | $\max(0, 1 - z)$ | Piecewise linear; **exactly zero** for $z \geq 1$ (correctly classified beyond the margin) |
| 0-1 loss (the "true" objective) | $\mathbb{1}[z < 0]$ | Non-convex, discontinuous — both above are convex relaxations |

Both losses are convex upper bounds on the 0-1 loss. The structural difference: **cross-entropy is everywhere positive**, so every training example pulls on $\mathbf{w}$ — hence no sparsity. **Hinge is exactly zero past the margin**, so well-classified examples drop out entirely — hence support vectors.

This single fact (zero vs strictly-positive loss past the margin) is the structural reason SVMs are sparse and logistic regression isn't.

## What They Have In Common

- **Linear boundary in $\phi$-space.** Both are linear classifiers; both gain non-linearity via [[non-linear-transformation|basis expansion]] or [[kernel-trick|kernels]].
- **Convex optimisation.** All three lead to convex objectives. Unique global optimum.
- **Discriminative.** All three model the boundary or conditional probability directly, not the joint distribution. Compare with naive Bayes / LDA, which are generative.
- **Need feature scaling.** All three are sensitive to feature magnitudes — standardise before training.

## What Goes Wrong

| Model | Failure mode |
|---|---|
| Logistic regression | Linearly separable data → unbounded weights (sigmoid saturates); fix with regularisation. Multicollinear features → unstable coefficients. |
| Hard-margin SVM | Non-separable data → no feasible solution. Single noisy point → boundary contortion with tiny margin. |
| Soft-margin SVM | $C$ too large → behaves like hard-margin (overfit). $C$ too small → boundary collapses (underfit). |
| All three | Wrong feature scale dominates kernel/distance computations. High-dim data without regularisation overfits. |

## The Sparsity Question

A key practical difference: **what does deployment look like?**

- Logistic regression: ship $\mathbf{w}$. Inference is one dot product. Compact, fast, embedded-friendly.
- SVM: ship the support vectors and their multipliers. Inference is $\sum_{n \in S} a^{(n)} y^{(n)} k(\mathbf{x}, \mathbf{x}^{(n)})$ — proportional to the number of support vectors. With Gaussian kernels and large datasets, $|S|$ can be substantial.

When deployment cost matters (mobile, edge), this can decide. When training-time accuracy matters more, SVMs with kernels often win.

## What's Probabilistic and What Isn't

Logistic regression's $p_1$ is a real probability — under the model's assumption, it's calibrated. SVMs have no native probability output. **Platt scaling** (fitting a sigmoid to SVM scores via held-out data) is the standard hack to extract probabilities, but it's a post-hoc calibration, not a principled output. If you genuinely need probabilities, prefer logistic regression — or pay the calibration tax.

## Where We're Headed

The module continues with:
- **Regression** — same hyperplane, continuous output, MSE loss instead of cross-entropy. Bayesian regression adds priors to regularise.
- **Generalisation theory** — formalises why margin maximisation and regularisation help, via VC dimension and bias-variance.
- **Validation** — how to actually pick $C$ (and other hyperparameters) without peeking at the test set.

## Connections

- [[logistic-regression]] — probabilistic discriminative classifier; MLE-fit; cross-entropy loss.
- [[support-vector-machine]] — hard-margin SVM; constrained QP; sparse via complementary slackness.
- [[soft-margin-svm]] — slack-relaxed SVM; box constraint $a \leq C$; the practical default.
- [[slack-variables]] — the mechanism that makes SVMs work on non-separable data.
- [[margin]] — the geometric quantity SVMs maximise.
- [[decision-boundary-ml|decision boundary]] — the geometric object all three classifiers produce.
- [[kernel-trick]] — what makes SVMs vastly more flexible than the linear case suggests.
- [[discriminative-vs-generative-models]] — why all three are discriminative, not generative.
