---
type: concept
sources:
  - raw/Wk_1_Lec_1-1.pdf
  - raw/week-01/Monday_ September 29_ 2025 at 3_43_36 PM_Captions_English (United States).txt
status: draft
updated: 2026-04-20
---

*The ability of a learned hypothesis to perform well on new, unseen data drawn from the same distribution as the training set — the property that separates genuine learning from memorization.*

## Definition

A hypothesis $g$ **generalizes** if its error on the full data distribution $p(\mathbf{x}, y)$ is low, not just its error on the finite training set $T$. Formally, the **generalization error** (or test error) is:

- **Classification:** $E(g) = P_{(\mathbf{x}, y) \sim p}\big[\mathbb{1}[g(\mathbf{x}) \neq y]\big]$ (misclassification probability)
- **Regression:** $E(g) = \mathbb{E}_{(\mathbf{x}, y) \sim p}\big[(g(\mathbf{x}) - y)^2\big]$ (expected squared error)

We never observe $E(g)$ directly because we don't have access to $p(\mathbf{x}, y)$. We can only estimate it from held-out test data.

## Why Memorization Fails

![[77e9e348-3917-4096-824e-9f996ce6a61b.png]]

A model that perfectly fits every training point may simply be reciting the training set rather than capturing the underlying pattern. Consider a lookup table that maps each training input to its training label: it achieves zero training error but tells us nothing about inputs it hasn't seen. The gap between **training error** (error on $T$) and **generalization error** (error on the full distribution) is the central tension in machine learning.

## What Makes Generalization Possible

Two assumptions make generalization tractable:

1. **i.i.d. sampling** — Training and test data are drawn independently from the same distribution $p(\mathbf{x}, y)$. If the test distribution differs (distribution shift), generalization guarantees break down.
2. **Restricted hypothesis set** — The learning algorithm searches within a limited family $\mathcal{H}$, not over all possible functions. A smaller $\mathcal{H}$ is easier to learn from limited data but risks excluding the true function. This trade-off is formalized by VC dimension and bias–variance analysis (covered in weeks 8–10).

## Two Pillars: Closeness and Attainability

A useful way to break the generalization problem in two: it succeeds only when **both** of the following hold.

> [!info] ASIDE — The marble jar
> Imagine a giant jar full of marbles, where the fraction of red marbles is $\mu$ — the *true* error rate of our hypothesis on the entire distribution. We can't count them all. Instead we pull out a small handful (the training set) and measure $\nu$, the fraction of red marbles in our hand — the in-sample error $E_\text{in}$. Generalization rests on whether $\nu$ tells us anything reliable about $\mu$.

**Pillar 1 — Closeness:** *Is the in-sample error a faithful estimate of the out-of-sample error?* That is, $E_\text{in} \approx E_\text{out}$. This is a property of *sampling* and *hypothesis-set complexity*, not of the model's accuracy. The i.i.d. assumption gives us probabilistic tools (Hoeffding's inequality, VC theory) that bound how far apart $E_\text{in}$ and $E_\text{out}$ can be. Roughly: with enough data and a not-too-flexible $\mathcal{H}$, the marble fraction in our hand reflects the fraction in the jar.

**Pillar 2 — Attainability:** *Did we actually find a hypothesis with low $E_\text{in}$?* This is what the learning algorithm $\mathcal{A}$ delivers — picking a $g \in \mathcal{H}$ that fits the training data well. A hypothesis set that's too restrictive may make low $E_\text{in}$ unreachable (underfitting); a hypothesis that we couldn't find via optimisation also fails this pillar.

Both pillars are necessary. Either alone is useless:

| Closeness | Attainability | Outcome |
|---|---|---|
| ✓ ($E_\text{in} \approx E_\text{out}$) | ✓ (low $E_\text{in}$) | **Learning succeeded** — low test error |
| ✓ | ✗ (high $E_\text{in}$) | **Underfitting** — model too simple |
| ✗ ($E_\text{in} \ll E_\text{out}$) | ✓ | **Overfitting** — memorised noise |
| ✗ | ✗ | Total failure |

The two pillars are in tension. Making $\mathcal{H}$ richer (more flexible) helps **attainability** — there's a better hypothesis available — but hurts **closeness**, because more flexible models can fit training noise and pull $E_\text{in}$ below $E_\text{out}$. This is exactly the bias-variance / capacity-control trade-off; the sweet spot balances both pillars.

## Connection to Occam's Razor

Simpler hypotheses — those from smaller hypothesis sets — are less likely to be fitting noise and more likely to generalize. This is an informal statement of what VC theory and regularization make precise later in the module.

## Related

- [[supervised-learning]] — the framework where generalization is the central objective
- [[decision-boundary-ml|decision boundary]] — the geometric object whose quality is measured by generalization error

## Active Recall

> [!question]- A model achieves 100% accuracy on the training set but 55% on the test set. What has gone wrong, and what does this reveal about the hypothesis set?
> The model has **overfitted**: it memorized the training data (including noise) rather than learning the underlying pattern. The large gap between training and test accuracy suggests the hypothesis set $\mathcal{H}$ is too expressive for the amount of training data available — the model has enough capacity to fit noise.

> [!question]- Why is the i.i.d. assumption critical for generalization? Give a concrete scenario where violating it causes trouble.
> The i.i.d. assumption ensures that the training data is representative of the data the model will encounter at test time. If violated — for example, training a house-price model on data from London and then deploying it in Dubai — the test distribution differs from the training distribution (distribution shift), and the model's learned patterns may not transfer.

> [!question]- Explain the tension between making the hypothesis set $\mathcal{H}$ smaller versus larger, in terms of generalization.
> A smaller $\mathcal{H}$ is easier to learn from limited data (less risk of overfitting) but may not contain a good approximation of the true function $f$ (underfitting). A larger $\mathcal{H}$ can represent more complex relationships but requires more data to reliably pick the right hypothesis — with limited data, the algorithm may latch onto noise. The sweet spot depends on both the complexity of the true function and the amount of available training data.
