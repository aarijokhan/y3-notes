---
type: concept
sources:
  - raw/Wk_1_Lec_1-1.pdf
  - raw/week-01/Monday_ September 29_ 2025 at 3_43_36 PM_Captions_English (United States).txt
status: draft
updated: 2026-04-20
---

*A learning paradigm where a model is trained on input–output pairs and evaluated on its ability to predict outputs for unseen inputs.*

## Definition

Supervised learning is the task of learning a function $g: X \to Y$ from a set of training examples $T = \{(\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \ldots, (\mathbf{x}^{(N)}, y^{(N)})\}$, where each $\mathbf{x}^{(i)} \in X$ is an input and $y^{(i)} \in Y$ is the corresponding output (label). The goal is for $g$ to approximate an unknown target function $f: X \to Y$ well enough to [[generalization|generalize]] to new data drawn from the same distribution.

## The Learning Framework

Every supervised learning problem has five components:

1. **Unknown target function** $f: X \to Y$ — the true relationship between inputs and outputs. We never see $f$ directly; we only observe noisy samples from it.
2. **Training data** $T$ — a finite set of $(x, y)$ pairs drawn i.i.d. from an unknown joint distribution $p(\mathbf{x}, y)$.
3. **Hypothesis set** $\mathcal{H}$ — the family of candidate functions the algorithm is allowed to consider (e.g., all linear functions, all polynomials of degree $\leq 3$). This is the assumption we bring to the table about what $f$ might look like.
4. **Learning algorithm** $\mathcal{A}$ — the procedure that searches $\mathcal{H}$ for the hypothesis that best fits the training data.
5. **Final hypothesis** $g \in \mathcal{H}$ — the output of $\mathcal{A}$; our best approximation of $f$.

The choice of $\mathcal{H}$ is critical. Too small and $f$ may lie outside it; too large and the algorithm may overfit. This tension runs through the entire module.

## Input and Output Spaces

The **input space** $X$ is typically $d$-dimensional. Each dimension (feature) can be:

- **Numeric** — e.g., age, salary (already real-valued).
- **Ordinal** — e.g., expertise $\in \{\text{low}, \text{medium}, \text{high}\}$ (ordered categories, often mapped to numbers like $\{0, 0.5, 1\}$).
- **Categorical** — e.g., car brand $\in \{\text{Fiat}, \text{VW}, \text{Toyota}\}$ (no natural ordering). Typically encoded via **one-hot encoding**: each category becomes a binary dimension.

The **output space** $Y$ determines the task type:

- **Regression**: $Y = \mathbb{R}$ (predict a continuous value, e.g., house price).
- **Classification**: $Y$ is a finite set of categories. Binary classification ($|Y| = 2$) is the most common; multi-class ($|Y| > 2$) extends naturally.

## Noisy Targets

In practice the training data rarely comes from a deterministic $f$. Instead, $y$ is drawn from a conditional distribution $p(y \mid \mathbf{x})$, meaning the same input can map to different outputs. This noise is not a bug — it reflects genuine uncertainty in the real world. The target distribution $p(y \mid \mathbf{x})$ subsumes the deterministic case (where $p$ is a point mass on $f(\mathbf{x})$).

## Other Learning Paradigms

Supervised learning is one of three major paradigms:

| Paradigm | Data | Goal |
|---|---|---|
| Supervised | $\{(\mathbf{x}^{(i)}, y^{(i)})\}$ | Learn $f: X \to Y$ | 
| Unsupervised | $\{\mathbf{x}^{(i)}\}$ (no labels) | Find structure (clusters, density) |
| Reinforcement | States, actions, rewards | Learn policy that maximizes cumulative reward |

![[9bdf846f-b664-4162-a282-4da4aea4c902.png]]

## Related

- [[logistic-regression]] — first supervised classification algorithm in this module
- [[generalization]] — the property that separates learning from memorization
- [[decision-boundary-ml|decision boundary]] — geometric view of classification hypotheses

## Active Recall

> [!question]- Name the five components of the supervised learning framework and explain what each one contributes.
> (1) Unknown target function $f$ — the true mapping we want to approximate. (2) Training data $T$ — the finite sample of input–output pairs we observe. (3) Hypothesis set $\mathcal{H}$ — the family of functions the algorithm is allowed to search. (4) Learning algorithm $\mathcal{A}$ — the procedure that picks the best hypothesis from $\mathcal{H}$. (5) Final hypothesis $g$ — the learned approximation of $f$.

> [!question]- Why does the choice of hypothesis set matter? What goes wrong if it is too small or too large?
> If $\mathcal{H}$ is too small, the true function $f$ may not be representable within it, so no amount of data will produce a good approximation (underfitting). If $\mathcal{H}$ is too large, the algorithm can fit the training noise and fail on unseen data (overfitting). The art is choosing $\mathcal{H}$ large enough to contain a good approximation of $f$ but small enough that the algorithm can reliably find it from limited data.

> [!question]- A dataset has a "car brand" feature with values {Fiat, VW, Toyota}. Explain why you cannot feed these directly into a model that expects numeric inputs, and describe the standard fix.
> The categories have no natural numeric ordering — assigning Fiat = 1, VW = 2, Toyota = 3 would falsely imply that Toyota is "greater than" Fiat. One-hot encoding creates a separate binary dimension per category (e.g., $x_\text{Fiat} \in \{0,1\}$, $x_\text{VW} \in \{0,1\}$, $x_\text{Toyota} \in \{0,1\}$), preserving the fact that categories are unordered.

> [!question]- What does it mean for the target to be a distribution $p(y \mid \mathbf{x})$ rather than a deterministic function $f(\mathbf{x})$?
> It means the same input $\mathbf{x}$ can produce different outputs $y$ on different occasions — there is inherent noise or uncertainty in the data. The deterministic case is a special case where $p(y \mid \mathbf{x})$ is a point mass on a single value. In practice, the learning algorithm must cope with this noise rather than trying to fit every training point exactly.
