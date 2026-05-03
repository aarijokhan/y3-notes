---
type: concept
sources:
  - raw/Wk_1_Lec_2-1.pdf
  - raw/week-01/Tuesday_ September 30_ 2025 at 10_01_17 AM_Captions_English (United States).txt
status: stub
updated: 2026-04-20
---

*Two paradigms for building classifiers: discriminative models learn the boundary between classes directly, while generative models learn how each class produces data and infer the boundary from that.*

## Definition

- **Discriminative models** learn $P(y \mid \mathbf{x})$ directly — given features, what is the probability of each class? They focus on what distinguishes the classes from each other.
- **Generative models** learn $P(\mathbf{x} \mid y) \cdot P(y)$ — how likely is this data given each class, and how common is each class? Classification then follows from Bayes' rule: $P(y \mid \mathbf{x}) \propto P(\mathbf{x} \mid y) \cdot P(y)$.

## Comparison

| | Discriminative | Generative |
|---|---|---|
| Models | $P(y \mid \mathbf{x})$ | $P(\mathbf{x} \mid y)$ and $P(y)$ |
| Goal | Find the boundary | Model the data-generating process |
| Example | [[logistic-regression]] | Naive Bayes, Bayesian classifiers |
| Interpretability | Weights show which features discriminate | Can generate synthetic examples |

## When Each Is Used

Discriminative models tend to achieve higher classification accuracy when training data is plentiful, because they focus all modelling capacity on the decision boundary. Generative models are more useful when data is scarce (priors help), when you need to detect outliers, or when understanding the data distribution itself is the goal.

This distinction becomes concrete in weeks 7–8 when Bayesian methods are introduced as generative counterparts to the discriminative classifiers covered in weeks 1–5.

## Related

- [[logistic-regression]] — the primary discriminative classifier in weeks 1–5
- [[supervised-learning]] — the framework both paradigms operate within

## Active Recall

> [!question]- Logistic regression is a discriminative classifier. What exactly does it model, and what does it *not* model?
> It models $P(y \mid \mathbf{x})$ — the probability of each class given the input features. It does *not* model $P(\mathbf{x} \mid y)$ — how the features are distributed within each class. It has no concept of what a "typical" class-1 input looks like; it only knows which side of the boundary an input falls on.

> [!question]- A generative classifier models $P(\mathbf{x} \mid y)$ and $P(y)$. How does it use these to classify a new input $\mathbf{x}$?
> It applies Bayes' rule: $P(y \mid \mathbf{x}) \propto P(\mathbf{x} \mid y) \cdot P(y)$. For each class $y$, it computes the likelihood of observing $\mathbf{x}$ under that class times the prior probability of the class, then predicts the class with the highest posterior probability.

> [!question]- Why might a discriminative model outperform a generative model when training data is plentiful?
> A discriminative model devotes all its modelling capacity to learning the decision boundary — the quantity that directly determines classification accuracy. A generative model must also accurately model the full feature distribution $P(\mathbf{x} \mid y)$, which is a harder problem and may waste capacity on aspects of the distribution that don't affect the boundary. With enough data, the discriminative model's focused objective tends to yield better classification performance.
