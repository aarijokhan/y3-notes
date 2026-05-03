---
type: week
week: 1
title: "What Does It Mean to Learn from Data?"
dates: 2025-09-29 to 2025-10-05
sources:
  - raw/Wk_1_Lec_1-1.pdf
  - raw/Wk_1_Lec_2-1.pdf
  - raw/week-01/Monday_ September 29_ 2025 at 3_43_36 PM_Captions_English (United States).txt
  - raw/week-01/Tuesday_ September 30_ 2025 at 10_01_17 AM_Captions_English (United States).txt
  - raw/week-01/1b-logistic-regression-hypothesis-set-exercises.pdf
  - raw/week-01/1b-logistic-regression-hypothesis-set-answers.pdf
concepts:
  - "[[supervised-learning]]"
  - "[[logistic-regression]]"
  - "[[sigmoid-function-ml|sigmoid function]]"
  - "[[generalization]]"
  - "[[decision-boundary-ml|decision boundary]]"
  - "[[discriminative-vs-generative-models]]"
status: draft
updated: 2026-04-20
---

> [!question]+ THE CRUX: What does it actually mean to "learn from data" — and how do we build a machine that classifies things it has never seen?

*We formalise learning as searching a hypothesis set for a function that generalises beyond the training data, then build our first concrete classifier — logistic regression — which turns a linear score into a probability via the sigmoid function.*

---

Imagine you work at a bank. Every day, people apply for credit cards. Some will pay their bills; some won't. You have a decade of historical data — ages, salaries, employment histories, repayment outcomes. Can you write a program that looks at a *new* applicant and predicts which group they fall into?

You could try writing rules by hand: "if salary > 50k and years employed > 3, approve." But there are dozens of features and the relationships between them are tangled. What you really want is a program that *figures out the rules on its own* by studying the historical data. That's the promise of machine learning.

## The Framework

Let's be precise about what "learning" means. Behind the data, there's some unknown function $f$ that maps applicant features to outcomes. We'll never see $f$ — we only get a noisy sample of input–output pairs. Our job is to pick a hypothesis $g$ from some family of candidates $\mathcal{H}$ that approximates $f$ well enough to be useful on *new* applicants.

This gives us the five-piece [[supervised-learning]] framework that the entire module is built on: an unknown target $f$, training data $T$, a hypothesis set $\mathcal{H}$, a learning algorithm $\mathcal{A}$, and a final hypothesis $g$. Every algorithm we meet — from logistic regression to SVMs to Bayesian regressors — is just a different choice of $\mathcal{H}$ and $\mathcal{A}$.

> [!info] ASIDE — Definitions through the decades
> Arthur Samuel (1959) called ML "the field giving computers the ability to learn without being explicitly programmed." Tom Mitchell (1998) sharpened it: a program *learns* if its performance on task $T$, measured by $P$, improves with experience $E$. The Mitchell definition is the one that matters for this module — it forces you to specify what "improvement" means.

The framework also highlights a crucial assumption: the training data and future test data come from the *same* distribution $p(\mathbf{x}, y)$, drawn independently. If the world shifts between training and deployment, all bets are off.

## The Real Goal: Generalisation

Here's the trap. A model that memorises every training point gets zero training error — and might be completely useless on new data. A lookup table is not learning.

[[Generalization]] is the property we actually care about: low error on *unseen* examples from $p(\mathbf{x}, y)$. The tension between fitting the training data and performing well on new data is the thread that runs through the entire module — we'll formalise it with VC dimensions and bias–variance analysis in weeks 8–10, but keep it in mind from day one.

> [!question]- If a model achieves 100% training accuracy but 55% test accuracy, what has gone wrong — and which component of the framework is the likely culprit?
> The model has overfitted: it memorised the training data rather than learning the underlying pattern. The hypothesis set $\mathcal{H}$ is likely too large (too expressive) for the amount of training data available, allowing the learning algorithm to fit noise.

## Our First Classifier: Logistic Regression

With the framework in place, let's build something. We want a binary classifier: given features $\mathbf{x}$, output a probability that the applicant belongs to class 1 (will pay).

A natural first thought: model $P(y = 1 \mid \mathbf{x})$ as a linear combination $\mathbf{w}^\top \mathbf{x}$. The problem is immediate — $\mathbf{w}^\top \mathbf{x}$ can be any real number, but probabilities live in $[0, 1]$. You can't have a probability of $-500$.

[[Logistic-regression]] solves this with a two-step trick:

1. Model the **log-odds** (logit) as the linear combination: $\ln \frac{p_1}{1 - p_1} = \mathbf{w}^\top \mathbf{x}$. The logit maps $(0, 1)$ to $(-\infty, +\infty)$, so both sides live on the same scale.
2. Invert via the [[sigmoid-function-ml|sigmoid function]]: $p_1 = \sigma(\mathbf{w}^\top \mathbf{x}) = 1 / (1 + e^{-\mathbf{w}^\top \mathbf{x}})$.

The sigmoid is the key piece of machinery. It's S-shaped, smooth, and squashes any real number into a valid probability. At $\mathbf{w}^\top \mathbf{x} = 0$ it returns $0.5$ — maximum uncertainty. As $\mathbf{w}^\top \mathbf{x}$ grows large (positive or negative), the probability saturates toward 1 or 0.

> [!tip] TIP — The name is misleading
> Despite the word "regression," logistic regression is a classifier. The "regression" refers to the fact that we're regressing the log-odds on the features. If someone asks "is logistic regression a regression algorithm?" on an exam, the answer is no — it's a classification algorithm.

Classification is straightforward: the [[decision-boundary-ml|decision boundary]] is the hyperplane $\mathbf{w}^\top \mathbf{x} = 0$. Points on the positive side are class 1; points on the negative side are class 0. The further a point is from the boundary, the more confident the model is.

> [!question]- You've trained a logistic regression model with $\mathbf{w}^\top = (0.1, 0.2, 0.6)$. A new applicant arrives with features $\mathbf{x}^\top = (1, 1, 3)$ (where $x_0 = 1$ is the dummy variable). What class does the model predict, and how confident is it?
> $\mathbf{w}^\top \mathbf{x} = 0.1 + 0.2 + 1.8 = 2.1 \geq 0$, so the model predicts class 1. The probability is $p_1 = e^{2.1} / (1 + e^{2.1}) \approx 0.891$ — about 89% confidence.

One useful interpretability feature: each weight $w_j$ has a direct meaning. A unit increase in $x_j$ multiplies the odds of class 1 by $e^{w_j}$. If $w_j = 0.5$, a one-unit bump in that feature increases the odds by about 65%. This is why logistic regression is popular in the social and natural sciences — you can point to a specific feature and say *this is how much it matters*.

> [!info] ASIDE — Discriminative vs. generative
> Logistic regression is a [[discriminative-vs-generative-models|discriminative]] classifier: it models $P(y \mid \mathbf{x})$ directly without asking how the features themselves are distributed. A generative classifier would model $P(\mathbf{x} \mid y) \cdot P(y)$ — "what does a typical class-1 input look like?" — and derive $P(y \mid \mathbf{x})$ via Bayes' rule. We'll meet generative classifiers in weeks 7–8.

## What's Missing

We've defined the hypothesis set $\mathcal{H} = \{\sigma(\mathbf{w}^\top \mathbf{x}) \mid \mathbf{w} \in \mathbb{R}^{d+1}\}$, but we haven't said how to *find* the best $\mathbf{w}$. That's the learning algorithm $\mathcal{A}$ — and it's the subject of week 2, where we'll set up maximum likelihood estimation and gradient descent.

We also haven't addressed what happens when the data isn't linearly separable. Logistic regression draws a straight boundary, and if the true classes can't be separated by a hyperplane, it will do its best but inevitably misclassify some points. Non-linear transformations and SVMs (weeks 3–5) handle this.

> [!question]- The supervised learning framework has five components. We've now specified $\mathcal{H}$ for logistic regression. Which component is still missing, and what will it need to do?
> The learning algorithm $\mathcal{A}$ — the procedure for finding the weight vector $\mathbf{w}$ within $\mathcal{H}$ that best fits the training data. It will need to define a loss function (measuring how wrong a given $\mathbf{w}$ is) and an optimisation method for minimising that loss (e.g., gradient descent).

---

## Concepts Introduced This Week

- [[supervised-learning]] — the five-component framework: target $f$, data $T$, hypothesis set $\mathcal{H}$, algorithm $\mathcal{A}$, final hypothesis $g$.
- [[generalization]] — the central challenge: performing well on unseen data, not just the training set.
- [[logistic-regression]] — binary classifier that models log-odds as a linear combination of features.
- [[sigmoid-function-ml|sigmoid function]] — the S-shaped function $\sigma(z) = 1/(1+e^{-z})$ that maps linear scores to probabilities.
- [[decision-boundary-ml|decision boundary]] — the hyperplane $\mathbf{w}^\top \mathbf{x} = 0$ separating the two classes.
- [[discriminative-vs-generative-models]] — two paradigms for classification; logistic regression is discriminative.

## Connections

- **Sets up** [[week-02]]: How do we actually learn $\mathbf{w}$? Maximum likelihood estimation, gradient descent, and their pitfalls.

## Open Questions

- How do we find the optimal $\mathbf{w}$? (Answered in week 2: MLE + gradient descent.)
- What if the data can't be separated by a straight line? (Answered in weeks 3–5: non-linear transforms, kernels, SVMs.)
- Is learning *always* possible, or are there problems no algorithm can solve? (Answered in weeks 8–10: VC theory, bias–variance.)
