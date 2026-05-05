---
week: 1
topic: "What Does It Mean to Learn from Data?"
deck: MachineLearning::Week-01
---

TARGET DECK
MachineLearning::Week-01

## Supervised Learning Framework

> [!question]- What are the five components of the supervised learning framework?
> A learning problem is specified by:
> - **Unknown target distribution** $P(y \mid \mathbf{x})$ — what we're trying to learn
> - **Training data** $\mathcal{T} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$ — drawn i.i.d. from a fixed joint $P(\mathbf{x}, y)$
> - **Hypothesis set** $\mathcal{H}$ — the family of candidate functions
> - **Learning algorithm** $\mathcal{A}$ — the procedure for picking one $h \in \mathcal{H}$
> - **Final hypothesis** $g$ — the chosen function, $g \approx f$
>
> Every algorithm in the module differs only in $\mathcal{H}$ and $\mathcal{A}$.

> [!question]- What is the i.i.d. assumption and why does it matter for supervised learning?
> **Independent and identically distributed**: training and test examples are drawn independently from the *same* joint $P(\mathbf{x}, y)$. It matters because all generalisation guarantees rest on it — if the deployment distribution differs from training (distribution shift), there is no theoretical reason a learned model should perform well, regardless of how good training error looked.

> [!question]- What is **generalisation**, and why is training accuracy alone not enough?
> Generalisation is the ability to perform well on *unseen* examples drawn from the same $P(\mathbf{x}, y)$. A model that memorises the training set achieves zero training error but may be useless on new data — that's a **lookup table, not learning**. Generalisation is the actual goal; training accuracy is a (sometimes misleading) proxy.

## Logistic Regression

> [!question]- Why can't we just model $P(y=1 \mid \mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ directly?
> Because $\mathbf{w}^\top \mathbf{x}$ ranges over $(-\infty, +\infty)$ but a probability must lie in $[0, 1]$. The linear combination has no built-in constraint preventing values like $-500$ or $42$, neither of which is a valid probability.

> [!question]- What is the **logit** function and why is it useful for logistic regression?
> The **logit** maps a probability to the real line:
> $$\text{logit}(p) = \ln \frac{p}{1 - p}$$
> $p \in (0, 1)$ but $\text{logit}(p) \in (-\infty, +\infty)$. Modelling $\text{logit}(p_1) = \mathbf{w}^\top \mathbf{x}$ puts both sides on the same scale. Inverting gives the sigmoid: $p_1 = \sigma(\mathbf{w}^\top \mathbf{x}) = 1/(1 + e^{-\mathbf{w}^\top \mathbf{x}})$.

> [!question]- What is the formula for the **sigmoid** function and what are its key properties?
> $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
> - Range: $(0, 1)$ — outputs are valid probabilities
> - $\sigma(0) = 0.5$ — maximum uncertainty at the boundary
> - Monotonically increasing, smooth, S-shaped
> - Saturates: $\sigma(z) \to 1$ as $z \to \infty$ and $\sigma(z) \to 0$ as $z \to -\infty$
>
> Used to convert linear scores $\mathbf{w}^\top \mathbf{x}$ into class-1 probabilities.

> [!question]- For a logistic regression model with $\mathbf{w}^\top = (0.1, 0.2, 0.6)$ and input $\mathbf{x}^\top = (1, 1, 3)$, what is the predicted class and confidence?
> Compute the linear score: $\mathbf{w}^\top \mathbf{x} = 0.1 + 0.2 + 1.8 = 2.1$.
> Since $2.1 \geq 0$, predict **class 1**.
> Confidence: $p_1 = \sigma(2.1) = e^{2.1}/(1 + e^{2.1}) \approx 0.891$ — about **89%**.

> [!question]- How do you interpret a logistic regression weight $w_j$ in terms of odds?
> A unit increase in feature $x_j$ multiplies the odds of class 1 by $e^{w_j}$:
> $$\frac{p_1}{1 - p_1} \to e^{w_j} \cdot \frac{p_1}{1 - p_1}$$
> If $w_j = 0.5$, a one-unit bump in $x_j$ raises the odds by $e^{0.5} \approx 1.65$ — a 65% increase. This is why logistic regression is popular in social/natural sciences: the coefficients are directly interpretable.

## Decision Boundary

> [!question]- What is the **decision boundary** of a logistic regression classifier?
> The hyperplane $\mathbf{w}^\top \mathbf{x} = 0$. Points where $\mathbf{w}^\top \mathbf{x} > 0$ are predicted as class 1 (since $\sigma(z) > 0.5$); points where $\mathbf{w}^\top \mathbf{x} < 0$ are class 0. The further a point is from the boundary, the more confident the prediction.

> [!question]- Why is logistic regression called a **linear** model when the sigmoid is non-linear?
> "Linear" refers to **linearity in the parameters $\mathbf{w}$**, not in the inputs $\mathbf{x}$. The decision boundary $\mathbf{w}^\top \mathbf{x} = 0$ is a hyperplane (linear in $\mathbf{x}$), and the cross-entropy loss is convex in $\mathbf{w}$. The sigmoid is a fixed non-linearity wrapping a linear-in-$\mathbf{w}$ score.

## Discriminative vs Generative

> [!question]- What is the difference between **discriminative** and **generative** classifiers?
> - **Discriminative**: model $P(y \mid \mathbf{x})$ directly. Doesn't ask how features are distributed; just learns the decision boundary. *Logistic regression, SVM, neural networks.*
> - **Generative**: model $P(\mathbf{x} \mid y) \cdot P(y)$, then derive $P(y \mid \mathbf{x})$ via Bayes' rule. Asks "what does a class-1 input look like?" *Naïve Bayes, Gaussian discriminant analysis.*
>
> Discriminative is generally better for raw classification accuracy; generative supplies more information (sampling, missing-feature handling) at the cost of stronger modelling assumptions.

> [!question]- A model gets 100% training accuracy but 55% test accuracy. What has gone wrong?
> The model has **overfitted**: it memorised the training data instead of learning generalisable structure. Likely cause: $\mathcal{H}$ is too expressive relative to the amount of training data, letting the algorithm fit noise. Remedies (covered later): regularisation, more data, simpler model class, validation-based hyperparameter selection.
