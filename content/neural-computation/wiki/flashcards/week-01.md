---
week: 1
topic: "The Perceptron and the Optimisation Problem"
deck: NeuralComputation::Week-01
---

TARGET DECK
NeuralComputation::Week-01

## Perceptron

> [!question]- What is a **perceptron**, mathematically?
> A perceptron computes $\hat{y} = \text{sgn}(b + \mathbf{w} \cdot \mathbf{x})$, where $\mathbf{x}$ is the input vector, $\mathbf{w}$ is a learned weight vector, $b$ is a learned **bias**, and $\text{sgn}$ outputs $+1$ or $-1$. It is the simplest single-neuron classifier.

> [!question]- How does removing the sign function from a perceptron change its job?
> Without $\text{sgn}$, the output $\hat{y} = b + \mathbf{w} \cdot \mathbf{x}$ is a continuous real number — the model is now doing **linear regression** instead of classification. Same weights, same bias, same architecture, just a different output type.

> [!question]- What does the **dot product** $\mathbf{w} \cdot \mathbf{x}$ measure geometrically in a perceptron?
> The signed perpendicular distance (scaled by $\|\mathbf{w}\|$) from the point $\mathbf{x}$ to the decision hyperplane. Positive means $\mathbf{x}$ lies on the same side as $\mathbf{w}$; negative means the opposite side. The sign function turns this distance into a hard $\pm 1$ classification.

> [!question]- In a perceptron, what controls the **tilt** vs the **position** of the decision boundary?
>
> - **Tilt (orientation):** the weight vector $\mathbf{w}$ — the boundary is always perpendicular to $\mathbf{w}$.
> - **Position (offset from origin):** the bias $b$ — the boundary sits at signed distance $-b/\|\mathbf{w}\|$ from the origin along $\mathbf{w}$.

## Linear separability

> [!question]- What does it mean for a dataset to be **linearly separable**, and why does this matter for a single perceptron?
> A dataset is linearly separable if a single hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$ can split the classes correctly. A single perceptron can *only* solve linearly separable problems — XOR and concentric-circle patterns cannot be classified by any single perceptron, no matter how the weights are chosen.

## Loss & optimisation

> [!question]- Why is "learning is optimisation" the central idea of week 1?
> Learning means picking the parameters $(\mathbf{w}, b)$ that make predictions as close as possible to the truth. We formalise "close" with a **loss function** $L$, then solve $\mathbf{w}^*, b^* = \arg\min_{\mathbf{w}, b} L(\mathbf{w}, b)$. Every later technique (gradient descent, backprop, etc.) is just machinery for solving this minimisation.

> [!question]- What is the difference between $\arg\min$ and $\min$?
> $\min_x f(x)$ returns the smallest value $f$ takes. $\arg\min_x f(x)$ returns the input $x$ at which that smallest value is achieved. In ML we want the *parameters* that minimise loss, not the loss value itself — so we always write $\arg\min$.

## MLE

> [!question]- What does **maximum likelihood estimation (MLE)** ask?
> $$\hat{\theta}_{\text{MLE}} = \arg\max_{\hat{\theta}} \; P(\text{data} \mid \hat{\theta})$$
> Out of all possible parameter values, MLE picks the one that would have made the observed data most probable. The data is treated as fixed (we already saw it); $\hat{\theta}$ is what we sweep over.

> [!question]- What is the difference between **probability** and **likelihood**, given the same formula $P(\text{data} \mid \theta)$?
> Both use the same formula but flip what is fixed:
>
> - **Probability:** fix $\theta$, vary the data — "if the true mean is 20°C, what readings might we see?"
> - **Likelihood:** fix the data, vary $\theta$ — "given we observed 23°C, which $\theta$ best explains it?"
>
> MLE is a likelihood problem: data observed, parameter swept.

> [!question]- Under what assumptions does maximising the likelihood reduce to minimising the **sum of squared errors**?
> When (1) observations are **independent** and (2) noise around the true value is **Gaussian**. The Gaussian PDF contains $\exp(-(x_i - \hat{x})^2 / 2\sigma^2)$, so taking $\log$ of the product gives a sum of $-(x_i - \hat{x})^2$ terms (plus constants). Dropping constants and flipping the sign turns $\arg\max$ into $\arg\min \sum (x_i - \hat{x})^2$.

> [!question]- Why is it valid to apply $\log$ to the likelihood before optimising?
> Because $\log$ is **monotonically increasing**: if $f(a) > f(b)$ then $\log f(a) > \log f(b)$. The location of the maximum doesn't move. Three concrete benefits: it turns products into sums (easier to differentiate), cancels the Gaussian's $\exp$, and avoids floating-point underflow when multiplying many tiny probabilities.

> [!question]- In the MLE-to-SSE derivation, why is it valid to **drop** the $\frac{1}{\sqrt{2\pi}\sigma}$ and $2\sigma^2$ factors?
> We are optimising over $\hat{x}$, not $\sigma$. Those factors are constants with respect to $\hat{x}$, so they shift or scale the loss curve uniformly — every candidate $\hat{x}$ moves by the same amount. The location of the optimum is unchanged.

> [!question]- Why do we **flip the sign** in the final step of the MLE-to-SSE derivation?
> The log-likelihood is $-\sum (x_i - \hat{x})^2$. MLE *maximises* this, but ML training conventionally *minimises* loss. Maximising $-f$ is equivalent to minimising $f$, so we negate and switch from $\arg\max$ to $\arg\min$, giving $\arg\min \sum (x_i - \hat{x})^2$.

> [!question]- A thermometer reads $\{19, 17, 24\}$°C. Under MLE with Gaussian noise, what is the best estimate of the true temperature, and why?
> The **mean**: $(19 + 17 + 24)/3 = 20$°C. For Gaussian noise, the MLE is exactly the sample mean — the value that minimises $\sum (x_i - \hat{x})^2$. Setting the derivative to zero gives $\hat{x} = \bar{x}$.

## Why this matters

> [!question]- Why is MLE called the "bridge between probability and training"?
> MLE turns a probabilistic model of the data into a concrete loss function automatically:
>
> - Gaussian noise on real-valued targets $\to$ **MSE / SSE**
> - Bernoulli labels (binary) $\to$ **binary cross-entropy**
> - Categorical labels (multi-class) $\to$ **cross-entropy**
>
> We don't pick the loss arbitrarily — MLE *derives* it from the assumed distribution.
