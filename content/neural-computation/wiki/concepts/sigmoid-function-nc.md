---
name: sigmoid-function
description: Smooth S-shaped activation σ(z) = 1/(1+e^{-z}) that maps real numbers to (0,1); differentiable replacement for the sign function, enabling gradient-based learning for classification.
type: concept
sources:
  - raw/week-02/w02-l05-transcript.txt
  - raw/week-02/w02-l06-transcript.txt
  - raw/week-02/w02-slides.pdf
status: stable
updated: 2026-04-24
---

*A smooth, squashed version of the sign function. Crucially, it is differentiable — which is what makes gradient-based learning for classification actually work.*

## Definition

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Some facts:

- **Range:** $\sigma(z) \in (0, 1)$ for all real $z$. Output never exactly reaches 0 or 1.
- **Symmetry:** $\sigma(0) = 0.5$. Centred at the origin.
- **Saturation:** As $z \to +\infty$, $\sigma(z) \to 1$. As $z \to -\infty$, $\sigma(z) \to 0$. Plateaus where the gradient is nearly zero.
- **Derivative:** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Always positive, maximal at $z = 0$ (value $0.25$), decays toward 0 in the tails.

## Why it exists: the sign function is broken for gradient descent

The vanilla [[perceptron]] classifier uses $\hat{y} = \text{sgn}(b + \mathbf{w} \cdot \mathbf{x})$. The sign function's derivative is zero everywhere (and undefined at zero), so when you plug the perceptron into a squared loss and apply the chain rule, you get

$$\frac{\partial L}{\partial w_i} = -\sum_j 2(y^j - \hat{y}^j) \underbrace{\text{sgn}'(\cdot)}_{= 0} x_i^j = 0.$$

The gradient is identically zero. [[gradient-descent-nc|gradient descent]]'s update $\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \eta \cdot \mathbf{0} = \boldsymbol{\theta}^t$ never changes the parameters. Learning is impossible.

Swapping sign for sigmoid fixes this: sigmoid has a well-defined, mostly-nonzero derivative, so $\nabla L$ is meaningful and gradient descent can make progress.

## The sigmoid perceptron (aka soft perceptron, aka logistic regression)

Replace the sign activation with sigmoid:

$$\hat{y} = \sigma\!\left(b + \sum_{i=1}^{D} w_i x_i\right)$$

This is the same model you may have met in a stats course as **logistic regression**. Don't let the name mislead you — logistic regression is a *classification* algorithm, not a regression algorithm.

| Perceptron variant | Activation | Output |
|---|---|---|
| **Hard perceptron** | sign | $\pm 1$ (hard decision) |
| **Soft perceptron** | sigmoid | value in $(0, 1)$ (probability) |

"Hard" and "soft" refer to the abruptness of the activation — sign flips instantaneously, sigmoid transitions smoothly.

## Probabilistic interpretation

Because the output sits in $(0, 1)$, we can read it as a probability. For a binary classification problem with class labels $y \in \{0, 1\}$:

$$p_{\mathbf{w}, b}(y = 1 \mid \mathbf{x}) = \hat{y} = \sigma(b + \mathbf{w} \cdot \mathbf{x})$$

and the probability of the other class is $p(y = 0 \mid \mathbf{x}) = 1 - \hat{y}$. Together:

$$p_{\mathbf{w}, b}(y \mid \mathbf{x}) = \begin{cases} \hat{y} & \text{if } y = 1 \\ 1 - \hat{y} & \text{if } y = 0 \end{cases}$$

This probabilistic view is what lets [[maximum-likelihood-estimation-nc|maximum likelihood estimation]] derive [[binary-cross-entropy]] as the natural loss function.

> [!info] ASIDE — Why labels flipped from $\{+1, -1\}$ to $\{0, 1\}$
> With the sign activation, outputs are $\pm 1$, so labelling the two classes $\pm 1$ lines up with the model's output. With the sigmoid activation, outputs live in $(0, 1)$, so labelling the classes $0$ and $1$ makes the math (and the probability interpretation) work out cleanly. The choice of label encoding is driven by mathematical convenience, not anything fundamental — it's still a binary problem either way.

## The decision rule

To turn a sigmoid probability into a hard classification, threshold at 0.5:

$$\hat{y}_{\text{class}} = \begin{cases} 1 & \text{if } \sigma(z) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

Since $\sigma(z) = 0.5$ iff $z = 0$, this is equivalent to thresholding the pre-activation $z = b + \mathbf{w} \cdot \mathbf{x}$ at zero — so the decision boundary is still the hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$, just as it was for the hard perceptron. The difference is that sigmoid additionally reports *how confident* it is.

## Weight magnitude controls confidence, not the boundary

A subtle but important point: scaling $\mathbf{w}$ and $b$ by the same factor leaves the **decision boundary unchanged** but changes how *confident* the sigmoid is near that boundary. Concretely, take two models:

- **Model A:** $\mathbf{w} = (1, 1)$, $b = -25$. Decision boundary: $x_1 + x_2 = 25$.
- **Model B:** $\mathbf{w}' = (2, 2)$, $b' = -50$. Decision boundary: $2x_1 + 2x_2 = 50 \Leftrightarrow x_1 + x_2 = 25$.

**Same boundary.** But evaluate at three penguins (flipper length $x_1$ cm, body mass $x_2$ kg):

| Penguin | $(x_1, x_2)$ | $z_A = x_1 + x_2 - 25$ | $\sigma(z_A)$ | $z_B = 2 z_A$ | $\sigma(z_B)$ |
|---|---|---|---|---|---|
| 1 | $(22, 2)$ | $-1$ | $0.27$ | $-2$ | $0.12$ |
| 2 | $(22, 3)$ | $0$ | $0.50$ | $0$ | $0.50$ |
| 3 | $(23, 4)$ | $+2$ | $0.88$ | $+4$ | $0.98$ |

Same classifications (above-boundary → class 1, below → class 0), but Model B is *much more confident*: penguins near the boundary get probabilities pushed closer to 0 and 1.

The take-away:

- **Direction of $\mathbf{w}$** — orientation of the decision boundary.
- **$b$ relative to $\|\mathbf{w}\|$** — position of the boundary (specifically, $-b/\|\mathbf{w}\|$ along $\mathbf{w}$ from the origin).
- **Magnitude of $\mathbf{w}$** — *steepness* of the sigmoid transition. Larger $\|\mathbf{w}\|$ → sharper transition, more confident predictions on either side. The boundary doesn't move; the model just becomes more decisive about it.

Geometrically: $z = \mathbf{w} \cdot \mathbf{x} + b$ measures distance from the boundary in units of $\|\mathbf{w}\|$. Doubling $\|\mathbf{w}\|$ doubles those distances, so the same input now lands twice as far from the boundary in the sigmoid's eye, and gets squashed harder toward 0 or 1.

This also gives a different angle on [[regularization-nc|weight decay]]: penalising $\|\mathbf{w}\|^2$ doesn't directly stop the model from finding the right boundary — it just stops the model from being *over-confident* about it. Smoother transitions mean more cautious predictions, which usually generalise better.

## Limitations

- **Saturation kills gradients.** For $|z| \gg 0$, the curve is nearly flat, so $\sigma'(z) \approx 0$. Parameters whose pre-activation lands in the saturated region stop learning. This is the *vanishing gradient* problem — severe in deep networks. Modern networks often use ReLU or variants to avoid it.
- **Still only linear boundaries.** The sigmoid perceptron is a linear classifier — a single straight line (or hyperplane) in input space. It cannot solve XOR or other non-linearly-separable problems. The fix is combining multiple neurons into a multi-layer network (week 3).

## Related

- [[perceptron]] — the model in which sigmoid replaces sign
- [[gradient-descent-nc|gradient descent]] — the algorithm that needs a differentiable activation
- [[binary-cross-entropy]] — the loss derived from the probabilistic interpretation of sigmoid
- [[maximum-likelihood-estimation-nc|maximum likelihood estimation]] — the principle that connects sigmoid to cross-entropy

## Active Recall

> [!question]- Write down the sigmoid function and its derivative, and state the derivative's maximum value.
> $\sigma(z) = 1/(1 + e^{-z})$ with derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. The derivative is maximised at $z = 0$ where $\sigma(0) = 0.5$, giving $\sigma'(0) = 0.25$.

> [!question]- Why can't gradient descent learn a perceptron with the sign activation and squared loss, and how does sigmoid fix it?
> The sign function's derivative is 0 (or undefined at 0), so by the chain rule $\nabla L$ is identically zero and the parameters never update. Sigmoid has a positive, mostly-nonzero derivative, so $\nabla L$ is a meaningful vector that actually moves the parameters downhill.

> [!question]- A sigmoid perceptron outputs $\hat{y} = 0.82$ for input $\mathbf{x}$. Interpret this value and give the hard-threshold classification.
> The output is interpretable as $p(y = 1 \mid \mathbf{x}) = 0.82$ — the model's estimated probability that $\mathbf{x}$ belongs to class 1. With threshold 0.5, the hard classification is $y = 1$, with relatively high confidence.

> [!question]- What is the *vanishing gradient* problem and when does it bite for sigmoid?
> Sigmoid saturates in the tails: for $|z|$ large, $\sigma'(z) \approx 0$. Any parameter whose pre-activation is in the saturated region receives near-zero gradient and updates very slowly — effectively stops learning. This is especially damaging in deep networks where gradients multiply layer-by-layer and shrink exponentially toward the input.

> [!question]- Why do we switch the class label encoding from $\{+1, -1\}$ (hard perceptron) to $\{0, 1\}$ (sigmoid perceptron)?
> It's a mathematical convenience. Sigmoid outputs lie in $(0, 1)$, so encoding classes as 0 and 1 aligns labels with outputs and lets the output be read as a probability $p(y=1 \mid \mathbf{x})$. With sign outputs of $\pm 1$, the $\pm 1$ label encoding is the natural match. Nothing changes about the problem — it's still binary classification.

> [!question]- Two sigmoid classifiers have parameters $(\mathbf{w}, b) = ((1, 1), -25)$ and $(\mathbf{w}', b') = ((2, 2), -50)$. Do they classify points differently? What does change?
> They classify points *identically* — both have the same decision boundary $x_1 + x_2 = 25$, because scaling $\mathbf{w}$ and $b$ by the same factor leaves the equation $\mathbf{w} \cdot \mathbf{x} + b = 0$ unchanged. What changes is *confidence*: the second model has $\|\mathbf{w}\|$ twice as large, so its sigmoid transition is twice as steep. Points near the boundary get pushed harder toward 0 or 1, producing more extreme (more confident) probabilities. The direction and position of the boundary are set by $\mathbf{w}$ and $b/\|\mathbf{w}\|$; the magnitude $\|\mathbf{w}\|$ controls only the sharpness.
