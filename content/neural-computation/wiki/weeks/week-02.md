---
type: week
week: 2
title: "Gradient Descent: How Learning Actually Happens"
dates: 2025-10-06 to 2025-10-12
sources:
  - raw/week-02/w02-l04-transcript.txt
  - raw/week-02/w02-l05-transcript.txt
  - raw/week-02/w02-l06-transcript.txt
  - raw/week-02/w02-slides.pdf
concepts:
  - "[[gradient-descent-nc|gradient descent]]"
  - "[[learning-rate]]"
  - "[[sigmoid-function-nc|sigmoid function]]"
  - "[[binary-cross-entropy]]"
  - "[[gradient-descent-variants]]"
status: stable
updated: 2026-04-24
---

> [!question]+ THE CRUX: Week 1 said learning is optimisation. How do we actually *do* the optimisation — and what has to change about our model for the algorithm to work?

*Gradient descent walks downhill on the loss landscape: compute the slope, step in its negation, repeat. This works cleanly for regression, but the classification perceptron's sign activation has zero derivative everywhere and freezes the algorithm — so we swap sign for sigmoid and squared error for cross-entropy. All three (sigmoid, cross-entropy, and gradient descent itself) come back from maximum likelihood under the right probability model.*

## Where we left off

Week 1 reframed machine learning as optimisation: pick a [[loss-function]], find the parameters that minimise it. That gave us *what* to minimise but not *how*. With models having potentially billions of parameters, brute-force search is hopeless. We need an algorithm.

## The core idea: follow the slope

The [[gradient-descent-nc|gradient descent]] update is one equation you should commit to memory:

$$\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^{t} - \eta \, \nabla L_t$$

The gradient $\nabla L$ — the vector of partial derivatives of the loss with respect to each parameter — points in the direction of steepest *increase*. Since we want to *decrease* the loss, we step in the opposite direction. The scalar [[learning-rate]] $\eta$ controls step size.

> [!tip] TIP — Blindfolded hiker
> Imagine walking down a mountain in thick fog. You can't see the valley, but you can feel which way is downhill under your feet. Take a step, feel the slope again, take another step. That's gradient descent.

## A worked 1D example

For the three-thermometer problem from week 1 (measurements $\{19, 17, 24\}$) with the absolute-error loss $L = \sum_i |x_i - \hat{x}|$, starting from $\hat{x}^0 = 24.5$ and $\eta = 1$:

$24.5 \to 21.5 \to 20.5 \to 19.5 \to 18.5 \to 19.5 \to 18.5 \to \dots$

The iterate oscillates forever around the true optimum $\hat{x}^* = 19$ because the step size is too large relative to the distance from the minimum. This is a preview of two themes of the week: (1) the [[learning-rate]] is a knob that matters, and (2) gradient descent solves optimisation but not perfectly.

## Scaling up: from 1D to billions of dimensions

The algorithm doesn't care about dimensionality. For linear regression on the commute-time problem with weight $w$ and bias $b$, $\boldsymbol{\theta} = (w, b)$ is 2D and the gradient has two components:

$$\nabla L = \begin{pmatrix} \partial L / \partial w \\ \partial L / \partial b \end{pmatrix}$$

Add a second feature (distance *and* day-of-week) and you get $\boldsymbol{\theta} = (w_1, w_2, b)$ — three components. Scale to an image with 784 pixels and you get 785 components. The update equation $\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \eta \nabla L$ is identical. This is why gradient descent underpins both a 2-parameter regression and a 100-billion-parameter language model.

## Max likelihood returns: why squared error (again)

In week 1 we showed that, under Gaussian noise on measurements, maximum likelihood reduces to minimising the sum of squared errors. For regression we now extend this: assume the *conditional* distribution $p(y^j \mid \mathbf{x}^j)$ is Gaussian centred on the model's prediction $\hat{y}^j = b + \mathbf{w} \cdot \mathbf{x}^j$:

$$p_{\mathbf{w}, b}(y^j \mid \mathbf{x}^j) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(\hat{y}^j - y^j)^2}{2\sigma^2}\right)$$

Run the same derivation (product → log → drop constants → flip sign) and you get

$$\mathbf{w}^*, b^* = \arg\min_{\mathbf{w}, b} \sum_{j=1}^{n} (\hat{y}^j - y^j)^2$$

Squared error isn't an arbitrary choice — it's what max likelihood recommends when you believe the noise on your targets is Gaussian.

## The problem with classification

Take the perceptron classifier $\hat{y} = \text{sgn}(b + \mathbf{w} \cdot \mathbf{x})$, combine it with squared error as a loss, and try to run gradient descent. Applying the chain rule to compute $\partial L / \partial w_i$ produces a factor $\text{sgn}'(\cdot)$ — the derivative of the sign function.

Problem: the sign function is flat almost everywhere (derivative zero), and undefined at zero. So $\text{sgn}'(\cdot) = 0$ for all practical purposes, which means

$$\nabla L = \mathbf{0}$$

identically. The gradient descent update becomes $\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \eta \cdot \mathbf{0} = \boldsymbol{\theta}^t$. The parameters never change. Learning is impossible.

> [!question]- Why does replacing the loss function not fix this — why do we need to change the *activation*?
> Because the zero derivative comes from the sign function, not from the squared-error loss. Whatever loss you compose with sign, the chain rule multiplies by $\text{sgn}'(\cdot) = 0$ and wipes out the gradient. You have to swap sign for a differentiable activation.

## The fix: sigmoid + cross-entropy

Replace sign with the [[sigmoid-function-nc|sigmoid function]] $\sigma(z) = 1 / (1 + e^{-z})$:

- Output is in $(0, 1)$, so it can be read as $p(y = 1 \mid \mathbf{x})$.
- Smooth and differentiable — $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, non-zero except in the saturated tails.

With sigmoid turning the model into a probability estimator, applying maximum likelihood with the new encoding $y \in \{0, 1\}$ yields — after the usual log/product/flip-sign manipulations — the [[binary-cross-entropy]] loss:

$$L(\boldsymbol{\theta}) = -\sum_{j=1}^{n} \Big[ y^j \ln \hat{y}^j + (1 - y^j) \ln(1 - \hat{y}^j) \Big]$$

This is the classification counterpart to squared error. Same MLE recipe, different probability model (Bernoulli instead of Gaussian).

The soft (sigmoid) perceptron trained with cross-entropy is exactly **logistic regression** — a classification algorithm despite the misleading name.

> [!info] ASIDE — The label encoding swap
> With sign-activated perceptrons, class labels are $\pm 1$ to match the outputs. With sigmoid-activated models, the labels are re-encoded as $\{0, 1\}$ to match the sigmoid's $(0, 1)$ range. The problem is still binary classification — only the label convention changes, for mathematical convenience.

## Properties of gradient descent

### When to stop

Two standard criteria:
1. **Convergence** — loss no longer improves across iterations.
2. **Max iterations** — a manually set budget (e.g. 100 epochs), because complex models may never fully converge.

### Learning rate sensitivity

| Rate | Symptom |
|---|---|
| Too small | Very slow convergence — the curve trends down but inches |
| Too large | Oscillations; the loss may even increase |
| Just right | Fast smooth decrease to a low loss |

The "just right" value is typically found by trial and error. In practice $\eta \in \{10^{-3}, 10^{-2}, 10^{-1}\}$ — not 1.

### Local minima

Gradient descent finds a *local* minimum, not necessarily the *global* one. Which local minimum it finds depends on where you randomly initialised. In high dimensions we have no way to map the loss landscape, so there's no guarantee you're in the deepest valley. You can re-initialise and retry, or use methods (like momentum) that are better at escaping shallow basins.

## Variants: addressing vanilla GD's weaknesses

See [[gradient-descent-variants]] for the details. The short version:

- **Stochastic / mini-batch GD** — instead of summing the loss over all $n$ training samples per step, sample a subset $B$ of size $m \ll n$. Much faster per iteration; noisier trajectory. The noise actually helps escape local minima.
- **Momentum** — add a velocity term that decays previous gradients in, so the optimiser carries momentum through flat regions. $\mathbf{v}^t = \eta \nabla L_t + \beta \mathbf{v}^{t-1}$.
- **Adam** — combines momentum with per-parameter adaptive learning rates. The de facto default in modern deep learning.

## Summary: three ideas tied together by max likelihood

| Task | Activation | Loss | Derived from MLE under |
|---|---|---|---|
| Regression | *(none)* | Squared error $\sum_j (\hat{y}^j - y^j)^2$ | Gaussian noise |
| Classification | Sigmoid | Cross-entropy $-\sum_j [y^j \ln \hat{y}^j + (1 - y^j) \ln(1 - \hat{y}^j)]$ | Bernoulli labels |

Both are trained by gradient descent. The only real change between regression and classification is which probability model generated the data, which in turn fixes the correct loss and activation pair.

## Concepts introduced this week

- [[gradient-descent-nc|gradient descent]] — the iterative algorithm that drives all of deep learning
- [[learning-rate]] — the single scalar that determines whether training works or fails
- [[sigmoid-function-nc|sigmoid function]] — the differentiable activation that makes gradient descent work for classification
- [[binary-cross-entropy]] — the classification loss derived from MLE under Bernoulli labels
- [[gradient-descent-variants]] — SGD, mini-batch, momentum, Adam

## Connections

- **Builds on** [[week-01]]: week 1 posed learning as minimising a [[loss-function]]; this week provides the algorithm that does the minimising. The MLE framework from week 1 is reused — same recipe, different probability model.
- **Sets up** week 3: we still only have single neurons. Real-world data (e.g. XOR-like patterns) is not linearly separable, so we'll stack multiple perceptrons into layers. Training them needs backpropagation — which is gradient descent, just more carefully accounted for through the chain rule across layers.

## Open questions

- The gradient of binary cross-entropy composed with sigmoid is beautifully clean — $(\hat{y} - y) \mathbf{x}$ — but we didn't derive it explicitly in lecture. Worth working through for practice.
- The precise role of $\beta_1$ and $\beta_2$ in Adam, and why $\beta_2 = 0.999$ is a near-universal default, was only sketched — a revisit when optimisers come up again in later weeks would help.
- Saturation in sigmoid causes vanishing gradients in deep networks. This won't bite yet (we're still on single neurons), but it motivates ReLU and other activations that appear later.
