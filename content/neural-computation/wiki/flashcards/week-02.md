---
week: 2
topic: "Gradient Descent"
deck: NeuralComputation::Week-02
---

TARGET DECK
NeuralComputation::Week-02

## Gradient descent fundamentals

> [!question]- What is the **gradient descent** update rule?
> $$\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^{t} - \eta \, \nabla L_t$$
> where $\boldsymbol{\theta}^t$ is the parameter vector at step $t$, $\eta$ is the **learning rate** (a small positive scalar), and $\nabla L_t$ is the gradient of the loss with respect to the parameters at $\boldsymbol{\theta}^t$.

> [!question]- Why do we step in the direction of $-\nabla L$ instead of $+\nabla L$?
> The gradient $\nabla L$ points in the direction of steepest **increase** of the loss. Since we want to *decrease* the loss, we step in the opposite direction. The minus sign in the update rule encodes "downhill".

> [!question]- What does the **learning rate** $\eta$ control, and what happens at extreme values?
> $\eta$ is the step size along the negative gradient.
>
> - **Too small:** training is very slow; loss decreases inch-by-inch.
> - **Too large:** the optimiser overshoots, oscillates, or even *increases* the loss.
> - **Just right:** smooth steady decrease. Typical values are $\eta \in \{10^{-3}, 10^{-2}, 10^{-1}\}$, found by trial and error.

> [!question]- What two stopping criteria are used to terminate gradient descent?
>
> 1. **Convergence** — the loss stops improving across iterations (changes are below a threshold).
> 2. **Maximum iterations** — a manually set budget (e.g. 100 epochs), used because complex models may never fully converge.

> [!question]- Why does gradient descent only find a **local** minimum, not necessarily the **global** one?
> The algorithm only sees the local slope at the current point — it walks downhill from wherever it started. If the loss surface has multiple basins, GD lands in whichever one its initialisation sits inside. In high dimensions we cannot map the landscape, so there is no guarantee we are in the deepest valley. Re-initialising or using momentum-based methods can help escape shallow basins.

## Why classification breaks vanilla gradient descent

> [!question]- Why does gradient descent **fail** when training a sign-activated perceptron with squared-error loss?
> The chain rule introduces a factor $\text{sgn}'(\cdot)$, the derivative of the sign function. Sign is flat almost everywhere (derivative 0) and undefined at 0, so $\nabla L = \mathbf{0}$ identically. The update rule becomes $\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \eta \cdot \mathbf{0} = \boldsymbol{\theta}^t$ — the parameters never change. Learning is impossible.

> [!question]- Why does swapping the **loss** not fix the sign-activation gradient problem — why must the **activation** change?
> The zero gradient comes from the sign function, not the loss. Whatever loss you compose with sign, the chain rule multiplies by $\text{sgn}'(\cdot) = 0$ and wipes out the gradient. The activation has to be replaced with something differentiable.

## Sigmoid

> [!question]- What is the **sigmoid function**, and what is its derivative?
> $$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$
> Sigmoid maps $\mathbb{R} \to (0, 1)$ smoothly, so its output can be read as a probability $p(y=1 \mid \mathbf{x})$. Its derivative is non-zero everywhere except in the saturated tails, which is why gradient descent can flow through it.

> [!question]- What does it mean for sigmoid to **saturate**, and why is it a problem?
> For very large $|z|$, $\sigma(z) \approx 0$ or $1$ and $\sigma'(z) \approx 0$. When the gradient passes through a saturated sigmoid, it is multiplied by ~0 — the upstream parameters get a near-zero update and effectively stop learning. This is the **vanishing-gradient** problem; it bites hardest in deep networks stacked from sigmoids.

## Binary cross-entropy

> [!question]- What is the **binary cross-entropy** loss?
> $$L(\boldsymbol{\theta}) = -\sum_{j=1}^{n} \left[ y^j \ln \hat{y}^j + (1 - y^j) \ln(1 - \hat{y}^j) \right]$$
> $y^j \in \{0, 1\}$ is the true label of example $j$ and $\hat{y}^j \in (0, 1)$ is the model's predicted probability. For each example, only one of the two terms is non-zero (whichever matches the true class), and that term is $-\ln(\text{probability assigned to the truth})$.

> [!question]- From which probabilistic assumption does **binary cross-entropy** fall out of MLE?
> A **Bernoulli** model for the labels: $p(y \mid \mathbf{x}) = \hat{y}^{\,y} (1 - \hat{y})^{1-y}$ with $\hat{y} = \sigma(b + \mathbf{w} \cdot \mathbf{x})$. Taking the log of the product across $n$ independent observations and flipping the sign produces exactly the BCE expression. Same MLE recipe as MSE — just with Bernoulli instead of Gaussian.

> [!question]- A sigmoid-activated perceptron trained with cross-entropy is also known by another name. What is it?
> **Logistic regression.** Despite the name, it is a *classification* algorithm — the "regression" refers to fitting a continuous probability $\hat{y} \in (0, 1)$, which is then thresholded for the class decision.

## Loss / activation pairings

> [!question]- Summarise the **task / activation / loss / probability model** correspondences for the two cases covered in week 2.
>
> | Task | Activation | Loss | MLE under |
> |---|---|---|---|
> | Regression | (none, linear) | Squared error | Gaussian noise |
> | Binary classification | Sigmoid | Binary cross-entropy | Bernoulli labels |
>
> Both are trained by gradient descent. The change is the assumed data distribution — that fixes the loss-activation pair.

## GD variants

> [!question]- What is the difference between **batch**, **stochastic**, and **mini-batch** gradient descent?
>
> - **Batch GD:** compute the gradient over **all** $n$ training samples per step. Accurate gradient, very slow per step.
> - **Stochastic GD:** compute the gradient on **one** randomly chosen sample. Very fast per step, very noisy trajectory.
> - **Mini-batch GD:** use a random subset of size $m \ll n$ (typically 32–512). Best of both — fast per step, much less noisy than SGD. The de facto default.

> [!question]- Why does the **noise** in stochastic / mini-batch GD sometimes help training?
> The gradient estimate fluctuates from step to step, so the optimiser does not follow a smooth path. This noise can knock the parameters out of shallow local minima and saddle points, helping to find better solutions than a perfectly smooth descent would.

> [!question]- What does **momentum** add to gradient descent, and what is its update rule?
> Momentum maintains a velocity $\mathbf{v}$ that decays previous gradients in:
> $$\mathbf{v}^t = \eta \nabla L_t + \beta \mathbf{v}^{t-1}, \qquad \boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \mathbf{v}^t$$
> Typical $\beta \approx 0.9$. The optimiser carries inertia through flat regions and dampens oscillations across narrow valleys, both speeding up convergence and helping escape shallow minima.

> [!question]- What does **Adam** combine, and why is it the modern default?
> Adam combines two ideas:
>
> 1. **Momentum** — exponentially decaying running mean of gradients (first moment).
> 2. **Per-parameter adaptive learning rates** — exponentially decaying running mean of squared gradients (second moment), used to scale each parameter's step.
>
> The result is robust across many problems with little tuning, which is why it became the default optimiser for most deep-learning training loops.
