---
name: gradient-descent-variants
description: Practical variations on vanilla gradient descent — stochastic/mini-batch sampling, momentum, and Adam — that address cost, local minima, and learning-rate tuning.
type: concept
sources:
  - raw/week-02/w02-l06-transcript.txt
  - raw/week-02/w02-slides.pdf
status: stable
updated: 2026-04-24
---

*Vanilla gradient descent works but is slow, memory-hungry, and gets stuck. The variants below each address a specific weakness, and Adam — which combines several of them — is the de facto default in modern deep learning.*

## The cost problem: sampling variants

Vanilla [[gradient-descent-nc|gradient descent]] computes the loss over the *entire* training set at every iteration:

$$L(\boldsymbol{\theta}) = \sum_{j=1}^n \big(y^j - \hat{y}^j\big)^2 \quad (\text{regression example})$$

When $n$ is millions (or billions), one step can be minutes of compute. The fix is to estimate the gradient from a subset.

| Variant                 | Samples per step                                                    | Character                                                                                                                |
| ----------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Full-batch GD**       | All $n$ samples                                                     | Accurate gradient, expensive, smooth convergence path                                                                    |
| **Stochastic GD (SGD)** | Single random sample (so addresses the cost issue of full-batch DG) | Cheap gradient, very noisy, bounces around the optimum                                                                   |
| **Mini-batch GD**       | Random subset $B$ of size $m$, with $m \ll n$                       | Best of both — reasonable cost, acceptable noise.  In other words its the middle ground of full-batch and stochastic GD. |

$$L(\boldsymbol{\theta}) = \sum_{j \in B} \big(y^j - \hat{y}^j\big)^2$$

Note the extreme cases recover the other two: $m = 1$ is SGD, $m = n$ is full-batch GD. Common batch sizes are 32, 64, 128.

Because mini-batch exists as a contrast, **vanilla gradient descent is often called "full-batch gradient descent"** — the name emphasises that it uses the *full* training set on every step. If you see "full-batch GD" in a paper or textbook, it means the same thing as plain gradient descent.

> [!info] ASIDE — "Stochastic" in practice
> Strictly, SGD picks one sample at a time. But "SGD" is often used loosely to mean mini-batch GD as well, especially in deep learning frameworks. When someone says "we trained with SGD", they almost always mean mini-batch SGD with some batch size.

### Epochs vs iterations

An **iteration** is one parameter update (one forward/backward pass on one batch). An **epoch** is one full pass through the training set. With 1,000 samples and batch size 50, one epoch = 20 iterations.

### Convergence behaviour

Full-batch GD moves steadily along a smooth trajectory toward the minimum — each step is informed by the entire dataset. Mini-batch / stochastic GD moves noisily, oscillating around its path because each step sees only a rough estimate of the true gradient. The noise has an upside: it can **jump out of shallow local minima** that would trap vanilla GD.

## The local minima problem: momentum

Vanilla GD's update only uses the *current* gradient — it has no memory of where it was heading. Narrow valleys cause it to oscillate across the steep walls; shallow basins trap it.

**Momentum** accumulates a velocity vector that blends the current gradient with the previous direction:

$$\mathbf{v}^t = \eta \nabla L_t + \beta \, \mathbf{v}^{t-1}, \qquad \boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^{t} - \mathbf{v}^t$$

- $\beta \in (0, 1)$ is the **decay** (typical value: 0.9). Past velocity gradually fades.
- $\mathbf{v}^0 = \mathbf{0}$.
- At $\beta = 0$, momentum reduces to vanilla GD.
- At $\beta = 1$, no decay — past gradients accumulate indefinitely, behaviour becomes unstable.

**Intuition:** a ball rolling downhill carries momentum from previous steps, so small uphill bumps don't stop it dead — it coasts over shallow local minima toward deeper ones. Steep consistent directions get amplified (past velocity reinforces current gradient); oscillating directions cancel out (signs flip between steps).

Basically, if  a person has confidence in where they are going they can make bigger moves and land at solution faster e.g. if you are driving and have clarity on the road you tend to drive faster as opposed to newer roads

## The learning-rate problem: adaptive methods

Tuning a single global $\eta$ is brittle — different parameters may benefit from different step sizes. Adaptive optimisers adjust a *per-parameter* effective learning rate based on the history of gradients.

### AdaGrad, RMSProp (sketch)

- **AdaGrad** divides each parameter's update by the square root of the accumulated squared gradients for that parameter. Directions that have moved a lot get slowed down; directions that haven't get amplified. Downside: the accumulator grows forever, so learning eventually stalls.
- **RMSProp** replaces the cumulative sum with an exponentially-decayed moving average, which prevents the slowdown.

### Adam (the default)

**Adam** (*Adaptive Moment Estimation*) combines momentum with RMSProp-style adaptive learning rates:

$$\mathbf{m}^t = \beta_1 \mathbf{m}^{t-1} + (1-\beta_1)\nabla L_t \qquad \text{(1st moment: momentum)}$$
$$\mathbf{s}^t = \beta_2 \mathbf{s}^{t-1} + (1-\beta_2)(\nabla L_t \odot \nabla L_t) \qquad \text{(2nd moment: squared gradient)}$$
$$\hat{\mathbf{m}}^t = \frac{\mathbf{m}^t}{1-\beta_1^t}, \quad \hat{\mathbf{s}}^t = \frac{\mathbf{s}^t}{1-\beta_2^t} \qquad \text{(bias correction)}$$
$$\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^{t} - \eta \frac{\hat{\mathbf{m}}^t}{\sqrt{\hat{\mathbf{s}}^t} + \varepsilon}$$

Typical hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 10^{-3}$, $\varepsilon = 10^{-8}$.

Adam is the de facto standard optimiser in modern deep learning. It works well out of the box on a very wide range of problems.

## Comparison summary

| Optimiser | Gradient info | Learning rate | Main strength | Main cost |
|---|---|---|---|---|
| **Vanilla GD** | Current only | Fixed global $\eta$ | Simple; predictable | Slow, stuck in local minima |
| **Momentum** | Current + exp-weighted past | Fixed global $\eta$ | Faster, escapes shallow minima | One extra hyperparameter ($\beta$) |
| **AdaGrad** | Current + all past squared | Per-parameter, monotonically shrinking | Auto-tunes per parameter | Learning eventually dies |
| **RMSProp** | Current + exp-weighted past squared | Per-parameter, bounded | Fixes AdaGrad's decay | No momentum |
| **Adam** | Both (momentum + RMSProp) | Per-parameter, adaptive | Best of both; robust default | Most hyperparameters to tune |

## When to use what

- **Small dataset, want maximum accuracy:** full-batch GD.
- **Large dataset, need speed per step:** mini-batch SGD (batch size 32–128).
- **Non-convex loss with local minima:** momentum or Adam.
- **Default modern choice:** Adam with mini-batch.

## Related

- [[gradient-descent-nc|gradient descent]] — the base algorithm these variants build on
- [[learning-rate]] — the $\eta$ these methods (sometimes) adapt

## Active Recall

> [!question]- What is the difference between an *epoch* and an *iteration* when training with mini-batch gradient descent?
> An iteration is one parameter update (one batch processed). An epoch is one complete pass through the training set — so if there are $n$ samples and batch size $m$, one epoch contains $n/m$ iterations.

> [!question]- Write the momentum update rule and say what the decay parameter $\beta$ does.
> $\mathbf{v}^t = \eta \nabla L_t + \beta \mathbf{v}^{t-1}$ then $\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \mathbf{v}^t$. $\beta \in (0, 1)$ controls how much of the previous velocity is carried forward. $\beta = 0$ forgets all past direction (vanilla GD); $\beta$ close to 1 keeps most of it, giving a heavier "ball rolling downhill" behaviour. Typical value $\beta = 0.9$.

> [!question]- Why does stochastic gradient descent sometimes escape local minima that vanilla gradient descent gets stuck in?
> SGD's gradient estimate is noisy — it's based on a single sample (or small batch), not the full dataset. The noise adds random perturbations to each step, which can push the iterate over the rim of a shallow local basin into a different (potentially deeper) one. Vanilla GD uses the exact full-data gradient, which points straight into the nearest minimum with no randomness to break out of it.

> [!question]- For a dataset of 10,000 samples and batch size 100, how many iterations are in one epoch?
> 100. Each iteration consumes 100 samples; $10{,}000 / 100 = 100$ iterations to see the full dataset once.

> [!question]- What two ideas does Adam combine, and what does each contribute?
> Adam combines **momentum** (tracks an exponentially-decayed average of past gradients, giving smooth, accelerated motion in consistent directions) with **RMSProp-style adaptive learning rates** (tracks an exponentially-decayed average of squared gradients, so parameters with historically large gradients get smaller effective step sizes). The result is per-parameter step sizes that auto-tune *and* benefit from momentum — a strong default optimiser.
