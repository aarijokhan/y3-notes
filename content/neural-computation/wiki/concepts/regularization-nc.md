---
name: regularization
description: An explicit penalty term added to the loss that discourages large weights, reducing a model's effective capacity and preventing overfitting.
type: concept
sources:
  - raw/week-03/w03-l08-transcript.txt
  - raw/week-03/w03-slides.pdf
status: stable
updated: 2026-04-24
---

*A simple idea with a big effect: if a model with huge weights is at risk of overfitting, include the size of the weights in the loss itself. The optimiser then has to balance fitting the data against keeping the weights small — a trade-off controlled by a single hyperparameter.*

## The idea

Without regularisation, gradient descent minimises only the fit-the-data loss:

$$L(\boldsymbol{\theta}) = L_{\text{orig}}(\boldsymbol{\theta})$$

where $L_{\text{orig}}$ is squared error, cross-entropy, or whatever matches the task. The optimiser is free to set weights arbitrarily large if that improves training fit — even if those large weights are chasing noise rather than signal.

Regularisation adds a **penalty term** that grows with the size of the weights:

$$L(\boldsymbol{\theta}) = L_{\text{orig}}(\boldsymbol{\theta}) + \lambda \sum_{j=1}^{N} \theta_j^2$$

Now the optimiser has to simultaneously:
- Minimise the data-fit loss $L_{\text{orig}}$ — pushes toward solutions that explain the training data.
- Minimise the regularisation term $\lambda \sum \theta_j^2$ — pushes toward solutions with small weights.

The balance point has smaller weights than the unregularised optimum, and empirically, smaller weights mean smoother models that generalise better.

## The $\lambda$ hyperparameter

$\lambda \geq 0$ controls the strength of regularisation:

| $\lambda$ | Effect |
|---|---|
| $0$ | No regularisation. Back to vanilla training. |
| Very small | Light nudge toward smaller weights. Probably still overfits if unregularised model does. |
| Moderate | Meaningful trade-off. Weights stay bounded, generalisation improves. |
| Very large | Data-fit term is dominated by the penalty. Weights driven to zero; model underfits. |

$\lambda$ is a hyperparameter, tuned on the validation set. Like learning rate, it's typically set by trial over a log-scale grid: $10^{-4}$, $10^{-3}$, $10^{-2}$, $10^{-1}$.

## Why "weight decay"?

The $L_2$ penalty form above is often called **weight decay** because of what it does during gradient descent. Taking the gradient of the full loss:

$$\nabla L = \nabla L_{\text{orig}} + 2 \lambda \boldsymbol{\theta}$$

The update rule becomes:

$$\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \eta (\nabla L_{\text{orig}} + 2 \lambda \boldsymbol{\theta}^t) = (1 - 2 \eta \lambda) \boldsymbol{\theta}^t - \eta \nabla L_{\text{orig}}$$

The factor $(1 - 2\eta\lambda) < 1$ **shrinks every weight toward zero at every step**, before the data-fit gradient gets applied. Each step *decays* the weights by a constant fraction — hence the name.

## Why small weights help

A network with very large weights is highly *reactive*: a small change in the input produces a large change in the pre-activation, and (near a sigmoid's transition region) a huge change in the output. This reactivity is exactly what lets a model thread through every noisy training point — and exactly what harms generalisation.

Bounding the weights by penalising their magnitude limits this reactivity. The decision surface smooths out, fewer sharp wiggles form, and the model's predictions become less sensitive to individual training examples.

> [!info] ASIDE — L1 vs L2 regularisation
> The $\sum \theta_j^2$ penalty is called **L2 regularisation** (penalises the squared L2 norm of the parameter vector). An alternative, **L1 regularisation**, uses $\sum |\theta_j|$ instead. L1 tends to drive weights *exactly to zero* (inducing sparsity), while L2 keeps all weights small but non-zero. This module uses L2. Both are used in practice.

## What gets regularised?

By convention, regularisation penalises **weights** but not **biases**. Weights are the multiplicative connections between layers; biases are additive offsets. Penalising biases tends not to help generalisation and can cause systematic underfitting of the output level.

Some implementations use $\sum \theta_j^2$ over everything for simplicity; others are careful to exclude bias terms. In the week 3 formulation, $\theta_j$ ranges over both but the practical effect is dominated by the weights since they're the numerous parameters.

## Connection to overfitting remedies

Regularisation sits among several complementary overfitting fixes (see [[overfitting-nc|overfitting]]):

- **More data** — attacks the cause (insufficient signal-to-noise).
- **Smaller model** — reduces capacity directly.
- **Early stopping** — limits effective capacity by limiting training time.
- **Regularisation** — reduces effective capacity by penalising weight magnitude.
- **[[dropout]]** — randomly disable neurons during training so the network can't rely on any specific unit; reduces effective capacity another way.
- **[[data-augmentation]]** — synthesise extra training examples to increase the dataset's effective diversity; forces invariant features.

These are not alternatives; they stack. Real-world training often uses data augmentation + dropout + early stopping + L2 regularisation simultaneously.

## In practice

For the week 3 Python exercise, you'll train an MLP on non-linear classification data with and without regularisation and compare the decision boundaries. Expect:
- Without regularisation: tighter, wigglier decision boundary that fits training points precisely.
- With regularisation: smoother boundary that generalises better to unseen test points.

The visual difference on a 2D classification problem is often striking — it makes the abstract idea "regularisation controls complexity" immediately concrete.

## Related

- [[overfitting-nc|overfitting]] — the problem regularisation solves; sits alongside early stopping and more-data as remedies
- [[dropout]] — regularisation by randomly disabling neurons; common alternative/complement to L2
- [[data-augmentation]] — regularisation by synthesising more training examples
- [[gradient-descent-nc|gradient descent]] — the algorithm; regularisation just modifies the loss it minimises
- [[loss-function]] — regularisation is just an extra term added to the standard loss

## Active Recall

> [!question]- Write down the L2-regularised loss function. What does each symbol do?
> $L(\boldsymbol{\theta}) = L_{\text{orig}}(\boldsymbol{\theta}) + \lambda \sum_{j=1}^{N} \theta_j^2$. $L_{\text{orig}}$ is the original data-fit loss (e.g. squared error or cross-entropy). $\lambda \geq 0$ is the regularisation strength hyperparameter. $\sum \theta_j^2$ is the squared L2 norm of the parameter vector — a penalty that grows with the size of the weights. Minimising this combined loss trades off fitting the training data against keeping the weights small.

> [!question]- Why is L2 regularisation also called "weight decay"?
> Because in gradient descent, the L2 penalty contributes a gradient of $2 \lambda \boldsymbol{\theta}$ that, after the update rule, multiplies every weight by a factor $(1 - 2 \eta \lambda) < 1$ each step — shrinking it toward zero before the data-fit gradient is applied. The weights *decay* by a constant fraction per step, hence "weight decay".

> [!question]- What happens to the trained model if you set $\lambda$ very large? Very small? Zero?
> $\lambda = 0$: no regularisation, model behaves as if unregularised (may overfit). Very small $\lambda$: weak penalty, limited effect. Moderate $\lambda$: meaningful trade-off, weights stay bounded, usually improves generalisation. Very large $\lambda$: penalty dominates data-fit; weights shrink toward zero; the model effectively becomes a constant predictor and underfits severely. $\lambda$ is tuned on the validation set to find the sweet spot.

> [!question]- Why does penalising large weights improve generalisation rather than just making the fit worse?
> Large weights make the model highly reactive to input changes, which is what lets an overfitted model thread through every noisy training point. Small weights smooth out the decision surface so the model relies on general patterns rather than sensitivity to individual training examples. The training loss gets marginally worse (the model can't chase every point) but the test loss improves because the smoother model transfers better to unseen data — the classic bias-variance trade-off.

> [!question]- If regularisation helps, why don't we always crank $\lambda$ up to a very large value?
> Because too much regularisation drives weights to zero and makes the model unable to fit the data at all — you've overshot from overfitting into underfitting. The goal is the *balance point* where the penalty is strong enough to suppress overfitting but weak enough to let the model still learn the signal. That balance depends on the model, the data, and the problem, which is why $\lambda$ is tuned on the validation set rather than set to a fixed value.
