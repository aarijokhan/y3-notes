---
name: learning-rate
description: The scalar hyperparameter η that controls the step size in gradient descent; too small causes slow convergence, too large causes oscillation or divergence.
type: concept
sources:
  - raw/week-02/w02-l04-transcript.txt
  - raw/week-02/w02-l06-transcript.txt
  - raw/week-02/w02-slides.pdf
status: stable
updated: 2026-04-24
---

*The knob that decides how big a step gradient descent takes. It is not learned from data — you set it, tune it, and often regret it.*

## Definition

In the [[gradient-descent-nc|gradient descent]] update

$$\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^{t} - \eta \, \nabla L_t,$$

the learning rate $\eta > 0$ scales the gradient before it is subtracted from the parameters. It controls how far each step moves in the downhill direction.

$\eta$ is a **hyperparameter** — not a learnable parameter. The model doesn't adjust it during training. You choose it (or use a schedule), typically by trial and error.

## Three regimes

| Regime | Symptom | What happens |
|---|---|---|
| **Too small** | Loss decreases smoothly but very slowly | Training takes forever; may not converge within the iteration budget |
| **Too large** | Loss oscillates or diverges | Each step overshoots the minimum; can bounce further away every iteration |
| **Just right** | Loss decreases quickly and stabilises at a low value | Fast convergence to a good solution |

The "just right" value is found by trial and error. Typical starting points in practice: $\eta = 0.1$, $0.01$, or $0.001$ — *not* $\eta = 1$, which is only used in toy examples because it makes the arithmetic clean.

## Worked example: oscillation at a kink

In the week 2 1D example with $L(\hat{x}) = \sum_i |x_i - \hat{x}|$, measurements $\{19, 17, 24\}$ and $\eta = 1$, gradient descent goes $24.5 \to 21.5 \to 20.5 \to 19.5 \to 18.5 \to 19.5 \to 18.5 \to \dots$ and never reaches the true optimum $\hat{x}^* = 19$. The step size of 1 is larger than the distance from 19.5 (or 18.5) to 19, so every step straddles the minimum.

Reducing $\eta$ (say, to $0.1$) lets the iterates inch in closer rather than leap across.

## Why not just learn it?

You can't easily learn $\eta$ via gradient descent itself — the gradient of the loss *with respect to* $\eta$ doesn't give you a useful signal about whether the learning rate is well-tuned for the problem. In practice people use:

- **Manual tuning** — try a few orders of magnitude and pick the best.
- **Learning rate schedules** — start high, decay over time (e.g. $\eta_t = \eta_0 / (1 + \alpha t)$). See below.
- **Adaptive optimisers** — Adam, RMSProp, and AdaGrad effectively assign a *per-parameter* learning rate that adapts based on the history of gradients (see [[gradient-descent-variants]]).

## Learning rate schedules

A *constant* $\eta$ is rarely optimal: early in training, when the parameters are far from the minimum, large steps make rapid progress; late in training, near the minimum, large steps overshoot and oscillate. A **schedule** changes $\eta$ over time, ideally large early and small late.

The gradient descent update with a schedule:

$$\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^{t} - \eta(t) \, \nabla L_t$$

where $\eta(t)$ is now a function of the iteration or epoch.

### Common schedules

- **Step decay.** Reduce $\eta$ by a factor (typically 10) every fixed number of epochs: $\eta = 0.1$ for epochs 1–30, then $\eta = 0.01$ for epochs 30–60, then $\eta = 0.001$, etc. Simple, common, surprisingly effective.
- **Exponential decay.** $\eta(t) = \eta_0 \cdot \gamma^t$ with $\gamma < 1$. Smooth, monotonic decrease.
- **Reduce-on-plateau.** Watch the validation loss; whenever it stops improving for a fixed number of epochs (the *patience*), drop $\eta$ by a factor. The schedule adapts to what the network is actually doing instead of following a fixed timetable.
- **Cosine annealing.** $\eta(t)$ follows a half-cosine from $\eta_0$ down to (near) zero across the training run. Smooth and often slightly better than step decay in practice.

### The intuition

Picture the loss landscape as a valley with the minimum somewhere in the middle. With a large $\eta$ you take big strides — useful when you're far from the bottom and need to traverse the slopes quickly. As you approach the minimum, big strides keep overshooting; you need to *tiptoe* into the centre. The schedule encodes this physical intuition: stride first, tiptoe later.

> [!tip] TIP — The helicopter-on-helipad picture
> Imagine landing a helicopter on a small helipad in a dark valley. Move *too fast* (high $\eta$) and you overshoot the pad on every approach, bouncing around without settling. Move *too slow* (low $\eta$) and you're precise but take an eternity — or worse, you settle into a small pothole on the way down (a local minimum) because you don't have the speed to climb out. A schedule gives you both: full speed across the open valley, throttle back as the pad gets close.

> [!tip] TIP — When loss plateaus, drop the learning rate
> A common training-curve diagnostic: validation loss decreases for many epochs, then plateaus. If you keep training at the same $\eta$, nothing changes — the optimiser is bouncing around the minimum, never settling. Dropping $\eta$ by 10× often produces a sudden additional drop in loss, then another plateau. This is reduce-on-plateau in action; it's the basis of the schedule of the same name.

## Related

- [[gradient-descent-nc|gradient descent]] — the algorithm where $\eta$ lives
- [[gradient-descent-variants]] — variants that adapt the learning rate automatically

## Active Recall

> [!question]- What are the three qualitative regimes for $\eta$ and the symptom of each?
> Too small: very slow convergence (loss decreases but barely). Too large: oscillation or divergence (loss bounces around or grows). Just right: fast smooth decrease to a low loss. Found by trial and error.

> [!question]- Why can't you just learn $\eta$ the way you learn weights?
> $\eta$ is a meta-level control on the learning process, not a parameter of the model. The gradient of the loss with respect to $\eta$ doesn't cleanly tell you whether the step size is appropriate for the current curvature of the landscape. Instead we tune it manually, use a schedule, or let an adaptive optimiser (Adam, RMSProp) effectively adjust a per-parameter step size.

> [!question]- A training loss curve decreases for the first 5 epochs, then oscillates wildly without improving. What's a likely cause and a fix?
> The learning rate is too large — once near the minimum, each step overshoots. A fix is a learning rate schedule: decay $\eta$ over time (e.g. divide by 10 every few epochs), which gives fast progress early and fine-grained convergence later.

> [!question]- A network's validation loss has been flat for 10 epochs. What does a "reduce-on-plateau" schedule do, and why does that often help?
> When the validation loss stops improving for a fixed number of epochs (the patience window), reduce-on-plateau drops the learning rate by a factor (typically 10). The intuition: the optimiser is making large steps that overshoot the minimum, bouncing around without settling. A smaller step size lets it actually descend the remaining distance into the basin. After dropping $\eta$, you typically see another sharp loss decrease followed by a new plateau — repeat until further drops stop helping. This is more responsive than a fixed step-decay schedule because the schedule reacts to what the network is actually doing.
