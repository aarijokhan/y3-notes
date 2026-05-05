---
type: week
week: 10
title: "Diagnosing Overfitting and Putting On the Brakes"
dates: 2025-12-01 to 2025-12-07
sources:
  - raw/week-10/Wk_10_Lec_1-1.pdf
  - raw/week-10/Wk_10_Lec_2-1.pdf
  - raw/week-10/Thursday_ December 4_ 2025 at 8_10_06 AM_Captions_English (United States).txt
  - raw/week-10/Thursday_ December 4_ 2025 at 8_54_53 AM_Captions_English (United States).txt
  - raw/week-10/Thursday_ December 4_ 2025 at 4_05_35 PM_Captions_English (United States).txt
  - raw/week-10/Tutorial_Wk_10.pdf
  - raw/week-10/ML_Exercise_Sheet_10a.pdf
  - raw/week-10/ML_Exercise_Sheet_10a_solution.pdf
  - raw/week-10/ML_Exercise_Sheet_10b.pdf
  - raw/week-10/ML_Exercise_Sheet_10b_solution.pdf
concepts:
  - "[[overfitting-ml|overfitting]]"
  - "[[regularization-ml|regularization]]"
  - "[[lasso-regression]]"
status: draft
updated: 2026-05-04
---

> [!question]+ THE CRUX: We've spent two weeks building learning theory — VC dimension, generalisation bound, bias–variance — that explains *why* models overfit. **Now what do we actually do about it?** Specifically: how do we move from theoretical understanding (model complexity should match data) to practical procedure (this matrix, this $\lambda$, this code)? **(1)** What does overfitting actually look like, and what causes it? **(2)** How does the constrained-optimisation view of "smaller hypothesis set" become the unconstrained augmented error $E_{\text{in}} + (\lambda/N)\Omega(\mathbf{w})$, and how does that recover ridge regression?

*The two halves of week 10 each answer one. **(1)** [[overfitting-ml|Overfitting]] is the regime where $E_{\text{in}}$ keeps falling while $E_{\text{out}}$ rises — the model captures noise rather than structure. Driven by some combination of data scarcity, model complexity, **stochastic noise** ($\epsilon$ in the labels), and **deterministic noise** (target complexity that the hypothesis class can't represent). The "two-learners" experiment shows the surprising result: even when both learners know the target is degree-10, a degree-2 learner beats a degree-10 learner on test error if $N$ is small — *because* the right hypothesis class isn't enough; you need data to constrain it. **(2)** [[regularization-ml|Regularisation]] is the structural cure. Start from the constrained problem $\min E_{\text{in}}$ s.t. $\|\mathbf{w}\|^2 \leq C$ — geometrically, a ball-shaped hypothesis set with radius $\sqrt{C}$. The Lagrangian transforms this into the unconstrained **augmented error** $E_{\text{in}}(\mathbf{w}) + (\lambda/N) \mathbf{w}^\top \mathbf{w}$, with $\lambda$ inversely related to $C$. Solving this for linear regression gives $\mathbf{w}_{\text{REG}} = (\mathbf{Z}^\top \mathbf{Z} + \lambda \mathbf{I})^{-1} \mathbf{Z}^\top \mathbf{y}$ — exactly [[ridge-regression|ridge regression]]. The L1 analogue, [[lasso-regression|lasso]], replaces the round constraint ball with a corner-bearing diamond and produces sparse solutions.*

---

## Part 1: What Overfitting Actually Looks Like

### Two Pictures, One Disease

Recall from week 9 that the VC bound shows model complexity has a U-shape against $E_{\text{out}}$: too simple → underfitting (high bias, $E_{\text{in}}$ already large), too complex → overfitting (low $E_{\text{in}}$ but $E_{\text{out}}$ blows up). The minimum is somewhere in the middle.

A canonical example, lifted from the lecture: $N = 10$ noisy points sampled from a smooth target.

| Model | $E_{\text{in}}$ | $E_{\text{out}}$ |
|---|---|---|
| Degree-2 polynomial | $0.029$ | $0.120$ |
| Degree-10 polynomial | $10^{-5}$ | $7680$ |

The degree-10 model fits the training points to numerical accuracy and produces test error six orders of magnitude worse. The training fit is a function with extreme oscillations, hitting every training point but exploding everywhere else.

The diagnostic signature: **low $E_{\text{in}}$, high $E_{\text{out}}$ — that's [[overfitting-ml|overfitting]]**. Both errors high together is **underfitting**. Both errors low is the goal.

### The Six Causes

The lecture lists six common drivers of overfitting. Each compounds the others, and real overfitting usually involves several.

- **Model too complex.** A million-parameter network on 1{,}000 examples memorises rather than generalises.
- **Too little training data.** A sentiment model on 50 customer reviews fits unique phrases instead of patterns.
- **Too many training epochs.** Validation error bottoms out, then climbs; training error keeps falling. (Early stopping is the cure.)
- **Lack of regularisation.** Without constraints, weights take whatever values minimise $E_{\text{in}}$ — including extreme values that fit noise.
- **High-variance features or noisy labels.** Random ID-number features become decision-tree splits; mislabelled examples drag the boundary.
- **Poor data processing.** Unstandardised inputs, leakage from validation into training, imbalanced classes.

### The Two-Learners Experiment

The lecture's most striking demonstration — and the one most worth internalising — is the comparison of **two learners** on the same target:

- **Learner $O$ (Overfit):** picks $g_{10} \in \mathcal{H}_{10}$ — a degree-10 polynomial.
- **Learner $R$ (Restrict):** picks $g_2 \in \mathcal{H}_2$ — a degree-2 polynomial.

Both are *told the truth*: the target is degree-10. Yet $R$ deliberately uses an under-expressive model class. Run the experiment with $N = 15$ training points:

| Target | $\mathcal{H}_2$ $E_{\text{out}}$ | $\mathcal{H}_{10}$ $E_{\text{out}}$ |
|---|---|---|
| Noisy degree-10 target | $0.127$ | $9.00$ |
| Noiseless degree-50 target | $0.120$ | $7680$ |

**Even when $\mathcal{H}_2$ structurally cannot fit the target, it wins by orders of magnitude on test error.**

> [!info]+ ASIDE — A counterintuitive lesson worth pausing on
> Knowing the right hypothesis class is not enough. With insufficient data, the high-capacity learner has so much variance that any bias saving evaporates. This is the bias–variance trade-off shouting at you: **deliberate underfitting beats accurate complexity-matching when $N$ is small.** The implication for practice: start simple, scale capacity *only as data grows*. Modern deep learning works precisely because it pairs huge capacity with huge $N$ — not because capacity alone is virtuous.

> [!question]- The two-learners experiment shows degree-2 wins even when the target is degree-10. Where, conceptually, does the win come from?
> Variance. With $N = 15$, the degree-10 learner has 11 parameters fitting 15 points — barely constrained, so the trained polynomial swings wildly between training points and produces huge test error. The degree-2 learner has 3 parameters fitting 15 points — heavily constrained, so the fit is stable across training sets. The bias the degree-2 learner pays (it can't represent the degree-10 truth) is dwarfed by the variance the degree-10 learner pays (it fits noise enthusiastically). At larger $N$, the picture flips — variance shrinks and bias starts to matter — but at $N = 15$, low capacity wins.

### Stochastic and Deterministic Noise

The bias–variance decomposition with noise has the form

$$\mathbb{E}[E_{\text{out}}] = \text{bias} + \text{var} + \sigma^2.$$

The $\sigma^2$ floor — irreducible **stochastic noise** — comes from randomness in the labels. But there's a second kind:

**Deterministic noise** is the part of the target $f$ that the chosen $\mathcal{H}$ cannot represent. Try to fit a degree-50 target with $\mathcal{H}_2$: the residual $f - $ (best degree-2 approximation in $\mathcal{H}_2$) is a fixed function of $\mathbf{x}$ — perfectly predictable in principle, but invisible to the learner because nothing in $\mathcal{H}_2$ matches it.

Both look identical to the trained model: residuals it can't reduce. Both encourage overfitting. The four sources of "serious overfitting" (from the lecture's $(\sigma^2, Q_f, N)$ heat-map experiment):

| Factor | Direction |
|---|---|
| Data size $N$ ↓ | Overfitting ↑ |
| Stochastic noise $\sigma^2$ ↑ | Overfitting ↑ |
| Target complexity $Q_f$ ↑ (deterministic noise) | Overfitting ↑ |
| Excessive model power | Overfitting ↑ |

> [!tip]+ TIP — Why high-capacity models overfit even on noiseless data
> A degree-2 model fitting a noiseless degree-50 target *still* overfits with $N = 15$. The "noise" the model is fitting isn't randomness — it's the part of the target that exceeds the model's representable structure, but it's still pulling the fit in spurious directions. Even noiseless complex targets need regularisation when capacity is mismatched.

### Why MLE Cannot Help

[[maximum-likelihood-estimation-ml|MLE]] minimises in-sample negative log-likelihood. There's nothing in MLE that prefers simple hypotheses to complex ones — the likelihood always rewards in-sample fit. So MLE on a high-capacity model interpolates the training data and overfits.

The Bayesian fix: combine the likelihood with a *prior* over $\mathbf{w}$ that favours simpler hypotheses. The MAP estimate balances likelihood (data fit) against prior (simplicity) — and for a Gaussian prior, MAP is exactly [[ridge-regression|ridge regression]]. **Regularisation isn't a hack — it's MAP estimation.** The probabilistic mechanics of why we should regularise are already in the framework from week 8.

## Part 2: How to Actually Put On the Brakes

### From Constrained to Augmented

The VC bound's slogan: $E_{\text{out}} \leq E_{\text{in}} + \Omega(\mathcal{H})$. Two levers, both worth pulling.

The pragmatic implementation, slowly built up:

**Hard constraint.** Force certain weights to zero — say $w_3 = w_4 = \cdots = w_{10} = 0$. This is just "use $\mathcal{H}_2$ instead of $\mathcal{H}_{10}$" — a discrete capacity reduction. Crude but effective.

**Looser constraint.** Force at least 8 of the 10 weights to be zero, but let the algorithm pick *which* 8: $\mathcal{H}_2' = \{\mathbf{w} : \text{at least 8 of } w_q = 0\}$. More expressive than $\mathcal{H}_2$, less risky than $\mathcal{H}_{10}$. This is **sparsity**.

**Soft constraint.** Combinatorial sparsity is hard to optimise. Replace with a continuous proxy: $\mathbf{w}^\top \mathbf{w} \leq C$, a closed ball of radius $\sqrt{C}$. As $C$ ranges from 0 to $\infty$, $\mathcal{H}(C)$ smoothly interpolates between $\{\mathbf{0}\}$ and $\mathbb{R}^{Q+1}$.

So the regularised problem is

$$\min_{\mathbf{w}} \;\; E_{\text{in}}(\mathbf{w}) \quad \text{subject to} \quad \mathbf{w}^\top \mathbf{w} \leq C.$$

### The Lagrangian Move

Constrained optimisation is harder than unconstrained. The [[lagrangian|Lagrangian]] turns one into the other.

Geometrically: at the optimum $\mathbf{w}_{\text{REG}}$ (assuming the constraint is active, $\mathbf{w}^\top \mathbf{w} = C$), the gradient $-\nabla E_{\text{in}}$ must be parallel to the surface's outward normal — otherwise we could slide along the surface and decrease $E_{\text{in}}$. The normal at $\mathbf{w}_{\text{REG}}$ on the sphere is just $\mathbf{w}_{\text{REG}}$ itself, so

$$\nabla E_{\text{in}}(\mathbf{w}_{\text{REG}}) + \frac{2 \lambda}{N} \mathbf{w}_{\text{REG}} = \mathbf{0}$$

for some $\lambda \geq 0$. This is exactly the gradient of the **augmented error**

$$\boxed{E_{\text{aug}}(\mathbf{w}) = E_{\text{in}}(\mathbf{w}) + \frac{\lambda}{N} \mathbf{w}^\top \mathbf{w}.}$$

So solving $\min E_{\text{in}}$ s.t. $\|\mathbf{w}\|^2 \leq C$ is equivalent to *unconstrained* minimisation of $E_{\text{aug}}$ — for some $\lambda$ that depends monotonically on $C$ (larger budget $\leftrightarrow$ smaller penalty).

This is the **two-form duality** of regularisation: pick whichever form is convenient.

### The Solution

For linear regression, plug $E_{\text{in}}(\mathbf{w}) = (1/N)(\mathbf{Z} \mathbf{w} - \mathbf{y})^\top (\mathbf{Z} \mathbf{w} - \mathbf{y})$ into $E_{\text{aug}}$ and set the gradient to zero:

$$\nabla E_{\text{aug}}(\mathbf{w}) = \frac{2}{N} (\mathbf{Z}^\top \mathbf{Z} + \lambda \mathbf{I}) \mathbf{w} - \frac{2}{N} \mathbf{Z}^\top \mathbf{y} = \mathbf{0}.$$

Rearranging:

$$\boxed{\mathbf{w}_{\text{REG}} = (\mathbf{Z}^\top \mathbf{Z} + \lambda \mathbf{I})^{-1} \mathbf{Z}^\top \mathbf{y}.}$$

The unconstrained OLS gives $\mathbf{w}_{\text{lin}} = (\mathbf{Z}^\top \mathbf{Z})^{-1} \mathbf{Z}^\top \mathbf{y}$. The regularised solution differs by a single $\lambda \mathbf{I}$ inside the inverse — the **ridge regression** solution. Adding $\lambda \mathbf{I}$ also fixes ill-conditioning: every eigenvalue of $\mathbf{Z}^\top \mathbf{Z}$ shifts up by $\lambda$, ensuring invertibility even when $\mathbf{Z}^\top \mathbf{Z}$ alone is singular.

The same closed form was derived in week 8 from the Bayesian MAP perspective ($\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})$). Here it appears from constrained optimisation. **Two derivations, same answer**: regularisation is structural, with multiple equivalent justifications.

### Effective VC Dimension

Why does this help generalisation? Recall the VC bound: $E_{\text{out}} \leq E_{\text{in}} + \Omega(\mathcal{H})$, with $\Omega$ growing in $d_{\text{VC}}(\mathcal{H})$.

The nominal $\mathcal{H} = \mathbb{R}^{\tilde{d}+1}$ has $d_{\text{VC}} = \tilde{d} + 1$. But the regularised algorithm only navigates within $\mathcal{H}(C)$ — the ball of radius $\sqrt{C}$ — so the effective complexity is $d_{\text{EFF}}(\mathcal{H}, \mathcal{A}) = d_{\text{VC}}(\mathcal{H}(C))$, smaller than the nominal $d_{\text{VC}}$.

The slogan: **$d_{\text{VC}}(\mathcal{H})$ large, while $d_{\text{EFF}}(\mathcal{H}, \mathcal{A})$ small if $\mathcal{A}$ is regularised.** The hypothesis class is structurally rich, but the algorithm only uses a constrained subset — the SVM's fat-hyperplane story from week 9, generalised.

### Choosing $\lambda$ — The U-Curve

Plot expected $E_{\text{out}}$ against $\lambda$: the shape is U-shaped. Sharp drop at small $\lambda$ (overfitting cured), minimum at some $\lambda^\ast$, slow rise (underfitting setting in). Practical observation: the U is **steep on the left and shallow on the right** — better to err on the side of slightly *more* regularisation than too little.

The optimal $\lambda^\ast$ depends on three things you don't directly observe:

- **Stochastic noise** $\sigma^2$: more noise → more regularisation needed (more bumps, more brakes).
- **Deterministic noise** (target complexity): more → more regularisation.
- **Data size $N$**: more $N$ → less need to regularise.

Since none of these are observable, $\lambda$ is chosen by **validation** — the second of the lecture's "two cures" and the topic of next week. Try a grid of $\lambda$'s, hold out a portion of the training data, and pick the $\lambda$ minimising held-out error.

> [!question]- Why is the U-curve of $E_{\text{out}}$ vs $\lambda$ steep on the left and shallow on the right?
> On the left ($\lambda$ near zero), the model is barely regularised and overfitting is severe; small increases in $\lambda$ rapidly fix the gap. On the right (large $\lambda$), the model is heavily constrained and the marginal cost of *more* regularisation is small — you're already close to the underfitting minimum. The asymmetry suggests that when you don't know $\lambda^\ast$, biasing your choice slightly higher than your guess is safer than biasing lower — the cost of mild underfitting is much smaller than the cost of mild overfitting.

### L1 vs L2 — Different Geometries

The L2 regulariser $\Omega(\mathbf{w}) = \mathbf{w}^\top \mathbf{w}$ is the natural choice for differentiable, closed-form optimisation. But it's not the only one. The general $L_p$ family is

$$\Omega(\mathbf{w}) = \sum_q |w_q|^p.$$

The case $p = 1$ — [[lasso-regression|lasso]] — is qualitatively different. The constraint set $\|\mathbf{w}\|_1 \leq C$ is a **diamond** (cross-polytope) with corners on the coordinate axes. The optimum (where the elliptical $E_{\text{in}}$ contours first touch the diamond) tends to land on a corner — a *sparse vector* with some coordinates exactly zero.

The L2 ball is round, no corners; the optimum is generically a smooth point with all coordinates non-zero. The L1 diamond is corner-rich, with corners precisely at sparse vectors; the optimum lands there generically.

**Practical implication.** Use lasso when you believe the true model is sparse — most features irrelevant — and you want automatic feature selection. Use ridge when you expect a dense model where most features matter. Elastic net (L1 + L2) hybrids combine sparsity with stability under correlated features.

> [!info]+ ASIDE — All of these have Bayesian counterparts
> L2 ↔ Gaussian prior. L1 ↔ Laplace prior (sharp peak at zero, drives sparsity). Mixtures ↔ elastic net. The Bayesian view explains *why* L1 is sparse — the Laplace's sharp peak at zero pins the MAP estimate there in the absence of strong data evidence — and why L2 is smooth (the Gaussian's smooth peak doesn't pin anything specifically). Each regulariser is a probabilistic statement about which weight vectors are *a priori* plausible.

### Recovering the Bayesian View

The MAP estimate from week 8's [[bayesian-linear-regression]] derivation:

$$\mathbf{w}_{\text{MAP}} = \arg\min_{\mathbf{w}} \;\; \frac{1}{2} \sum_i (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 + \frac{\alpha}{2 \beta} \mathbf{w}^\top \mathbf{w}$$

is *identical* to ridge regression with $\lambda = \alpha/\beta$. So:

- The constrained-optimisation derivation (this week, geometric/Lagrangian) and
- The Bayesian MAP derivation (week 8, probabilistic/prior)

both arrive at the same closed form $\mathbf{w}_{\text{REG}} = (\mathbf{Z}^\top \mathbf{Z} + \lambda \mathbf{I})^{-1} \mathbf{Z}^\top \mathbf{y}$. Each derivation illuminates a different facet:

- Constrained view: regularisation is "use a smaller hypothesis set" written as Lagrangian penalty.
- Bayesian view: regularisation is MAP estimation under a prior on $\mathbf{w}$.

> [!tip]+ TIP — The lecture's practical advice in one slogan
> "Whenever you train a model, try including regularisation." The benefit (variance reduction, more graceful behaviour with noise) almost always outweighs the cost (slight bias). Given a sensible $\lambda$ from validation, regularisation is one of the lowest-effort, highest-impact moves in practical ML.

---

## Concepts Introduced This Week

- [[overfitting-ml|overfitting]] — low $E_{\text{in}}$, high $E_{\text{out}}$. Caused by model complexity, data scarcity, stochastic noise, deterministic noise. The two-learners example shows that even degree-2 beats degree-10 when $N$ is small. Underfitting is the symmetric failure (both errors high).
- [[regularization-ml|regularization]] — augmenting $E_{\text{in}}$ with a complexity penalty to bias the optimiser away from overfitting. Constrained $\min E_{\text{in}}$ s.t. $\Omega(\mathbf{w}) \leq C$ ↔ unconstrained $\min E_{\text{in}} + (\lambda/N) \Omega(\mathbf{w})$ via Lagrangian. The augmented error is a better proxy for $E_{\text{out}}$. Effective VC dimension is smaller than nominal when the algorithm is regularised.
- [[lasso-regression]] — L1 regularisation. The diamond-shaped constraint set has corners on coordinate axes, so the optimum is sparse. Useful when the true model is sparse; corresponds to a Laplace prior in the Bayesian view.

## Connections

- **Builds on** [[week-08]]: the [[bayesian-linear-regression|Bayesian]] MAP derivation and [[ridge-regression]] are the same closed form we re-derive this week from constrained optimisation. Two paths, one destination.
- **Builds on** [[week-09]]: the VC bound's $\Omega(\mathcal{H})$ term motivates regularisation; the bias–variance decomposition explains *why* trading bias for variance is sensible. The "two-learners" experiment is bias–variance in action.
- **Builds on** [[non-linear-transformation]]: high-degree polynomial bases blow up the VC dimension, making regularisation essential. Without it, even moderate-degree polynomials overfit catastrophically on small $N$.
- **Sets up** validation and cross-validation (week 11): we keep saying "pick $\lambda$ by validation" — next week makes that procedure precise. Cross-validation, train/validation/test splits, model selection more broadly.

## Open Questions

- **How is $\lambda^\ast$ actually chosen?** "Validation" is the high-level answer; the practical procedure (k-fold CV, bias of the validation estimator, the 1-SE rule) is next week's content.
- **What other regularisation tricks does deep learning use?** Dropout, batch norm, data augmentation, weight tying, early stopping — each has a "what's the implicit $\Omega(\mathbf{w})$?" interpretation, often in terms of equivalent Bayesian priors. Active research.
- **How to choose between L1 and L2 in practice?** Heuristically: L1 if you suspect sparsity (most features irrelevant), L2 otherwise. Elastic net hedges. Cross-validate on both and let the data decide.
- **Why does deep learning generalise despite minimal explicit regularisation?** Implicit regularisation of SGD (toward flat minima), architectural inductive biases, and dataset-level structure — beyond classical VC + regularisation theory.
