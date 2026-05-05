---
type: concept
name: regularization
description: Adding a penalty on hypothesis complexity to the training objective, biasing the optimiser away from overfitting. Equivalent to constrained minimisation via the Lagrangian, justifies the augmented-error formulation, and reduces effective VC dimension below the nominal one.
sources:
  - raw/week-10/Wk_10_Lec_2-1.pdf
  - raw/week-10/Thursday_ December 4_ 2025 at 4_05_35 PM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*Augmenting the training objective $E_{\text{in}}(\mathbf{w})$ with a complexity penalty $\Omega(\mathbf{w})$, scaled by $\lambda \geq 0$, to bias the optimiser towards simpler hypotheses. The constrained form $\min E_{\text{in}}$ s.t. $\Omega(\mathbf{w}) \leq C$ and the unconstrained form $\min E_{\text{in}} + (\lambda/N) \Omega(\mathbf{w})$ are Lagrangian-equivalent. The augmented objective is a better proxy for $E_{\text{out}}$ than $E_{\text{in}}$ alone, with $\Omega(\mathbf{w})$ standing in for the model-complexity penalty $\Omega(\mathcal{H})$ in the [[generalization-bound|VC bound]].*

## The Motivation

The [[generalization-bound|VC bound]] says

$$E_{\text{out}}(h) \leq E_{\text{in}}(h) + \Omega(\mathcal{H})$$

where $\Omega(\mathcal{H})$ is the complexity of the hypothesis set. Two ways to keep $E_{\text{out}}$ small:

1. Make $E_{\text{in}}$ small — fit the training data.
2. Make $\Omega(\mathcal{H})$ small — use a simpler hypothesis set.

Pure ERM (empirical risk minimisation) only attacks the first. **Regularisation attacks both**: it minimises $E_{\text{in}}$ subject to a constraint on hypothesis complexity, or equivalently, minimises $E_{\text{in}} + (\lambda/N) \Omega(\mathbf{w})$ where $\Omega(\mathbf{w})$ measures the complexity of an *individual* hypothesis (a proxy for the set $\Omega(\mathcal{H})$ that the algorithm actually selects from).

## The Constrained View

The regularisation idea, slowly introduced:

**Hard constraint.** Force certain weights to zero: minimise $E_{\text{in}}$ subject to $w_3 = w_4 = \cdots = w_{10} = 0$. This is equivalent to using $\mathcal{H}_2$ instead of $\mathcal{H}_{10}$ — a discrete jump in capacity.

**Looser constraint.** Force at least 8 of $w_q$ to zero — but let the algorithm decide *which* ones. Now $\mathcal{H}_2' = \{\mathbf{w} : \text{at least 8 of } w_q = 0\}$. More expressive than $\mathcal{H}_2$ (you can choose which dimensions matter), less risky than $\mathcal{H}_{10}$ (still constrained). Mathematically a sparsity constraint.

**Soft constraint.** The combinatorial sparsity is hard to optimise. Replace with a continuous proxy:

$$\mathcal{H}(C) = \{\mathbf{w} : \mathbf{w}^\top \mathbf{w} \leq C\}.$$

The hypothesis set is a closed ball of radius $\sqrt{C}$. As $C$ ranges from $0$ to $\infty$, $\mathcal{H}(C)$ smoothly interpolates between $\{\mathbf{0}\}$ and $\mathbb{R}^{Q+1}$. The training problem becomes

$$\min_{\mathbf{w}} E_{\text{in}}(\mathbf{w}) \quad \text{subject to} \quad \mathbf{w}^\top \mathbf{w} \leq C.$$

## From Constraint to Lagrangian

Constrained optimisation is harder than unconstrained. The [[lagrangian|Lagrangian]] turns it into the latter.

Geometrically: the optimal $\mathbf{w}_{\text{REG}}$ either sits inside the ball ($\mathbf{w}^\top \mathbf{w} < C$, the constraint is inactive) or on the surface ($\mathbf{w}^\top \mathbf{w} = C$, active). When active, two facts:

1. $\nabla E_{\text{in}}(\mathbf{w}_{\text{REG}})$ is perpendicular to the surface (otherwise we could slide along the surface and decrease $E_{\text{in}}$ without violating the constraint).
2. The normal to the surface $\mathbf{w}^\top \mathbf{w} = C$ at $\mathbf{w}_{\text{REG}}$ is the vector $\mathbf{w}_{\text{REG}}$ itself.

Combining: $-\nabla E_{\text{in}}(\mathbf{w}_{\text{REG}})$ is parallel to $\mathbf{w}_{\text{REG}}$, i.e.,

$$\nabla E_{\text{in}}(\mathbf{w}_{\text{REG}}) + \frac{2 \lambda}{N} \mathbf{w}_{\text{REG}} = \mathbf{0}$$

for some $\lambda \geq 0$. This is exactly the gradient of the **augmented error**

$$E_{\text{aug}}(\mathbf{w}) = E_{\text{in}}(\mathbf{w}) + \frac{\lambda}{N} \mathbf{w}^\top \mathbf{w}.$$

So solving the constrained problem with budget $C$ is equivalent to *unconstrained* minimisation of $E_{\text{aug}}$ with some specific $\lambda$. The correspondence: $C \uparrow$ ↔ $\lambda \downarrow$ — bigger budget means lighter penalty.

> [!tip]+ TIP — Two-form duality is convenient
> The constrained form is interpretable ("keep $\|\mathbf{w}\|^2$ below $C$"); the augmented form is solvable (just gradient-step on $E_{\text{aug}}$). Use the form that fits your purpose. They give the same $\mathbf{w}_{\text{REG}}$ for paired $C, \lambda$ values.

## The Augmented Error

$$\boxed{E_{\text{aug}}(\mathbf{w}) = E_{\text{in}}(\mathbf{w}) + \frac{\lambda}{N} \Omega(\mathbf{w})}$$

The two components:

- $\Omega(\mathbf{w})$ — the **regulariser**: a function of the hypothesis itself (e.g., $\mathbf{w}^\top \mathbf{w}$).
- $\lambda$ — the **regularisation parameter**: how aggressively to apply the brakes.

**Heuristic interpretation.** $\Omega(\mathbf{w})$ stands in for the hypothesis-set complexity $\Omega(\mathcal{H})$ in the VC bound. We can't bound $\Omega(\mathcal{H})$ directly during training (it's a property of the whole set), but $\Omega(\mathbf{w})$ for the chosen $\mathbf{w}$ is computable and correlates with what hypothesis-set complexity we'd need to "actually contain" $\mathbf{w}$.

**Practical interpretation.** Minimising $E_{\text{aug}}$ uses a *better proxy for $E_{\text{out}}$* than $E_{\text{in}}$ alone. We can't measure $E_{\text{out}}$, but we can simulate the bound's two terms (data fit + complexity) and minimise their sum.

## Effective VC Dimension

The nominal hypothesis set $\mathcal{H} = \mathbb{R}^{\tilde{d}+1}$ has $d_{\text{VC}} = \tilde{d} + 1$ — *all* weight vectors are considered candidates. But after regularisation, the algorithm only navigates within $\mathcal{H}(C)$ — the ball of radius $\sqrt{C}$ — so the *effective* VC dimension

$$d_{\text{EFF}}(\mathcal{H}, \mathcal{A}) = d_{\text{VC}}(\mathcal{H}(C))$$

is smaller. This is the formal sense in which *regularised algorithms generalise*: they implicitly select from a smaller hypothesis set than the nominal $\mathcal{H}$, even though the nominal $\mathcal{H}$ remains expressive enough to capture complex targets when regularisation is loose.

The slogan: **$d_{\text{VC}}(\mathcal{H})$ large, while $d_{\text{EFF}}(\mathcal{H}, \mathcal{A})$ small if $\mathcal{A}$ is regularised.**

## L2: Weight Decay

The most common regulariser is

$$\Omega(\mathbf{w}) = \mathbf{w}^\top \mathbf{w} = \|\mathbf{w}\|_2^2.$$

For linear regression with this penalty, the augmented error is

$$E_{\text{aug}}(\mathbf{w}) = \frac{1}{N} (\mathbf{Z} \mathbf{w} - \mathbf{y})^\top (\mathbf{Z} \mathbf{w} - \mathbf{y}) + \frac{\lambda}{N} \mathbf{w}^\top \mathbf{w}.$$

Setting $\nabla E_{\text{aug}} = 0$:

$$\boxed{\mathbf{w}_{\text{REG}} = (\mathbf{Z}^\top \mathbf{Z} + \lambda \mathbf{I})^{-1} \mathbf{Z}^\top \mathbf{y}.}$$

Compare to the OLS solution $\mathbf{w}_{\text{lin}} = (\mathbf{Z}^\top \mathbf{Z})^{-1} \mathbf{Z}^\top \mathbf{y}$: the only change is $\lambda \mathbf{I}$ added inside the inverse. This is [[ridge-regression|ridge regression]].

The L2 penalty is called **weight decay** because, in iterative optimisers, each step subtracts a multiple of $\mathbf{w}$ — every weight "decays" toward zero. Larger $\lambda$ means stronger decay, shorter $\mathbf{w}$, smaller effective $C$.

## Choosing $\lambda$

The trade-off is sensitive: too little regularisation lets overfitting through, too much produces underfitting.

The shape of expected $E_{\text{out}}$ vs $\lambda$ is U-shaped: a sharp drop at small $\lambda$ (fixing overfitting), a minimum at some optimal $\lambda^\ast$, then a slow rise (underfitting). The optimal $\lambda^\ast$ depends on:

- **Stochastic noise.** More noise → larger $\lambda^\ast$ (more brakes for a bumpier road).
- **Deterministic noise** (target complexity beyond $\mathcal{H}$). More → larger $\lambda^\ast$.
- **Data size $N$.** More data → smaller $\lambda^\ast$ (less need to regularise heavily).

In practice $\lambda^\ast$ is *unknown* — neither $\sigma^2$ nor target complexity is observable. The standard procedure is **validation**: try a grid of $\lambda$ values, hold out part of the training data, and pick the $\lambda$ minimising held-out error. This is Lec 1's cross-validation half of the "two cures" — covered in detail next week.

## General Regularisers

The L2 norm isn't the only choice. The general $L_p$ family is

$$\Omega(\mathbf{w}) = \sum_i |w_i|^p.$$

| $p$ | Name | Effect |
|---|---|---|
| $p = 2$ | weight decay / [[ridge-regression\|ridge]] | Shrinks weights smoothly toward 0 |
| $p = 1$ | [[lasso-regression\|lasso]] (sparsity) | Drives some weights to *exactly* 0 |
| $p < 1$ | (non-convex) | Aggressive sparsity, harder to optimise |

L2 has a unique closed-form solution and is differentiable everywhere; L1 is non-differentiable at zero but produces sparse solutions, making it the right choice when you suspect the true model is sparse (most coefficients exactly zero). The $p < 1$ regularisers are non-convex and rarely used outside specialised contexts.

The Bayesian view: each regulariser corresponds to a prior on $\mathbf{w}$. L2 ↔ Gaussian, L1 ↔ Laplace, $p < 1$ ↔ super-Gaussian (heavy tails *and* sharp peaks).

## Practical Tips

The lecture's "tricks and tips":

- **Try regularisation by default.** It rarely hurts if $\lambda$ is reasonable. Sweep a grid and let validation pick.
- **Higher noise → more regularisation.** Heuristic argument: noise is "high frequency", complex targets are also high frequency, so a low-frequency-favouring regulariser helps.
- **Modern deep learning is full of regularisation.** L2 weight decay, dropout, batch norm, data augmentation, early stopping — all variants of the same idea. Different $\Omega$'s, different $\lambda$'s.

## Related

- [[ridge-regression]] — L2 regularisation specifically applied to linear regression.
- [[lasso-regression]] — L1 regularisation; sparse solutions.
- [[bayesian-linear-regression]] — regularisation as MAP estimation under a prior.
- [[generalization-bound]] — the VC bound that motivates regularisation.
- [[overfitting-ml|overfitting]] — the disease that regularisation cures.
- [[lagrangian]] — the constrained-to-unconstrained translation underlying $E_{\text{aug}}$.

## Active Recall

> [!question]- Show that the constrained optimisation $\min E_{\text{in}}$ s.t. $\mathbf{w}^\top \mathbf{w} \leq C$ is equivalent to the unconstrained $\min E_{\text{in}}(\mathbf{w}) + (\lambda/N) \mathbf{w}^\top \mathbf{w}$ for some $\lambda \geq 0$. Explain the geometric intuition.
> At the constrained optimum $\mathbf{w}_{\text{REG}}$ (assuming the constraint is active, i.e. on the surface $\mathbf{w}^\top \mathbf{w} = C$), $-\nabla E_{\text{in}}$ must be parallel to the outward normal of the surface — otherwise we could move along the surface to decrease $E_{\text{in}}$. The outward normal at $\mathbf{w}_{\text{REG}}$ is $\mathbf{w}_{\text{REG}}$ itself, so $-\nabla E_{\text{in}} \propto \mathbf{w}_{\text{REG}}$, i.e. $\nabla E_{\text{in}}(\mathbf{w}_{\text{REG}}) + (2\lambda/N) \mathbf{w}_{\text{REG}} = 0$ for some $\lambda \geq 0$. This is exactly the stationarity condition for $E_{\text{aug}}(\mathbf{w}) = E_{\text{in}}(\mathbf{w}) + (\lambda/N) \mathbf{w}^\top \mathbf{w}$. The correspondence $C \leftrightarrow \lambda$ is monotone: larger $C$ (looser constraint) ↔ smaller $\lambda$ (lighter penalty).

> [!question]- Why is the augmented error $E_{\text{aug}}(\mathbf{w}) = E_{\text{in}}(\mathbf{w}) + (\lambda/N) \Omega(\mathbf{w})$ a better proxy for $E_{\text{out}}$ than $E_{\text{in}}$ alone?
> The VC bound has two terms: $E_{\text{in}}$ (fit) and $\Omega(\mathcal{H})$ (set complexity). Pure ERM minimises only the first; $E_{\text{out}}$ depends on both. $E_{\text{aug}}$ adds a term proxying the complexity penalty, so minimising it minimises a *closer approximation* to the upper bound on $E_{\text{out}}$. The proxy is heuristic — $\Omega(\mathbf{w})$ for the chosen weights, not $\Omega(\mathcal{H})$ for the whole set — but for "well-chosen" regularisers (e.g., $\Omega(\mathbf{w}) = \|\mathbf{w}\|^2$) it correlates with the hypothesis-set complexity that the algorithm effectively uses.

> [!question]- Compute the closed-form regularised solution for L2 linear regression. Why does adding $\lambda \mathbf{I}$ to $\mathbf{Z}^\top \mathbf{Z}$ also fix numerical conditioning issues?
> $\nabla E_{\text{aug}}(\mathbf{w}) = \frac{2}{N} \mathbf{Z}^\top (\mathbf{Z} \mathbf{w} - \mathbf{y}) + \frac{2\lambda}{N} \mathbf{w} = 0$. Multiplying through and rearranging: $(\mathbf{Z}^\top \mathbf{Z} + \lambda \mathbf{I}) \mathbf{w}_{\text{REG}} = \mathbf{Z}^\top \mathbf{y}$, so $\mathbf{w}_{\text{REG}} = (\mathbf{Z}^\top \mathbf{Z} + \lambda \mathbf{I})^{-1} \mathbf{Z}^\top \mathbf{y}$. Adding $\lambda \mathbf{I}$ shifts every eigenvalue of $\mathbf{Z}^\top \mathbf{Z}$ up by $\lambda$, so a near-zero eigenvalue (the source of ill-conditioning) becomes $\lambda$, well away from zero. The condition number — ratio of largest to smallest eigenvalue — improves, and the inverse is numerically stable even when $\mathbf{Z}^\top \mathbf{Z}$ alone would be singular or near-singular.

> [!question]- What is the *effective* VC dimension of an L2-regularised hypothesis set, and why is it usually smaller than the nominal $d_{\text{VC}}$?
> The nominal $d_{\text{VC}}(\mathcal{H}) = \tilde{d} + 1$ counts all weight vectors as candidates. With a regulariser, the algorithm only converges to weights inside the ball $\mathcal{H}(C) = \{\mathbf{w} : \mathbf{w}^\top \mathbf{w} \leq C\}$ — a strictly smaller set, with smaller VC dimension. The *effective* VC dimension is $d_{\text{EFF}}(\mathcal{H}, \mathcal{A}) = d_{\text{VC}}(\mathcal{H}(C))$, where $C$ is the implicit budget set by $\lambda$. So even though the nominal model class is highly expressive, the regularised algorithm effectively selects from a smaller class — explaining why the regularised generalisation gap is smaller than a naive $d_{\text{VC}}$ analysis would predict.

> [!question]- What's the practical procedure for choosing $\lambda$, and why can't you read it off the data directly?
> $\lambda^\ast$ depends on $\sigma^2$ (stochastic noise), target complexity (deterministic noise), and $N$ — none of which is directly observable. The standard procedure is **validation**: try a grid of $\lambda$ values, train on a portion of the training data, evaluate held-out error on the remainder, and pick the $\lambda$ minimising mean held-out error (often via $K$-fold cross-validation). The held-out error is an unbiased estimate of $E_{\text{out}}$, so the chosen $\lambda$ is the one that empirically generalises best — without requiring knowledge of the noise levels.
