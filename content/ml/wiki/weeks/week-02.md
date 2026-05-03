---
type: week
week: 2
title: "Solving Logistic Regression: From MLE to IRLS"
dates: 2025-10-06 to 2025-10-12
sources:
  - raw/week-02/Wk_2_Lec_1-1.pdf
  - raw/week-02/Wk_2_Lec_2-1.pdf
  - raw/week-02/Monday_ October 6_ 2025 at 4_14_08 PM_Captions_English (United States).txt
  - raw/week-02/Tuesday_ October 7_ 2025 at 10_01_35 AM_Captions_English (United States).txt
  - raw/week-02/Tutorial_Wk_2.pdf
  - raw/week-02/2a-MLE_gradient_descent-exercises.pdf
  - raw/week-02/2a-MLE_gradient_descent-answers.pdf
  - raw/week-02/2b-newton_raphson_IRLS-exercises.pdf
  - raw/week-02/2b-newton_raphson_IRLS-answers.pdf
concepts:
  - "[[maximum-likelihood-estimation-ml|maximum likelihood estimation]]"
  - "[[cross-entropy-loss]]"
  - "[[gradient-descent-ml|gradient descent]]"
  - "[[convex-function]]"
  - "[[taylor-polynomial]]"
  - "[[hessian-matrix]]"
  - "[[newton-raphson-method]]"
  - "[[iteratively-reweighted-least-squares]]"
status: draft
updated: 2026-04-26
---

> [!question]+ THE CRUX: We have a hypothesis set $\mathcal{H}$ for logistic regression, but no learning algorithm $\mathcal{A}$. How do we actually *find* the weights $\mathbf{w}$ that best fit the data — and what do we do when the obvious approach is too slow?

*We turn "best fit" into a precise objective using maximum likelihood, derive the cross-entropy loss, and minimise it with gradient descent. Then we confront gradient descent's failure modes — differential curvature and learning-rate sensitivity — and fix them with Newton-Raphson, which becomes IRLS when applied to logistic regression.*

---

Last week we built logistic regression as a hypothesis set: $\mathcal{H} = \{\sigma(\mathbf{w}^\top \mathbf{x}) \mid \mathbf{w} \in \mathbb{R}^{d+1}\}$. We can score any candidate $\mathbf{w}$ on a test point. But "given training data, find the best $\mathbf{w}$" is still an empty box. This week fills it in.

There are two distinct pieces to nail down:

1. **What does "best" mean?** We need an objective function — a loss $E(\mathbf{w})$ that quantifies how badly a given $\mathbf{w}$ fits the data.
2. **How do we minimise it?** We need an algorithm that searches the space of weights for the one that drives $E(\mathbf{w})$ to its minimum.

Both pieces are non-trivial, and the standard answer to (2) — gradient descent — has subtle failure modes that motivate a more sophisticated approach.

## Step 1: What Should the Loss Be?

It might be tempting to count misclassifications. But this is a discrete, non-differentiable function — moving $\mathbf{w}$ slightly typically doesn't change the count, until suddenly it does. Useless for optimisation.

A better idea: since the model outputs *probabilities*, we should reward weights that assign high probability to the actual training labels. This is the principle of [[maximum-likelihood-estimation-ml|maximum likelihood estimation]]: pick the $\mathbf{w}$ for which the observed training data is most probable.

For a single example $(\mathbf{x}^{(i)}, y^{(i)})$, the model assigns probability $p_{y^{(i)}} = p(y^{(i)} \mid \mathbf{x}^{(i)}, \mathbf{w})$ to the actual label. Assuming i.i.d. examples, the joint probability of the entire training set is the product:

$$\mathcal{L}(\mathbf{w}) = \prod_{i=1}^{N} p(y^{(i)} \mid \mathbf{x}^{(i)}, \mathbf{w})$$

We want $\mathbf{w}^* = \arg\max_\mathbf{w} \mathcal{L}(\mathbf{w})$.

> [!info] ASIDE — Probability vs likelihood
> Though they share a formula, the words mean different things. *Probability* treats the parameters $\mathbf{w}$ as fixed and asks how likely the data is. *Likelihood* treats the data as fixed and asks how plausible different $\mathbf{w}$ values are. We're in likelihood mode: the data is given, $\mathbf{w}$ is what varies.

## From Likelihood to Cross-Entropy

Products of probabilities are numerically nasty — multiplying $N$ small numbers gives an absurdly tiny result that underflows in floating-point arithmetic. The fix is to take the logarithm, which converts products into sums and is monotonic (so $\arg\max$ is preserved):

$$\ln \mathcal{L}(\mathbf{w}) = \sum_{i=1}^{N} \ln p(y^{(i)} \mid \mathbf{x}^{(i)}, \mathbf{w})$$

Convention prefers minimising over maximising, so we negate to get the [[cross-entropy-loss]]:

$$E(\mathbf{w}) = -\sum_{i=1}^{N} \left[ y^{(i)} \ln p_1(\mathbf{x}^{(i)}, \mathbf{w}) + (1 - y^{(i)}) \ln (1 - p_1(\mathbf{x}^{(i)}, \mathbf{w})) \right]$$

The two terms are a switch: when $y^{(i)} = 1$, only the first term contributes ($-\ln p_1$); when $y^{(i)} = 0$, only the second ($-\ln (1 - p_1)$). Either way, the loss is small when the model assigns high probability to the correct class, and explodes toward $+\infty$ when it confidently predicts the wrong class.

> [!question]- The model assigns $p_1 = 0.99$ to a training example whose true label is $y = 1$. What does that example contribute to the loss? What if $p_1 = 0.01$ instead?
> When $y = 1$, the contribution is $-\ln p_1$. So $p_1 = 0.99$ contributes $-\ln(0.99) \approx 0.01$ — almost nothing, since the model is correct and confident. But $p_1 = 0.01$ contributes $-\ln(0.01) \approx 4.6$ — a heavy penalty for a confident wrong prediction.

The cross-entropy loss for logistic regression is **strictly [[convex-function|convex]]** in $\mathbf{w}$ — it has a single, unique global minimum. This is the property that makes everything else work.

## Step 2: How Do We Minimise It?

There's no closed-form $\mathbf{w}^*$ for logistic regression's cross-entropy (unlike linear regression with squared error). We need an iterative procedure.

The natural one is [[gradient-descent-ml|gradient descent]]: stand at a point, compute which way is "downhill" (the direction of steepest descent — the negative gradient), take a step, repeat.

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla E(\mathbf{w})$$

where $\eta > 0$ is the **learning rate**. For our cross-entropy loss, the gradient has a clean form:

$$\nabla E(\mathbf{w}) = \sum_{i=1}^{N} (p_1(\mathbf{x}^{(i)}, \mathbf{w}) - y^{(i)}) \mathbf{x}^{(i)}$$

It's just the prediction error $(p_1 - y)$ weighted by the input vector, summed over the training set. Beautifully simple.

> [!info] ASIDE — Why gradient descent works at all
> The gradient $\nabla E$ points in the direction of steepest *increase*. The negative gradient points in the direction of steepest *decrease*. The definition of derivative guarantees that for a small enough step, moving in the direction of $-\nabla E$ strictly reduces $E$. This is a *local* property — gradient descent has no global view of the loss landscape, only the slope under its feet.

Because cross-entropy for logistic regression is strictly convex, gradient descent converging to a critical point converges to the *global* minimum. No worries about local minima.

## The Trouble with $\eta$

The learning rate $\eta$ is a hyperparameter — *we* pick it, the algorithm does not. And it matters enormously:

- $\eta$ **too large**: the step overshoots the minimum, possibly landing on the *opposite* slope at a higher loss. Iterations bounce around or even diverge.
- $\eta$ **too small**: each step moves a tiny distance. Convergence is technically guaranteed but takes thousands of iterations to make progress that a moderate $\eta$ would make in tens.

Picking $\eta$ is partly trial and error, but there's a deeper problem lurking.

## The Real Problem: Differential Curvature

Even with a perfectly tuned $\eta$, gradient descent struggles when the loss surface curves at different rates along different axes. Imagine the contour plot of $E(\mathbf{w}) = w_1^2 + 4 w_2^2$ — concentric ellipses, stretched along $w_1$ because the curvature in $w_2$ is four times stronger.

What does gradient descent do here? At any point, the gradient is much larger in the $w_2$ direction than in $w_1$. A single $\eta$ has to handle both. Set it small enough to avoid overshooting in $w_2$, and progress along $w_1$ becomes glacial. The trajectory zig-zags across the narrow valley while creeping slowly toward the optimum.

This isn't a curvy edge case — it's the typical situation when input features have different scales. Standardising features (subtracting the mean, dividing by the standard deviation) helps but doesn't fully cure the issue: the loss landscape's curvature is determined by feature *interactions*, not just feature scales.

> [!question]- A loss surface has contours that look like long, narrow ellipses aligned with the $w_1$ axis. Where will gradient descent waste most of its iterations, and why?
> The narrow direction is $w_2$ (steep walls), the long direction is $w_1$ (gentle slope along the valley). Gradient descent will zigzag back and forth across the narrow $w_2$ direction while inching slowly along $w_1$ toward the minimum. The single $\eta$ can't simultaneously be aggressive in $w_1$ and conservative in $w_2$.

## The Insight: Use Curvature Information

Gradient descent uses only *first-order* information — the slope. But the slope tells you nothing about how it itself is changing. If the slope is changing rapidly in some direction, that's a sign you should take *smaller* steps; if it's barely changing, you can afford a larger step.

The **second-order derivative** captures exactly this: how quickly the gradient is changing. If we can build that into our update rule, we get an algorithm that automatically adapts its step size to the local curvature.

The vehicle is the [[taylor-polynomial]]. A degree-2 Taylor approximation of $E$ around the current weight $w_0$ is a quadratic that matches the true loss in value, slope, and curvature at $w_0$:

$$E(w) \approx E(w_0) + (w - w_0) E'(w_0) + \tfrac{(w - w_0)^2}{2} E''(w_0)$$

This quadratic has a closed-form minimum. Setting its derivative to zero and solving for $w$ gives the **[[newton-raphson-method|Newton-Raphson]]** update:

$$w \leftarrow w - \frac{E'(w)}{E''(w)}$$

Notice what's happening: the first derivative still gives the *direction*, but the step size is now $1/E''(w)$ — automatic, no $\eta$ to tune. Where curvature is high (second derivative large), steps shrink; where curvature is low, steps grow.

## Multivariate: The Hessian

For multi-dimensional $\mathbf{w}$, the second derivative becomes the [[hessian-matrix|Hessian]] — a matrix of all second-order partial derivatives:

$$H_{ij} = \frac{\partial^2 E}{\partial w_i \, \partial w_j}$$

It captures not just curvature along each axis but the *interactions* between axes. The Newton-Raphson update generalises to:

$$\mathbf{w} \leftarrow \mathbf{w} - H_E^{-1}(\mathbf{w}) \, \nabla E(\mathbf{w})$$

For logistic regression specifically, this update is called **[[iteratively-reweighted-least-squares|Iteratively Reweighted Least Squares (IRLS)]]**:

$$H_E(\mathbf{w}) = \sum_{i=1}^{N} p_1(\mathbf{x}^{(i)}, \mathbf{w})(1 - p_1(\mathbf{x}^{(i)}, \mathbf{w})) \mathbf{x}^{(i)} \mathbf{x}^{(i)\top}$$

Both gradient and Hessian depend on $\mathbf{w}$, so we update $\mathbf{w}$, recompute, update again — hence "iterative." The procedure typically converges in far fewer iterations than gradient descent.

> [!tip] TIP — Why is it called "Reweighted Least Squares"?
> The Newton update can be rewritten as a weighted least-squares problem at each iteration, where the weights are $p_1(1 - p_1)$ — i.e., proportional to the model's *uncertainty* about each example. Examples the model is unsure about (probability near 0.5) get the highest weight; examples it's already very confident about contribute little. The "reweighting" changes at every iteration as $\mathbf{w}$ changes.

## Trade-offs

IRLS isn't free. Each iteration requires inverting a $(d+1) \times (d+1)$ Hessian — $O(d^3)$ work. For high-dimensional problems (many features), this becomes prohibitive, and gradient descent (which is $O(d)$ per step but needs more steps) wins.

Also, the quadratic approximation is only locally accurate. Far from the minimum, Newton-Raphson can overshoot or even move uphill if the Hessian is non-positive-definite. For logistic regression with cross-entropy this isn't a problem (loss is convex, Hessian is positive semi-definite), but it's a real concern in non-convex settings like neural networks — which is why deep learning relies on first-order methods despite their slower per-iteration progress.

> [!question]- Newton-Raphson needs no learning rate, while gradient descent does. Yet deep learning practitioners rarely use Newton-Raphson. Why?
> Two reasons. First, computing and inverting the Hessian is $O(d^3)$ per iteration; for a network with millions of parameters, that's infeasible. Second, deep network losses are non-convex, so the quadratic approximation can mislead — the Hessian may not be positive-definite, and the "Newton step" may point uphill. Gradient descent (and its variants like Adam) trades per-iteration progress for tractable per-iteration *cost*.

---

## Concepts Introduced This Week

- [[maximum-likelihood-estimation-ml|maximum likelihood estimation]] — the principle: pick parameters that make the observed data most probable.
- [[cross-entropy-loss]] — the negative log-likelihood for logistic regression; strictly convex in $\mathbf{w}$.
- [[gradient-descent-ml|gradient descent]] — iterative first-order optimisation; updates $\mathbf{w}$ in the direction of steepest descent.
- [[convex-function]] — guarantees that local optima are global; lets gradient descent succeed.
- [[taylor-polynomial]] — local polynomial approximation; degree-2 version underpins Newton-Raphson.
- [[hessian-matrix]] — matrix of second-order partial derivatives; captures curvature in multi-dim.
- [[newton-raphson-method]] — second-order optimisation using a quadratic approximation; no learning rate needed.
- [[iteratively-reweighted-least-squares]] — Newton-Raphson applied to logistic regression's cross-entropy loss.

## Connections

- **Builds on** [[week-01]]: completes the supervised learning framework's missing piece — the algorithm $\mathcal{A}$.
- **Sets up** later weeks: gradient descent is the workhorse for nearly every algorithm in the module. Cross-entropy reappears in neural networks. The MLE principle generalises to most probabilistic models.

## Open Questions

- What if the data isn't linearly separable? (Answered in weeks 3–5: non-linear transforms, kernels, SVMs.)
- How do we control overfitting when $\mathcal{H}$ is too expressive? (Answered later: regularisation, validation, VC theory.)
- For very large datasets, can we avoid summing over all $N$ examples per gradient step? (Answered when stochastic gradient descent is introduced.)
