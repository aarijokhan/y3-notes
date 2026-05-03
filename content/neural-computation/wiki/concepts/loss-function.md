---
type: concept
sources:
  - raw/week-01/w01-l03-transcript.txt
  - raw/week-01/w01-slides.pdf
status: stable
updated: 2026-04-20
---

*A function that measures how far a model's predictions are from the truth — the thing we minimise when we train.*

## Definition

A **loss function** (also called a cost function or objective function) maps a model's predictions and the true values to a single non-negative number. Smaller values mean better predictions. Training a model means finding parameters that minimise the loss:

$$\mathbf{w}^*, b^* = \arg\min_{\mathbf{w}, b} \; L(\mathbf{w}, b)$$

The notation $\arg\min$ means "the argument (value) that produces the minimum" — not the minimum itself, but the parameter values that achieve it.

## Why we need loss functions

A [[perceptron]] has parameters $\mathbf{w}$ and $b$, but how do we find the right values? We need a way to score any candidate set of parameters: "these weights give a total error of 9; those give 7; those give 3." The loss function provides that score. Once we have it, learning becomes an optimisation problem.

## Concrete example: temperature estimation

Suppose three noisy thermometers read $x_1 = 19°$C, $x_2 = 17°$C, $x_3 = 24°$C. We want to estimate the true temperature $\hat{x}$.

Using absolute error, an estimate of $\hat{x} = 21$ gives:

$$L = |19 - 21| + |17 - 21| + |24 - 21| = 2 + 4 + 3 = 9$$

An estimate of $\hat{x} = 22$ gives $L = 3 + 5 + 2 = 10$ — worse. We sweep through all candidates and pick the one with the lowest total loss.

## Common loss functions

### Mean Absolute Error (MAE)

$$L_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

The average of absolute differences. Treats all errors equally regardless of direction. The loss-vs-estimate curve is V-shaped (for a single data point) or piecewise linear.

### Sum of Squared Errors (SSE)

$$L_{\text{SSE}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Squaring penalises large errors more heavily than small ones. The loss-vs-estimate curve is a smooth parabola (for a single parameter), which makes it easier to optimise with calculus. SSE is not an arbitrary choice — [[maximum-likelihood-estimation-nc|maximum likelihood estimation]] shows it falls out naturally from assuming Gaussian noise.

### Mean Squared Error (MSE)

$$L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

SSE divided by the number of data points. Normalising by $n$ makes the loss comparable across datasets of different sizes but does not change which parameters minimise it.

## A special case where we don't need gradient descent: the analytical solution for SSE

For most loss functions and model architectures, optimisation is iterative — gradient descent climbs down the loss landscape one step at a time. But for the specific case of estimating a single value $\hat{x}$ from measurements $x_1, \dots, x_n$ under sum-of-squared-errors, calculus gives a closed-form answer directly.

**Setup.** Find $\hat{x}$ minimising $L(\hat{x}) = \sum_{i=1}^{n} (\hat{x} - x_i)^2$.

**Take the derivative.** Expanding $(\hat{x} - x_i)^2 = \hat{x}^2 - 2 \hat{x} x_i + x_i^2$ and differentiating term-by-term:

$$\frac{\partial L}{\partial \hat{x}} = \sum_{i=1}^{n} (2\hat{x} - 2 x_i) = 2n \hat{x} - 2 \sum_{i=1}^{n} x_i$$

**Set it to zero** (necessary condition for a minimum on a smooth convex function):

$$2n \hat{x} = 2 \sum_{i=1}^{n} x_i \quad \Longrightarrow \quad \hat{x}^* = \frac{1}{n} \sum_{i=1}^{n} x_i$$

The optimal estimate is **the arithmetic mean** of the measurements. For three thermometer readings $\{19, 17, 24\}$, the SSE-optimal estimate is $(19 + 17 + 24)/3 = 20$.

This is the same answer [[maximum-likelihood-estimation-nc|maximum likelihood estimation]] gives for Gaussian noise, but derived without any probability — purely as the calculus solution to the SSE minimisation. (The two derivations agreeing is what we'd expect: SSE is what MLE recommends under a Gaussian assumption.)

> [!info] ASIDE — Why we still need gradient descent
> Closed-form solutions exist only for simple losses with simple models. As soon as the model has non-linearities, multiple parameters with non-trivial dependencies, or a non-quadratic loss, the "set the derivative to zero and solve" approach produces equations you can't rearrange by hand. Iterative gradient descent works in *every* case where the loss is differentiable, regardless of how messy the closed-form solution would be — which is why it's the workhorse algorithm for neural networks.

## Why optimisation is hard

For a single parameter, we could plot the loss and visually find the minimum. But real models have many parameters — modern neural networks have billions. Each parameter can take any real value, creating an effectively infinite search space. Brute-force search is impossible; we need efficient algorithms like gradient descent (week 2).

## Related

- [[perceptron]] — the model whose parameters we optimise
- [[maximum-likelihood-estimation-nc|maximum likelihood estimation]] — derives squared error loss from probability theory

## Active Recall

> [!question]- What is the difference between $\min$ and $\arg\min$, and why do we care about $\arg\min$ in machine learning?
> $\min L$ gives the smallest value of the loss. $\arg\min L$ gives the *parameter values* that achieve that smallest loss. In ML we care about $\arg\min$ because we need the actual weight values, not just how low the loss can go.

> [!question]- Why does squared error penalise predictions differently from absolute error, and when might that matter?
> Squaring magnifies large errors: an error of 4 contributes $16$ to SSE but only $4$ to MAE. This means SSE is more sensitive to outliers — a single wildly wrong prediction can dominate the total loss. MAE treats all errors proportionally to their magnitude.

> [!question]- Three sensors read 10, 14, and 16. Compute the MAE and SSE for an estimate of 13.
> MAE: $(|10-13| + |14-13| + |16-13|)/3 = (3 + 1 + 3)/3 = 7/3 \approx 2.33$. SSE: $(10-13)^2 + (14-13)^2 + (16-13)^2 = 9 + 1 + 9 = 19$.

> [!question]- Why can't we find optimal parameters by trying every possible value of $\mathbf{w}$ and $b$?
> Each parameter is a real number with infinitely many possible values. A model with $D$ weights has a $D+1$ dimensional search space (weights plus bias). Modern networks have billions of parameters, making brute-force search computationally impossible. We need algorithms like gradient descent that navigate the loss landscape efficiently.

> [!question]- Derive the value of $\hat{x}$ that minimises $L(\hat{x}) = \sum_{i=1}^n (\hat{x} - x_i)^2$.
> Differentiate: $\frac{\partial L}{\partial \hat{x}} = \sum (2\hat{x} - 2 x_i) = 2n\hat{x} - 2\sum x_i$. Set to zero: $\hat{x}^* = \frac{1}{n}\sum x_i$ — the sample mean. The squared-error optimum *is* the average of the measurements. This is one of the few cases where the optimal parameter has a closed form; in general, gradient descent is needed because the analytical solution is intractable.
