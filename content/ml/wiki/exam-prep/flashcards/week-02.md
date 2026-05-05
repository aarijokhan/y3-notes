---
week: 2
topic: "Solving Logistic Regression: From MLE to IRLS"
deck: MachineLearning::Week-02
---

TARGET DECK
MachineLearning::Week-02

## Maximum Likelihood Estimation

> [!question]- What is the **principle of maximum likelihood estimation** (MLE)?
> Pick the parameters $\mathbf{w}$ for which the observed training data is most probable:
> $$\mathbf{w}^\ast = \arg\max_{\mathbf{w}} \prod_{i=1}^N p(y^{(i)} \mid \mathbf{x}^{(i)}, \mathbf{w})$$
> Treat the data as fixed; treat $\mathbf{w}$ as variable. Choose $\mathbf{w}$ that makes the data look most plausible under the model.

> [!question]- Why do we minimise the **negative log-likelihood** instead of maximising the likelihood directly?
> Two reasons:
> - **Numerical**: products of small probabilities underflow in floating-point arithmetic. Logarithm converts $\prod p_i$ into $\sum \log p_i$ — a sum of moderate-magnitude numbers.
> - **Convention**: optimisers minimise. Negating gives a loss to minimise without changing the $\arg\max$ ($\log$ is monotonic).

> [!question]- What is the **cross-entropy loss** for logistic regression?
> $$E(\mathbf{w}) = -\sum_{i=1}^N \left[ y^{(i)} \ln p_1^{(i)} + (1 - y^{(i)}) \ln (1 - p_1^{(i)}) \right]$$
> where $p_1^{(i)} = \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})$.
> The two terms act as a switch: when $y = 1$, only $-\ln p_1$ contributes; when $y = 0$, only $-\ln (1 - p_1)$ contributes. Strictly **convex** in $\mathbf{w}$, so a unique global minimum exists.

> [!question]- A logistic regression model assigns $p_1 = 0.99$ to a true-label-1 example. What does it contribute to the loss? What if $p_1 = 0.01$ instead?
> When $y = 1$, the contribution is $-\ln p_1$.
> - $p_1 = 0.99 \Rightarrow -\ln(0.99) \approx 0.01$ — almost zero (correct, confident).
> - $p_1 = 0.01 \Rightarrow -\ln(0.01) \approx 4.6$ — large penalty (wrong, confident).
>
> Cross-entropy heavily punishes confidently-wrong predictions; the loss diverges to $+\infty$ as $p \to 0$ for the true class.

## Gradient Descent

> [!question]- What is the **gradient descent** update rule, and what is the role of $\eta$?
> $$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla E(\mathbf{w})$$
> - $\nabla E(\mathbf{w})$ is the gradient — direction of steepest *increase*.
> - The negative gradient points downhill.
> - $\eta > 0$ is the **learning rate** — step size hyperparameter.
>
> Each step moves $\mathbf{w}$ a distance proportional to $\|\nabla E\|$ in the direction that locally reduces $E$ fastest.

> [!question]- What is the gradient of the cross-entropy loss for logistic regression?
> $$\nabla E(\mathbf{w}) = \sum_{i=1}^N (p_1^{(i)} - y^{(i)}) \, \mathbf{x}^{(i)}$$
> The prediction error $(p_1 - y)$ scales each input vector. Beautifully simple: when the prediction matches the label, that example contributes nothing to the gradient.

> [!question]- What goes wrong when the gradient descent learning rate $\eta$ is too large or too small?
> - **$\eta$ too large**: the step overshoots the minimum, possibly landing on the opposite slope at *higher* loss. Iterations bounce around or diverge.
> - **$\eta$ too small**: each step makes negligible progress; convergence is technically guaranteed but might take thousands of iterations to do what a moderate $\eta$ does in tens.
>
> No single $\eta$ is universally correct — it requires tuning, and the right value depends on the loss landscape's curvature.

> [!question]- What is **differential curvature** and why does it cause gradient descent to zig-zag?
> When the loss surface curves at different rates along different axes — e.g. $E(\mathbf{w}) = w_1^2 + 4w_2^2$ — the gradient is much larger in the steep direction than the gentle one. A single $\eta$ has to handle both: small enough to avoid overshoot in the steep direction, but then progress is glacial in the gentle direction. The trajectory zig-zags across the narrow valley while creeping toward the optimum.

## Newton-Raphson and IRLS

> [!question]- What is the **Newton-Raphson** update rule (1D), and why does it not need a learning rate?
> $$w \leftarrow w - \frac{E'(w)}{E''(w)}$$
> The first derivative gives the direction; the second derivative scales the step. Where curvature is high ($E''$ large), steps shrink automatically; where it's low, steps grow. The step size is *adapted to local curvature* — no $\eta$ to tune. Comes from minimising the degree-2 Taylor approximation of $E$ exactly.

> [!question]- What is the **multivariate Newton-Raphson** update, and what does the Hessian capture?
> $$\mathbf{w} \leftarrow \mathbf{w} - H_E^{-1}(\mathbf{w}) \, \nabla E(\mathbf{w})$$
> The **Hessian** $H_{ij} = \partial^2 E / (\partial w_i \, \partial w_j)$ is the matrix of second-order partial derivatives. It captures curvature along each axis *and* interactions between axes. Inverting it produces direction-aware step sizes that handle differential curvature automatically.

> [!question]- What is **Iteratively Reweighted Least Squares (IRLS)**?
> Newton-Raphson applied to logistic regression's cross-entropy loss. The Hessian is:
> $$H_E(\mathbf{w}) = \sum_{i=1}^N p_1^{(i)} (1 - p_1^{(i)}) \, \mathbf{x}^{(i)} \mathbf{x}^{(i)\top}$$
> The "reweighting" weight $p_1(1 - p_1)$ is largest when $p_1 \approx 0.5$ (model uncertain) and small when $p_1 \approx 0$ or $1$ (model confident). Each iteration is a weighted least-squares step where the weights track current uncertainty.

> [!question]- Why is IRLS rarely used for deep neural networks despite needing no learning rate?
> Two reasons:
> - **Cost**: each iteration inverts a $(d+1) \times (d+1)$ Hessian — $O(d^3)$. For millions of parameters, infeasible.
> - **Non-convexity**: deep network losses are non-convex, so the local quadratic Taylor approximation can mislead. The Hessian may not be positive-definite, and the "Newton step" may point uphill.
>
> Gradient descent (and variants like Adam) trades per-iteration progress for tractable per-iteration cost.

## Convexity

> [!question]- What does it mean for a function to be **convex**, and why does convexity matter for optimisation?
> A function $f$ is convex if for any two points $\mathbf{w}_1, \mathbf{w}_2$ and $\lambda \in [0, 1]$:
> $$f(\lambda \mathbf{w}_1 + (1-\lambda) \mathbf{w}_2) \leq \lambda f(\mathbf{w}_1) + (1 - \lambda) f(\mathbf{w}_2)$$
> Convexity matters because it guarantees **every local minimum is a global minimum** — gradient descent (or any descent method) converging to a stationary point converges to *the* optimum. No worries about getting stuck in suboptimal valleys.

> [!question]- A loss has long, narrow elliptical contours aligned with the $w_1$ axis. Where will gradient descent struggle, and why?
> The narrow direction is $w_2$ (steep walls); the long direction is $w_1$ (gentle slope along the valley). Gradient descent zig-zags across $w_2$ while inching slowly along $w_1$. A single $\eta$ can't simultaneously be aggressive enough in $w_1$ and conservative enough in $w_2$. **Newton-Raphson would handle this automatically** by using curvature-adapted step sizes.
