---
type: week
week: 7
title: "Linear Regression and Why Squared Error Isn't Arbitrary"
dates: 2025-11-10 to 2025-11-16
sources:
  - raw/week-07/Wk_7_Lec_1-1.pdf
  - raw/week-07/Wk_7_Lec_2-1.pdf
  - raw/week-07/Monday_ November 10_ 2025 at 4_03_29 PM_Captions_English (United States).txt
  - raw/week-07/Tuesday_ November 11_ 2025 at 10_06_15 AM_Captions_English (United States).txt
  - raw/week-07/Tutorial_Wk_7-1.pdf
  - raw/week-07/ML_Exercise_Sheet_7a.pdf
  - raw/week-07/ML_Exercise_Sheet_7a_solution.pdf
  - raw/week-07/ML_Exercise_Sheet_7b.pdf
  - raw/week-07/ML_Exercise_Sheet_7b_solution.pdf
concepts:
  - "[[linear-regression]]"
  - "[[ordinary-least-squares]]"
  - "[[design-matrix]]"
  - "[[gaussian-distribution]]"
  - "[[bayes-law]]"
status: draft
updated: 2026-04-27
---

> [!question]+ THE CRUX: We've spent six weeks on classification — logistic regression, hard- and soft-margin SVMs. This week pivots to **regression**: predicting continuous outputs. Linear regression sounds trivial — fit a line — but two questions sit underneath. (1) How do we actually compute the answer? (2) *Why* do we use squared error and not absolute error or anything else?

*The first answer is **Ordinary Least Squares (OLS)**: minimise the sum of squared residuals, which has a closed-form solution via the **normal equation** $\mathbf{w} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}$. One matrix inversion, no iterations, no learning rate. The second answer is the **probabilistic interpretation**: assume $y = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, and the MLE for $\mathbf{w}$ is exactly the OLS solution. Squared error isn't arbitrary — it's the negative log-likelihood of a Gaussian noise model, up to constants. This is the regression analog of week 2's logistic-regression-as-MLE result: the noise model determines the loss.*

---

## Part 1: Linear Regression

### From Classification to Regression

For six weeks we've been doing **classification**: $\mathcal{Y}$ is a finite set of categories. This week, $\mathcal{Y} = \mathbb{R}$ — the output is continuous. Examples:

- BMI → cardiac ejection fraction
- Customer features → recommended credit line (in dollars, not yes/no)
- Ad spend → revenue

The hypothesis form looks identical to logistic regression's pre-sigmoid output:

$$h(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$$

But there's no sigmoid, no threshold. The output *is* the prediction. What changes is the loss function and how we fit.

### The Model

[[linear-regression|Linear regression]] assumes:

$$\hat{y}(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^M w_j \phi_j(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x})$$

where $\boldsymbol{\phi}$ is a vector of basis functions (with $\phi_0 \equiv 1$ for the intercept). The "linear" in linear regression refers to **linearity in the parameters $\mathbf{w}$** — not in $\mathbf{x}$. The following are all linear regression models:

$$\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 \qquad \text{(polynomial)}$$
$$\hat{y} = w_0 + w_1 e^{x_1} + w_2 e^{x_2} \qquad \text{(exponential basis)}$$
$$\hat{y} = w_0 + \sum_j w_j \exp\left(-\tfrac{(x - \mu_j)^2}{2 s^2}\right) \qquad \text{(Gaussian RBF basis)}$$

What unites them: $\partial \hat{y} / \partial w_j$ depends on $\mathbf{x}$ alone, not on $\mathbf{w}$. This is what OLS exploits to give a closed-form solution — no matter how curvy the basis functions, the *fitting* problem is linear-algebraic.

### Ordinary Least Squares

Define the **residual** $r_i = y_i - \hat{y}_i$ and the objective:

$$R(\mathbf{w}) = \sum_{i=1}^N r_i^2 = \sum_{i=1}^N (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2$$

[[ordinary-least-squares|OLS]] picks the $\mathbf{w}$ that minimises this. Setting $\nabla R = 0$ gives a linear system; collecting it in matrix form using the [[design-matrix|design matrix]] $\boldsymbol{\Phi}$:

$$\boldsymbol{\Phi}^\top \boldsymbol{\Phi} \, \mathbf{w} = \boldsymbol{\Phi}^\top \mathbf{y} \qquad \text{(normal equation)}$$

Solving:

$$\boxed{\mathbf{w}_{\text{OLS}} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y} = \boldsymbol{\Phi}^{\dagger} \mathbf{y}}$$

where $\boldsymbol{\Phi}^{\dagger}$ is the **Moore–Penrose pseudoinverse**. One matrix inversion gives the exact answer.

### Worked Example — Degree-1 Polynomial

For $\hat{y} = w_0 + w_1 x$, expanding $R$ and setting $\partial R / \partial w_0 = \partial R / \partial w_1 = 0$ gives:

$$\sum_i y_i = w_0 N + w_1 \sum_i x_i$$
$$\sum_i x_i y_i = w_0 \sum_i x_i + w_1 \sum_i x_i^2$$

In matrix form:

$$\begin{pmatrix} \sum_i y_i \\ \sum_i x_i y_i \end{pmatrix} = \begin{pmatrix} N & \sum_i x_i \\ \sum_i x_i & \sum_i x_i^2 \end{pmatrix} \begin{pmatrix} w_0 \\ w_1 \end{pmatrix}$$

Inverting the $2 \times 2$ matrix gives $w_0$ and $w_1$.

### Basis Expansion — The Bridge to Non-Linear

This is the same trick we used for SVMs in week 4: replace $\mathbf{x}$ with $\boldsymbol{\phi}(\mathbf{x})$ to gain non-linear capacity while staying within linear-fitting machinery. The model is "linear in $\boldsymbol{\phi}$-space" but bends in input space.

Common basis families:

| Basis | Form | When |
|---|---|---|
| Polynomial | $\phi_j(x) = x^j$ | Low-degree smooth |
| Gaussian RBF | $\phi_j(x) = e^{-(x - \mu_j)^2 / (2 s^2)}$ | Local bumps |
| Sigmoidal | $\phi_j(x) = (1 + e^{-(x - \mu_j)/s})^{-1}$ | Saturation |
| tanh | $\phi_j(x) = \tanh((x - \mu_j)/s)$ | Symmetric saturation |

Higher polynomial degree → tighter fit but more risk of overfitting. The classic plot: a degree-1 fit on 10 points gives a line that misses local structure; a degree-6 fit hits every point but oscillates wildly between them. **Validation** (next weeks) is how we pick the right degree.

### Iterative Alternative — Gradient Descent

OLS fails when $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ is too large to invert (millions of features) or near-singular (collinear features). Then [[gradient-descent-ml|gradient descent]] handles the same objective iteratively:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla R(\mathbf{w})$$

where $\nabla R = -2 \boldsymbol{\Phi}^\top (\mathbf{y} - \boldsymbol{\Phi} \mathbf{w})$. Convex objective → guaranteed convergence to the global optimum, but requires learning-rate tuning.

The gradient-descent example from the lecture: for $f(x, y) = x^3 + 2 y^2 - y$ at $(1, 0)$ with $\eta = 0.1$, $\nabla f = (3, -1)$, so:

$$x_{t+1} = (1, 0) - 0.1 \cdot (3, -1) = (0.7, 0.1)$$

Function value drops from $f(1,0) = 1$ to $f(0.7, 0.1) = 0.263$ — the step improved the objective.

## Part 2: The Probabilistic View — Why Squared Error?

### The Question OLS Doesn't Answer

OLS says "minimise squared residuals." But why squared? Why not absolute residuals, or fourth-power residuals? The OLS objective itself doesn't say. We need a separate argument.

The answer comes from **probabilistic modelling**. Assume the targets are generated by:

$$y_i = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$$

That is: the underlying relationship is linear (in $\boldsymbol{\phi}$-space), but observations are corrupted by additive [[gaussian-distribution|Gaussian noise]].

### MLE for the Regression Weights

Under this model, $y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i), \sigma^2)$. The likelihood of the dataset is:

$$\mathcal{L}(\mathbf{w}) = \prod_i \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2}{2 \sigma^2}\right)$$

Taking the log:

$$\ln \mathcal{L}(\mathbf{w}) = -\frac{N}{2} \ln(2 \pi \sigma^2) - \frac{1}{2 \sigma^2} \sum_i (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2$$

The only $\mathbf{w}$-dependent term is the **sum of squared residuals**, with a negative coefficient. So:

$$\arg\max_{\mathbf{w}} \ln \mathcal{L}(\mathbf{w}) = \arg\min_{\mathbf{w}} \sum_i (y_i - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i))^2 = \mathbf{w}_{\text{OLS}}$$

> [!info]+ MLE = OLS under additive Gaussian noise
> The maximum-likelihood weights are **identical** to the OLS weights, when noise is i.i.d. Gaussian. This is the missing justification: squared error isn't an arbitrary choice of loss — it's the negative log-likelihood of a Gaussian noise model. Picking any other loss is implicitly assuming a different noise distribution.

The MLE for $\sigma^2$ falls out of the same derivation: $\hat{\sigma}^2_{\text{MLE}} = \text{RSS} / N$ where $\text{RSS} = \|\mathbf{y} - \boldsymbol{\Phi} \mathbf{w}_{\text{OLS}}\|^2$.

### The Pattern Connecting Week 2 and Week 7

This is the same recipe we saw in week 2 for [[logistic-regression]]:

| Model | Noise / output model | MLE objective | Reduces to |
|---|---|---|---|
| Linear regression | $y = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}) + \varepsilon$, $\varepsilon \sim \mathcal{N}$ | Maximise Gaussian likelihood | **Minimise squared error** (= OLS) |
| Logistic regression | $y \sim \text{Bernoulli}(\sigma(\mathbf{w}^\top \mathbf{x}))$ | Maximise Bernoulli likelihood | **Minimise [[cross-entropy-loss\|cross-entropy]]** |

Same recipe (negative log-likelihood as loss), different distribution → different loss function. The "right" loss is determined by what noise/output model you're willing to assume.

### Frequentist vs Bayesian — Setting Up Future Weeks

The MLE view treats $\mathbf{w}$ as a *fixed unknown* to be point-estimated. The **Bayesian view** (preview for later weeks) treats $\mathbf{w}$ as a *random variable* with a prior distribution; observed data updates this to a posterior via [[bayes-law|Bayes' law]]:

$$p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) \cdot p(\mathbf{w})$$

Posterior ∝ likelihood × prior.

Under a Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1} \mathbf{I})$, the **MAP estimate** turns out to be **ridge regression** — adding $\alpha \|\mathbf{w}\|^2$ to the OLS objective. So L2 regularisation has a Bayesian interpretation: it's a Gaussian prior on the weights. The "regularisation" lever and the "prior belief" lever are the same lever, viewed differently.

This connection (regularisation ↔ priors) is one of the cleanest unifying ideas in the module.

## Part 3: Evaluating Regression Models

Common metrics for evaluating $\hat{y}$ vs $y$ on test data:

| Metric | Formula | Notes |
|---|---|---|
| Mean Squared Error (MSE) | $\tfrac{1}{N} \sum (y_i - \hat{y}_i)^2$ | What OLS minimises (up to scale) |
| Root Mean Squared Error (RMSE) | $\sqrt{\text{MSE}}$ | Same units as $y$; interpretable |
| Mean Absolute Error (MAE) | $\tfrac{1}{N} \sum |y_i - \hat{y}_i|$ | Robust to outliers |
| Coefficient of Determination ($R^2$) | $1 - \tfrac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Fraction of variance explained; 1 = perfect |

$R^2 = 1$ means the model explains all variance in $y$; $R^2 = 0$ means it does no better than the constant predictor $\bar{y}$; $R^2 < 0$ means it's worse than the mean baseline.

## What Could Go Wrong

- **Ill-conditioned $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$.** Collinear features or polynomial bases on a narrow range produce nearly singular matrices. Inversion blows up; small data changes flip the weights wildly. Mitigations: drop redundant features, standardise inputs, add ridge regularisation.
- **More parameters than examples ($M > N$).** $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ is rank-deficient; OLS has no unique solution. Use regularisation or reduce $M$.
- **Outliers.** Squared loss penalises large residuals quadratically — a single mislabelled point can dominate the fit. Robust alternatives: MAE, Huber loss, RANSAC.
- **Wrong polynomial degree.** Too low → underfitting (linear fit on a quadratic relationship); too high → overfitting (degree-6 polynomial on 10 points oscillates wildly). Validation picks the sweet spot.
- **Wrong noise model.** If true noise is heteroscedastic (variance depends on $\mathbf{x}$) or heavy-tailed, the MLE-equivalence argument no longer applies. OLS is still computable; it's just no longer the optimal estimator.

---

## Concepts Introduced This Week

- [[linear-regression]] — the model: $\hat{y} = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x})$, linear in parameters but not necessarily in inputs.
- [[ordinary-least-squares]] — the fitting criterion: minimise $\sum (y_i - \hat{y}_i)^2$, closed-form via the normal equation.
- [[design-matrix]] — the $N \times (M+1)$ matrix $\boldsymbol{\Phi}$ that encodes "all training inputs through all basis functions."
- [[gaussian-distribution]] — univariate and multivariate normal distributions; the noise model that justifies squared error.
- [[bayes-law]] — posterior ∝ likelihood × prior; the foundation for upcoming Bayesian regression.

## Connections

- **Pivots from** [[week-01]]–[[week-06]]: the same supervised-learning framework, but with a continuous output instead of class labels.
- **Mirrors** [[logistic-regression]] in week 2: same MLE recipe, different noise model. Bernoulli noise → cross-entropy loss; Gaussian noise → squared loss.
- **Reuses** the [[non-linear-transformation|basis-expansion]] trick from week 3: the model is linear in $\boldsymbol{\phi}$-space, non-linear in $\mathbf{x}$-space.
- **Sets up** later weeks: regularisation (ridge, lasso) as Bayesian priors; Bayesian linear regression with closed-form posteriors; generalisation theory for choosing model complexity; cross-validation for picking hyperparameters like polynomial degree.

## Open Questions

- How do we pick the polynomial degree (or RBF width / kernel hyperparameters) without cheating on the test set? **Validation** — covered next.
- What if $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ is singular or ill-conditioned? **Regularisation** (ridge regression) — and its Bayesian interpretation as a Gaussian prior.
- Can we *sample* from a posterior over $\mathbf{w}$ and quantify uncertainty in our predictions, not just point-estimate? **Bayesian linear regression** — posterior over weights is closed-form Gaussian under conjugate priors.
- What if the noise *isn't* Gaussian? Robust regression (MAE, Huber) for heavy tails; weighted regression for heteroscedasticity.
