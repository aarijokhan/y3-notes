---
type: concept
name: learning curve
description: A plot of expected in-sample and out-of-sample error against training-set size N. Shape distinguishes simple from complex models — and reveals overfitting, underfitting, and the asymptotic noise floor.
sources:
  - raw/week-09/Wk_9_Lec_2-1.pdf
  - raw/week-09/Tuesday_ November 25_ 2025 at 9_42_04 AM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*A **learning curve** plots $\mathbb{E}_{\mathcal{D}}[E_{\text{in}}(g^{(\mathcal{D})})]$ and $\mathbb{E}_{\mathcal{D}}[E_{\text{out}}(g^{(\mathcal{D})})]$ as functions of training-set size $N$. The two curves converge as $N \to \infty$ — to the noise floor for the right model class. The shape (gap and slope) reveals whether a model is underfitting, overfitting, or hitting irreducible error.*

## The Two Curves

For a fixed hypothesis set $\mathcal{H}$ and learning algorithm $\mathcal{A}$:

- $\mathbb{E}_{\mathcal{D}}[E_{\text{in}}(g^{(\mathcal{D})})]$ — expected training error, averaging over training sets $\mathcal{D}$ of size $N$.
- $\mathbb{E}_{\mathcal{D}}[E_{\text{out}}(g^{(\mathcal{D})})]$ — expected test error of the same trained hypothesis on new data.

Plot both as functions of $N$. Universal qualitative behaviour:

- $E_{\text{in}}$ **increases** with $N$. With few points the model interpolates; adding points forces compromises.
- $E_{\text{out}}$ **decreases** with $N$. More data means a more representative training set and a hypothesis that generalises better.
- Both curves converge to the same asymptote — the **best achievable error** within $\mathcal{H}$, which equals $\sigma^2$ (irreducible noise) for a well-specified model.

## Simple vs Complex Model

The shape of the curves changes qualitatively with model complexity:

**Simple model (low VC dimension).**

- Curves come together quickly — small gap even for small $N$.
- Both stabilise high above zero error: the model is structurally limited (high bias).
- Asymptote: $\sigma^2 + \text{bias}^2$, with the bias contribution often dominant.

**Complex model (high VC dimension).**

- Curves are far apart for small $N$ — large generalisation gap. With $N$ less than $d_{\text{VC}}$, $E_{\text{in}}$ can be near zero (model interpolates) while $E_{\text{out}}$ is huge.
- Need $N \gg d_{\text{VC}}$ before they meet.
- Asymptote: lower than the simple model — closer to $\sigma^2$ — because expressive enough to fit $f$.

This visualises the regime distinction:

- **Underfitting region**: simple model, both curves near asymptote, asymptote is high.
- **Overfitting region**: complex model with $N$ too small, large gap.
- **Sweet spot**: complex model with $N$ large enough that the gap has closed.

## Linear Regression (Closed Form)

For linear regression with a noisy target $y = \mathbf{w}^{*\top} \mathbf{x} + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$, with $d + 1$ parameters and $N \geq d + 1$, the learning curve has an exact form:

$$\mathbb{E}_{\mathcal{D}}[E_{\text{in}}] = \sigma^2 \left(1 - \frac{d+1}{N}\right), \qquad \mathbb{E}_{\mathcal{D}}[E_{\text{out}}] = \sigma^2 \left(1 + \frac{d+1}{N}\right).$$

Reading off:

- $E_{\text{in}}$ starts at $0$ when $N = d + 1$ (perfect fit) and rises towards $\sigma^2$.
- $E_{\text{out}}$ starts above $\sigma^2$ and decays towards $\sigma^2$.
- **Generalisation gap**: $\mathbb{E}_{\mathcal{D}}[E_{\text{out}}] - \mathbb{E}_{\mathcal{D}}[E_{\text{in}}] = 2\sigma^2 \frac{d+1}{N}$, decaying as $1/N$.

Each parameter costs you a $\sigma^2 / N$ unit of generalisation gap. With $N \gg d + 1$, the gap is negligible and both errors sit at $\sigma^2$.

## VC vs Bias–Variance View

Two equivalent ways to draw the same curves:

**VC view.** Shade the region between the curves as the **generalisation error** — what the VC bound controls. The bound $E_{\text{out}} \leq E_{\text{in}} + \Omega$ is a uniform statement about how far the curves can be apart.

**Bias–variance view.** Draw a horizontal line at $\text{bias}$ separating the area under $E_{\text{in}}$ (= bias) from the area between the curves (= variance, plus noise asymptote). Now the picture decomposes the error into the structural minimum (bias) and the data-dependent fluctuation (variance).

Both pictures live on top of the same two curves — they're complementary lenses, not competing analyses.

## Diagnostic Use

Plotting empirical learning curves (using held-out validation as a proxy for $E_{\text{out}}$) is one of the most useful debugging tools in ML:

- **Both curves high, close together**: high bias. Try a more expressive model.
- **Training curve low, validation curve high**: high variance. Get more data, regularise, or simplify.
- **Both flatten out at the same value**: hitting the noise floor — diminishing returns from more data.
- **Curves still descending at the right edge**: more data would help.

This is also how people decide whether to invest in collecting more data: if the validation curve has plateaued, more data won't help; if it's still falling, it might.

## Related

- [[bias-variance-decomposition]] — the conceptual framework for what learning curves show.
- [[generalization-bound]] — the worst-case bound on the gap between curves.
- [[vc-dimension]] — controls the size of the gap.
- [[ridge-regression]] — regularisation flattens the variance contribution, narrowing the gap.

## Active Recall

> [!question]- For linear regression with $d + 1$ parameters and noisy target $y = \mathbf{w}^{*\top} \mathbf{x} + \epsilon$ ($\epsilon \sim \mathcal{N}(0, \sigma^2)$), the expected in-sample error is $\sigma^2(1 - (d+1)/N)$. Why does it start at zero when $N = d + 1$ and rise rather than fall as $N$ increases?
> When $N = d + 1$, the model has exactly enough parameters to interpolate $N$ points — $E_{\text{in}} = 0$. As $N$ grows, the model can no longer interpolate every point, so the residuals are no longer zero on average. The factor $(1 - (d+1)/N)$ captures the fraction of "noise variance left over" after the OLS fit absorbs $d + 1$ degrees of freedom. As $N \to \infty$, $E_{\text{in}} \to \sigma^2$ — the irreducible noise floor.

> [!question]- Why are the in-sample and out-of-sample learning curves of a complex model far apart for small $N$, but close together for large $N$?
> A complex model has many parameters; with few training points, it can fit the training data essentially perfectly ($E_{\text{in}}$ near zero) by aligning to noise. The same model on new data would produce predictions inconsistent with the noisy fit, so $E_{\text{out}}$ is large. As $N$ grows past the model's effective capacity ($N \gg d_{\text{VC}}$), the model can no longer interpolate, $E_{\text{in}}$ rises towards the noise floor, $E_{\text{out}}$ falls towards the same floor, and the gap closes. The generalisation gap formula $2\sigma^2 (d+1)/N$ for linear regression is a clean instance: gap $\propto 1/N$.

> [!question]- You plot empirical learning curves and find that both training and validation error are about 0.30 and very close together, even after $N = 10{,}000$. Diagnose the issue and suggest a remedy.
> Both errors high and close together is the classical *high-bias* (underfitting) signature. The model class isn't expressive enough to capture the structure of $f$ — the average hypothesis $\bar{g}$ is far from the truth. Adding more data won't help (the curves have already converged). Remedies: increase model capacity (more parameters, deeper network, higher-degree polynomial, richer kernel), reduce regularisation, or add useful features. If after these the error is still 0.30, you may be hitting the irreducible noise floor — though 0.30 is high enough that you should suspect bias first.
