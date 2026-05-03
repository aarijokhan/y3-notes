---
type: concept
sources:
  - raw/week-03/Wk_3_Lec_1-1.pdf
  - raw/week-03/Monday_ October 13_ 2025 at 3_59_13 PM_Captions_English (United States).txt
status: draft
updated: 2026-04-26
---

*A pre-processing map $\phi: \mathbb{R}^d \to \mathbb{R}^D$ (often with $D > d$) that turns the inputs into a higher-dimensional feature space. A linear model run on $\phi(\mathbf{x})$ can produce a non-linear decision boundary in the original space, while keeping its optimisation machinery and convexity properties unchanged.*

## The Trick

Many classification problems aren't linearly separable. A single hyperplane in the original input space cannot capture circles, XOR-like patterns, or polynomial boundaries.

The fix is to send each input $\mathbf{x}$ through a non-linear map $\phi$ — also called a **basis expansion** or **feature transform** — and apply a linear model in the resulting space:

$$\mathbf{x} \in \mathbb{R}^d \;\longrightarrow\; \phi(\mathbf{x}) \in \mathbb{R}^D$$

In the new space, the model is *still* a linear function:

$$h(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x})$$

But $h$, viewed as a function of the original $\mathbf{x}$, is non-linear. A line in $\phi$-space can correspond to a curve, a circle, a polynomial, anything determined by $\phi$.

The key reframing: **what looks like a linear model in $\phi$-space is a highly non-linear model when viewed back in the original input space.** The non-linearity is hidden inside the pre-processing step $\phi$, leaving the model itself — and all its convexity-based optimisation guarantees — perfectly linear.

### Toy Example

Data on $\mathbb{R}$: blue points at $x_1 \in \{-3, -2, 4, 5\}$, orange at $x_1 \in \{0, 1\}$. No linear function $w_0 + w_1 x_1$ can separate them — a line on $\mathbb{R}$ has at most one zero crossing.

![[image 4.png]]

Apply $\phi(x_1) = (1, x_1, x_1^2)^\top$ and the points become $(1, -3, 9), (1, -2, 4), (1, 0, 0), (1, 1, 1), (1, 4, 16), (1, 5, 25)$ in $\mathbb{R}^3$.

![[JPEG_image-4DB9-BEF3-1D-0.jpeg]] A linear function $w_0 + w_1 x_1 + w_2 x_1^2$ can now separate them — for example, $w_0 = 10, w_1 = -7, w_2 = 1$ gives $h(x_1) = x_1^2 - 7 x_1 + 10 = (x_1 - 2)(x_1 - 5)$, which is positive outside $[2, 5]$ and negative inside. The decision boundary in the original 1-D space is the *pair of points* $x_1 = 2$ and $x_1 = 5$ — a non-linear "boundary" produced by a linear model in lifted space.

## Polynomial Basis Expansion

For polynomial decision boundaries of degree $\leq p$, include all monomials of total degree up to $p$:

| Inputs     | Degree | $\phi(\mathbf{x})$                                                         |
| ---------- | ------ | -------------------------------------------------------------------------- |
| $x_1$      | 2      | $(1, x_1, x_1^2)$                                                          |
| $x_1, x_2$ | 2      | $(1, x_1, x_2, x_1^2, x_2^2, x_1 x_2)$                                     |
| $x_1, x_2$ | 3      | $(1, x_1, x_2, x_1^2, x_2^2, x_1 x_2, x_1^3, x_2^3, x_1 x_2^2, x_1^2 x_2)$ |

Cross terms like $x_1 x_2$ are essential for capturing interactions — without them you can only express decision boundaries that are sums of single-variable functions.

The number of monomials of degree $\leq p$ in $d$ variables is $\binom{d + p}{p}$, growing roughly as $d^p$ for fixed $p$. With 100 features and $p = 3$, $\phi(\mathbf{x})$ has more than 170,000 components.

## Other Bases

You're not limited to polynomials. Useful non-polynomial expansions include:

- **Exponential / logarithmic**: $\phi(\mathbf{x}) = (1, x_1, e^{x_1}, \ln(1 + |x_1|))$ for data with multiplicative structure.
- **Sinusoidal**: $\phi(\mathbf{x}) = (1, x_1, \cos(x_1), \sin(x_1))$ for periodic data; the basis of Fourier methods.
- **Indicator functions**: $\phi(\mathbf{x})$ has one component per region of input space; "one-hot" features for discrete cuts.
- **Radial basis functions**: $\phi_i(\mathbf{x}) = \exp(-\|\mathbf{x} - \mathbf{c}_i\|^2 / 2\sigma^2)$ centred on each training point or a chosen set of centres.

The choice encodes a guess about the shape of the truth. Polynomial expansions are popular because they're simple and connect cleanly with [[taylor-polynomial|Taylor approximation]] intuition: any sufficiently smooth function looks polynomial locally.

## Linearity Is Preserved Where It Matters

Think of $\phi$ as data pre-processing: from the optimisation algorithm's point of view, the inputs are $\phi(\mathbf{x}^{(i)})$ rather than $\mathbf{x}^{(i)}$, and that is the *only* change. Concretely, for [[logistic-regression]] with [[cross-entropy-loss]]:

$$E(\mathbf{w}) = -\sum_{i=1}^{N} \left[ y^{(i)} \ln \sigma(\mathbf{w}^\top \phi(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \ln (1 - \sigma(\mathbf{w}^\top \phi(\mathbf{x}^{(i)}))) \right]$$

$$\nabla E(\mathbf{w}) = \sum_{i=1}^{N} (p_1(\phi(\mathbf{x}^{(i)}), \mathbf{w}) - y^{(i)}) \phi(\mathbf{x}^{(i)})$$

The loss is still strictly [[convex-function|convex]] in $\mathbf{w}$. [[gradient-descent-ml|gradient descent]] and [[newton-raphson-method|IRLS]] still find the global optimum. *Nothing in the optimiser changes.*

> [!info] ASIDE — "Linear in $\mathbf{w}$" vs "linear in $\mathbf{x}$"
> When we call logistic regression a "linear model," we mean linear in the **parameters** $\mathbf{w}$, not linear in the inputs. After basis expansion, the model is still linear in $\mathbf{w}$ — only nonlinear in $\mathbf{x}$. This is the property that preserves the convex optimisation. If we had instead made the model non-linear in $\mathbf{w}$ (as in a neural network with multiple weight layers), we would lose convexity entirely.

## Caveats

Two costs to weigh against the gained expressiveness:

- **Dimensionality**: $\phi(\mathbf{x})$ may have far more components than $\mathbf{x}$, increasing memory, computation, and (especially) sample complexity. The number of training examples needed to fit a model reliably scales with the feature dimension.
- **Overfitting**: more flexible models can fit training noise. A degree-4 polynomial classifier can perfectly classify any reasonable training set while generalising worse than a linear classifier with a few mistakes. Capacity must be justified by evidence — the central concern of [[generalization|generalisation]] theory.

## The Kernel Trick (Preview)

For algorithms whose optimisation depends on inputs only through inner products — notably [[support-vector-machine|SVMs]] in their dual form — there is a way to use $\phi$ without ever computing it explicitly. A *kernel function* $k(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^\top \phi(\mathbf{x}')$ can sometimes be evaluated in $O(d)$ time even when $\phi$ has thousands of components. This makes very high (even infinite) dimensional basis expansions practical. Covered in week 4.

## Related

- [[support-vector-machine]] — combines basis expansion with maximum-margin classification
- [[logistic-regression]] — accepts basis expansion transparently
- [[generalization]] — high-dim $\phi$ raises overfitting risk
- [[decision-boundary-ml|decision boundary]] — what becomes non-linear in the original space

## Active Recall

> [!question]- Why does applying a basis expansion $\phi$ to logistic regression *not* break the convexity of the loss?
> The cross-entropy loss is convex as a function of $\mathbf{w}$, regardless of what the inputs are. After basis expansion, the loss is $E(\mathbf{w}) = -\sum_i [y^{(i)} \ln \sigma(\mathbf{w}^\top \phi(\mathbf{x}^{(i)})) + \ldots]$, which is still convex in $\mathbf{w}$ — the new inputs $\phi(\mathbf{x}^{(i)})$ are constants from the optimiser's perspective. We've changed the inputs, not the function class of the loss in $\mathbf{w}$.

> [!question]- For two input variables $x_1, x_2$, write the basis expansion for polynomial decision boundaries up to degree 2. Why must the cross term $x_1 x_2$ be included?
> $\phi(\mathbf{x}) = (1, x_1, x_2, x_1^2, x_2^2, x_1 x_2)^\top$. The cross term captures *interactions* between features. Without it, the model can only express boundaries that are sums of separate single-variable functions of $x_1$ and $x_2$ — for example, an axis-aligned ellipse $x_1^2 + x_2^2 = c$, but not a rotated one $(x_1 - x_2)^2 = c$, which expands to include $-2 x_1 x_2$. Generally, no rotation-non-invariant quadratic boundary can be expressed without the cross term.

> [!question]- A degree-4 polynomial basis expansion lets a linear classifier perfectly classify all training data, while a linear classifier without expansion misclassifies a few points. Which is likely to generalise better, and why?
> Often the un-expanded linear classifier. Training accuracy is not the goal — generalisation is. The high-degree expansion gives the classifier enough capacity to memorise the training noise, drawing wildly contorted boundaries that fit specific training points but don't reflect the underlying structure. With limited data, the simpler model whose mistakes are evenly distributed (suggesting irreducible noise rather than systematic structure it's missing) often generalises better. Capacity must be matched to the evidence in the data — a tension formalised by VC dimension and bias-variance analysis.
