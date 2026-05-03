---
type: week
week: 3
title: "Beyond the Line: Basis Expansion and Maximum Margin"
dates: 2025-10-13 to 2025-10-19
sources:
  - raw/week-03/Wk_3_Lec_1-1.pdf
  - raw/week-03/Wk_3_Lec_2-1.pdf
  - raw/week-03/Monday_ October 13_ 2025 at 3_59_13 PM_Captions_English (United States).txt
  - raw/week-03/Tuesday_ October 14_ 2025 at 10_05_17 AM_Captions_English (United States).txt
  - raw/week-03/Tuesday_ October 14_ 2025 at 10_48_34 AM_Captions_English (United States).txt
  - raw/week-03/Tutorial_Wk_3.pdf
  - raw/week-03/3a-nonlinear-transformations-exercises.pdf
  - raw/week-03/3a-nonlinear-transformations-answers.pdf
  - raw/week-03/3b-svm-primal-exercises.pdf
  - raw/week-03/3b-svm-primal-answers.pdf
concepts:
  - "[[non-linear-transformation]]"
  - "[[support-vector-machine]]"
  - "[[margin]]"
  - "[[quadratic-programming]]"
status: draft
updated: 2026-04-26
---

> [!question]+ THE CRUX: Logistic regression draws a straight line. What do we do when the truth isn't a straight line — and among all straight lines that *do* work, how do we pick the best one?

*We extend linear models to nonlinear problems via basis expansion: send the inputs through a nonlinear map $\phi$ and let the linear model live in the new space. Then we tackle a separate question — when many hyperplanes separate the data, the best one is the one that sits as far as possible from the closest training points. That insight gives us Support Vector Machines.*

---

We finished week 2 with logistic regression in good working order: a hypothesis set, a cross-entropy loss, and two ways to optimise it (gradient descent and IRLS). But the underlying classifier still draws a straight line — a hyperplane in the original feature space. Two problems with that:

1. **Many problems aren't linearly separable.** Concentric rings, points arranged like an XOR, polynomially-curved boundaries — a single hyperplane can't capture them.
2. **Even when the data *is* linearly separable, there are infinitely many separating hyperplanes**, and our existing methods give no principled way to choose between them.

This week tackles each problem in turn. They turn out to be loosely related: both fixes preserve the linear *machinery* but reinterpret what "linear" applies to.

## Part 1: Basis Expansion — Linear Models in Disguise

Suppose your data has a quadratic boundary in 1-D: blue on the outside, orange in the middle on the $x_1$ axis. No linear function $w_0 + w_1 x_1$ can separate them — a line on $\mathbb{R}$ has at most one zero crossing.

But consider the map $\phi(x_1) = (1, x_1, x_1^2)^\top$. Each point is sent into 3-D, and a *linear* function in the new space, $w_0 + w_1 x_1 + w_2 x_1^2$, is exactly the quadratic we wanted. The trick: by climbing into a higher-dimensional space, we make a non-linearly separable problem linearly separable. The original logistic regression model, run on $\phi(\mathbf{x})$ instead of $\mathbf{x}$, now produces a curved decision boundary in the original input space.

This is the [[non-linear-transformation|non-linear transformation]] (or *basis expansion*) idea. Substitute $\mathbf{x} \to \phi(\mathbf{x})$ everywhere it appears:

$$\text{logit}(p_1) = \mathbf{w}^\top \phi(\mathbf{x}) \qquad p_1 = \sigma(\mathbf{w}^\top \phi(\mathbf{x}))$$

The cross-entropy loss, gradient, Hessian, and update rules are identical — just with $\phi(\mathbf{x}^{(i)})$ in place of $\mathbf{x}^{(i)}$. From the optimisation algorithm's point of view, the data has been pre-processed; everything else carries on unchanged.

> [!info] ASIDE — "Linear" in what?
> When we say logistic regression is a "linear" model, we mean linear *in its parameters* $\mathbf{w}$, not linear in the inputs. After basis expansion, the model is still linear in $\mathbf{w}$ — just nonlinear in $\mathbf{x}$. This is what makes it tractable: the optimisation problem doesn't change shape (the cross-entropy loss is still strictly convex in $\mathbf{w}$), only the input vectors do.

### Building the Right $\phi$

For polynomial decision boundaries of degree $\leq p$, include all monomials of degree up to $p$. With one input variable and degree 2:

$$\mathbf{x} = (1, x_1)^\top \;\to\; \phi(\mathbf{x}) = (1, x_1, x_1^2)^\top$$

With two inputs and degree 2 you also need the cross term $x_1 x_2$:

$$\mathbf{x} = (1, x_1, x_2)^\top \;\to\; \phi(\mathbf{x}) = (1, x_1, x_2, x_1^2, x_2^2, x_1 x_2)^\top$$

For degree 3 the list grows further. The number of monomials of degree $\leq p$ in $d$ variables grows roughly as $d^p$ — quickly painful for high-dimensional inputs.

You're not limited to polynomials. Any non-linear map is fair game: $\phi(\mathbf{x}) = (1, x_1, e^{x_1})^\top$ if you suspect exponential structure, sinusoidal bases for periodic data, indicator functions for discrete cuts, and so on. The choice encodes a guess about the shape of the truth.

### What Could Go Wrong

Two caveats sit on the other side of the trick:

- **High dimension.** Polynomial expansion explodes the number of features. With 100 inputs and degree-3 expansion, $\phi(\mathbf{x})$ has tens of thousands of components. Storage, computation, and (especially) sample complexity all suffer.
- **Overfitting.** A high-degree polynomial can fit any training set perfectly while wiggling wildly between points. Fitting the training data is no guarantee of [[generalization|generalising]] — and a 4th-order polynomial may generalise *worse* than a linear fit despite explaining the training data more accurately. This problem is the central concern of weeks 8–10.

> [!question]- A 4th-order polynomial perfectly classifies all training data. A linear classifier misclassifies a few. Which is the better choice, and why isn't training accuracy the deciding factor?
> Often the linear classifier. Training accuracy is *not* what we ultimately care about — we care about [[generalization|generalisation]] to unseen examples. A 4th-order polynomial has enough capacity to memorise the training noise, drawing wildly contorted boundaries to capture every point. Those contortions are unlikely to reflect the true underlying structure, so the model performs worse on new data than a simpler classifier that "lets" a few outliers be misclassified. Capacity must be matched to evidence — a principle formalised by VC dimension and bias-variance analysis later in the module.

## Part 2: Among Many Separators, Which is Best?

Now switch problem. Suppose the data *is* linearly separable. Logistic regression and several other algorithms can find *some* separating hyperplane, but they don't all find the *same* one. Two hyperplanes might both achieve zero training error while sitting in very different positions. Which is preferable?

Geometric intuition: a separator that just barely splits the classes — passing close to a training point — is fragile. A small perturbation in that point (or any new test point in its neighbourhood) could land on the wrong side. A separator that runs through the *middle* of the no-man's-land between classes, leaving wide buffer zones on both sides, is more robust.

Formalise the buffer as the [[margin]]: the perpendicular distance from the decision boundary to the closest training example.

### The Maximum Margin Idea

Among all hyperplanes that correctly separate the data, choose the one that **maximises the margin**. The closest training points — the ones that touch the maximum-margin boundary's "envelope" — are the [[support-vector-machine|support vectors]]. They alone determine the boundary; every other point could be removed without changing the answer. This is what defines a [[support-vector-machine|Support Vector Machine]] (SVM).

> [!tip] TIP — Why support vectors are special
> If you took an SVM, deleted every training point *except* the support vectors, and re-trained, you'd get exactly the same hyperplane. The non-support-vectors are slack — they sit safely on their side of the boundary and contribute nothing to the optimum. Compare with logistic regression, where every training point contributes to the gradient at every iteration. SVMs achieve a kind of "data efficiency" by structurally identifying which examples actually matter for the decision boundary.

### Setting Up the Optimisation

For a hyperplane $h(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$, the perpendicular distance from a point $\mathbf{x}^{(n)}$ to the hyperplane is:

$$\text{dist}(h, \mathbf{x}^{(n)}) = \frac{|h(\mathbf{x}^{(n)})|}{\|\mathbf{w}\|}$$

The denominator $\|\mathbf{w}\|$ is the Euclidean norm — this normalises away the fact that scaling $\mathbf{w}$ and $b$ by any constant $\kappa$ doesn't change the geometric hyperplane.

The optimisation problem is then:

$$\arg\max_{\mathbf{w}, b} \left\{ \min_n \frac{|h(\mathbf{x}^{(n)})|}{\|\mathbf{w}\|} \right\} \quad \text{subject to } y^{(n)} h(\mathbf{x}^{(n)}) > 0 \text{ for all } n$$

Reading the formula: of all $(\mathbf{w}, b)$ that classify everything correctly, find the pair where the *closest* training point is as far from the boundary as possible.

The constraint $y^{(n)} h(\mathbf{x}^{(n)}) > 0$ uses labels $y \in \{+1, -1\}$ — different from logistic regression's $\{0, 1\}$ convention. The product is positive when the prediction agrees with the label and negative when it disagrees. Note that $y^{(n)} h(\mathbf{x}^{(n)}) = |h(\mathbf{x}^{(n)})|$ once the constraint is satisfied, so the absolute value can be dropped:

$$\arg\max_{\mathbf{w}, b} \left\{ \min_n \frac{y^{(n)} h(\mathbf{x}^{(n)})}{\|\mathbf{w}\|} \right\}$$

### The Canonical Form

The argmax-min nested optimisation looks intimidating. There's a clever rescaling that flattens it.

Notice that scaling $(\mathbf{w}, b) \to (\kappa \mathbf{w}, \kappa b)$ leaves the hyperplane geometry unchanged: same boundary, same distances. So we can fix the scale by demanding that for the closest training point, $y^{(n)} h(\mathbf{x}^{(n)}) = 1$. This is the **canonical representation** of the hyperplane.

Under this normalisation the constraint becomes $y^{(n)} h(\mathbf{x}^{(n)}) \geq 1$ for all $n$ (with equality at the closest point), and the inner min in the objective becomes $1$. The whole problem collapses to:

$$\arg\max_{\mathbf{w}, b} \frac{1}{\|\mathbf{w}\|} \quad \text{subject to } y^{(n)} h(\mathbf{x}^{(n)}) \geq 1 \text{ for all } n$$

Maximising $1/\|\mathbf{w}\|$ is equivalent to minimising $\|\mathbf{w}\|$, which is equivalent to minimising $\tfrac{1}{2}\|\mathbf{w}\|^2$ (the half is decorative — its derivative cancels the 2 from differentiating the square; the squared norm is preferred because it's smooth and convex). The final clean form:

$$\boxed{\arg\min_{\mathbf{w}, b} \tfrac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to } y^{(n)} (\mathbf{w}^\top \mathbf{x}^{(n)} + b) \geq 1 \;\; \forall n}$$

This is a [[quadratic-programming|quadratic program]]: a convex quadratic objective with linear inequality constraints. Off-the-shelf QP solvers handle it efficiently, and convexity guarantees a unique global optimum.

> [!question]- Why does maximising margin reduce to *minimising* $\|\mathbf{w}\|$? It seems backwards — bigger weights should give a bigger margin.
> The intuition reverses once you account for normalisation. After canonical rescaling, the closest training point has $y^{(n)} h(\mathbf{x}^{(n)}) = 1$, which means its perpendicular distance to the hyperplane is $1 / \|\mathbf{w}\|$. To make that distance large, $\|\mathbf{w}\|$ must be small. The weights determine the *steepness* of the linear function, not the geometric position of the hyperplane — a steeper function can only achieve $h = \pm 1$ if you're closer to the boundary, so a steep $\mathbf{w}$ corresponds to a *small* margin.

### Combining the Two Ideas

Nothing in the SVM derivation requires that the data be linearly separable in the original space. Substitute $\mathbf{x} \to \phi(\mathbf{x})$ via a [[non-linear-transformation|basis expansion]] and you get an SVM whose decision boundary is non-linear in the original space:

$$\arg\min_{\mathbf{w}, b} \tfrac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to } y^{(n)} (\mathbf{w}^\top \phi(\mathbf{x}^{(n)}) + b) \geq 1 \;\; \forall n$$

The classical SVM example — concentric rings of two classes, impossible to separate linearly — becomes trivially separable in $\phi(\mathbf{x}) = (\|\mathbf{x}\|^2, x_1, x_2, \ldots)$ space. The maximum-margin separator in the lifted space corresponds to a circular (or polynomial) decision boundary in the original space. The support vectors are the points of each class closest to the boundary — usually a handful of points, even when the dataset is large.

Combining SVMs with high-dimensional $\phi$ raises the same dimensionality concern as before, plus a more practical worry: does the QP scale? In week 4 we'll see that there's a *dual* formulation in which the optimisation depends on $\phi$ only through inner products $\phi(\mathbf{x}^{(i)})^\top \phi(\mathbf{x}^{(j)})$ — which we can compute via *kernel functions* without ever explicitly forming $\phi$. That's the **kernel trick**, and it's what makes SVMs practical in very high dimensions.

---

## Concepts Introduced This Week

- [[non-linear-transformation]] — basis expansion $\phi(\mathbf{x})$; lifts data into a higher-dimensional space where linear methods can produce non-linear decision boundaries in the original space.
- [[support-vector-machine]] — maximum-margin linear classifier; the hyperplane is determined entirely by the closest training points (the support vectors).
- [[margin]] — perpendicular distance from the decision boundary to the closest training example; the quantity SVMs maximise.
- [[quadratic-programming]] — a convex optimisation problem class with quadratic objective and linear constraints; the canonical SVM problem reduces to one.

## Connections

- **Builds on** [[week-02]]: basis expansion plugs straight into the same gradient-descent and IRLS machinery; the loss landscape stays convex in $\mathbf{w}$.
- **Builds on** [[week-01]]: directly answers the open question "what if the data isn't linearly separable?"
- **Sets up** later weeks: the dual SVM formulation and kernels (week 4); soft-margin SVMs for non-separable data; the bias-variance / VC-dimension framing of overfitting risk for high-dimensional $\phi$ (weeks 8–10).

## Open Questions

- High-dimensional $\phi$ is expensive — can we avoid computing it explicitly? (Answered next week: the kernel trick uses inner products only.)
- What if the data isn't linearly separable even after basis expansion? (Soft-margin SVMs, with slack variables.)
- How do we choose the right $\phi$ — degree, basis family, dimension? (Cross-validation, regularisation, model selection — later in the module.)
