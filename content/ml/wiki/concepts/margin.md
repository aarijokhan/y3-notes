---
type: concept
sources:
  - raw/week-03/Wk_3_Lec_1-1.pdf
  - raw/week-03/Wk_3_Lec_2-1.pdf
  - raw/week-03/Tuesday_ October 14_ 2025 at 10_05_17 AM_Captions_English (United States).txt
status: draft
updated: 2026-04-26
---

*The perpendicular distance from a linear decision boundary to the closest training example. Maximising it is the defining principle of [[support-vector-machine|Support Vector Machines]].*

## Definition

For a linear classifier $h(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ and training set $\mathcal{T}$, the **margin** $\gamma$ is:

$$\gamma = \min_{n} \frac{|h(\mathbf{x}^{(n)})|}{\|\mathbf{w}\|}$$

In words: of all the training points, how far is the *closest one* from the boundary $h = 0$? That distance is the margin.

The factor $1/\|\mathbf{w}\|$ converts the algebraic value $|h(\mathbf{x})|$ into a geometric distance. The same hyperplane can be described by infinitely many $(\mathbf{w}, b)$ pairs related by a positive scalar; the norm in the denominator removes that ambiguity.

## Distance From a Point to a Hyperplane

The perpendicular distance from any point $\mathbf{x}^{(n)}$ to the hyperplane $\mathbf{w}^\top \mathbf{x} + b = 0$ is:

$$\text{dist}(h, \mathbf{x}^{(n)}) = \frac{|\mathbf{w}^\top \mathbf{x}^{(n)} + b|}{\|\mathbf{w}\|}$$

**Why $\|\mathbf{w}\|$ in the denominator?** $\mathbf{w}$ is normal to the hyperplane (it points in the direction of steepest increase of $h$). The unit normal is $\mathbf{w}/\|\mathbf{w}\|$. The signed distance from any point to the hyperplane along the normal direction is $h(\mathbf{x})/\|\mathbf{w}\|$; the absolute value drops the sign.

![[margin-distance-to-hyperplane.png]]

## Why Maximise the Margin?

Suppose two hyperplanes both achieve zero training error. The one running through the *middle* of the gap between classes is more robust than one that just barely separates them:

- **Robustness to noise.** If a training point's position is uncertain (measurement noise, label noise), a wide margin makes it less likely that the point — or any new point in its neighbourhood — ends up on the wrong side of the boundary.
- **Generalisation.** Loosely, a wide margin reflects strong evidence that the classes really are separated by this gap. There is a formal version of this intuition in VC-theory and PAC-Bayes bounds: roughly, classifiers with larger margins on the training data have provably tighter generalisation guarantees.

The classifier that maximises the margin among all hyperplanes correctly classifying the training data is the **maximum-margin classifier**, also known as the (hard-margin) [[support-vector-machine|Support Vector Machine]].

## Support Vectors

The training points that *attain* the minimum in $\gamma = \min_n \frac{|h(\mathbf{x}^{(n)})|}{\|\mathbf{w}\|}$ — the ones sitting on the margin's edge — are the **support vectors**. They alone determine the optimal hyperplane: removing any non-support-vector and re-fitting gives the same boundary.

This is striking. For a typical SVM trained on thousands of points, only a handful end up as support vectors. The rest sit comfortably on their side of the boundary and contribute nothing to the optimum. SVMs achieve a kind of structural data efficiency: they identify, automatically, *which* training examples actually pin the decision down.

## The Optimisation

The natural formulation is a nested max-min:

$$\arg\max_{\mathbf{w}, b} \left\{ \min_n \frac{y^{(n)} h(\mathbf{x}^{(n)})}{\|\mathbf{w}\|} \right\} \quad \text{subject to } y^{(n)} h(\mathbf{x}^{(n)}) > 0 \text{ for all } n$$

(Here $y^{(n)} \in \{-1, +1\}$ and $y^{(n)} h(\mathbf{x}^{(n)}) > 0$ means correct classification, so we can drop the absolute value.)

This nested form is awkward to solve directly, but it admits a clever **canonical rescaling**: since $(\mathbf{w}, b)$ and $(\kappa \mathbf{w}, \kappa b)$ describe the same hyperplane, we can set the scale so that the closest training point has $y^{(n)} h(\mathbf{x}^{(n)}) = 1$. Under this rescaling the problem becomes:

$$\arg\min_{\mathbf{w}, b} \tfrac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to } y^{(n)} h(\mathbf{x}^{(n)}) \geq 1 \;\; \forall n$$

The width of the resulting margin is $1/\|\mathbf{w}\|$ — so minimising $\|\mathbf{w}\|$ maximises the margin. See [[support-vector-machine]] for the full derivation.

![[margin-max-margin-classifier.png]]

## A Conventional Subtlety

The "margin" is sometimes defined as the *width of the corridor* — the distance between the two parallel hyperplanes $h = +1$ and $h = -1$ that touch the support vectors on each side. That width is $2/\|\mathbf{w}\|$. This module uses "margin" to mean the per-side perpendicular distance $1/\|\mathbf{w}\|$ — so the corridor width is twice the margin. Either convention is fine as long as you're consistent.

## Related

- [[support-vector-machine]] — the algorithm that finds the maximum-margin hyperplane
- [[decision-boundary-ml|decision boundary]] — the hyperplane $h = 0$ whose margin we measure
- [[generalization]] — wider margins correlate with better generalisation
- [[quadratic-programming]] — the convex optimisation form the maximum-margin problem reduces to

## Active Recall

> [!question]- Compute the perpendicular distance from the point $\mathbf{x} = (3, 1)^\top$ to the hyperplane $h(\mathbf{x}) = 2 x_1 + x_2 - 5 = 0$.
> $h(\mathbf{x}) = 2(3) + 1 - 5 = 2$. $\|\mathbf{w}\| = \sqrt{2^2 + 1^2} = \sqrt{5}$. Distance $= |h(\mathbf{x})| / \|\mathbf{w}\| = 2/\sqrt{5} \approx 0.894$.

> [!question]- A maximum-margin classifier has been trained on 1000 points, of which only 5 are support vectors. If we delete one of the 995 non-support-vector points and retrain, what happens to the decision boundary, and why?
> Nothing changes — the boundary is identical. Non-support-vectors satisfy $y^{(n)} h(\mathbf{x}^{(n)}) > 1$ strictly (they sit beyond the margin's envelope), so they don't enter the active constraint set of the optimisation. Removing them changes neither the objective $\tfrac{1}{2}\|\mathbf{w}\|^2$ nor the binding constraints, so the optimum $(\mathbf{w}^*, b^*)$ is unchanged. This is why support vectors are the "critical" elements of the training set — they're the only ones the boundary actually depends on.

> [!question]- Why does the formula for distance from a point to a hyperplane $\mathbf{w}^\top \mathbf{x} + b = 0$ have $\|\mathbf{w}\|$ in the denominator?
> $\mathbf{w}$ is normal to the hyperplane. The unit normal is $\mathbf{w} / \|\mathbf{w}\|$. The signed distance from a point $\mathbf{x}^{(n)}$ to the hyperplane along the normal direction is the projection of $\mathbf{x}^{(n)} - \mathbf{x}_0$ (any point on the plane) onto the unit normal, which works out to $h(\mathbf{x}^{(n)}) / \|\mathbf{w}\|$. The denominator normalises away the fact that the same hyperplane can be represented by infinitely many algebraically distinct $(\mathbf{w}, b)$ pairs that differ by a scalar.
