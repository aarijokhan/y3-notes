---
type: concept
sources:
  - raw/week-01/w01-l02-transcript.txt
  - raw/week-01/w01-l03-transcript.txt
  - raw/week-01/w01-slides.pdf
status: stable
updated: 2026-04-20
---


*The surface in input space that separates one predicted class from another — for a perceptron, this is always a hyperplane.*

## Definition

A **decision boundary** is the set of all points $\mathbf{x}$ where a classifier's output changes from one class to another. For a [[perceptron]], the decision boundary is defined by:

$$\mathbf{w} \cdot \mathbf{x} + b = 0$$

This equation describes a **hyperplane** — a flat surface that divides input space into two half-spaces.

## Hyperplanes across dimensions

| Input dimension $D$ | Decision boundary | Example |
|---|---|---|
| 1 | A single point | $w_1 x_1 + b = 0 \Rightarrow x_1 = -b/w_1$ |
| 2 | A straight line | $w_1 x_1 + w_2 x_2 + b = 0$ |
| 3 | A flat plane | $w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0$ |
| $D$ | A $(D{-}1)$-dimensional hyperplane | $\mathbf{w} \cdot \mathbf{x} + b = 0$ |

The math is identical in every dimension — only the geometric visualisation changes.

## What the parameters control

The weight vector $\mathbf{w}$ and bias $b$ fully determine the boundary:

**Orientation.** The hyperplane is always perpendicular to $\mathbf{w}$. Changing $\mathbf{w}$ tilts or rotates the boundary. In the 2D analogy $y = mx + c$, $\mathbf{w}$ plays the role of the slope $m$.

**Position.** The bias $b$ shifts the hyperplane along the direction of $\mathbf{w}$. Specifically, the boundary sits at a perpendicular distance of $-b/\|\mathbf{w}\|$ from the origin. In the 2D analogy, $b$ plays the role of the intercept $c$.

- $b = 0$: the hyperplane passes through the origin
- $b < 0$: the hyperplane shifts in the direction of $\mathbf{w}$
- $b > 0$: the hyperplane shifts against the direction of $\mathbf{w}$

To verify the sign of $b$ geometrically: check the origin $(0, 0, \dots)$. Evaluating $\mathbf{w} \cdot \mathbf{0} + b = b$. If $b > 0$, the origin is on the $+1$ side; if $b < 0$, the origin is on the $-1$ side.

## Linear separability

A dataset is **linearly separable** if there exists some hyperplane that perfectly separates the two classes — all positive points on one side, all negative points on the other. A single perceptron can only solve linearly separable problems.

Many real-world problems are **not** linearly separable. The classic example is XOR: four points at $(\pm 1, \pm 1)$ where diagonally opposite corners share the same label. No single line can separate them.

To handle non-linearly separable data, we need to:
1. Combine multiple perceptrons into layers (each drawing its own linear boundary)
2. Stack the layers so that a final perceptron classifies based on the intermediate outputs

This gives rise to **multi-layer perceptrons** (MLPs), covered in week 3. The XOR problem, for instance, can be solved with just 3 perceptrons: two in a first layer and one combining their outputs.

## Related

- [[perceptron]] — the model that produces a linear decision boundary
- [[dot-product]] — computes the signed distance from a point to the boundary

## Active Recall

> [!question]- How does the bias $b$ affect the position of the decision boundary, and what happens when $b = 0$?
> The bias shifts the hyperplane along $\mathbf{w}$ by a distance $-b/\|\mathbf{w}\|$ from the origin. When $b = 0$, the hyperplane passes through the origin. A negative $b$ pushes the boundary in the direction of $\mathbf{w}$; a positive $b$ pushes it against $\mathbf{w}$.

> [!question]- A perceptron in 3D input space has its decision boundary described by $w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0$. What is the dimensionality of this boundary, and why?
> The boundary is a 2-dimensional plane embedded in 3D space. In general, a hyperplane in $D$-dimensional space has $D-1$ dimensions — it "uses up" one dimension to divide the space into two halves.

> [!question]- Why can't the XOR problem be solved by a single perceptron, and what is the minimum number of perceptrons needed?
> XOR has positive points at $(-1, -1)$ and $(1, 1)$, negative at $(-1, 1)$ and $(1, -1)$. These sit at opposite corners, so no single straight line can separate them. You need at least 3 perceptrons: two in a first layer (each drawing a different line) and one in a second layer combining their outputs. This is the simplest multi-layer perceptron.

> [!question]- Given $\mathbf{w} = (0, 1)$ and $b = -1$, describe the decision boundary. On which side does the point $(3, 0)$ fall?
> The boundary is $0 \cdot x_1 + 1 \cdot x_2 - 1 = 0$, i.e. the horizontal line $x_2 = 1$. For $(3, 0)$: $\mathbf{w} \cdot \mathbf{x} + b = 0 + 0 - 1 = -1 < 0$, so the point falls on the $-1$ side (below the line).
