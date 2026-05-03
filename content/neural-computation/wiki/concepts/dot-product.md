---
type: concept
sources:
  - raw/week-01/w01-l02-transcript.txt
  - raw/week-01/w01-slides.pdf
status: stable
updated: 2026-04-20
---

*The dot product measures how much two vectors point in the same direction — and in neural networks, it is the operation that decides which side of a boundary a data point falls on.*

## Definition

For two vectors $\mathbf{x} = (x_1, \dots, x_D)$ and $\mathbf{w} = (w_1, \dots, w_D)$, the dot product is:

$$\mathbf{x} \cdot \mathbf{w} = \sum_{i=1}^{D} x_i \, w_i$$

This is also called the **inner product** or **scalar product**.

## Geometric interpretation

The dot product has an equivalent geometric form:

$$\mathbf{x} \cdot \mathbf{w} = \|\mathbf{x}\| \cdot \|\mathbf{w}\| \cdot \cos(\theta)$$

where $\theta$ is the angle between the two vectors and $\|\cdot\|$ denotes the Euclidean length (norm).

The sign of $\cos(\theta)$ determines the sign of the dot product:

| Condition | Angle | Dot product |
|---|---|---|
| Vectors point in the same direction | $\theta < 90°$ | Positive |
| Vectors are perpendicular | $\theta = 90°$ | Zero |
| Vectors point in opposite directions | $\theta > 90°$ | Negative |

## Signed distance to a hyperplane

This is where the dot product becomes central to classification. Consider a hyperplane passing through the origin, defined by a normal vector $\mathbf{w}$. For any point $\mathbf{x}$:

$$\mathbf{x} \cdot \mathbf{w} = \|\mathbf{w}\| \cdot d$$

where $d$ is the **signed perpendicular distance** from $\mathbf{x}$ to the hyperplane. Points on the same side as $\mathbf{w}$ have $d > 0$; points on the opposite side have $d < 0$; points on the hyperplane have $d = 0$.

When $\|\mathbf{w}\| = 1$ (unit vector), the dot product *equals* the signed distance directly. In general, the dot product gives the distance scaled by $\|\mathbf{w}\|$.

## Role in the perceptron

The [[perceptron]] computes $\hat{y} = \text{sgn}(\mathbf{w} \cdot \mathbf{x} + b)$. The dot product $\mathbf{w} \cdot \mathbf{x}$ measures the signed distance from $\mathbf{x}$ to the [[decision-boundary-nc|decision boundary]] $\mathbf{w} \cdot \mathbf{x} + b = 0$. The sign function then maps this distance to a class label:

- Positive distance (same side as $\mathbf{w}$) $\to$ class $+1$
- Negative distance (opposite side) $\to$ class $-1$

The bias $b$ shifts the hyperplane away from the origin, but the dot product still does the core geometric work.

## Related

- [[perceptron]] — uses the dot product as its core computation
- [[decision-boundary-nc|decision boundary]] — the hyperplane whose signed distance the dot product measures

## Active Recall

> [!question]- Why does the sign of $\mathbf{w} \cdot \mathbf{x}$ tell you which side of the hyperplane (through the origin) a point is on?
> $\mathbf{w} \cdot \mathbf{x} = \|\mathbf{w}\| \|\mathbf{x}\| \cos\theta$. The sign depends on $\cos\theta$: if $\theta < 90°$, the point is on the same side as $\mathbf{w}$ (positive); if $\theta > 90°$, it's on the opposite side (negative); if $\theta = 90°$, the point lies exactly on the hyperplane.

> [!question]- Given $\mathbf{w} = (1, 0)$, what does the hyperplane look like, and what does $\mathbf{w} \cdot \mathbf{x}$ compute for any point $\mathbf{x} = (x_1, x_2)$?
> The hyperplane is the vertical line $x_1 = 0$ (the $x_2$-axis). The dot product $\mathbf{w} \cdot \mathbf{x} = x_1$, which is simply the horizontal coordinate — i.e., the signed distance from the point to the $x_2$-axis.

> [!question]- When does the dot product equal the exact perpendicular distance (without any scaling factor)?
> When the weight vector has unit length, $\|\mathbf{w}\| = 1$. In that case, $\mathbf{w} \cdot \mathbf{x} = \|\mathbf{w}\| \cdot d = d$, so the dot product is the signed perpendicular distance directly.

> [!question]- How are the two formulas for the dot product — $\sum x_i w_i$ and $\|\mathbf{x}\|\|\mathbf{w}\|\cos\theta$ — used differently in practice?
> The algebraic form $\sum x_i w_i$ is how we *compute* the dot product (just multiply and add). The geometric form $\|\mathbf{x}\|\|\mathbf{w}\|\cos\theta$ is how we *interpret* it (it tells us about angles and distances). Both give the same number; the algebraic form is for calculation, the geometric form is for understanding.
