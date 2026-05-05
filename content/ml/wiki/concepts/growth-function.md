---
type: concept
name: growth function
description: The maximum number of distinct labellings that hypothesis set H can realise on N inputs, taken over the worst-case input arrangement. Replaces |H| as the effective complexity measure in generalisation bounds, and is finite even when |H| is infinite.
sources:
  - raw/week-09/Wk_9_Lec_1-1.pdf
  - raw/week-09/Monday_ November 24_ 2025 at 4_04_22 PM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*The **growth function** $m_{\mathcal{H}}(N) = \max_{\mathbf{x}_1, \ldots, \mathbf{x}_N} |\mathcal{H}(\mathbf{x}_1, \ldots, \mathbf{x}_N)|$ is the largest number of distinct [[dichotomy|dichotomies]] that $\mathcal{H}$ can produce on any choice of $N$ inputs. Bounded by $2^N$ in the worst case, often polynomial in $N$ for structured hypothesis sets — and the quantity that replaces $M$ in the generalisation bound when $|\mathcal{H}| = \infty$.*

## Definition

For a hypothesis set $\mathcal{H}$ of binary classifiers on $\mathcal{X}$, define

$$m_{\mathcal{H}}(N) = \max_{\mathbf{x}_1, \ldots, \mathbf{x}_N \in \mathcal{X}} |\mathcal{H}(\mathbf{x}_1, \ldots, \mathbf{x}_N)|$$

where $|\mathcal{H}(\mathbf{x}_1, \ldots, \mathbf{x}_N)|$ is the number of distinct labellings $\mathcal{H}$ can produce on those specific inputs.

Two key properties:

- $m_{\mathcal{H}}(N) \leq 2^N$ — there are only $2^N$ binary labellings of $N$ points.
- $m_{\mathcal{H}}(N)$ depends *only* on $\mathcal{H}$ and $N$. It does not depend on the input distribution $p(\mathbf{x})$, the learning algorithm, or the target function.

The "max over inputs" eliminates dependence on the specific training set. We're asking: *with the most adversarial possible placement of $N$ points*, how many distinct labellings can $\mathcal{H}$ realise?

## Worked Examples

### Positive Rays

$\mathcal{H} = \{h(x) = \text{sign}(x - a) : a \in \mathbb{R}\}$ — threshold functions on the real line.

Place $N$ points on $\mathbb{R}$, sorted: $x_1 < x_2 < \cdots < x_N$. Sliding the threshold $a$ from $-\infty$ to $+\infty$, the labelling pattern changes only when $a$ crosses one of the points. There are $N + 1$ "regions" for $a$, each producing a distinct dichotomy:

$$m_{\mathcal{H}}(N) = N + 1.$$

The "$N$" comes from the $N$ flip points; the "$+1$" from the two ends.

### Positive Intervals

$\mathcal{H} = \{h(x) = +1 \text{ iff } x \in [a, b]\}$.

Place $N$ points on $\mathbb{R}$. A dichotomy is determined by which two of the $N + 1$ "gaps between/around points" the endpoints $a$ and $b$ fall into. Choosing 2 gaps from $N + 1$ with repetition gives

$$m_{\mathcal{H}}(N) = \binom{N+1}{2} + 1 = \frac{N^2}{2} + \frac{N}{2} + 1.$$

The extra "$+1$" is the all-negative labelling (interval entirely outside the points).

### 2D Perceptrons

$\mathcal{H} = $ all linear classifiers in $\mathbb{R}^2$. By direct enumeration:

| $N$ | $m_{\mathcal{H}}(N)$ | $2^N$ |
|---|---|---|
| 1 | 2 | 2 |
| 2 | 4 | 4 |
| 3 | 8 | 8 |
| 4 | **14** | 16 |

At $N = 4$ the count drops below $2^N$ for the first time. With more careful analysis $m_{\mathcal{H}}(N) = O(N^3)$ for 2D perceptrons — polynomial, not exponential.

### Convex Sets in $\mathbb{R}^2$

$\mathcal{H} = \{h(\mathbf{x}) = +1 \text{ iff } \mathbf{x} \in C, C \text{ convex}\}$.

Place $N$ inputs on a circle. For *any* labelling, take the convex hull of the $+1$ points (slightly extended): every dichotomy is realisable. So

$$m_{\mathcal{H}}(N) = 2^N \quad \text{for all } N.$$

Convex sets are maximally expressive on circles — they shatter any $N$. The hypothesis set has no [[break-point]] and infinite [[vc-dimension|VC dimension]].

## The Break Point

A **break point** for $\mathcal{H}$ is the smallest $k$ such that no set of $k$ inputs is shattered — i.e.,

$$m_{\mathcal{H}}(k) < 2^k.$$

If $k$ is a break point, every $k' > k$ is also a break point: if you can't shatter $k$ inputs, you can't shatter $k'$ either (a counter-example to shattering on $k$ points sits inside any superset).

Break points for the examples above:

| $\mathcal{H}$ | Growth function | Break point |
|---|---|---|
| Positive rays | $N + 1$ | $k = 2$ ($m_{\mathcal{H}}(2) = 3 < 4$) |
| Positive intervals | $N^2/2 + N/2 + 1$ | $k = 3$ ($m_{\mathcal{H}}(3) = 7 < 8$) |
| 2D perceptrons | $\leq O(N^3)$ | $k = 4$ ($m_{\mathcal{H}}(4) = 14 < 16$) |
| Convex sets in $\mathbb{R}^2$ | $2^N$ | None |

## The Polynomial Bound

The deep result of VC theory: **if $\mathcal{H}$ has any break point $k$, then $m_{\mathcal{H}}(N)$ is bounded by a polynomial in $N$.** Concretely, with $d_{\text{VC}} = k - 1$ (the [[vc-dimension|VC dimension]]):

$$m_{\mathcal{H}}(N) \leq \sum_{i=0}^{d_{\text{VC}}} \binom{N}{i} \leq N^{d_{\text{VC}}} + 1.$$

The growth from exponential ($2^N$) to polynomial ($N^{d_{\text{VC}}}$) is what makes the generalisation bound non-vacuous for infinite hypothesis sets. Plug a polynomial $m_{\mathcal{H}}(N)$ into

$$\mathbb{P}[|E_{\text{in}} - E_{\text{out}}| > \epsilon] \leq 4 m_{\mathcal{H}}(2N) e^{-\frac{1}{8} \epsilon^2 N}$$

(the **[[vc-dimension|VC bound]]**), and the right-hand side $\to 0$ as $N \to \infty$ — learning is feasible.

Without a break point ($d_{\text{VC}} = \infty$, like convex sets), $m_{\mathcal{H}}(N) = 2^N$ and the bound is vacuous: the hypothesis set is too expressive to generalise from finite data.

## Why Max Over Inputs?

The growth function takes the *maximum* over all input arrangements, not the typical case. Three reasons:

1. **Worst-case bound.** The generalisation bound must hold uniformly over all distributions $p(\mathbf{x})$ — no probabilistic assumption rules out adversarial input placement.
2. **Algorithm- and distribution-free.** Removing the dependence on $p(\mathbf{x})$ makes $m_{\mathcal{H}}(N)$ a property of $\mathcal{H}$ alone, computable without reference to data.
3. **Structural complexity, not statistical.** $m_{\mathcal{H}}(N)$ measures *what $\mathcal{H}$ is capable of expressing*, divorced from what data we happen to see.

## Related

- [[dichotomy]] — the basic object being counted.
- [[vc-dimension]] — the largest $N$ for which $m_{\mathcal{H}}(N) = 2^N$.
- [[break-point]] — the smallest $N$ at which $m_{\mathcal{H}}(N) < 2^N$.
- [[generalization-bound]] — what the polynomial growth function rescues for infinite $\mathcal{H}$.

## Active Recall

> [!question]- Compute $m_{\mathcal{H}}(N)$ for positive rays $h(x) = \text{sign}(x - a)$ and explain where the "$+1$" in $N + 1$ comes from.
> Sort the $N$ inputs along $\mathbb{R}$. The threshold $a$ creates a distinct dichotomy depending on which "gap" between consecutive inputs (or below the smallest, or above the largest) it sits in. There are $N + 1$ such gaps, hence $m_{\mathcal{H}}(N) = N + 1$. The "$+1$" corresponds to either of the two extreme gaps where the threshold is below all points (all $+1$) or above all (all $-1$) — collapsed into one because both extremes give labellings that are realisable, but the count of slid-into regions is $N + 1$.

> [!question]- Why is the growth function for convex sets in $\mathbb{R}^2$ exactly $2^N$ for all $N$, while for 2D perceptrons it drops below $2^N$ at $N = 4$?
> Convex sets are far more expressive than half-planes. Place $N$ points on a circle: for any binary labelling, take the convex hull of the $+1$ points — that's a convex region containing exactly the $+1$ points and no others. Every labelling is realisable, so $m_{\mathcal{H}}(N) = 2^N$. Lines in $\mathbb{R}^2$ partition the plane into two half-spaces; XOR-style labellings on 4 corners of a square cannot be realised because the $+1$ points and $-1$ points each form non-collinear pairs that would require a non-convex separator. The structural restriction "decision boundary is a hyperplane" begins to bite at $N = 4$.

> [!question]- Suppose $m_{\mathcal{H}}(5) = 31 < 2^5 = 32$. What is the smallest $k$ that *might* be a break point for $\mathcal{H}$, and why is "*might*" the right word?
> The smallest possible break point is $k \leq 5$. We've shown $m_{\mathcal{H}}(5) < 2^5$, so $k = 5$ is at least a break point — but maybe shattering already fails for fewer points. We'd need to check $m_{\mathcal{H}}(4), m_{\mathcal{H}}(3)$ etc. against $2^4, 2^3$ to find the *smallest* $k$. Concretely, the break point is the smallest $k$ for which $m_{\mathcal{H}}(k) < 2^k$; once you find it, all larger values are break points too.

> [!question]- Why does the max-over-inputs definition of $m_{\mathcal{H}}(N)$ make the resulting generalisation bound *worst-case*, and what does that imply about how we should interpret it in practice?
> The max ensures the count holds regardless of how training inputs are drawn — even if an adversary placed them. Real data is typically not adversarial, so empirical generalisation is often much better than the bound predicts. The bound is a *sufficient* condition for generalisation, not an estimate of typical-case behaviour. In practice, this is why we still rely on cross-validation for model selection: the worst-case bound is too conservative to drive practical decisions for complex models.
