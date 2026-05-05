---
type: concept
name: break point
description: The smallest N at which a hypothesis set H can no longer shatter any input set. Existence of a break point implies polynomial growth function and finite VC dimension — the conditions under which infinite-H learning generalises.
sources:
  - raw/week-09/Wk_9_Lec_1-1.pdf
  - raw/week-09/Monday_ November 24_ 2025 at 4_04_22 PM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*A **break point** for a hypothesis set $\mathcal{H}$ is any $k$ such that no set of $k$ inputs is shattered — equivalently, the [[growth-function|growth function]] satisfies $m_{\mathcal{H}}(k) < 2^k$. Once $\mathcal{H}$ has a break point, every larger $k'$ is also a break point, $m_{\mathcal{H}}(N)$ is bounded by a polynomial of degree $k - 1$, and the VC bound becomes non-vacuous.*

## Definition

A break point for $\mathcal{H}$ is a positive integer $k$ such that

$$m_{\mathcal{H}}(k) < 2^k.$$

The smallest such $k$ is **the** break point. By convention if no such $k$ exists ($\mathcal{H}$ shatters arbitrarily large input sets), $\mathcal{H}$ has *no* break point, equivalently $d_{\text{VC}}(\mathcal{H}) = \infty$.

The break point and [[vc-dimension|VC dimension]] are linked exactly:

$$\text{break point} = d_{\text{VC}} + 1.$$

If $d_{\text{VC}} = N$ then $\mathcal{H}$ shatters some set of $N$ inputs ($m_{\mathcal{H}}(N) = 2^N$) but not any set of $N + 1$ ($m_{\mathcal{H}}(N+1) < 2^{N+1}$), making $N + 1$ the break point.

## Examples

| Hypothesis set | $m_{\mathcal{H}}(N)$ | Break point | $d_{\text{VC}}$ |
|---|---|---|---|
| Positive rays | $N + 1$ | $2$ ($m_{\mathcal{H}}(2) = 3 < 4$) | $1$ |
| Positive intervals | $\frac{N^2}{2} + \frac{N}{2} + 1$ | $3$ ($m_{\mathcal{H}}(3) = 7 < 8$) | $2$ |
| 2D perceptrons | $\leq O(N^3)$ | $4$ ($m_{\mathcal{H}}(4) = 14 < 16$) | $3$ |
| Convex sets in $\mathbb{R}^2$ | $2^N$ | none | $\infty$ |

## Why It Matters

The Sauer–Shelah lemma turns a break point at $k$ into a polynomial bound on the growth function:

$$m_{\mathcal{H}}(N) \leq \sum_{i=0}^{k-1} \binom{N}{i} \leq N^{k-1} + 1.$$

The transition from "could be exponential" ($2^N$) to "guaranteed polynomial" ($N^{k-1}$) is the crucial step that makes generalisation provable for infinite hypothesis sets:

$$\mathbb{P}[|E_{\text{in}} - E_{\text{out}}| > \epsilon] \leq 4 m_{\mathcal{H}}(2N) e^{-\frac{1}{8} \epsilon^2 N}$$

shrinks to zero as $N \to \infty$ if $m_{\mathcal{H}}(N)$ is polynomial. If there's no break point, $m_{\mathcal{H}}(N) = 2^N$ and the right-hand side is bounded below by a constant — learning is not guaranteed to generalise.

A single discrete fact ("does the hypothesis set fail to shatter even one input set of size $k$?") thus controls the entire qualitative behaviour of generalisation.

## Once a Break Point, Always

If $k$ is a break point for $\mathcal{H}$, so is every $k' > k$. Why? Suppose for contradiction that $\mathcal{H}$ shatters some set $S$ of size $k' > k$. Take any subset $S' \subset S$ of size $k$ — every dichotomy of $S'$ extends to a dichotomy of $S$, so $\mathcal{H}$ shatters $S'$ too, contradicting that $k$ is a break point.

This monotonicity is why we focus on the **smallest** break point: it determines when shattering first fails, and everything beyond inherits the failure.

## Related

- [[growth-function]] — the function whose break point we're finding.
- [[vc-dimension]] — equals one less than the break point.
- [[dichotomy]] — labelling patterns whose count $m_{\mathcal{H}}(N)$ saturates at $2^N$ before the break point.

## Active Recall

> [!question]- Why is the break point of "all lines in $\mathbb{R}^2$" equal to 4 rather than 5, and what does this say about the VC dimension?
> At $N = 4$, you can place 4 points (corners of a square) such that the XOR labelling $(+, -, -, +)$ is not realisable by any line — so $m_{\mathcal{H}}(4) = 14 < 16$ and $4$ is a break point. At $N = 3$, three non-collinear points are shattered: all $2^3 = 8$ labellings are realisable. So 3 is not a break point; 4 is the smallest. By the relation break point $= d_{\text{VC}} + 1$, $d_{\text{VC}} = 3$.

> [!question]- Suppose $\mathcal{H}$ has break point $k = 5$. Is $k' = 7$ also a break point? Justify.
> Yes. If no set of 5 inputs is shattered, no set of 7 inputs can be either: any 7-point set contains a 5-point subset, and shattering the 7-set would imply shattering the 5-set (every dichotomy of the subset extends to a dichotomy of the superset). The "no break point" property propagates upward, so once shattering fails at any size, it fails at all larger sizes.

> [!question]- Convex sets in $\mathbb{R}^2$ have no break point. What does this imply about generalisation when learning with this hypothesis set, and why is it intuitive?
> No break point means $m_{\mathcal{H}}(N) = 2^N$ for all $N$, so the VC bound is vacuous — no finite $N$ guarantees small generalisation gap. Intuitively, given any training set with a labelling, the convex-hull-of-positives produces a hypothesis with $E_{\text{in}} = 0$ — but the learned set tells you nothing about how unseen points should be classified. The hypothesis set is too expressive to constrain the outcome, so the worst-case behaviour can be arbitrarily bad. This is the formal version of "any function works on the training data, so we have no information about new data".
