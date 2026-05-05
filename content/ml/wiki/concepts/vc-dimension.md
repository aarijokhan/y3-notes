---
type: concept
name: VC dimension
description: The largest N for which a hypothesis set H can shatter some set of N inputs. A finite, distribution-free measure of model capacity that replaces |H| in the generalisation bound and certifies feasibility of learning for infinite hypothesis sets.
sources:
  - raw/week-09/Wk_9_Lec_1-1.pdf
  - raw/week-09/Monday_ November 24_ 2025 at 4_04_22 PM_Captions_English (United States).txt
  - raw/week-09/Tuesday_ November 25_ 2025 at 9_42_04 AM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*The **Vapnik–Chervonenkis dimension** $d_{\text{VC}}(\mathcal{H})$ is the largest $N$ such that some set of $N$ inputs is shattered by $\mathcal{H}$ — i.e., $m_{\mathcal{H}}(N) = 2^N$. Finite $d_{\text{VC}}$ guarantees that learning generalises: the [[growth-function|growth function]] is polynomial of degree $d_{\text{VC}}$, the VC bound contracts as $N \to \infty$, and the sample complexity scales as $N \approx 10 d_{\text{VC}}$ in practice.*

## Definition

$$d_{\text{VC}}(\mathcal{H}) = \max \{N : m_{\mathcal{H}}(N) = 2^N\}.$$

Equivalently:

- For $N \leq d_{\text{VC}}$: there exist $N$ inputs that $\mathcal{H}$ can [[dichotomy|shatter]] (realise all $2^N$ labellings).
- For $N > d_{\text{VC}}$: no set of $N$ inputs is shattered. $N$ is a [[break-point]] for $\mathcal{H}$.

The VC dimension is a property of $\mathcal{H}$ alone — it does not depend on the input distribution $p(\mathbf{x})$, the learning algorithm $\mathcal{A}$, or the target function $f$. It captures a hypothesis set's *capacity*: the most labellings of any input set it can produce.

## Examples

| Hypothesis set | VC dimension | Reason |
|---|---|---|
| Thresholds on $\mathbb{R}$, $h(x) = \text{sign}(x - a)$ | $1$ | Shatters 1 point; cannot shatter 2 (can't realise $-, +$ if $x_1 < x_2$). |
| Positive intervals on $\mathbb{R}$ | $2$ | Shatters 2 points; cannot shatter 3 (the alternating $-, +, -$ labelling fails). |
| Linear classifiers in $\mathbb{R}^d$ ("perceptron in $d$ dimensions") | $d + 1$ | Shatters $d+1$ points in general position; $d+2$ are too constrained. |
| Axis-aligned rectangles in $\mathbb{R}^2$ | $4$ | Shatters 4 points in a "diamond" arrangement; 5 fails because the interior point cannot be the unique negative one. |
| Convex sets in $\mathbb{R}^2$ | $\infty$ | Points on a circle: every labelling is realised by the convex hull of the positives. |
| Neural network with $W$ weights | $\Theta(W)$ | Roughly linear in the parameter count for many architectures. |

## The Perceptron Theorem

For the perceptron in $\mathbb{R}^d$ with bias (i.e., inputs $\mathbf{x} = (1, x_1, \ldots, x_d)^\top$):

$$d_{\text{VC}}(\text{perceptron in } \mathbb{R}^d) = d + 1.$$

The "$+1$" is the bias term $w_0$. So a 2D perceptron has $d_{\text{VC}} = 3$: it shatters any 3 non-collinear points, but cannot shatter 4 points in general position (the XOR labellings on a square are unrealisable).

The bound $d_{\text{VC}} = d + 1$ matches the number of free parameters of a $d$-dimensional perceptron — $d$ slope coefficients plus 1 bias. This is no coincidence: the VC dimension of many "smooth" parameterised hypothesis sets is approximately the number of effective parameters.

## The VC Bound

The replacement of $M$ by the growth function in the generalisation bound, with a careful argument that handles dependence between training and test sets, gives the **VC bound**:

$$\mathbb{P}[|E_{\text{in}}(g) - E_{\text{out}}(g)| > \epsilon] \leq 4 m_{\mathcal{H}}(2N) e^{-\frac{1}{8} \epsilon^2 N}.$$

Substituting the polynomial bound $m_{\mathcal{H}}(N) \leq N^{d_{\text{VC}}} + 1$:

$$\mathbb{P}[|E_{\text{in}}(g) - E_{\text{out}}(g)| > \epsilon] \leq 4 (2N)^{d_{\text{VC}}} e^{-\frac{1}{8} \epsilon^2 N}.$$

Setting the RHS equal to $\delta$ and solving for $\epsilon$ gives the high-probability form:

$$\boxed{E_{\text{out}}(g) \leq E_{\text{in}}(g) + \sqrt{\frac{8}{N} \log \frac{4 ((2N)^{d_{\text{VC}}} + 1)}{\delta}}}$$

with probability at least $1 - \delta$.

The penalty term $\sqrt{\frac{8}{N} \log \frac{4 ((2N)^{d_{\text{VC}}} + 1)}{\delta}}$ — called $\Omega(N, \mathcal{H}, \delta)$ — is the **model-complexity penalty**. It grows with $d_{\text{VC}}$ and shrinks with $N$.

## Reading the Bound

| Direction | Effect |
|---|---|
| $d_{\text{VC}} \uparrow$ | $E_{\text{in}}$ down (more expressive, fits training data better), $\Omega$ up (worse generalisation gap) |
| $d_{\text{VC}} \downarrow$ | $\Omega$ down (tighter generalisation), $E_{\text{in}}$ up (less able to fit) |
| Best $d_{\text{VC}}^{\ast}$ | Some intermediate value where the sum $E_{\text{in}} + \Omega$ is minimised |

This is the formal source of the **bias-variance / model-complexity trade-off**. Simple models underfit ($E_{\text{in}}$ large); complex models overfit ($\Omega$ large). The optimum is in between.

## Sample Complexity

Inverting the VC bound: for fixed accuracy $\epsilon$, confidence $1 - \delta$, and VC dimension $d_{\text{VC}}$,

$$N \geq \frac{8}{\epsilon^2} \log \frac{4 ((2N)^{d_{\text{VC}}} + 1)}{\delta}$$

is the number of samples needed. This is implicit in $N$, so iterate to find a self-consistent solution.

**Worked example.** $d_{\text{VC}} = 3$, $\epsilon = 0.1$, $\delta = 0.1$:

- Plug in $N = 1000$: RHS $\approx 8 / 0.01 \cdot \log(4 \cdot 8 \cdot 10^9 / 0.1) \approx 21{,}193$.
- Iterate with $N = 21{,}193$: converges to $N \approx 30{,}000$.
- For $d_{\text{VC}} = 4$: $N \approx 40{,}000$.

So *roughly* $N \approx 10{,}000 \cdot d_{\text{VC}}$ in theory. **Practical rule of thumb**: $N \approx 10 \cdot d_{\text{VC}}$ is usually enough — the bound is loose. Real models generalise better than theory promises.

## Margin and the SVM

The vanilla VC bound for a $d$-dimensional perceptron gives $d_{\text{VC}} = d + 1$, which after polynomial basis expansion to degree $Q$ becomes $\binom{Q + d}{d} = O(Q^d)$ — large.

[[support-vector-machine|SVMs]] are still linear classifiers, so naively their $d_{\text{VC}}$ is $d + 1$. But SVMs add a *margin* constraint: they restrict to hyperplanes of width at least $\rho$. The margin restricts the hypothesis set, and a smaller hypothesis set has a smaller VC dimension:

$$d_{\text{VC}}(\rho) \leq \left\lceil \frac{R^2}{\rho^2} \right\rceil + 1$$

where $R$ is the radius of a ball containing all data. This is **independent of $d$** — the margin gives data-dependent generalisation, and explains why SVMs work even with very high-dimensional kernel feature spaces.

## What VC Dimension Doesn't Capture

The VC bound is the worst-case, distribution-free generalisation bound. It doesn't account for:

- **Algorithm bias.** Different learning algorithms may converge to different hypotheses within the same $\mathcal{H}$; a regulariser like ridge implicitly restricts capacity below $d_{\text{VC}}$.
- **Distribution structure.** Real data tends not to be adversarial; effective complexity is often much smaller than $d_{\text{VC}}$.
- **Implicit regularisation.** Stochastic gradient descent on neural networks finds flat minima that generalise far better than parameter-counting predicts — a phenomenon outside classical VC theory.

For these reasons, modern deep learning routinely violates the naive prediction "models with $d_{\text{VC}} \gg N$ should overfit catastrophically." Networks with millions of parameters trained on hundreds of thousands of examples generalise well — for reasons beyond what VC dimension alone explains.

## Related

- [[dichotomy]] — the labelling pattern that VC dimension counts.
- [[growth-function]] — $m_{\mathcal{H}}(N)$, with $m_{\mathcal{H}}(d_{\text{VC}}) = 2^{d_{\text{VC}}}$ but $m_{\mathcal{H}}(d_{\text{VC}} + 1) < 2^{d_{\text{VC}} + 1}$.
- [[break-point]] — equals $d_{\text{VC}} + 1$.
- [[generalization-bound]] — the simpler $M$-based bound that VC dimension generalises.
- [[bias-variance-decomposition]] — the average-case alternative to VC's worst-case analysis.
- [[support-vector-machine]] — SVMs achieve generalisation by margin-based restriction of effective $d_{\text{VC}}$.

## Active Recall

> [!question]- Show that the VC dimension of a perceptron in $\mathbb{R}^2$ (with bias) is 3 by demonstrating both a set of 3 points it shatters and a set of 4 points it cannot.
> *Shattering 3 points.* Place 3 non-collinear points as the vertices of a triangle. For any of the $2^3 = 8$ labellings, draw a line that puts the $+1$ vertices on one side and $-1$ on the other — possible because the points aren't collinear. So $m_{\mathcal{H}}(3) = 8 = 2^3$. *Failing on 4 points.* Place 4 points at the corners of a square. The labelling $(+, -, -, +)$ on the diagonally opposite corners (XOR pattern) cannot be realised by any line: a line has a single linear separator, but the $+$ corners and $-$ corners are interleaved diagonally. So $m_{\mathcal{H}}(4) \leq 14 < 16$, and $d_{\text{VC}} = 3$.

> [!question]- The VC dimension of a perceptron in $\mathbb{R}^d$ is $d + 1$. Where does the "$+1$" come from, and what does this say about the relationship between VC dimension and parameter count?
> The "$+1$" is the bias term $w_0$. With inputs augmented as $\mathbf{x} = (1, x_1, \ldots, x_d)^\top$, the perceptron has $d + 1$ free parameters $(w_0, w_1, \ldots, w_d)$, each contributing a degree of freedom to the decision boundary. For perceptrons specifically, $d_{\text{VC}}$ equals the number of free parameters exactly. For other smooth parameterised models — neural networks especially — VC dimension is approximately linear in the parameter count, $d_{\text{VC}} \approx \Theta(W)$, but not always equal. Parameter counting is a useful heuristic, not a precise rule.

> [!question]- Why does using a polynomial basis expansion of degree $Q$ on a $d$-dimensional perceptron increase the VC dimension to $\binom{Q + d}{d}$, and what's the practical consequence?
> A degree-$Q$ polynomial basis has $\tilde{d} = \binom{Q+d}{d} - 1$ non-bias features (all monomials up to degree $Q$). The classifier is a perceptron in this $\tilde{d}$-dimensional feature space, so $d_{\text{VC}} \leq \tilde{d} + 1 = \binom{Q+d}{d}$. For $d = 100, Q = 3$: $\binom{103}{3} \approx 176{,}851$. The model-complexity penalty in the VC bound grows with this large $d_{\text{VC}}$, demanding hugely more samples for the same generalisation gap. Practically: high-degree polynomial expansions overfit easily and require either heavy regularisation or massive data.

> [!question]- Why is the SVM's $d_{\text{VC}}$ bound $\lceil R^2 / \rho^2 \rceil + 1$ — depending on data geometry through $R$ and $\rho$ — and how does this differ from the perceptron's $d + 1$?
> The plain perceptron bound $d + 1$ depends only on the input dimension; it ignores the data. The SVM constrains the hypothesis class to fat hyperplanes of margin at least $\rho$, restricting the classifier from labelling closely-spaced points arbitrarily. With data confined to a ball of radius $R$, an "$R^2/\rho^2$" bound emerges from a packing argument: only so many "fat" hyperplanes can fit before they must overlap. The bound has a flavor of *data-dependent capacity*: large margin (relative to data scale) implies low effective capacity. This is the formal expression of "SVMs generalise because of the margin" and is independent of input dimension — the kernel trick can blow $d$ up to infinity without spoiling generalisation.

> [!question]- A model has $d_{\text{VC}} = 50$. Roughly how many training samples does the practical rule of thumb suggest you need for good generalisation, and how does this compare to the theoretical VC sample complexity?
> Practical rule: $N \approx 10 \cdot d_{\text{VC}} = 500$ samples. The theoretical VC bound says you need $N \approx 10{,}000 \cdot d_{\text{VC}} = 500{,}000$ — twenty times more. The huge gap reflects that the VC bound is worst-case and distribution-free; real distributions are far more benign, so empirical generalisation kicks in much earlier than theory promises.
