---
type: week
week: 9
title: "From M to VC Dimension, and Bias–Variance as the Average Case"
dates: 2025-11-24 to 2025-11-30
sources:
  - raw/week-09/Wk_9_Lec_1-1.pdf
  - raw/week-09/Wk_9_Lec_2-1.pdf
  - raw/week-09/Monday_ November 24_ 2025 at 4_04_22 PM_Captions_English (United States).txt
  - raw/week-09/Tuesday_ November 25_ 2025 at 9_42_04 AM_Captions_English (United States).txt
  - raw/week-09/Tutorial_Wk_9-1.pdf
  - raw/week-09/ML_Exercise_Sheet_9a.pdf
  - raw/week-09/ML_Exercise_Sheet_9a_solution.pdf
  - raw/week-09/ML_Exercise_Sheet_9b.pdf
  - raw/week-09/ML_Exercise_Sheet_9b_solution.pdf
concepts:
  - "[[dichotomy]]"
  - "[[growth-function]]"
  - "[[break-point]]"
  - "[[vc-dimension]]"
  - "[[bias-variance-decomposition]]"
  - "[[learning-curve]]"
status: draft
updated: 2026-05-04
---

> [!question]+ THE CRUX: Last week's [[generalization-bound|generalisation bound]] $\mathbb{P}[|E_{\text{in}} - E_{\text{out}}| > \epsilon] \leq 2 M e^{-2\epsilon^2 N}$ said that learning generalises if the hypothesis set is finite. But every model we actually use — perceptrons, SVMs, neural networks — has infinite $|\mathcal{H}|$, so naively the bound says nothing. **(1)** What is the *right* finite quantity to replace $M$ with, and how does it lead to a non-vacuous bound? **(2)** Worst-case bounds give pessimistic guarantees that hold uniformly over all distributions. Is there a complementary, average-case picture of generalisation that's more useful in practice?

*The two halves of week 9 each answer one. **(1)** The fix is to count [[dichotomy|dichotomies]] — distinct labellings $\mathcal{H}$ produces on $N$ specific inputs — rather than distinct hypotheses. The maximum dichotomy count, the **[[growth-function|growth function]]** $m_{\mathcal{H}}(N)$, is bounded by $2^N$. The **[[vc-dimension|VC dimension]]** $d_{\text{VC}}(\mathcal{H})$ is the largest $N$ for which $m_{\mathcal{H}}(N) = 2^N$ — the most points $\mathcal{H}$ can shatter. Finite $d_{\text{VC}}$ implies polynomial $m_{\mathcal{H}}(N)$, the VC bound $E_{\text{out}} \leq E_{\text{in}} + \sqrt{(8/N) \log(4 m_{\mathcal{H}}(2N)/\delta)}$ contracts as $N \to \infty$, and learning is feasible. For the perceptron in $\mathbb{R}^d$, $d_{\text{VC}} = d + 1$ — the number of free parameters. **(2)** Switching from 0–1 loss to squared loss enables an exact decomposition of $\mathbb{E}_{\mathcal{D}}[E_{\text{out}}]$ into **[[bias-variance-decomposition|bias and variance]]**. Bias is how far the average hypothesis $\bar{g}$ is from the truth $f$; variance is how much hypotheses fluctuate across training sets. With noise, an additional irreducible $\sigma^2$ term appears. Complex models lower bias but raise variance — the formal trade-off in choosing model complexity, and the average-case companion to VC's worst-case bound.*

---

## Part 1: Replacing $M$ with Something Finite

### The Problem

The generalisation bound in week 8 ended with a problem: it depends on $M = |\mathcal{H}|$. For finite hypothesis sets the bound is

$$\mathbb{P}[|E_{\text{in}}(g) - E_{\text{out}}(g)| > \epsilon] \leq 2 M e^{-2 \epsilon^2 N}$$

— useful and informative. But every interesting model has $M = \infty$: linear classifiers in $\mathbb{R}^d$ have a continuum of weight vectors, so do SVMs and neural networks. Plugging $M = \infty$ in gives a vacuous bound that says nothing. Are infinite hypothesis sets simply doomed?

The answer is no, and the fix is conceptually beautiful: most of the $M$ in the union bound is double-counting. Two hypotheses that label the training set identically have *almost identical* "bad events" — the events that their training and test errors disagree by more than $\epsilon$. Treating them as $M = 2$ rather than $M = 1$ wastes half the budget.

### Dichotomies and the Growth Function

The right quantity is what hypotheses look like *on a fixed training set*, not what they look like as full functions on $\mathcal{X}$. A [[dichotomy]] is a labelling pattern $(h(\mathbf{x}_1), \ldots, h(\mathbf{x}_N))$ that some $h \in \mathcal{H}$ produces on $N$ specific inputs. Two hypotheses with different parameters but identical labellings on the training set are the *same dichotomy*.

The number of distinct dichotomies on $N$ specific inputs is at most $2^N$ — finite even when $|\mathcal{H}|$ is infinite. To remove dependence on the specific input choice, take the maximum over all input sets:

$$m_{\mathcal{H}}(N) = \max_{\mathbf{x}_1, \ldots, \mathbf{x}_N \in \mathcal{X}} |\mathcal{H}(\mathbf{x}_1, \ldots, \mathbf{x}_N)|.$$

This is the **[[growth-function|growth function]]** — a property of $\mathcal{H}$ alone, distribution-free, algorithm-independent, and bounded by $2^N$.

### Examples

Let's compute $m_{\mathcal{H}}(N)$ for some hypothesis sets:

| $\mathcal{H}$ | $m_{\mathcal{H}}(N)$ |
|---|---|
| Positive rays $h(x) = \text{sign}(x - a)$ on $\mathbb{R}$ | $N + 1$ |
| Positive intervals $h(x) = +1$ iff $x \in [a, b]$ | $N^2/2 + N/2 + 1$ |
| 2D perceptrons (lines in $\mathbb{R}^2$) | $\leq O(N^3)$, with $m_{\mathcal{H}}(4) = 14$ |
| Convex sets in $\mathbb{R}^2$ | $2^N$ (always) |

The first three grow polynomially. The last is exponential — and it stays at $2^N$ forever, no matter how large $N$ gets.

### Shattering, Break Points, and VC Dimension

If $m_{\mathcal{H}}(N) = 2^N$, we say $\mathcal{H}$ **shatters** some $N$-point input set: it can produce every conceivable labelling of those points. A [[break-point]] is the smallest $k$ at which shattering fails, $m_{\mathcal{H}}(k) < 2^k$. Once $\mathcal{H}$ has any break point, every larger $N$ inherits it.

The **[[vc-dimension|VC dimension]]** $d_{\text{VC}}(\mathcal{H})$ is the largest $N$ at which $\mathcal{H}$ can still shatter — equivalently, one less than the break point.

> [!info]+ ASIDE — Why "break point" is the right name
> Imagine sliding $N$ from 1 upward. Initially $\mathcal{H}$ keeps up with the explosion of binary labellings ($m_{\mathcal{H}}(N) = 2^N$). At some specific $k$ — the break point — the hypothesis set "breaks", unable to keep producing every labelling. Once it breaks, it stays broken: the structural restriction propagates. The break point marks the *moment* the hypothesis set's geometry kicks in.

For 2D perceptrons: $m_{\mathcal{H}}(3) = 8$ (any 3 non-collinear points are shattered) and $m_{\mathcal{H}}(4) = 14 < 16$ (the XOR labelling on a square is impossible). Break point 4, $d_{\text{VC}} = 3$.

### The Polynomial Bound

The deep fact: **a single break point $k$ forces $m_{\mathcal{H}}(N)$ to grow polynomially**. Concretely (Sauer–Shelah):

$$m_{\mathcal{H}}(N) \leq \sum_{i=0}^{d_{\text{VC}}} \binom{N}{i} \leq N^{d_{\text{VC}}} + 1.$$

This is the moment the entire learning-theory story closes. We started with an exponential $2^N$ that ruined the bound; we end with a polynomial $N^{d_{\text{VC}}}$. The VC bound becomes

$$\mathbb{P}[|E_{\text{in}}(g) - E_{\text{out}}(g)| > \epsilon] \leq 4 m_{\mathcal{H}}(2N) e^{-\frac{1}{8} \epsilon^2 N},$$

and with polynomial $m_{\mathcal{H}}$ the right-hand side $\to 0$ as $N \to \infty$. **Finite VC dimension ⇒ learning generalises.**

> [!question]- Apply the Sauer–Shelah bound: a hypothesis set has $d_{\text{VC}} = 5$ and you have $N = 100$ samples. Roughly how big can $m_{\mathcal{H}}(2N)$ be?
> $m_{\mathcal{H}}(2N) = m_{\mathcal{H}}(200) \leq 200^5 + 1 \approx 3.2 \times 10^{11}$. Compared to $2^{200}$ this is *astronomically* smaller — by a factor of about $10^{49}$. That's why polynomial growth saves the bound: the hypothesis set may have infinite cardinality, but its effective complexity on any specific dataset is far smaller.

### The Perceptron Result

For the perceptron in $\mathbb{R}^d$ with bias (inputs $\mathbf{x} = (1, x_1, \ldots, x_d)^\top$):

$$d_{\text{VC}} = d + 1.$$

The "$+1$" is the bias term. So the VC dimension equals the number of free parameters — $d$ slope coefficients plus 1 intercept. For "smooth" parameterised hypothesis sets, this rough equivalence (capacity ≈ parameter count) holds approximately.

### The Sample Complexity Verdict

Inverting the VC bound: $N \geq \frac{8}{\epsilon^2} \log \frac{4((2N)^{d_{\text{VC}}} + 1)}{\delta}$. This is implicit in $N$, so iterate.

For $d_{\text{VC}} = 3$, $\epsilon = 0.1$, $\delta = 0.1$: converges to $N \approx 30{,}000$. For $d_{\text{VC}} = 4$: $N \approx 40{,}000$. The pattern: $N \approx 10{,}000 \cdot d_{\text{VC}}$ in theory.

In practice, the bound is loose. **Rule of thumb: $N \approx 10 \cdot d_{\text{VC}}$ is usually enough.** Real distributions are far more benign than worst-case theory presumes — the bound's distribution-free guarantee comes at a heavy quantitative price.

### Margins, Fat Hyperplanes, and Why SVMs Generalise

A perceptron's $d_{\text{VC}} = d + 1$ scales with the input dimension, which becomes large after polynomial basis expansion (dimension $\binom{Q+d}{d}$ for degree $Q$). Naively, kernelised SVMs have astronomical $d_{\text{VC}}$ — yet they generalise well. Why?

Because the SVM doesn't use *all* hyperplanes — only those with margin at least $\rho$. Restricting to **fat hyperplanes** shrinks the hypothesis set, which shrinks the VC dimension:

$$d_{\text{VC}}(\rho) \leq \left\lceil \frac{R^2}{\rho^2} \right\rceil + 1$$

where $R$ is the radius of a ball containing the data. This is **independent of $d$** — the kernel feature space can have infinite dimension without harming generalisation. The SVM's geometric criterion (maximise the margin) is exactly the criterion that controls capacity. This is the formal version of "SVMs work because of the margin."

## Part 2: An Average-Case View

### Why a Different Decomposition?

The VC bound is *worst-case* and uniform over distributions. It tells you: even in the worst dataset, $E_{\text{out}}$ is at most $E_{\text{in}} + \Omega$. That's a strong guarantee — and a pessimistic one.

A complementary perspective asks: *what is the typical $E_{\text{out}}$ when training sets are drawn from a fixed distribution?* This is an average over $\mathcal{D}$ rather than a worst case, and (with squared error rather than 0–1) admits a clean closed-form decomposition.

### The Decomposition

Let $g^{(\mathcal{D})}$ be the hypothesis fit to dataset $\mathcal{D}$, and let

$$\bar{g}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}[g^{(\mathcal{D})}(\mathbf{x})]$$

be the **average hypothesis** — what you'd get by training on infinitely many independently-drawn datasets and averaging. Inserting $\bar{g}$ into the squared error and expanding:

$$\mathbb{E}_{\mathcal{D}}[(g^{(\mathcal{D})}(\mathbf{x}) - f(\mathbf{x}))^2] = \underbrace{\mathbb{E}_{\mathcal{D}}[(g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x}))^2]}_{\text{var}(\mathbf{x})} + \underbrace{(\bar{g}(\mathbf{x}) - f(\mathbf{x}))^2}_{\text{bias}(\mathbf{x})}.$$

The cross term vanishes because $\mathbb{E}_{\mathcal{D}}[g^{(\mathcal{D})} - \bar{g}] = 0$ by definition of $\bar{g}$. Taking expectation over $\mathbf{x}$ as well gives the **[[bias-variance-decomposition|bias–variance decomposition]]**:

$$\mathbb{E}_{\mathcal{D}}[E_{\text{out}}(g^{(\mathcal{D})})] = \text{bias} + \text{var}.$$

With label noise $y = f(\mathbf{x}) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$, an extra $\sigma^2$ floor appears:

$$\mathbb{E}_{\mathcal{D}, \epsilon}[E_{\text{out}}] = \text{bias} + \text{var} + \sigma^2.$$

### What Bias and Variance Mean

- **Bias.** *How far from the truth is the typical hypothesis $\mathcal{H}$ produces?* Low if $\mathcal{H}$ contains hypotheses near $f$ and the algorithm finds them on average. High if $\mathcal{H}$ is structurally too simple — even averaging over infinite datasets, you can't approximate $f$.
- **Variance.** *How much do individual trained hypotheses fluctuate across training sets?* Low if the algorithm is stable. High if it's overfitting random structure in each training set.

The classic dart-board picture: the bullseye is the truth, each dart is a learned hypothesis trained on a different dataset. Bias is the offset of the average dart from the bullseye; variance is the spread.

### The Trade-Off, Concretely

Target $f(x) = \sin(\pi x)$, two samples per training set, two hypothesis sets:

- $\mathcal{H}_0 = \{h(x) = b\}$ — constant lines. Best fit: the average of the two $y$-values.
- $\mathcal{H}_1 = \{h(x) = ax + b\}$ — sloped lines. Fit: the unique line through the two points.

Compute bias and variance over many random pairs of samples:

| Hypothesis set | bias | var | $\mathbb{E}_{\mathcal{D}}[E_{\text{out}}]$ |
|---|---|---|---|
| $\mathcal{H}_0$ (constant) | $0.50$ | $0.25$ | $0.75$ |
| $\mathcal{H}_1$ (line) | $0.21$ | $1.69$ | $1.90$ |

The **constant** wins despite having higher bias — it's so much more stable that the variance saving wipes out the bias penalty.

> [!tip]+ TIP — Choose model complexity by sample size
> The above example reverses if you sample more points. With $N = 5$:
>
> - $\mathcal{H}_0$: bias $0.50$, var $0.10$, total $0.60$.
> - $\mathcal{H}_1$: bias $0.21$, var $0.21$, total $0.42$.
>
> The line now wins. **Optimal model complexity scales with $N$.** Tiny datasets → simple models; abundant data → complex models. This generalises beyond the toy example: more data shrinks variance, eventually unmasking bias as the dominant error and rewarding flexibility.

### VC vs Bias–Variance: Two Lenses

The two analyses describe the same phenomenon from different angles:

| | VC analysis | Bias–variance |
|---|---|---|
| Loss | 0–1 | Squared |
| Over $\mathcal{D}$ | Worst case | Average |
| Output | $E_{\text{out}} \leq E_{\text{in}} + \Omega(d_{\text{VC}})$ | $\mathbb{E}_{\mathcal{D}}[E_{\text{out}}] = \text{bias} + \text{var} + \sigma^2$ |
| Distribution-free? | Yes | No |

VC tells you *whether* learning generalises uniformly across distributions; bias–variance tells you *what kind of error* is left, on average, with a specific algorithm. They live on top of the same [[learning-curve|learning curves]] — VC shades the gap between $E_{\text{in}}$ and $E_{\text{out}}$ as the worst-case generalisation gap; bias–variance splits the curves' contents into bias (the noise floor's structural component) and variance (the gap above it).

> [!question]- Diagnose: training error 0.05, validation error 0.45, with 1{,}000 examples. High bias or high variance? What would you try?
> Big gap between training and validation $\Rightarrow$ **high variance**. The model is fitting training noise rather than signal. Remedies, in order of difficulty: get more data (the surest fix; reduces variance directly), regularise more aggressively (e.g., increase $\lambda$ in ridge), simplify the model class (lower polynomial degree, fewer NN parameters), or ensemble multiple models. Don't add capacity — that would worsen variance.

### Learning Curves

Plotting expected $E_{\text{in}}$ and $E_{\text{out}}$ against $N$ — a [[learning-curve|learning curve]] — visualises everything above. Universal qualitative behaviour:

- $E_{\text{in}}$ rises as $N$ grows (model can no longer interpolate);
- $E_{\text{out}}$ falls as $N$ grows (more data, better generalisation);
- They converge to the same asymptote (the bias-plus-noise floor of the model class).

For linear regression with Gaussian noise, the curves have an exact closed form:

$$\mathbb{E}_{\mathcal{D}}[E_{\text{in}}] = \sigma^2 \left(1 - \frac{d+1}{N}\right), \qquad \mathbb{E}_{\mathcal{D}}[E_{\text{out}}] = \sigma^2 \left(1 + \frac{d+1}{N}\right).$$

The gap $2\sigma^2 (d+1)/N$ decays as $1/N$ — every parameter "costs" $\sigma^2/N$ of generalisation gap. With $N \gg d + 1$, both curves sit at the noise floor $\sigma^2$.

Empirical learning curves (with held-out validation as a proxy for $E_{\text{out}}$) are one of the most useful debugging tools in ML — they tell you whether more data, more capacity, or stronger regularisation is the right next step.

---

## Concepts Introduced This Week

- [[dichotomy]] — labelling pattern that a hypothesis produces on a finite set of inputs; the bridge from infinite $|\mathcal{H}|$ to finite count.
- [[growth-function]] — $m_{\mathcal{H}}(N) = \max_{\mathbf{x}_1, \ldots, \mathbf{x}_N} |\mathcal{H}(\mathbf{x}_1, \ldots, \mathbf{x}_N)|$; bounded by $2^N$, often polynomial for structured $\mathcal{H}$.
- [[break-point]] — smallest $k$ for which $m_{\mathcal{H}}(k) < 2^k$; equals $d_{\text{VC}} + 1$.
- [[vc-dimension]] — largest $N$ for which $\mathcal{H}$ shatters some $N$-point input set; for perceptron in $\mathbb{R}^d$, $d_{\text{VC}} = d + 1$. Finite $d_{\text{VC}}$ certifies feasibility of learning.
- [[bias-variance-decomposition]] — $\mathbb{E}_{\mathcal{D}}[E_{\text{out}}] = \text{bias} + \text{var} (+ \sigma^2)$ for squared loss; complements VC's worst-case analysis.
- [[learning-curve]] — $\mathbb{E}_{\mathcal{D}}[E_{\text{in}}], \mathbb{E}_{\mathcal{D}}[E_{\text{out}}]$ vs $N$; the shape distinguishes underfitting, overfitting, and noise floor.

## Connections

- **Builds on** [[week-08]]: the generalisation bound $\mathbb{P} \leq 2 M e^{-2 \epsilon^2 N}$ ended with the question "what to do when $M = \infty$?" This week answers it. Hoeffding + union bound generalises, with $M$ replaced by $m_{\mathcal{H}}(N)$ — the dichotomy-counting generalisation.
- **Builds on** [[non-linear-transformation]]: a $Q$-th order polynomial transform raises the VC dimension to roughly $\binom{Q+d}{d}$ — a quantitative cost for the non-linearity. Now we have the framework to say *why* over-aggressive basis expansion overfits.
- **Builds on** [[support-vector-machine]]: the margin restricts the hypothesis class to fat hyperplanes, lowering effective $d_{\text{VC}}$ to $\lceil R^2/\rho^2 \rceil + 1$ — independent of input dimension. This is the formal sense in which SVMs "regularise via margin".
- **Sets up** [[week-10]]: overfitting and underfitting examined empirically, plus regularisation and cross-validation as the practical tools to navigate the bias–variance trade-off you can't measure directly.

## Open Questions

- **Why do deep networks generalise despite enormous $d_{\text{VC}}$?** Modern architectures have millions of parameters trained on hundreds of thousands of examples — VC theory predicts catastrophic overfitting that doesn't happen. Answers from "implicit regularisation" of SGD, flat-minima theory, and PAC-Bayes bounds are active research.
- **Does the worst-case nature of VC analysis ever lead practitioners astray?** The factor of 1{,}000 between theoretical sample complexity ($N \approx 10{,}000 \cdot d_{\text{VC}}$) and the practical rule ($N \approx 10 \cdot d_{\text{VC}}$) is uncomfortable — if you trusted the bound, you'd give up on learning long before it becomes possible.
- **How do you actually estimate bias and variance in practice?** You can't from a single training set — bootstrap resampling or cross-validation give approximate estimates. The decomposition is more often used conceptually than measured.
- **What's the right way to think about effective capacity for regularised models?** Ridge with large $\lambda$ has the same $d_{\text{VC}}$ as unregularised but different effective behaviour. "Effective dimension" or "effective $d_{\text{VC}}$" formalisations attempt this; we'll touch on regularisation directly next week.
