---
type: week
week: 11
title: "Validation, Cross-Validation, and Three Principles for Honest Learning"
dates: 2025-12-08 to 2025-12-14
sources:
  - raw/week-11/Wk_11_Lec_1.pdf
  - raw/week-11/Wk_11_Lec_2.pdf
  - raw/week-11/Monday_ December 8_ 2025 at 4_03_57 PM_Captions_English (United States).txt
  - raw/week-11/Tuesday_ December 9_ 2025 at 9_56_03 AM_Captions_English (United States).txt
  - raw/week-11/Tuesday_ December 9_ 2025 at 10_42_13 AM_Captions_English (United States).txt
  - raw/week-11/Tutorial_Wk_11.pdf
  - raw/week-11/ML_Exercise_Sheet_11a.pdf
  - raw/week-11/ML_Exercise_Sheet_11a_solution.pdf
concepts:
  - "[[validation]]"
  - "[[cross-validation]]"
  - "[[learning-principles]]"
status: draft
updated: 2026-05-04
---

> [!question]+ THE CRUX: Last week's regularisation gave us a *family* of trained models indexed by $\lambda$ — but how do we actually pick the right $\lambda$? More broadly: with the regularisation parameter, model class, kernel, and other knobs all unknown, how do we choose between candidate models without the test set we promised never to touch? And once we've answered that — what general principles, beyond specific algorithms, govern whether a learning experiment was honest in the first place?

*The two halves of week 11 each answer one. **(1)** [[validation|Validation]] holds out a portion of the training data, trains on the rest, and uses the held-out part to estimate $E_{\text{out}}$ for each candidate model. The validation error $E_{\text{val}}(g^-)$ is an **unbiased estimate** of $E_{\text{out}}(g^-)$ with variance $\leq 1/(4K)$ for binary classification — large $K$ shrinks variance but starves the trained model. The standard $K = N/5$ rule splits the difference. After validation picks the winner, retrain on all $N$ examples to produce the final hypothesis. [[cross-validation|Cross-validation]] generalises validation to use *all* the data: leave-one-out trains $N$ times on $N - 1$ examples each, $V$-fold partitions into $V$ chunks ($V = 10$ is the practical default). **(2)** Three meta-principles wrap up the module's theory. **Occam's Razor**: simpler is better, where "simpler" means small VC dimension and "better" means lower $E_{\text{out}}$. **Sampling Bias**: training data from the wrong distribution → no theorem rescues you. **Data Snooping**: any influence of the test set on any decision contaminates it. Together they're the discipline that turns the module's machinery into honest learning.*

---

## Part 1: How Do You Actually Pick $\lambda$?

### The Model Selection Problem

Last week we built [[regularization-ml|regularisation]]: minimise $E_{\text{in}} + (\lambda/N) \|\mathbf{w}\|^2$ for some choice of $\lambda$. But "for some choice of $\lambda$" is doing heavy lifting. The optimal $\lambda^\ast$ depends on the noise level $\sigma^2$, target complexity $Q_f$, and sample size $N$ — all of which we don't observe.

More generally: we have $M$ candidate models $\mathcal{H}_1, \ldots, \mathcal{H}_M$ — different hypothesis classes (linear, quadratic, kernel SVM, ...), different regularisers, different $\lambda$ values, different feature transformations. **How do we pick?**

Three obvious-looking choices, all wrong:

- **Pick by smallest $E_{\text{in}}$**: always selects the most expressive class, since a richer $\mathcal{H}$ can always fit training data better. This is *overfitting through the back door* — minimising $E_{\text{in}}$ over $\mathcal{H}_1 \cup \cdots \cup \mathcal{H}_M$ pays the VC cost of the union, and the most complex member always wins.
- **Pick by smallest $E_{\text{test}}$**: would give an unbiased generalisation guarantee with $O(\sqrt{\log M / N_{\text{test}}})$ error, but $\mathcal{D}_{\text{test}}$ is locked in the boss's safe — using it for selection would be **infeasible and cheating**.
- **Pick by intuition / cross-paper folklore**: invisible bias accumulates; not a procedure.

### Validation: A Held-Out Slice of Training Data

The fix is structural: split $\mathcal{D}$ into a **training set** $\mathcal{D}_{\text{train}}$ ($N - K$ examples) and a **validation set** $\mathcal{D}_{\text{val}}$ ($K$ examples). Train on $\mathcal{D}_{\text{train}}$ to get $g^-$. Then compute

$$E_{\text{val}}(g^-) = \frac{1}{K} \sum_{\mathbf{x}_n \in \mathcal{D}_{\text{val}}} e(g^-(\mathbf{x}_n), y_n).$$

Three error types now sit on a spectrum:

| | Source | Status |
|---|---|---|
| $E_{\text{in}}$ | $\mathcal{D}$ | Feasible, **contaminated** (used for training) |
| $E_{\text{val}}$ | $\mathcal{D}_{\text{val}} \subset \mathcal{D}$ | Feasible, **clean** (held out before training) |
| $E_{\text{test}}$ | $\mathcal{D}_{\text{test}}$ | Infeasible, **clean** (locked away) |

The validation set is the on-hand simulation of the test set. **Discipline is critical**: once $\mathcal{D}_{\text{val}}$ is used to select something, it's no longer clean for any subsequent decision.

### Mean and Variance

Two key statistical properties of $E_{\text{val}}$:

**Unbiased.** $\mathbb{E}_{\mathcal{D}_{\text{val}}}[E_{\text{val}}(g^-)] = E_{\text{out}}(g^-)$. The validation error is an unbiased estimate of out-of-sample error. The proof is two lines of linearity-of-expectation, requiring only that $\mathcal{D}_{\text{val}}$ wasn't used to train $g^-$.

**Variance shrinks as $1/K$.** For i.i.d. validation samples,

$$\sigma_{\text{val}}^2 = \frac{\sigma^2(g^-)}{K} \leq \frac{1}{4K}$$

— the bound saturates when per-example errors are Bernoulli(1/2) (worst case for binary classification, since $p(1-p) \leq 1/4$).

### The K Trade-Off

$K$ large is good for the variance bound but bad for the training set:

$$E_{\text{out}}(g) \;\underbrace{\approx}_{\text{small } K} \; E_{\text{out}}(g^-) \;\underbrace{\approx}_{\text{large } K} \; E_{\text{val}}(g^-).$$

- Small $K$: $g^-$ trained on nearly $N$ examples is close to $g$. But $E_{\text{val}}$ has high variance — the estimate is noisy.
- Large $K$: $E_{\text{val}}$ is precise. But $g^-$ trained on $N - K$ examples is much worse than the $g$ you'd actually deploy.

The expected $E_{\text{val}}$ as a function of $K$ has wide error bars at both extremes, with a sweet spot in the middle. **Practical rule of thumb: $K = N/5$.**

> [!tip]+ TIP — Always retrain on all $N$ before reporting
> Validation chooses the hypothesis class $\mathcal{H}_{m^\ast}$. After that decision, retrain on $\mathcal{D}$ (all $N$ examples) to produce the final $g_{m^\ast}$. More data → better generalisation: $E_{\text{out}}(g_{m^\ast}) \leq E_{\text{out}}(g_{m^\ast}^-)$. Reporting $g^-$ instead is leaving signal on the table.

### Validation for Model Selection

The protocol:

1. Split $\mathcal{D} \to \mathcal{D}_{\text{train}}, \mathcal{D}_{\text{val}}$.
2. For each candidate $m = 1, \ldots, M$: train $g_m^- = \mathcal{A}_m(\mathcal{D}_{\text{train}})$, compute $E_m = E_{\text{val}}(g_m^-)$.
3. Pick $m^\ast = \arg\min_m E_m$.
4. Retrain $g_{m^\ast}$ on all of $\mathcal{D}$. Report it.

The generalisation guarantee:

$$E_{\text{out}}(g_{m^\ast}) \leq E_{\text{out}}(g_{m^\ast}^-) \leq E_{\text{val}}(g_{m^\ast}^-) + O\left(\sqrt{\frac{\log M}{K}}\right).$$

The $\log M$ rather than $M$ is benign — even a thousand candidates costs $\log 1000 \approx 7$ in the numerator.

> [!question]- Why does selecting by $E_{\text{val}}$ work, while selecting by $E_{\text{in}}$ does not?
> Selecting by $E_{\text{val}}$ uses an *unbiased estimate* of $E_{\text{out}}(g_m^-)$ for each candidate — comparing apples to apples. Selecting by $E_{\text{in}}$ uses a *biased* estimate (the data was used to fit $g_m$, so $E_{\text{in}}$ is downward-biased). Worse, the bias depends on hypothesis-set complexity: more expressive classes get more $E_{\text{in}}$ shrinkage, so the comparison systematically favours overcomplex models. Validation breaks the cycle by evaluating on data the algorithm hasn't seen.

### Validation vs Regularisation

Both attack the same equation:

$$E_{\text{out}}(h) = E_{\text{in}}(h) + \text{overfit penalty}.$$

- **Regularisation** estimates the *overfit penalty* directly (the augmented error $(\lambda/N) \Omega(\mathbf{w})$ proxies the term).
- **Validation** estimates *$E_{\text{out}}$ directly* (held-out data bypasses the penalty entirely).

Two complementary cures, two sides of the same equation. Regularisation gives you a family of trained models indexed by $\lambda$; validation picks the best $\lambda$ from that family. **Together they're a complete model-selection pipeline.**

## Part 2: Cross-Validation — Use All The Data

Single-split validation has a frustrating constraint: data spent on $\mathcal{D}_{\text{val}}$ isn't available for training. **Cross-validation** removes the trade-off by **using every example for both training and validation, just not at the same time**.

### Leave-One-Out

The extreme: for each $n \in \{1, \ldots, N\}$, hold out example $n$ alone and train on the remaining $N - 1$:

$$g_n^- = \mathcal{A}(\mathcal{D} \setminus \{(\mathbf{x}_n, y_n)\}), \qquad e_n = e(g_n^-(\mathbf{x}_n), y_n).$$

Average:

$$\boxed{E_{\text{cv}} = \frac{1}{N} \sum_{n=1}^N e_n.}$$

**Theorem.** $E_{\text{cv}}$ is an unbiased estimate of $\bar{E}_{\text{out}}(N - 1)$ — the expected out-of-sample error when training with $N - 1$ examples. The proof factorises the expectation: $\mathbb{E}_{\mathcal{D}}[e_n] = \mathbb{E}_{\mathcal{D} \setminus \{n\}} \mathbb{E}_{(\mathbf{x}_n, y_n)}[e_n] = \mathbb{E}_{\mathcal{D} \setminus \{n\}}[E_{\text{out}}(g_n^-)] = \bar{E}_{\text{out}}(N - 1)$.

For practical purposes — since $N - 1 \approx N$ for moderate $N$ — practitioners say $E_{\text{cv}}$ is "almost unbiased" for $E_{\text{out}}(g)$.

### The Computational Catch

LOOCV trains the model $N$ times. For a deep network, infeasible. **Special case**: linear regression has a closed-form LOOCV via the hat matrix:

$$E_{\text{cv}} = \frac{1}{N} \sum_{n=1}^N \left( \frac{\hat{y}_n - y_n}{1 - H_{n,n}(\lambda)} \right)^2$$

where $\mathbf{H}(\lambda) = \mathbf{A}(\mathbf{A}^\top \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^\top$. One ridge fit gives all $N$ leave-one-out errors. Useful trick — but doesn't generalise beyond linear regression.

### V-Fold: The Practical Compromise

Partition $\mathcal{D}$ into $V$ equal-size parts. For each fold $v$, train on the other $V - 1$ parts and validate on the $v$-th. Average the $V$ per-fold errors:

$$E_{\text{cv}}(\mathcal{H}, \mathcal{A}) = \frac{1}{V} \sum_{v=1}^V E_{\text{val}}^{(v)}(g_v^-).$$

LOOCV is the special case $V = N$. **Practical rule of thumb: $V = 10$**.

> [!info]+ ASIDE — Why 10-fold is the de facto standard
> 10-fold CV gives an estimate of $E_{\text{out}}$ that is statistically nearly identical to LOOCV (since each training set is $0.9N$, very close to $N - 1$ for large $N$), but at $1/100$th of LOOCV's compute for $N = 1000$. The variance is also often *lower* than LOOCV, because LOOCV's $N$ per-fold errors are highly correlated (training sets differ in 2 examples). 10-fold strikes the bias-variance-compute trade-off well enough that you should usually start there and only deviate with a specific reason.

### Cross-Validation for Hyperparameter Selection

The standard recipe for choosing $\lambda$ in [[ridge-regression|ridge]] or [[lasso-regression|lasso]]:

1. Pick a grid of candidate $\lambda$'s: e.g. $\{0.001, 0.01, 0.1, 1, 10, 100\}$.
2. For each $\lambda$, compute $E_{\text{cv}}(\mathcal{H}_\lambda, \mathcal{A}_\lambda)$ via 10-fold CV.
3. Pick $\lambda^\ast = \arg\min E_{\text{cv}}$.
4. Retrain on all of $\mathcal{D}$ with $\lambda^\ast$. Report.

This is the practical answer to "how do you pick $\lambda$?" — and it's been waiting since week 8 when ridge was first introduced.

## Part 3: Three Principles for Honest Learning

The module's theoretical machinery — VC bound, bias–variance, regularisation, validation — all rests on certain *assumptions*. The Tuesday lecture wraps up by formalising three meta-principles whose violation breaks every theorem.

### Occam's Razor — The Simplest Model That Fits Wins

The principle, paraphrased from William of Occam (1287–1347): *entities must not be multiplied beyond necessity.*

In ML terms: among hypotheses that fit the data, prefer the simplest. Two senses of "simple":

- **Simple hypothesis $h$**: small $\Omega(h)$ — few parameters.
- **Simple hypothesis set $\mathcal{H}$**: small $\Omega(\mathcal{H})$ — small VC dimension.

The two are related: a hypothesis drawn from a low-complexity set is automatically a low-complexity hypothesis. Both senses point toward the same practical advice: **start linear, then ask whether the data is being over-modelled before adding capacity**.

"Better" in this context means **better $E_{\text{out}}$**, not aesthetic elegance. The VC bound's $\Omega(\mathcal{H})$ term is smaller for simpler classes, so when $E_{\text{in}}$ is comparable, simpler wins.

### Sampling Bias — Wrong Distribution, No Recovery

> *If the data is sampled in a biased way, learning will produce a similarly biased outcome.*

The VC bound's i.i.d. assumption is load-bearing: **training and test data must come from the same joint distribution**. Violate it and the bound is silent — there's no theorem that says $E_{\text{out}}$ on the deployment distribution is close to $E_{\text{in}}$ on a different one.

The lecture's philosophical version: studying maths hard but being tested on English gives no strong test-performance guarantee. The classic real-world example: a 1936 magazine poll based on land-line phone owners predicted Landon over Roosevelt in a landslide; Roosevelt won by a landslide. The mismatch between the polled distribution and the voting distribution made the model useless, regardless of how "well-trained" it was on the (biased) data.

### Data Snooping — The Insidious One

> *If a data set has affected any step in the learning process, its ability to assess the outcome has been compromised.*

The strongest of the three. **Any** influence of the test set on any decision — preprocessing, feature engineering, hyperparameter choice — counts. Once the test set has been "snooped," it's no longer an unbiased estimator of $E_{\text{out}}$.

The lecture's vivid example: a financial trading strategy trained with snooping (test-period statistics used to normalise data) shows a cumulative profit of $30\%+$. The same strategy with proper normalisation (training-only) loses money. The "model" learned the test set's statistics, not signal.

The clean workflow:

1. Split $\mathcal{D} \to \mathcal{D}_{\text{train}}, \mathcal{D}_{\text{test}}$ at the very start.
2. Lock $\mathcal{D}_{\text{test}}$ in the safe. Don't normalise with it. Don't peek at it.
3. Use $\mathcal{D}_{\text{train}}$ for everything (with internal validation/CV splits as needed).
4. **One** evaluation on $\mathcal{D}_{\text{test}}$ at the end.

If you decide to "try one more thing" after that single test evaluation, the test set is now contaminated — for the next round, you'd need a fresh held-out set.

> [!warning] COMMON MISCONCEPTION — Validation = test
> Validation and test sets play different roles. **Validation** is part of the training pipeline; you use it to select hyperparameters, and once you've done so, $E_{\text{val}}$ is no longer a clean estimate of $E_{\text{out}}$. **Testing** is the final, untouched evaluation. If you've cross-validated to pick $\lambda$ and the cross-validation error was 0.05, that's the *selection* criterion, not an honest estimate of $E_{\text{out}}$ for the chosen $\lambda$. To honestly report $E_{\text{out}}$, you'd need a fresh held-out test set never used for selection. Many published papers conflate these.

### How the Three Interact

| Violation | Effect |
|---|---|
| Occam's Razor (model too complex) | Overfitting: $E_{\text{out}} \gg E_{\text{in}}$, bound is loose |
| Sampling Bias (wrong distribution) | $E_{\text{out}}$ on deployment $\gg E_{\text{out}}$ on training distribution; bound silent |
| Data Snooping (test set used) | Reported $E_{\text{test}}$ is optimistic; true $E_{\text{out}}$ is worse than reported |

All three corrupt the inference chain from training to deployment. The discipline of the three principles is what turns the module's machinery — VC, bias-variance, regularisation, validation — into honest learning.

---

## Concepts Introduced This Week

- [[validation]] — held-out subset of $\mathcal{D}$ for estimating $E_{\text{out}}$ unbiasedly. Variance $\leq 1/(4K)$ for binary classification. The standard tool for picking hyperparameters.
- [[cross-validation]] — generalises validation to use all the data: LOOCV with $N$ folds (almost unbiased; $N$ trainings), $V$-fold with $V$ folds ($V = 10$ practical default).
- [[learning-principles]] — three meta-principles: **Occam's Razor** (simpler wins on $E_{\text{out}}$), **Sampling Bias** (wrong distribution kills every guarantee), **Data Snooping** (any test-set influence contaminates).

## Connections

- **Builds on** [[week-10]]: regularisation produces a family of trained models indexed by $\lambda$; this week's validation/CV is *how* you actually choose $\lambda$. The two together complete the practical model-selection pipeline.
- **Builds on** [[week-09]]: the VC bound's three assumptions (bounded complexity, i.i.d. data, independent test set) become the three learning principles. The principles are the practical-discipline counterpart to the bound's mathematical hypotheses.
- **Builds on** [[generalization-bound]]: the validation generalisation bound $E_{\text{out}} \leq E_{\text{val}} + O(\sqrt{\log M / K})$ has the same Hoeffding-union-bound structure but with $M$ small (a finite candidate list) so the term is benign — far tighter than the $\sqrt{\log m_{\mathcal{H}}(2N)/N}$ structural bound.
- **Closes the module**: with validation, regularisation, and the three principles, we now have everything needed to take a problem from "data on disk" to "trained model deployed responsibly".

## Open Questions

- **The 1-SE rule**: when the cross-validation curve has multiple $\lambda$'s within one standard error of the minimum, which to pick? The "1-SE rule" picks the largest $\lambda$ within 1 SE of the min — preferring slightly more regularisation for robustness. Heuristic, common in practice.
- **Nested cross-validation**: when you both *select* via CV and *evaluate* via CV, the outer CV's error is not quite an unbiased estimate of $E_{\text{out}}$ for the chosen model. Nested CV (outer for evaluation, inner for selection) fixes this rigorously but at $V_1 \cdot V_2$ training cost.
- **Distribution shift in deployment**: the i.i.d. assumption is the sampling-bias principle's mathematical avatar, and it's violated almost everywhere in production ML. Active research area: distributionally robust optimisation, domain adaptation, online learning.
- **Implicit data snooping in benchmark culture**: if many researchers test on the same benchmark and publish only the wins, the community-level reported numbers are biased downward — even though no individual experiment snooped. How to honestly evaluate a field's progress is genuinely hard.
