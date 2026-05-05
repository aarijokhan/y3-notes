---
type: concept
name: cross-validation
description: Repeating validation across multiple held-out subsets and averaging — a way to use *all* the data for both training and validation. LOOCV uses $N$ folds of size 1; V-fold uses $V$ folds of size $N/V$, typically $V = 10$. Lower-variance estimate of $E_{\text{out}}$ than a single validation split.
sources:
  - raw/week-11/Wk_11_Lec_1.pdf
  - raw/week-11/Monday_ December 8_ 2025 at 4_03_57 PM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*Repeating the [[validation]] procedure across multiple held-out subsets ("folds") and averaging the per-fold validation errors. **Leave-one-out CV** (LOOCV) holds out one example at a time, training on $N - 1$ and evaluating on 1, repeated $N$ times: $E_{\text{cv}} = \frac{1}{N} \sum_n e_n$. **V-fold CV** partitions $\mathcal{D}$ into $V$ equal parts, trains on $V - 1$ of them, validates on the remaining one, repeats $V$ times. Typical $V = 10$. $E_{\text{cv}}$ is an (almost) unbiased estimate of $E_{\text{out}}$, and dramatically lower-variance than a single train/val split.*

## The Motivation

Single-split [[validation]] forces a trade-off: large $K$ means a precise validation estimate but a poorly-trained $g^-$; small $K$ means a well-trained $g^-$ but a noisy validation estimate. Cross-validation circumvents the trade-off by **using every example for both training and validation, just not at the same time**.

The cost: more computation (training $V$ or $N$ times instead of once) — but one estimate of $E_{\text{out}}$ that's both unbiased and low-variance.

## Leave-One-Out Cross-Validation (LOOCV)

For each $n = 1, \ldots, N$:

1. Define $\mathcal{D}_n = \mathcal{D} \setminus \{(\mathbf{x}_n, y_n)\}$ — all data except example $n$.
2. Train: $g_n^- = \mathcal{A}(\mathcal{D}_n)$.
3. Compute the single-point error: $e_n = e(g_n^-(\mathbf{x}_n), y_n)$.

Average over all $N$ runs:

$$\boxed{E_{\text{cv}} = \frac{1}{N} \sum_{n=1}^N e_n}.$$

Each $e_n$ is the validation error of a model trained on $N - 1$ examples evaluated on the held-out one. LOOCV uses every example as both training (in $N - 1$ runs) and validation (in 1 run).

### Unbiasedness

**Theorem.** $E_{\text{cv}}$ is an unbiased estimate of $\bar{E}_{\text{out}}(N - 1)$ — the expected $E_{\text{out}}$ when training with $N - 1$ points.

**Proof.** Linearity of expectation gives $\mathbb{E}_{\mathcal{D}}[E_{\text{cv}}] = \mathbb{E}_{\mathcal{D}}[e_n]$ (by symmetry, all $\mathbb{E}[e_n]$ are equal). Decompose: $\mathbb{E}_{\mathcal{D}}[e_n] = \mathbb{E}_{\mathcal{D}_n} \mathbb{E}_{(\mathbf{x}_n, y_n)}[e(g_n^-(\mathbf{x}_n), y_n)] = \mathbb{E}_{\mathcal{D}_n}[E_{\text{out}}(g_n^-)] = \bar{E}_{\text{out}}(N - 1)$. The inner expectation is unbiasedness of the validation error for $g_n^-$ (since $\mathcal{D}_n$ doesn't contain example $n$); the outer averages over the random training set of size $N - 1$.

**Almost unbiased for $\bar{E}_{\text{out}}(N)$**: practitioners say $E_{\text{cv}}$ is "almost unbiased" for $E_{\text{out}}(g)$ — strictly it estimates $\bar{E}_{\text{out}}(N - 1)$, but for moderate $N$ the gap between $\bar{E}_{\text{out}}(N - 1)$ and $\bar{E}_{\text{out}}(N)$ is tiny.

### Variance

The variance of $E_{\text{cv}}$ is harder to analyse: the $N$ training sets $\mathcal{D}_n$ overlap (each pair shares $N - 2$ examples), so the $e_n$'s aren't independent. In practice $E_{\text{cv}}$ has low variance — comparable to a much larger validation set — but not provably bounded by $1/(4N)$ as a single-split would suggest.

### Disadvantage: $N$ Trainings

$$E_{\text{cv}}(\mathcal{H}, \mathcal{A}) = \frac{1}{N} \sum_{n=1}^N e_n$$

requires training the model $N$ times. For a deep network with hours-long training, this is infeasible. **Special case**: linear regression has a closed-form LOOCV that costs the same as one fit. The hat matrix $\mathbf{H}(\lambda) = \mathbf{A}(\mathbf{A}^\top \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^\top$ gives:

$$E_{\text{cv}} = \frac{1}{N} \sum_{n=1}^N \left( \frac{\hat{y}_n - y_n}{1 - H_{n,n}(\lambda)} \right)^2$$

— evaluating LOOCV by re-using the original fit, scaled by per-point leverage. This trick works for ridge regression but rarely generalises.

## V-Fold Cross-Validation

Partition $\mathcal{D}$ into $V$ equal-size parts $\mathcal{D}^{(1)}, \ldots, \mathcal{D}^{(V)}$. For each fold $v$:

1. Train: $g_v^- = \mathcal{A}(\mathcal{D} \setminus \mathcal{D}^{(v)})$.
2. Validate: $E_{\text{val}}^{(v)}(g_v^-)$ on $\mathcal{D}^{(v)}$.

Average over folds:

$$E_{\text{cv}}(\mathcal{H}, \mathcal{A}) = \frac{1}{V} \sum_{v=1}^V E_{\text{val}}^{(v)}(g_v^-).$$

LOOCV is the special case $V = N$. **Practical rule of thumb: $V = 10$** — a good balance between estimate quality and computational cost.

### Why V-Fold Is Usually Preferred Over LOOCV

| Aspect | LOOCV ($V = N$) | V-fold ($V = 10$) |
|---|---|---|
| Trainings | $N$ — often infeasible | $V$ — manageable |
| Training set size | $N - 1$ | $N - N/V \approx N$ |
| Validation set size per fold | 1 — high per-fold variance | $N/V$ — moderate |
| Bias as estimate of $E_{\text{out}}(N)$ | Smaller (each fold uses $N-1$) | Slightly larger (each fold uses $N - N/V$) |
| Variance | Often high (correlated $e_n$'s) | Moderate |

For most practical problems, $V = 10$ gives essentially the same error estimate as LOOCV at a fraction of the cost. **Don't trade $V$-fold for LOOCV unless you have a reason to.**

## Cross-Validation for Model Selection

Same protocol as single-split [[validation]], but each candidate is scored by $E_{\text{cv}}$ instead of $E_{\text{val}}$:

1. For each candidate $\mathcal{H}_m$ (or each $\lambda$), compute $E_{\text{cv}}(\mathcal{H}_m, \mathcal{A}_m)$.
2. Pick $m^\ast = \arg\min_m E_{\text{cv}}(\mathcal{H}_m, \mathcal{A}_m)$.
3. **Retrain $g_{m^\ast}$ on all $N$ examples** and report it.

This is how you actually choose the regularisation parameter $\lambda$ for ridge or lasso — try a grid, score each via cross-validation, pick the minimum.

## Choosing $V$

| $V$ | Use |
|---|---|
| $V = 5$ | Quick checks, large datasets |
| $V = 10$ | Default — strikes the balance |
| $V = N$ (LOOCV) | Small $N$, or when LOOCV has a closed form (e.g., linear regression) |
| $V$ large but $< N$ | Specific computational constraints |

The choice rarely matters much for moderate $N$; the standard advice is **start with 10-fold and only deviate with reason**.

## Related

- [[validation]] — the single-split version that cross-validation generalises.
- [[regularization-ml|regularization]] — the typical hyperparameter family that cross-validation chooses among.
- [[ridge-regression]] — the special case where LOOCV has a closed form via the hat matrix.
- [[generalization-bound]] — the worst-case theory that cross-validation refines empirically.

## Active Recall

> [!question]- Define **leave-one-out cross-validation** error and explain why it requires $N$ separate training runs.
> $E_{\text{cv}} = \frac{1}{N} \sum_{n=1}^N e(g_n^-(\mathbf{x}_n), y_n)$, where $g_n^-$ is the model trained on $\mathcal{D} \setminus \{(\mathbf{x}_n, y_n)\}$ — all data except example $n$. Each $g_n^-$ is a *different* fit (different training set), so producing all $N$ values $e_n$ requires running the learning algorithm $N$ times. This is computationally expensive for any model whose training takes more than seconds. The exception is linear regression, where the LOOCV error has a closed-form expression in terms of the hat matrix that re-uses the single fit on all $N$ points.

> [!question]- Why is LOOCV said to be "almost unbiased" for $E_{\text{out}}(g)$ rather than exactly unbiased?
> Strictly, the theorem says $\mathbb{E}_{\mathcal{D}}[E_{\text{cv}}] = \bar{E}_{\text{out}}(N - 1)$ — the expected out-of-sample error when training with $N - 1$ examples. The hypothesis we actually report, $g$, is trained on all $N$ examples, so its expected error is $\bar{E}_{\text{out}}(N)$, which is slightly *smaller* than $\bar{E}_{\text{out}}(N - 1)$ (more data → less error on average). The gap is small for moderate $N$ — typically the difference is dominated by the noise in $E_{\text{cv}}$ itself — so practitioners treat LOOCV as essentially unbiased for $E_{\text{out}}(g)$.

> [!question]- For 10-fold cross-validation on a dataset of $N = 1{,}000$ examples, how many examples are used to train each fold's model, how many to validate, and how many total trainings are required?
> $V = 10$, so each fold has $N/V = 100$ validation examples and $N - N/V = 900$ training examples. The full procedure trains the model $V = 10$ times, once per fold, each time on a different 900-example subset. The final $E_{\text{cv}}$ is the average of 10 per-fold validation errors. After model selection picks the best $\lambda$, you'd typically retrain once more on all 1000 examples to produce the final hypothesis.

> [!question]- Why is $V$-fold cross-validation usually preferred over LOOCV in practice?
> Three reasons:
> - **Cost**: V-fold requires $V$ trainings (typically 10), LOOCV requires $N$ (often hundreds to millions). For non-trivial models, LOOCV is infeasible.
> - **Variance**: LOOCV's $N$ per-fold errors are highly correlated (each pair of training sets differs in only 2 examples), so its variance can be paradoxically *higher* than V-fold for some problems.
> - **Bias**: V-fold's training sets are slightly smaller than LOOCV's, but for moderate $N$ the bias gap is negligible.
>
> The default recommendation is $V = 10$ — only switch to LOOCV when you have a closed-form trick (linear regression) or specific reason.
