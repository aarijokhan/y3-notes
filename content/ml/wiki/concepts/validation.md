---
type: concept
name: validation
description: Holding out a portion of the training data to estimate out-of-sample error directly. Used for model selection (choosing $\lambda$, kernel, model class) without touching the test set. Unbiased but variance-limited; the variance bound is $1/(4K)$.
sources:
  - raw/week-11/Wk_11_Lec_1.pdf
  - raw/week-11/Monday_ December 8_ 2025 at 4_03_57 PM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*Splitting the dataset $\mathcal{D}$ into a **training set** $\mathcal{D}_{\text{train}}$ ($N - K$ examples) and a **validation set** $\mathcal{D}_{\text{val}}$ ($K$ examples). Train on $\mathcal{D}_{\text{train}}$ to get $g^-$, then compute $E_{\text{val}}(g^-) = \frac{1}{K} \sum_{\mathbf{x}_n \in \mathcal{D}_{\text{val}}} e(g^-(\mathbf{x}_n), y_n)$ — an **unbiased estimate** of $E_{\text{out}}(g^-)$ with variance $\leq 1/(4K)$ for binary classification. Used for model selection by training $M$ candidates, evaluating each on $\mathcal{D}_{\text{val}}$, and picking the minimum.*

## Where Validation Sits

Three errors, three roles:

| Error | Computed from | Status |
|---|---|---|
| **In-sample** $E_{\text{in}}$ | $\mathcal{D}$ | Feasible on hand, but **contaminated** — the algorithm already used it to select $g$. |
| **Test** $E_{\text{test}}$ | $\mathcal{D}_{\text{test}}$ | Unbiased but **infeasible** — locked away, never touched. |
| **Validation** $E_{\text{val}}$ | $\mathcal{D}_{\text{val}} \subset \mathcal{D}$ | Feasible *and* clean — *if* $\mathcal{D}_{\text{val}}$ was held out before training. |

The validation set is the on-hand simulation of the test set. The discipline is strict: feed only $\mathcal{D}_{\text{train}}$ to the learning algorithm, never $\mathcal{D}_{\text{val}}$. Once $\mathcal{D}_{\text{val}}$ has been used to select hyperparameters, it's no longer "clean" for any subsequent decision.

## Mean and Variance of $E_{\text{val}}$

**Validation error** for a hypothesis $g^-$ trained on $\mathcal{D}_{\text{train}}$:

$$E_{\text{val}}(g^-) = \frac{1}{K} \sum_{\mathbf{x}_n \in \mathcal{D}_{\text{val}}} e(g^-(\mathbf{x}_n), y_n)$$

where $e$ is the pointwise error: $e(g^-(\mathbf{x}), y) = \mathbb{1}[g^-(\mathbf{x}) \neq y]$ for classification, $(g^-(\mathbf{x}) - y)^2$ for regression.

**Unbiasedness.** $E_{\text{val}}(g^-)$ is an unbiased estimate of $E_{\text{out}}(g^-)$:

$$\mathbb{E}_{\mathcal{D}_{\text{val}}}[E_{\text{val}}(g^-)] = E_{\text{out}}(g^-).$$

Proof: by linearity, $\mathbb{E}_{\mathcal{D}_{\text{val}}}\left[\frac{1}{K} \sum e(g^-(\mathbf{x}_n), y_n)\right] = \frac{1}{K} \sum \mathbb{E}_{\mathbf{x}_n}[e(g^-(\mathbf{x}_n), y_n)] = E_{\text{out}}(g^-)$.

**Variance.** With i.i.d. validation samples,

$$\sigma_{\text{val}}^2 = \frac{1}{K^2} \sum_n \text{Var}_{\mathbf{x}_n}[e(g^-(\mathbf{x}_n), y_n)] = \frac{\sigma^2(g^-)}{K}.$$

For binary classification, the per-example error is Bernoulli with $p = \mathbb{P}[g^-(\mathbf{x}) \neq y]$, so $\sigma^2(g^-) = p(1-p) \leq 1/4$, giving

$$\boxed{\sigma_{\text{val}}^2 \leq \frac{1}{4K}}.$$

As $K \to \infty$, $\sigma_{\text{val}}^2 \to 0$ and $E_{\text{val}}$ converges to $E_{\text{out}}(g^-)$.

## The K Trade-Off

Bigger $K$ is good for the variance bound but bad for the trained model. The chain of approximations is:

$$E_{\text{out}}(g) \;\underbrace{\approx}_{\text{small }K} \; E_{\text{out}}(g^-) \;\underbrace{\approx}_{\text{large }K} \; E_{\text{val}}(g^-)$$

- **Large $K$** (lots of validation data, little training): $E_{\text{val}} \approx E_{\text{out}}(g^-)$ tightly, but $g^-$ is much worse than $g$ trained on all $N$ — the validation estimate is accurate but for a poor hypothesis.
- **Small $K$**: $g^-$ is close to $g$, but $E_{\text{val}}$ has high variance — the estimate of $E_{\text{out}}(g^-)$ is noisy.

The expected $E_{\text{val}}$ as a function of $K$ has wide error bars at both extremes, with a sweet spot in the middle. **Practical rule of thumb: $K = N/5$** (an 80/20 split).

> [!tip]+ TIP — Why "report $g$ trained on all $N$" rather than $g^-$
> After model selection picks $\mathcal{H}_{m^\ast}$ via validation, **retrain on the full $N$ examples** to produce $g_{m^\ast}$. The selected hypothesis class is fixed; using all the data gives a better fit. The bound $E_{\text{out}}(g_{m^\ast}) \leq E_{\text{out}}(g_{m^\ast}^-)$ usually holds because more training data means lower expected $E_{\text{out}}$. Reporting $g^-$ is leaving signal on the table.

## Validation for Model Selection

Given $M$ candidate hypothesis sets $\mathcal{H}_1, \ldots, \mathcal{H}_M$ (different model classes, different $\lambda$ values, different kernels, etc.):

1. Split $\mathcal{D}$ into $\mathcal{D}_{\text{train}}$ ($N - K$) and $\mathcal{D}_{\text{val}}$ ($K$).
2. For each $m$: train $g_m^- = \mathcal{A}_m(\mathcal{D}_{\text{train}})$ and compute $E_m = E_{\text{val}}(g_m^-)$.
3. Pick the winner: $m^\ast = \arg\min_m E_m$.
4. **Retrain** $g_{m^\ast} = \mathcal{A}_{m^\ast}(\mathcal{D})$ on the full dataset and report it.

The generalisation guarantee for the selected model:

$$E_{\text{out}}(g_{m^\ast}) \leq E_{\text{out}}(g_{m^\ast}^-) \leq E_{\text{val}}(g_{m^\ast}^-) + O\left(\sqrt{\frac{\log M}{K}}\right).$$

The $\sqrt{\log M / K}$ term comes from a finite-bin Hoeffding union bound over the $M$ models — it's analogous to the $M$-dependence in week 8's generalisation bound, but here $M$ is small (a finite list of candidates), so the term is benign.

## Why Selection by $E_{\text{in}}$ Fails

If you select the model with smallest $E_{\text{in}}$, you always pick the most expressive class:

- $\Phi_{1126}$ always preferred over $\Phi_1$ (more flexibility);
- $\lambda = 0$ always preferred over $\lambda = 0.1$ (no regularisation).

Reason: minimising $E_{\text{in}}$ over $\mathcal{H}_1 \cup \mathcal{H}_2$ pays the VC cost of the *union*, and the more expressive class wins. **Selection by $E_{\text{in}}$ is the same as overfitting through the back door** — the algorithm will always reach for capacity it shouldn't have.

## Why Selection by $E_{\text{test}}$ Is Infeasible (and "Cheating")

The test set is locked away. If you peek at it during selection — even *once* — it's no longer a test set; it's now another validation set, and you'd need a fresh test set to assess the final model. Using $\mathcal{D}_{\text{test}}$ for model selection is a form of [[data-snooping|data snooping]] that breaks every generalisation guarantee that depended on the test set being independent.

## Validation vs Regularisation

Both [[regularization-ml|regularisation]] and validation address the same equation:

$$E_{\text{out}}(h) = E_{\text{in}}(h) + \text{overfit penalty}.$$

- **Regularisation** estimates the *overfit penalty* directly via the augmented error $E_{\text{in}} + (\lambda/N) \Omega(\mathbf{w})$ — proxying the term that would otherwise be unobservable.
- **Validation** estimates *$E_{\text{out}}$ directly* via held-out data — bypassing the penalty term entirely.

Two different sides of the same equation. They're complementary: regularisation gives you a family of models indexed by $\lambda$; validation picks the best $\lambda$ from that family.

## Related

- [[cross-validation]] — generalises validation to use *all* the data for both training and validation.
- [[regularization-ml|regularization]] — the other cure for overfitting; validation picks its hyperparameters.
- [[generalization-bound]] — the worst-case bound that motivates the need for validation.
- [[overfitting-ml|overfitting]] — the disease validation diagnoses.
- [[data-snooping]] — what you must *not* do with the validation/test sets.

## Active Recall

> [!question]- Why is $E_{\text{val}}(g^-)$ an unbiased estimate of $E_{\text{out}}(g^-)$, and what assumption does the proof require?
> By definition $E_{\text{val}}(g^-) = \frac{1}{K} \sum_{\mathbf{x}_n \in \mathcal{D}_{\text{val}}} e(g^-(\mathbf{x}_n), y_n)$. Taking expectation over the randomness of $\mathcal{D}_{\text{val}}$ and using linearity: $\mathbb{E}_{\mathcal{D}_{\text{val}}}[E_{\text{val}}(g^-)] = \frac{1}{K} \sum_n \mathbb{E}_{\mathbf{x}_n}[e(g^-(\mathbf{x}_n), y_n)] = \frac{1}{K} \cdot K \cdot E_{\text{out}}(g^-) = E_{\text{out}}(g^-)$. The proof requires $\mathcal{D}_{\text{val}}$ to be drawn i.i.d. from the same $P(\mathbf{x}, y)$ as the test data, *and* that $\mathcal{D}_{\text{val}}$ was not used to train $g^-$ (otherwise $g^-$ depends on $\mathcal{D}_{\text{val}}$ and the expectation factorisation breaks).

> [!question]- Show that the variance of $E_{\text{val}}$ for binary classification satisfies $\sigma_{\text{val}}^2 \leq 1/(4K)$, and explain what this implies for choosing $K$.
> The per-example error $\mathbb{1}[g^-(\mathbf{x}) \neq y]$ is Bernoulli with parameter $p = \mathbb{P}[g^-(\mathbf{x}) \neq y]$, so its variance is $p(1-p)$. By independence of i.i.d. samples, $\sigma_{\text{val}}^2 = \frac{1}{K^2} \sum p(1-p) = \frac{p(1-p)}{K} \leq \frac{1}{4K}$ (the maximum of $p(1-p)$ on $[0,1]$ is $1/4$, achieved at $p = 1/2$). Implication: increasing $K$ shrinks the variance of the validation estimate at rate $1/K$, but also shrinks the training set, hurting the trained $g^-$. The trade-off has a sweet spot — the rule of thumb is $K \approx N/5$.

> [!question]- After validation picks the best model class $\mathcal{H}_{m^\ast}$, why should you retrain on the full $N$ examples rather than reporting $g_{m^\ast}^-$?
> $g_{m^\ast}^-$ was trained on only $N - K$ examples; $g_{m^\ast} = \mathcal{A}_{m^\ast}(\mathcal{D})$ uses all $N$. More data typically means better generalisation, so $E_{\text{out}}(g_{m^\ast}) \leq E_{\text{out}}(g_{m^\ast}^-)$ — reporting $g^-$ leaves signal on the table. The validation set's job was to choose *which* hypothesis class to use; once that decision is locked in, there's no reason not to use the full data for the final fit.

> [!question]- A student picks the model with the smallest $E_{\text{in}}$ rather than smallest $E_{\text{val}}$. Why does this systematically pick overcomplicated models?
> Minimising $E_{\text{in}}$ over $\mathcal{H}_1 \cup \cdots \cup \mathcal{H}_M$ pays the VC cost of the *union* (which is at least as expressive as the most complex member). The most expressive class always achieves the lowest $E_{\text{in}}$ — for example, $\lambda = 0$ beats any $\lambda > 0$ on training error, and a degree-1126 polynomial beats a degree-1. The student's procedure is structurally biased toward overfitting. Validation breaks the cycle by evaluating each candidate on data the algorithm never saw — providing an *unbiased* estimate of $E_{\text{out}}$ for each.

> [!question]- The generalisation bound for validation-based selection is $E_{\text{out}}(g_{m^\ast}) \leq E_{\text{val}}(g_{m^\ast}^-) + O(\sqrt{\log M / K})$. Why does only $\log M$ (not $M$) appear, and why is this comforting in practice?
> The $\log M$ comes from a Hoeffding union bound over $M$ candidates evaluated on the validation set: $\mathbb{P}[\exists m: |E_{\text{val}}(g_m^-) - E_{\text{out}}(g_m^-)| > \epsilon] \leq M \cdot 2 e^{-2\epsilon^2 K}$. Setting equal to $\delta$ and solving gives $\epsilon = O(\sqrt{\log(M/\delta)/K})$. The logarithm is comforting because it grows very slowly: doubling the candidate list adds only $\log 2 \approx 0.69$ to the numerator. Even with thousands of $\lambda$ candidates, the validation penalty is small compared to the validation error itself.
