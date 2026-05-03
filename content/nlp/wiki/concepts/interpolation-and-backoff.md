---
type: concept
sources:
  - raw/week-02/w02-slides.pdf
  - raw/week-02/w02-l06-transcript.txt
status: draft
updated: 2026-04-20
---

*Interpolation and backoff combine higher-order and lower-order n-gram models to handle unseen contexts; interpolation mixes all orders simultaneously, while backoff falls through to lower orders only when higher orders fail.*

## Motivation

[[Smoothing|Add-one smoothing]] is too aggressive for [[n-gram-language-models]]: it makes the probability distribution nearly uniform. We need a more principled way to handle unseen n-grams.

The core idea: when a trigram has never been seen, the bigram or unigram might still carry useful information. Rather than adding pseudo-counts, **use lower-order models as backup**. There are two ways to do this:

- **Interpolation** — always blend all orders together. Every prediction is a weighted mix of the trigram, bigram, and unigram models, regardless of whether the higher-order count is zero.
- **Backoff** — use the highest-order model that has evidence. If the trigram count is zero, fall through to the bigram; if that is also zero, fall through to the unigram.

## Linear Interpolation

Mix the predictions of unigram, bigram, and trigram models (i.e. take a weighted sum — a linear combination):

$$\hat{P}(w_n \mid w_{n-2}\, w_{n-1}) = \lambda_1 \cdot P(w_n \mid w_{n-2}\, w_{n-1}) + \lambda_2 \cdot P(w_n \mid w_{n-1}) + \lambda_3 \cdot P(w_n)$$

where $\lambda_1, \lambda_2, \lambda_3$ are **weights** controlling how much each model contributes to the final prediction. A higher $\lambda$ means more trust in that model order. They must satisfy $\lambda_1 + \lambda_2 + \lambda_3 = 1$ and $\lambda_i \geq 0$, so the result is still a valid probability.

All three models contribute to every prediction. Even if the trigram count is zero (contributing $\lambda_1 \times 0$), the bigram and unigram terms ensure $\hat{P} > 0$ (assuming the unigram is always non-zero, which it is for any word in the vocabulary).

### Context-dependent lambdas

With fixed lambdas, the same weights apply to every prediction regardless of how reliable the trigram estimate is. But reliability depends on how often the specific context has been seen: a common context like "New York ___" has many observed trigrams and a trustworthy estimate; a rare context like "zygote morphology ___" has almost none and is noisy. Context-dependent lambdas adapt to this — when the context is common, weight the trigram more heavily; when it is rare, lean more on the bigram and unigram.

The lambdas can vary based on context:

$$\hat{P}(w_n \mid w_{n-2}\, w_{n-1}) = \lambda_1(w_{n-2:n-1}) \cdot P(w_n \mid w_{n-2}\, w_{n-1}) + \lambda_2(w_{n-2:n-1}) \cdot P(w_n \mid w_{n-1}) + \lambda_3(w_{n-2:n-1}) \cdot P(w_n)$$

If the bigram context $(w_{n-2}, w_{n-1})$ is common, the trigram estimate is reliable and $\lambda_1$ should be large. If the context is rare, the trigram estimate is noisy and $\lambda_2, \lambda_3$ should dominate.

### How to set the lambdas?

The $\lambda$ values cannot be learned from the training data directly — the model would just put all weight on the highest-order n-gram since that is what the training counts favour. Instead, they are set using a separate **held-out set**:

1. Compute n-gram probabilities on the **training set** (these are fixed).
2. Hold out a separate **held-out set** (drawn from the same distribution as training and test).
3. Search for the $\lambda$ values that **maximize the probability** of the held-out set.
4. Fix those $\lambda$ values.
5. Evaluate [[perplexity-nlp|perplexity]] on the **test set** once.

This is the same logic as the train/dev/test split from [[evaluation-methodology]]: the held-out set plays the role of the dev set for the interpolation weights.

## Backoff

Backoff uses only one model order at a time, starting from the highest:

1. If $C(w_{n-2}, w_{n-1}, w_n) > 0$: use the trigram probability $P(w_n \mid w_{n-2}, w_{n-1})$.
2. Else if $C(w_{n-1}, w_n) > 0$: back off to bigram $P(w_n \mid w_{n-1})$.
3. Else: back off to unigram $P(w_n)$.

### The discounting requirement

Backoff has a subtle problem: the trigram probabilities for observed trigrams already sum to something close to 1. If we use them as-is and then also assign probability to unseen trigrams via backoff, the total exceeds 1 — an invalid probability distribution.

The fix: **discount** the probabilities of observed n-grams (multiply them by a factor $< 1$) to free up probability mass for the backed-off cases. This is the essence of **Katz backoff**, which applies a discounting factor (e.g., 0.4) before backing off.

### Stupid backoff

A simpler variant that skips the discounting:

$$S(w_i \mid w_{i-k+1:i-1}) = \begin{cases} \frac{C(w_{i-k+1:i})}{C(w_{i-k+1:i-1})} & \text{if } C(w_{i-k+1:i}) > 0 \\ 0.4 \times S(w_i \mid w_{i-k+2:i-1}) & \text{otherwise} \end{cases}$$

Each time you back off, multiply by 0.4. This is **not** a true probability distribution (the values may not sum to 1), but it is simple, fast, and works surprisingly well at scale — it was used by Google's Web LM with 1 trillion words of training data.

## Interpolation vs Backoff

Interpolation generally outperforms backoff because it always uses information from **all** model orders. When the trigram count is zero, backoff discards the trigram entirely and switches to the bigram. Interpolation still uses the trigram term (contributing $\lambda_1 \times 0 = 0$) but also always blends in the bigram and unigram, even when the trigram count is non-zero. This means interpolation can benefit from lower-order evidence even for well-attested trigrams.

## Related

- [[smoothing]] — the simpler approach (modify counts); interpolation and backoff are the more principled alternative
- [[n-gram-language-models]] — the model family that these techniques apply to
- [[perplexity-nlp|perplexity]] — the metric used to evaluate whether interpolation/backoff improves the model
- [[evaluation-methodology]] — held-out data for lambda estimation follows the same logic as train/dev/test

## Active Recall

> [!question]- In linear interpolation, what constraint must the lambda weights satisfy, and what happens if you set λ₁ = 1, λ₂ = λ₃ = 0?
> The lambdas must sum to 1: $\lambda_1 + \lambda_2 + \lambda_3 = 1$. Setting $\lambda_1 = 1$ makes the interpolated model identical to the trigram model alone — no bigram or unigram contribution. If the trigram has a zero count, the interpolated probability is also zero, and smoothing has achieved nothing. The whole point of interpolation is to have $\lambda_2, \lambda_3 > 0$ so that lower-order models can rescue zero-count trigrams.

> [!question]- Why does Katz backoff require discounting the probabilities of observed n-grams, whereas stupid backoff does not?
> Katz backoff aims to produce a valid probability distribution (summing to 1). The observed n-gram probabilities from MLE already sum to nearly 1, so adding probability for unseen n-grams via backoff would push the total above 1. Discounting frees up mass for the backed-off cases. Stupid backoff deliberately gives up on being a valid probability distribution — it produces scores, not probabilities. Since it makes no guarantee about summing to 1, it does not need to discount.

> [!question]- How are the interpolation lambdas learned? Why can't you simply optimize them on the training data?
> Lambdas are optimized on a held-out dataset: fix the n-gram probabilities (computed from training data), then search for the lambda values that maximize the probability of the held-out set. Optimizing on the training data would overfit — the model would learn to put all weight on the highest-order n-gram ($\lambda_1 \to 1$), since the training data contains exactly the counts the model was estimated from. The held-out set contains unseen n-grams, which forces the optimization to value the lower-order models that can handle novel contexts.

> [!question]- The lecturer says interpolation generally outperforms backoff. Explain why mixing all orders simultaneously could be better than using only the highest available order.
> Backoff uses the trigram when available and ignores bigram/unigram evidence for those cases. But a non-zero trigram count does not mean the trigram estimate is reliable — with a count of 1 or 2, the MLE is noisy. Interpolation hedges: even when the trigram is available, it blends in the bigram and unigram, which are estimated from more data and are therefore more stable. The lambdas control how much to trust each order. This robustness to noisy high-order estimates is the core advantage.
