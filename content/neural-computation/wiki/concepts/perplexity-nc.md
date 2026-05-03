---
name: perplexity
description: The standard intrinsic evaluation metric for any language model. Defined as the inverse probability of a held-out test set, normalised (geometric mean) by the number of words. Lower is better. Equivalent to "average branching factor" — how many equally-likely continuations the model considers at each step. Range is $[1, \infty)$; minimising perplexity equals maximising test-set probability.
type: concept
sources:
  - raw/week-09/w09-l20-transcript.txt
  - raw/week-09/w09-slides.pdf
status: stable
updated: 2026-04-27
---

*Perplexity is the standard yardstick for language models — used to compare n-gram LMs against neural LMs against modern LLMs on identical terms. It is the inverse probability of a held-out test set, normalised by the number of words. The intuition: a good language model is one that finds the held-out text unsurprising — it assigns high probability to the words that actually occurred. Perplexity is just that "expected probability" rescaled into a single, readable number where lower is better and where the absolute value carries an intuitive meaning (the average branching factor).*

## The definition

Given a held-out test set $W = w_1, w_2, \ldots, w_N$ of $N$ words, the **perplexity** of a language model under that test set is:

$$\text{PP}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N} = \sqrt[N]{\frac{1}{P(w_1, \ldots, w_N)}}$$

By the chain rule, this expands to:

$$\text{PP}(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i \mid w_1, \ldots, w_{i-1})}}$$

For a bigram model:

$$\text{PP}(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i \mid w_{i-1})}}$$

The structure is: take the inverse probability that the model assigned to the test set, then take the $N$-th root to make it a per-word geometric mean.

## Why these specific operations

Three design choices, each of which fixes a specific problem:

- **Inverse, not raw probability.** Probability lives in $[0, 1]$ where smaller is "less likely" — counter-intuitive when "less likely = worse model" since you want a metric where smaller = better. Inverting flips the polarity: now higher inverse probability = more surprising = worse model. Range becomes $[1, \infty)$.
- **$N$-th root for per-word normalisation.** Sentence probabilities are products over words, so they shrink exponentially with sentence length. Comparing models on test sets of different sizes would reward shorter texts. The geometric mean (the $N$-th root of the product) gives a *per-word* score that's invariant to test set length.
- **Geometric, not arithmetic, mean.** Because the underlying quantity is a *product* of probabilities, the right "average" is geometric, not arithmetic. Equivalently: perplexity is $e^H$ where $H$ is the per-word cross-entropy under the model — the original information-theoretic definition (Shannon).

> [!info] ASIDE — Why "perplexity"?
> The metric measures how *perplexed* the model is by the test set: a perplexed model assigns low probability (high inverse probability) to many words it should have predicted. The naming intuition fits the Shannon Game: if asked to fill in *"once upon a ____"*, a fluent English speaker is barely perplexed (almost certainly *"time"*); asked to fill in *"That is a picture of a ____"* without seeing the picture, the speaker has thousands of plausible continuations and is highly perplexed. Lower perplexity ↔ less hesitation ↔ better predictions.
>
> The lecturer notes that "perplexity" was a niche LM-evaluation term until the company Perplexity AI made it a household word — but its origin is this metric.

## The intuition: average branching factor

Perplexity has a direct interpretation as the **average effective branching factor** of the model. If $\text{PP}(W) = k$, the model is, on average across positions, "as uncertain as if it were choosing uniformly among $k$ candidate words."

| Setting | Perplexity | Interpretation |
|---|---|---|
| Uniform model over $|V|=10^5$ | $10^5$ | Knows nothing — every word equally likely |
| Unigram on Wall Street Journal | 962 | Knows word frequencies; not much else |
| Bigram on WSJ | 170 | Knows pairwise patterns |
| Trigram on WSJ | 109 | Adds local syntactic context |
| Modern LLM on standard test sets | 70–80 | State-of-the-art |
| Lower bound | 1 | Perfect predictions (probability 1 on each word) |

The progression (uniform → unigram → bigram → trigram → modern LLM) shows what each level of structure buys you. Going from a $10^5$-word vocabulary to perplexity 70 means the model effectively narrows the next-word distribution from "any word" to "one of ~70 plausible continuations" on average — a $\sim 1400\times$ reduction in uncertainty. That's the impact of all the LM machinery built on top of the basic chain rule.

> [!tip] TIP — Holding the test set constant
> Perplexity is only comparable *between models on the same test set*. Comparing perplexities computed on different test sets is meaningless — different sets have different underlying entropy, and a "harder" set legitimately yields higher perplexity for any model. Standard benchmarks (WSJ, WikiText, Penn Treebank) exist so progress is measurable across papers.

## Intrinsic vs. extrinsic evaluation

Two ways to evaluate a language model:

- **Extrinsic (in-vivo) evaluation.** Plug the LM into a real downstream task (machine translation, speech recognition, autocomplete) and measure end-to-end performance — translation BLEU score, ASR word error rate, etc. Two LMs are compared by which makes the downstream task work better.
- **Intrinsic (in-vitro) evaluation.** Compare LMs directly on probability they assign to held-out text — perplexity.

Extrinsic is "more honest" (it measures the thing you actually care about), but expensive and sometimes infeasible — you'd need a full machine translation pipeline trained on top of each LM, and the results may not transfer to other downstream tasks. Perplexity is cheap, model-agnostic (works for n-gram and neural LMs identically), and provides a single number for ranking. The two correlate but not perfectly: a model can have lower perplexity but worse downstream performance because perplexity weights all words equally and downstream tasks may care disproportionately about specific word categories.

## Practical considerations

- **Never compute by hand.** The naive product of probabilities underflows to zero on test sets of more than ~50 tokens. Compute log-probabilities instead and take $\text{PP}(W) = \exp\!\left(-\frac{1}{N} \sum_i \log P(w_i \mid \text{context})\right)$.
- **Zero probability = infinite perplexity.** If any word in the test set was assigned probability zero (the [[n-gram-language-model#The zeros problem|zeros problem]]), perplexity is undefined / infinity. This is a hard motivation for smoothing in n-gram LMs and (more elegantly) for the smoothness of neural LM output distributions.
- **Vocabulary mismatch is a trap.** Two LMs with different vocabularies cannot be compared by perplexity directly — the one with the smaller vocabulary has an artificially easier task. Always normalise vocabulary or use a common tokenisation.
- **Per-token vs. per-word perplexity.** Modern LLMs are trained on subword tokens (BPE, WordPiece), so reported perplexity is per-token. Rescaling to per-word requires knowing the average tokens-per-word ratio of the test set.

## Connections

- **Evaluates** [[language-model|language models]] of any architecture — the metric is parameterisation-agnostic.
- **Defined relative to** [[n-gram-language-model|n-gram models]] — first widely-used context, but applies identically to neural LMs.
- **Information-theoretic siblings**: perplexity = $e^H$ where $H$ is per-word cross-entropy. Cross-entropy is also the natural training loss (negative log-likelihood). So *minimising training loss* and *minimising perplexity* are the same operation under different names — one viewed as a training objective, the other as an evaluation metric.
