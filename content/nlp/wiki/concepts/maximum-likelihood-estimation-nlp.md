---
type: concept
sources:
  - raw/week-02/w02-slides.pdf
  - raw/week-03/w03-slides.pdf
status: draft
updated: 2026-04-24
---

*Maximum likelihood estimation picks parameter values that make the observed training data as probable as possible under the model. For the categorical distributions used throughout NLP, this reduces to "count and normalize."*

## The Principle

Given a model with parameters $\theta$ and training data $D = \{x_1, \ldots, x_N\}$, MLE picks the parameters that maximize the probability of $D$:

$$\hat{\theta}_{MLE} = \underset{\theta}{\arg\max}\ P(D \mid \theta)$$

"What parameter values would make me least surprised by the data I actually saw?" The data is fixed; we tune $\theta$. If the model assumes examples are independent:

$$P(D \mid \theta) = \prod_{i=1}^{N} P(x_i \mid \theta)$$

## Why It Reduces to Counting

Most NLP models estimate **categorical distributions** — probabilities over a vocabulary, a class set, or a set of transitions. MLE for a categorical always has the same closed-form solution: relative frequencies.

**Sketch.** Suppose you're estimating $P(w)$ for each word in vocabulary $V$, having observed counts $C(w)$ summing to $N$. The log-likelihood of the data is

$$\log P(D \mid \theta) = \sum_w C(w) \log \theta_w \quad \text{subject to} \quad \sum_w \theta_w = 1$$

Maximize under the simplex constraint (Lagrange multiplier) and you get

$$\hat{P}(w) = \frac{C(w)}{N}$$

The relative-frequency estimator isn't a rule of thumb — it is the MLE. Every "count this and divide by that" formula in the module is an instance of the same theorem.

## Examples Already in the Module

| Model | Parameter | MLE |
|---|---|---|
| Unigram LM | $P(w)$ | $C(w) / N$ |
| Bigram LM | $P(w_n \mid w_{n-1})$ | $C(w_{n-1}, w_n) / C(w_{n-1})$ |
| N-gram LM | $P(w_n \mid w_{n-N+1:n-1})$ | $C(w_{n-N+1:n}) / C(w_{n-N+1:n-1})$ |
| [[naive-bayes|Naive Bayes]] prior | $P(c)$ | $N_c / N_{\text{total}}$ |
| [[naive-bayes|Naive Bayes]] word likelihood | $P(w \mid c)$ | $C(w, c) / \sum_{w'} C(w', c)$ |

Each is the MLE of a categorical. **Conditional MLE** (estimating a distribution given some context, like the previous word or the class) is just unconditional MLE done separately for each conditioning context — you partition the data by context, then count and normalize within each partition. That's why the bigram formula has $C(w_{n-1})$ in the denominator: within the partition of bigrams starting with $w_{n-1}$, the MLE of $P(w_n \mid w_{n-1})$ is the relative frequency of $w_n$.

## Log-Likelihood

Because $\log$ is monotonic, $\arg\max P(D \mid \theta) = \arg\max \log P(D \mid \theta)$. You always use the log version:

- **Numerical stability** — $\prod_i P(x_i \mid \theta)$ underflows on any realistic corpus; the sum of log-probabilities stays in a sane range (this is also why n-gram LMs and [[naive-bayes|Naive Bayes]] decode in log space).
- **Gradient-friendly** — for models trained by gradient ascent (logistic regression, neural nets), log turns products into sums, so gradients decompose cleanly across examples.
- **Equivalent to cross-entropy minimization** — maximizing log-likelihood of the data under a model is the same objective as minimizing the cross-entropy between the empirical data distribution and the model distribution. The "cross-entropy loss" used to train most modern NLP systems is literally negative log-likelihood.

## The Zero-Probability Problem

MLE has a famous weakness: **if an event never appears in training, MLE assigns it probability zero.** In NLP, this is catastrophic because:

- Sentence probabilities are products of conditional probabilities. One zero anywhere zeroes the whole sentence.
- [[perplexity-nlp|perplexity]] is undefined when any test-set probability is zero.
- Even large training corpora leave most possible n-grams unseen (Heaps' Law guarantees the vocabulary keeps growing). Zeros are the rule, not the exception.

MLE's maxim — *"don't pretend you've seen what you haven't"* — is fair in the abstract but too harsh for finite training sets. The standard fix is [[smoothing]]: reserve some probability mass for unseen events. Smoothed estimators are no longer strictly MLE — they trade off training-data likelihood for held-out generalization. That trade-off is the whole game of statistical modelling.

## MLE vs MAP vs Fully Bayesian

Three ways to pick $\theta$, ordered by how much prior information they use:

- **MLE** — use only the data: $\hat{\theta} = \arg\max_\theta P(D \mid \theta)$.
- **Maximum a posteriori (MAP)** — combine data with a prior: $\hat{\theta} = \arg\max_\theta P(\theta \mid D) \propto P(D \mid \theta) P(\theta)$. Add-1 (Laplace) smoothing is exactly MAP estimation of a categorical under a uniform Dirichlet prior — as if you'd pre-observed one pseudo-count of every outcome. See [[bayes-rule]] for the inversion.
- **Fully Bayesian** — integrate over $\theta$ instead of picking a point estimate. Expensive but principled; central to topic models, Bayesian deep learning, and Gaussian processes.

For most of this module, **MLE with smoothing** is the default. The smoothing step is how MAP's prior sneaks back into an otherwise likelihood-only estimator.

## Related

- [[n-gram-language-models]] — MLE estimator for bigram, trigram, and higher-order probabilities
- [[naive-bayes]] — MLE estimator for the class prior and per-class word likelihoods
- [[smoothing]] — the standard fix for MLE's zero-probability problem
- [[perplexity-nlp|perplexity]] — the metric that collapses when any MLE estimate is zero
- [[bayes-rule]] — MAP estimation uses Bayes' rule to introduce a prior over parameters

## Active Recall

> [!question]- State the MLE principle in one sentence, then state the closed-form solution for a categorical distribution.
> **Principle**: choose the parameters that make the observed training data as probable as possible under the model — $\hat{\theta}_{MLE} = \arg\max_\theta P(D \mid \theta)$. **Closed form for a categorical** over vocabulary $V$ with counts $C(w)$ and total $N$: $\hat{P}(w) = C(w)/N$. Relative frequencies aren't a heuristic — they fall out of taking the derivative of the log-likelihood subject to the simplex constraint.

> [!question]- Why is every count-and-normalize formula in n-gram LMs and Naive Bayes the same theorem?
> Each one estimates a categorical distribution. Unigram LM = one categorical over $V$. Bigram LM = one categorical per previous word (the distribution over next words). Naive Bayes likelihood = one categorical per class (the distribution over words given that class). MLE of any categorical is relative frequency — count events and divide by the total in the conditioning context. The formulas differ only in what you condition on.

> [!question]- Why do we always maximize log-likelihood rather than likelihood itself?
> Three reasons. **Numerical**: products of many small probabilities underflow 64-bit floats; sums of logs don't. **Mathematical**: log is monotonic, so $\arg\max$ is preserved. **Algorithmic**: log turns products into sums, and gradients decompose cleanly across training examples — essential for any gradient-based learner. Minimizing cross-entropy is the same objective under a different name.

> [!question]- Why is MLE's zero-probability problem especially catastrophic in NLP?
> NLP models compute sentence probabilities as products of many conditional probabilities. A single zero in the product zeroes the whole sentence, regardless of how well every other factor scored. Heaps' Law guarantees that even huge training corpora leave most bigrams and n-grams unseen, so zeros are the common case, not an edge case. The fix is [[smoothing]] — move small amounts of probability mass onto unseen events.

> [!question]- How is add-1 (Laplace) smoothing related to MLE?
> Add-1 smoothing is **MAP estimation** of a categorical under a uniform Dirichlet(1) prior — it's mathematically equivalent to observing one pseudo-count of every outcome before the real data arrives, then applying MLE. MLE uses only observed counts; MAP combines observed counts with a prior. Moving from MLE to add-1 smoothing is moving from pure likelihood maximization to maximum-a-posteriori estimation with a mild, symmetric prior.

> [!question]- What's the difference between MLE and conditional MLE, and why is conditional MLE what you actually use for bigram LMs?
> **MLE** estimates a single distribution over all events. **Conditional MLE** estimates a separate distribution for each value of the conditioning variable. For bigrams you want $P(w_n \mid w_{n-1})$ — one distribution over next words for every possible previous word. Operationally, partition the corpus by $w_{n-1}$, then do standard MLE within each partition: $C(w_{n-1}, w_n) / C(w_{n-1})$. The denominator is the size of the partition, not the overall corpus — which is why the formula conditions on the previous word's count, not the total token count.
