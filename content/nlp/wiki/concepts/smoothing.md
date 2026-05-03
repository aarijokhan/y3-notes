---
type: concept
sources:
  - raw/week-02/w02-slides.pdf
  - raw/week-02/w02-l06-transcript.txt
  - raw/week-02/w02-lab-02.pdf
  - raw/week-02/w02-lab-02-solutions.pdf
status: draft
updated: 2026-04-20
---

*Smoothing assigns small non-zero probabilities to unseen n-grams, preventing the entire sentence probability from collapsing to zero when the model encounters an n-gram it never saw in training.*

## The Zero-Probability Problem

An [[n-gram-language-models|n-gram model]] estimated by [[maximum-likelihood-estimation-nlp|MLE]] assigns $P = 0$ to any n-gram not seen in training. One zero anywhere in the chain makes the entire sentence probability zero.

Suppose training contains "ate lunch", "ate dinner", "ate the", "ate a" — but never "ate breakfast". MLE sets $P(\text{breakfast} \mid \text{ate}) = 0$, so:

$$P(\langle s \rangle \text{ I ate breakfast} \langle /s \rangle) = \ldots \times \underbrace{P(\text{breakfast} \mid \text{ate})}_{= 0} \times \ldots = 0$$

If any test-set sentence gets probability zero, [[perplexity-nlp|perplexity]] is undefined (division by zero). The model cannot be evaluated at all.

This is not a minor edge case. In the Shakespeare corpus ($V = 29{,}066$ types), 99.96% of possible bigrams were never observed. Any realistic test set will contain unseen bigrams.

## Add-One (Laplace) Smoothing

The simplest fix: pretend every bigram was seen one extra time.

**Intuition.** MLE treats a count of 0 as proof of impossibility. But absence of evidence is not evidence of absence — "ate breakfast" just didn't appear in this corpus, not in all possible English. Laplace smoothing encodes a weaker belief: before seeing any data, assume every n-gram has been seen once. These are called **pseudocounts**. After observing the real corpus, you have actual counts + 1 for everything. No entry is ever zero, so no sentence ever gets probability zero.

The $+V$ in the denominator is a bookkeeping consequence: if you add 1 to each of the $V$ possible next words in a row, the row total grows by $V$, and you must divide by $C(w_{n-1}) + V$ to keep the row summing to 1.

**Bigram formula (MLE):**

$$P(w_n \mid w_{n-1}) = \frac{C(w_{n-1}\, w_n)}{C(w_{n-1})}$$

**Bigram formula (add-one smoothed):**

$$P^*_{\text{Laplace}}(w_n \mid w_{n-1}) = \frac{C(w_{n-1}\, w_n) + 1}{C(w_{n-1}) + V}$$

where $V$ is the vocabulary size (number of types). Adding 1 to every numerator requires adding $V$ to the denominator to keep the probabilities summing to 1.

**Trigram formula** (from lab):

$$P^*(w_3 \mid w_1, w_2) = \frac{C(w_1, w_2, w_3) + 1}{C(w_1, w_2) + V}$$

The same pattern generalizes to any n-gram order.

## Add-k Smoothing

Add-one is a special case of **add-k smoothing**: add a fractional count $k < 1$ instead of 1.

$$P^*_{k}(w_n \mid w_{n-1}) = \frac{C(w_{n-1}\, w_n) + k}{C(w_{n-1}) + kV}$$

Smaller $k$ means less aggressive redistribution. The choice of $k$ is a hyperparameter, often tuned on a dev set.

## Why Add-One Is Too Aggressive for N-grams

Add-one adds 1 to *every* cell in the count table. When $V$ is large (e.g., $V = 1{,}446$ in the BeRP corpus), the total mass added to zero cells vastly outweighs the original counts. The probability for observed events drops dramatically while the mass gets spread thin across hundreds of thousands of previously-zero entries.

The result: the model treats everything as nearly equally likely — precisely the opposite of what a good language model should do.

> [!warning] COMMON MISCONCEPTION
> Add-one smoothing is not useless — it works fine for **text classification** and other models where the number of zero entries is proportionally much smaller. It specifically fails for n-gram language models because the count table is overwhelmingly sparse (99.96% zeros for Shakespeare bigrams). The right tool depends on the sparsity regime.

## Worked Example 1: Smoothed vs Unsmoothed (BeRP Corpus)

Compute $P(\langle s \rangle$ i want chinese food $\langle /s \rangle)$ using the Berkeley Restaurant Project bigram tables.

**Unsmoothed** (from Figure 3.2):

$$P(\text{i} \mid \langle s \rangle) \times P(\text{want} \mid \text{i}) \times P(\text{chinese} \mid \text{want}) \times P(\text{food} \mid \text{chinese}) \times P(\langle /s \rangle \mid \text{food})$$
$$= 0.19 \times 0.33 \times 0.0065 \times 0.52 \times 0.40 = 0.0000848$$

**Add-one smoothed** (from Figure 3.7, $V = 1{,}446$):

$$= 0.19 \times 0.21 \times 0.0029 \times 0.052 \times 0.40 = 0.00000241$$

**The counterintuitive result:** smoothing *lowered* the probability of this observed sentence — from $8.48 \times 10^{-5}$ to $2.41 \times 10^{-6}$, a 35x decrease. Why? The probability mass that was redistributed to the hundreds of previously-zero bigrams had to come from somewhere — it came from the observed bigrams. P(want|i) dropped from 0.33 to 0.21; P(chinese|want) dropped from 0.0065 to 0.0029; P(food|chinese) dropped from 0.52 to 0.052.

This demonstrates why add-one is too aggressive: it steals too much mass from observed events to give to unobserved ones.

## Worked Example 2: P(Sam|am) with Add-One

**Corpus** (4 sentences):

```
<s> I am Sam </s>
<s> Sam I am </s>
<s> I am Sam </s>
<s> I do not like green eggs and Sam </s>
```

Treating `<s>` and `</s>` as regular tokens in the vocabulary.

**Count**: $C(\text{am, Sam}) = 3$ (appears in sentences 1, 2, 3). $C(\text{am}) = 3$.

**Vocabulary size**: $V = 14$ types (`<s>`, `</s>`, I, am, Sam, do, not, like, green, eggs, and, ...).

Wait — we need to count carefully. The exact $V$ depends on how boundary tokens are handled, but the lab gives $V = 14$.

$$P^*(\text{Sam} \mid \text{am}) = \frac{C(\text{am, Sam}) + 1}{C(\text{am}) + V} = \frac{3 + 1}{3 + 14} = \frac{4}{17} \approx 0.214$$

Without smoothing: $P(\text{Sam} \mid \text{am}) = 3/3 = 1.0$. Smoothing cuts it from certainty to 0.214 — the difference is dramatic because the corpus is tiny and $V$ is large relative to the counts.

## Absolute Discounting

Add-one steals too much. Add-$k$ steals a tunable amount. **Absolute discounting** steals a *fixed constant* — typically 0.75 — from every non-zero count and redistributes that mass to the unseen n-grams via a lower-order model.

**Where does $d = 0.75$ come from?** Church and Gale (1991) divided 22 million words of AP Newswire into a training set and a held-out set, then for each training-set bigram count $c$, they averaged the count of the same bigrams in the held-out set. The pattern was striking:

| Bigram count in training | Average count in held-out |
|---|---|
| 0 | 0.0000270 |
| 1 | 0.448 |
| 2 | 1.25 |
| 3 | 2.24 |
| 4 | 3.23 |
| 5 | 4.21 |
| 6 | 5.23 |
| 7 | 6.21 |
| 8 | 7.21 |
| 9 | 8.26 |

For counts 2 and above, the held-out count is consistently $c - 0.75$. So the MLE over-estimates each count by about 0.75 — subtracting that constant produces a much better estimate on unseen data.

**Absolute discounting with interpolation:**

$$P_{\text{AbsoluteDiscounting}}(w_i \mid w_{i-1}) = \underbrace{\frac{\max(C(w_{i-1}, w_i) - d,\; 0)}{C(w_{i-1})}}_{\text{discounted bigram}} + \underbrace{\lambda(w_{i-1})}_{\text{interpolation weight}} \cdot \underbrace{P(w_i)}_{\text{lower-order model}}$$

The $\max(\cdot, 0)$ ensures we never produce a negative count. The interpolation weight $\lambda(w_{i-1})$ is chosen so the total probability mass per context sums to 1 — it recovers exactly the mass we discounted. Some implementations keep separate $d$ values for counts of 1 and 2, and a single $d$ for all higher counts.

This sets up the question Kneser-Ney answers: *should the lower-order model really be the plain unigram $P(w)$?*

## Kneser-Ney Smoothing

Kneser-Ney is absolute discounting with a smarter lower-order model. It is the most widely used n-gram smoothing method when compute-cheap LMs are needed.

### The problem with unigrams as the backoff

Consider the Shannon game: *"I can't see without my reading ___"*. The unigram frequency of "Kong" is higher than "glasses" in many corpora — but "Kong" almost always appears after "Hong". It is a frequent word in *very few contexts*. The plain unigram $P(w)$ will happily recommend "Kong" as a fill-in whenever absolute discounting falls back to it, even though it is a terrible choice after "reading".

The key insight: **the unigram is used exactly when we have not seen this bigram before**. So what we really want is not "how frequent is $w$" but "how likely is $w$ to appear as a *novel continuation* — after a context where we haven't seen it before?"

### Continuation probability

Instead of $P(w)$, define:

$$P_{\text{CONTINUATION}}(w) \propto |\{w_{i-1} : C(w_{i-1}, w) > 0\}|$$

This counts the **number of distinct word types that $w$ has followed at least once** — the number of different bigram types $w$ completes. A word that appears in many different contexts has high continuation probability; a word locked into one context (like "Kong" after "Hong") has low continuation probability, regardless of its raw frequency.

Normalize by the total number of observed bigram types so it is a proper distribution:

$$P_{\text{CONTINUATION}}(w) = \frac{|\{w_{i-1} : C(w_{i-1}, w) > 0\}|}{|\{(w_{j-1}, w_j) : C(w_{j-1}, w_j) > 0\}|}$$

Numerator: number of word types that precede $w$. Denominator: total distinct bigram types in the corpus. Equivalently, the denominator is $\sum_{w'} |\{w'_{i-1} : C(w'_{i-1}, w') > 0\}|$ — the sum of numerators over all words.

**Why this fixes "Kong".** "Kong" appears in only one bigram context ("Hong Kong"), so its numerator is 1. "Glasses" appears after many words (reading, broken, dark, sun, his, her, ...), so its numerator is large. Continuation probability correctly prefers "glasses" even though raw unigram frequency prefers "Kong".

### The Kneser-Ney formula

Combine absolute discounting with continuation probability as the lower-order model:

$$P_{\text{KN}}(w_i \mid w_{i-1}) = \frac{\max(C(w_{i-1}, w_i) - d,\; 0)}{C(w_{i-1})} + \lambda(w_{i-1}) \cdot P_{\text{CONTINUATION}}(w_i)$$

The interpolation weight $\lambda(w_{i-1})$ is the normalizing constant that redistributes exactly the mass discounted by $d$:

$$\lambda(w_{i-1}) = \frac{d}{C(w_{i-1})} \cdot |\{w : C(w_{i-1}, w) > 0\}|$$

Reading this: $d / C(w_{i-1})$ is the per-event normalized discount; $|\{w : C(w_{i-1}, w) > 0\}|$ is the number of word types that can follow $w_{i-1}$, which equals the number of times we applied the discount. The product is the total mass we stole and now hand back to the continuation term.

### Recursive formulation

For higher-order n-grams, Kneser-Ney applies recursively:

$$P_{\text{KN}}(w_i \mid w_{i-n+1:i-1}) = \frac{\max(c_{\text{KN}}(w_{i-n+1:i}) - d,\; 0)}{c_{\text{KN}}(w_{i-n+1:i-1})} + \lambda(w_{i-n+1:i-1}) \cdot P_{\text{KN}}(w_i \mid w_{i-n+2:i-1})$$

The trick is the count function $c_{\text{KN}}$:

$$c_{\text{KN}}(\cdot) = \begin{cases} \text{count}(\cdot) & \text{for the highest order} \\ \text{continuation count}(\cdot) & \text{for lower orders} \end{cases}$$

At the top level (e.g. trigram), use ordinary counts. At every recursive step below, use **continuation counts** — the number of distinct single-word contexts in which the sequence appears. This propagates the continuation intuition all the way down the recursion.

**Extended Interpolated Kneser-Ney** (Chen and Goodman, 1998) uses three separate $d$ values for counts 1, 2, and 3+ — this is the method most commonly deployed in practice.

## N-gram Smoothing Summary

| Situation | Method |
|---|---|
| Text classification; few zeros | **Add-1 (Laplace)** |
| General-purpose n-gram LM | **Extended Interpolated Kneser-Ney** |
| Web-scale (trillions of words) | **Stupid backoff** (see [[interpolation-and-backoff]]) |

Add-1 is the pedagogical baseline — almost never used for n-gram LMs in practice because the sparsity regime is too punishing. Kneser-Ney dominates when model quality matters. Stupid backoff dominates when scale matters more than theoretical validity — at Google-web scale, discounting is expensive and the simple 0.4-multiplier heuristic outperforms more principled methods.

## Related

- [[n-gram-language-models]] — smoothing modifies the probability estimates of n-gram models
- [[perplexity-nlp|perplexity]] — zero probabilities make perplexity undefined; smoothing restores computability
- [[interpolation-and-backoff]] — the preferred alternative when add-one is too aggressive; stupid backoff for web-scale

## Active Recall

> [!question]- Why does a single unseen bigram in the test set make perplexity impossible to compute, and how does smoothing fix this?
> Perplexity involves the product $\prod 1/P(w_i \mid w_{i-1})$. If any $P(w_i \mid w_{i-1}) = 0$, one factor becomes $1/0 = \infty$, making perplexity undefined. Smoothing ensures every bigram has $P > 0$ by adding a small pseudo-count to all entries, so no factor in the product is infinite.

> [!question]- Given $V = 1000$, $C(\text{the, cat}) = 50$, $C(\text{the}) = 10{,}000$, compute both MLE and add-one smoothed $P(\text{cat} \mid \text{the})$. By what factor does smoothing change the estimate?
> MLE: $50 / 10{,}000 = 0.005$. Add-one: $(50 + 1) / (10{,}000 + 1{,}000) = 51 / 11{,}000 \approx 0.00464$. Smoothing reduced the estimate by a factor of $0.005 / 0.00464 \approx 1.08$ — about an 8% drop. For a common bigram with decent counts, the effect is mild. For rare bigrams the effect is proportionally larger; for zero-count bigrams, the change is from 0 to $1/11{,}000 \approx 0.000091$.

> [!question]- Add-one smoothing can actually lower the probability of an observed sentence compared to unsmoothed MLE. Explain why this happens and what it reveals about add-one's weakness.
> Adding 1 to every bigram count adds $V$ to each row's denominator. When $V$ is large (e.g., 1,446 in BeRP), the denominator increases substantially, so even observed bigrams get lower probability. The total mass removed from observed events equals the mass given to the $V - k$ previously-zero cells (where $k$ is the number of observed bigram types in that row). If almost all bigrams are zero — which is typical — most of the mass flows to unseen events, leaving observed events with much less. This reveals add-one's weakness: it treats all unseen events as equally worthy of probability mass, regardless of how many there are.

> [!question]- Why is add-one smoothing acceptable for text classification but not for n-gram language models?
> In text classification, the feature space (typically word-in-document indicators) is much denser relative to $V$ — most features have non-trivial counts. Adding 1 to each has a proportionally small effect. In n-gram language models, the space of possible n-grams is $V^N$, and 99%+ of entries are zero. Adding 1 to every zero entry redistributes a massive fraction of the total probability mass, making the model nearly uniform. The sparsity regime determines whether add-one is acceptable.

> [!question]- Write the add-one smoothed formula for trigram probability estimation. What goes in the denominator?
> $P^*(w_3 \mid w_1, w_2) = [C(w_1, w_2, w_3) + 1] / [C(w_1, w_2) + V]$. The denominator is the count of the *bigram prefix* $C(w_1, w_2)$ plus the vocabulary size $V$. The $V$ compensates for adding 1 to each of the $V$ possible continuations of the prefix, ensuring the probabilities sum to 1.

> [!question]- Church and Gale compared bigram training counts to their held-out counts and found a consistent pattern. What was it, and what method does it justify?
> For bigram training counts $c \geq 2$, the average held-out count was $c - 0.75$. MLE systematically over-estimates each count by about 0.75. This justifies **absolute discounting**: subtract a fixed constant $d \approx 0.75$ from every non-zero count and redistribute that mass to unseen n-grams via a lower-order model.

> [!question]- Why does Kneser-Ney replace the unigram $P(w)$ with a continuation probability? Use the "Kong" vs "glasses" example.
> The lower-order model is used exactly when the bigram is unseen, so it should predict words that are plausible novel continuations. In raw unigram frequency, "Kong" beats "glasses" because "Hong Kong" is common. But "Kong" appears in only *one* bigram context (after "Hong"), while "glasses" appears after many contexts. Continuation probability $P_{\text{CONT}}(w) \propto |\{w_{i-1} : C(w_{i-1}, w) > 0\}|$ counts the number of distinct contexts $w$ completes, correctly ranking "glasses" above "Kong" for the sentence "I can't see without my reading ___".

> [!question]- Write the Kneser-Ney bigram formula and identify the role of each term.
> $P_{\text{KN}}(w_i \mid w_{i-1}) = \max(C(w_{i-1}, w_i) - d, 0)/C(w_{i-1}) + \lambda(w_{i-1}) \cdot P_{\text{CONT}}(w_i)$. The first term is the **discounted bigram MLE** — subtract $d$ (typically 0.75) from the count, clip at 0. The second term is the **interpolation weight times continuation probability** — $\lambda(w_{i-1})$ redistributes exactly the mass discounted from all continuations of $w_{i-1}$, and $P_{\text{CONT}}$ scores words by how many distinct contexts they novelly continue.

> [!question]- In the recursive Kneser-Ney formulation, what changes at each level of the recursion?
> The count function $c_{\text{KN}}$ changes. At the highest order (e.g. trigram), $c_{\text{KN}}$ is the ordinary count. For every lower order reached by recursion, $c_{\text{KN}}$ is the **continuation count** — the number of distinct single-word contexts in which the sequence appears — rather than the raw count. This propagates the "novel continuation" intuition all the way down the backoff chain, not just at the bigram-to-unigram transition.

> [!question]- In one sentence each, when would you choose add-1, Kneser-Ney, or stupid backoff?
> **Add-1**: text classification or any domain where the zero rate is modest — it is simple and good enough. **Kneser-Ney (extended interpolated)**: general n-gram LM where model quality matters — the default choice for perplexity-driven work. **Stupid backoff**: web-scale training (trillions of words) where proper discounting is too expensive and a valid probability distribution is not required.
