---
name: n-gram-language-model
description: A statistical language model that approximates $P(w_n \mid w_1, \ldots, w_{n-1})$ by truncating context to the previous $N-1$ words (the Markov assumption with memory $N-1$). Probabilities are estimated by counting n-gram occurrences in a training corpus and normalising — pure MLE, no learned parameters. Practical maximum is 5-grams; suffers from the "zeros problem" (unseen n-grams get probability 0) and cannot model synonymy or long-distance dependencies.
type: concept
sources:
  - raw/week-09/w09-l19-transcript.txt
  - raw/week-09/w09-l20-transcript.txt
  - raw/week-09/w09-slides.pdf
  - raw/week-09/w09-problem-set.pdf
  - raw/week-09/w09-problem-set-solution.pdf
status: stable
updated: 2026-04-27
---

*The first practical [[language-model|language model]], and the right place to start because every later neural LM solves problems n-grams have. The recipe: take the chain rule of probability, replace the full history with the last $N-1$ words (the Markov assumption), estimate each conditional by counting in a corpus and dividing. The reason it works at all is that local word context carries most of the predictive signal. The reason it fails is that "local" means $N \le 5$ in practice, the table grows as $|V|^N$, and any combination you didn't see at training time gets probability zero.*

## The chain rule + Markov assumption

The chain rule of probability, applied to a sentence:

$$P(w_1, w_2, \ldots, w_n) = \prod_{k=1}^{n} P(w_k \mid w_{1:k-1})$$

This is exact, but unhelpful — the conditional $P(w_k \mid w_{1:k-1})$ requires knowing the joint over arbitrary-length histories, which is the exact intractability we wanted to escape.

The **Markov assumption** truncates: assume each word depends only on the previous $N-1$:

$$P(w_k \mid w_{1:k-1}) \approx P(w_k \mid w_{k-N+1:k-1})$$

For example, a **bigram** model (memory of 1) approximates:

$$P(w_n \mid w_1, \ldots, w_{n-1}) \approx P(w_n \mid w_{n-1})$$

So the joint becomes:

$$P(w_{1:n}) \approx \prod_{k=1}^{n} P(w_k \mid w_{k-1})$$

| Name | $N$ | Conditional |
|---|---|---|
| Unigram | 1 | $P(w_k)$ |
| Bigram | 2 | $P(w_k \mid w_{k-1})$ |
| Trigram | 3 | $P(w_k \mid w_{k-2}, w_{k-1})$ |
| 4-gram | 4 | $P(w_k \mid w_{k-3:k-1})$ |
| 5-gram | 5 | $P(w_k \mid w_{k-4:k-1})$ |

> [!tip] TIP — Why the Markov assumption is plausible
> Most predictive signal for the next word is short-range: *"once upon a"* → *"time"* needs three words of context, not three hundred. The assumption is that beyond a few words back, additional context contributes diminishing returns. It's wrong in important cases (*"The **soups** that I made from the new cookbook I bought yesterday **were** delicious"* — subject–verb agreement spans 12 words), but it's a starting point. Neural LMs exist precisely to relax it.

## Estimating probabilities: MLE by counting

The maximum-likelihood estimate of a bigram probability is:

$$\hat{P}(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})} = \frac{C(w_{n-1}, w_n)}{\sum_{w} C(w_{n-1}, w)}$$

Numerator: how many times $w_{n-1}$ was followed by $w_n$ in the corpus. Denominator: how many times $w_{n-1}$ appeared at all (equivalently, the row sum of the bigram count table). For unigrams, just $\hat{P}(w) = C(w) / \sum_{w'} C(w')$.

That's it. **No gradient descent, no parameters in the neural sense — just count and divide.** Training an n-gram LM is a single pass over the corpus building the count tables. The "model" *is* the count tables.

> [!info] ASIDE — Sentence boundary tokens
> A bigram LM needs to know how sentences begin and end, since the first word has no left context and the last has no right neighbour. The standard trick: prepend `<SOS>` (start-of-sentence) and append `<EOS>` (end-of-sentence) to every training sentence. Then $P(\text{first word} \mid \text{<SOS>})$ and $P(\text{<EOS>} \mid \text{last word})$ are well-defined, and the joint over a full sentence picks up multiplicative factors for "this is a plausible way to start" and "this is a plausible way to end." Without `<EOS>` the model has no notion of when to stop generating.

### Worked example: the Berkeley Restaurant corpus

A classic teaching dataset (recorded restaurant phone orders) gives raw bigram counts like:

| | i | want | to | eat | chinese | food | lunch | spend |
|---|---|---|---|---|---|---|---|---|
| i | 5 | 827 | 0 | 9 | 0 | 0 | 0 | 2 |
| want | 2 | 0 | 608 | 1 | 6 | 6 | 5 | 1 |
| to | 2 | 0 | 4 | 686 | 2 | 0 | 6 | 211 |
| eat | 0 | 0 | 2 | 0 | 16 | 2 | 42 | 0 |

Dividing by row sums (the unigram counts) gives $\hat{P}(w_n \mid w_{n-1})$:

| | i | want | to | eat | chinese | food | lunch | spend |
|---|---|---|---|---|---|---|---|---|
| i | 0.002 | 0.33 | 0 | 0.0036 | 0 | 0 | 0 | 0.00079 |
| want | 0.0022 | 0 | 0.66 | 0.0011 | 0.0065 | 0.0065 | 0.0054 | 0.0011 |
| to | 0.00083 | 0 | 0.0017 | 0.28 | 0.00083 | 0 | 0.0025 | 0.087 |
| eat | 0 | 0 | 0.0027 | 0 | 0.021 | 0.0027 | 0.056 | 0 |

You can read off useful patterns: $P(\text{to} \mid \text{want}) = 0.66$ (high — "want to" is common); $P(\text{eat} \mid \text{to}) = 0.28$ (high — "to eat" is common); $P(\text{food} \mid \text{to}) = 0$ (sensibly low — "to food" is ungrammatical). Sentence probability comes from chaining:

$$P(\text{<SOS>~i~want~english~food~<EOS>}) = P(\text{i}|\text{<SOS>}) \cdot P(\text{want}|\text{i}) \cdot P(\text{english}|\text{want}) \cdot \ldots \approx 3.1 \times 10^{-5}$$

## The size problem: why we stop at 5-grams

For vocabulary size $|V|$, an $N$-gram model has up to $|V|^N$ entries in its conditional table. With $|V| \approx 10^5$:

| $N$ | Table size | |
|---|---|---|
| 1 | $10^5$ | trivial |
| 2 | $10^{10}$ | feasible |
| 3 | $10^{15}$ | very large |
| 4 | $10^{20}$ | extreme |
| 5 | $10^{25}$ | enormous (English-only, a single machine struggles) |

Empirically the lecturer notes that at Microsoft Research circa 2008, the production LMs were 5-grams for English (largest language with enough data) and 4-grams for everything else. Training data and storage become limiting before modelling power does. Recent work (e.g., infini-grams, Liu et al. 2024) sidesteps this by storing the corpus in a **suffix array** instead of precomputing counts — letting you compute $N$-gram probabilities for *any* $N$ at lookup time over 5 trillion words of web text. Still, the fundamental scaling problem is a key motivation for [[language-model|neural language models]] that compress the next-word distribution into a (much smaller) parametric model.

## The zeros problem

For any $N \ge 2$, most syntactically valid n-grams *will not appear in the training corpus* simply because there are too many. A bigram count of zero forces the conditional probability to zero, which forces the *entire sentence probability* to zero whenever such a bigram appears (since the joint is a product). This wrecks both:

- **Generation** — words with zero probability are never sampled, so legal continuations are unreachable.
- **Evaluation** — perplexity involves $1/P$, so a zero in the test set gives perplexity infinity.

> [!example]+ The breakfast example
> Training corpus contains *"ate lunch", "ate dinner", "ate a", "ate the"* but never *"ate breakfast"*. Test corpus contains *"... ate breakfast"*. The bigram model says $P(\text{breakfast} \mid \text{ate}) = 0$, so any test sentence containing this bigram has probability zero, and perplexity blows up. Yet *"ate breakfast"* is plainly grammatical English — the corpus just didn't happen to include it.

The classical fix is **smoothing** (add-one / Laplace, Kneser-Ney, etc.) — redistribute a small probability mass to unseen n-grams. Smoothing is a substantial subfield in itself, with whole papers per technique. The deeper fix, however, is to abandon discrete counting and represent words by [[word-embedding|dense vectors]]: if "lunch" and "breakfast" have similar vectors, then $P(\text{breakfast} \mid \text{ate})$ can be high even without the exact bigram appearing in training. This is the n-gram → neural-LM transition.

## Generating from an n-gram LM (the Shannon visualisation)

Given a trained model, you can sample sequences. For a bigram LM:

1. Start with `<SOS>`.
2. Sample $w_1 \sim P(w \mid \text{<SOS>})$.
3. Sample $w_2 \sim P(w \mid w_1)$.
4. Continue until you sample `<EOS>`.

The sample quality depends sharply on $N$ and on training data. Trained on Shakespeare:

| $N$ | Sample |
|---|---|
| 1 | *To him swallowed confess hear both. Which. Of save on trail for are ay device and rote life have* |
| 2 | *Why dost stand forth thy canopy, forsooth; he is this palpable hit the King Henry. Live king. Follow.* |
| 3 | *Fly, and will rid me these news of price. Therefore the sadness of parting, as they say, 'tis done.* |
| 4 | *King Henry. What! I will go seek the traitor Gloucester. Exeunt some of the watch. A great banquet serv'd in;* |

Higher $N$ → more locally fluent but increasingly verbatim from training (memorisation, not learning). The lecturer notes that the corpus also fingerprints the style: trained on Wall Street Journal, the same setup produces financial-sounding gibberish ("*They also point to ninety nine point six billion dollars from two hundred four oh six three percent of the rates of interest*") — a side effect of an LM is **author identification**: train one LM per candidate author, score the disputed text under each, and the lowest-perplexity model wins.

For sampling mechanics (turning a probability distribution into an actual word draw), see [[decoding-strategies]].

## Two structural problems with n-grams (the real motivation for neural LMs)

The size problem and zeros problem are practical. Two deeper, structural shortcomings drive the neural transition:

1. **No long-distance dependencies.** Because $N \le 5$, the model cannot enforce constraints that span more than 5 words. Subject–verb agreement across a long relative clause (*"The **soups** that I made from the new cookbook I bought yesterday **were** amazing"*) is invisible to a 5-gram. Increasing $N$ further is precluded by the size problem.
2. **No notion of similarity between words.** Each word is a discrete, opaque token — *cat* is no more similar to *dog* than to *Wednesday*. So learning that *"the dog ran"* is a likely sentence teaches the model nothing about *"the cat ran"* or *"the puppy ran"*. Every n-gram must be observed for itself; there's no generalisation across semantically similar contexts.

Both problems dissolve once you represent words as [[word-embedding|dense vectors]] in $\mathbb{R}^d$. Vectors automatically encode similarity (close vectors = similar words), and the matrix multiplications + non-linearities used by neural LMs to combine them produce smooth, never-zero probability distributions over the next word. This is the bridge to the neural era.

## Connections

- **Foundational case of** [[language-model]] — the simplest concrete LM, useful as the conceptual baseline against which all later models are compared.
- **Motivates** [[word-embedding|word embeddings]] — needed to fix the zeros and synonymy problems.
- **Motivates** [[recurrent-neural-network|RNN-based LMs]] — needed for unbounded context and long-range dependencies.
- **Evaluated by** [[perplexity-nc|perplexity]] — the standard intrinsic metric, applied identically to n-gram and neural LMs.
- **Generation uses** [[decoding-strategies]] — the sampling/greedy/beam machinery applies to any autoregressive LM.

## Common pitfalls

- **Forgetting the `<EOS>` token** — without it, the LM cannot decide when to stop generating; sentence probabilities are also miscalibrated because they don't include the "this is a complete sentence" factor.
- **Confusing chain rule with Markov assumption** — the chain rule is *exact* and decomposes the joint into conditionals. The Markov assumption is the *additional* simplification (truncating each conditional to a finite history) that makes the model tractable. They are independent ideas; you need both.
- **Believing higher $N$ is always better** — past a point, larger $N$ overfits to the training corpus (memorising verbatim n-grams) rather than learning generalisable structure. The benefit also flattens: trigrams are dramatically better than bigrams; 5-grams are only marginally better than 4-grams on most tasks (compare the Shakespeare samples above).
