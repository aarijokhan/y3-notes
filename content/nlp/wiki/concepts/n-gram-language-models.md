---
type: concept
sources:
  - raw/week-02/w02-slides.pdf
  - raw/week-02/w02-l04-transcript.txt
  - raw/week-02/w02-l05-transcript.txt
  - raw/week-02/w02-lab-01.pdf
  - raw/week-02/w02-lab-01-solutions.pdf
status: draft
updated: 2026-04-20
---

*An n-gram language model predicts the next word from the previous N−1 words, estimating probabilities by counting n-gram frequencies in a corpus.*

## Definition

A **language model** (LM) assigns probabilities to sequences of words. It can answer two questions:

1. **What is the probability of a sentence?** $P(w_1, w_2, \ldots, w_n)$
2. **What is the probability of the next word?** $P(w_n \mid w_1, w_2, \ldots, w_{n-1})$

These two formulations are equivalent via the **chain rule of probability**.

## The Chain Rule Applied to Language

Any joint probability can be decomposed into a product of conditionals:

$$P(w_{1:n}) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_{1:2}) \cdots P(w_n \mid w_{1:n-1}) = \prod_{k=1}^{n} P(w_k \mid w_{1:k-1})$$

For example:

$$P(\text{the water of Walden Pond}) = P(\text{the}) \times P(\text{water} \mid \text{the}) \times P(\text{of} \mid \text{the water}) \times P(\text{Walden} \mid \text{the water of}) \times P(\text{Pond} \mid \text{the water of Walden})$$

The chain rule is exact — it makes no approximations. The problem is practical.

## Why the Chain Rule Alone Isn't Enough

To estimate $P(w_n \mid w_{1:n-1})$ from a corpus, you count: how many times did the full prefix $w_1 \ldots w_{n-1}$ appear, and of those, how many times was the next word $w_n$?

$$P(w_n \mid w_{1:n-1}) = \frac{C(w_1, \ldots, w_{n-1}, w_n)}{C(w_1, \ldots, w_{n-1})}$$

This works for very short prefixes. It fails catastrophically for long ones.

**The combinatorial explosion.** With a vocabulary of size $V$, there are $V^{n-1}$ possible distinct contexts of length $n-1$. For English-sized vocabularies ($V \approx 50{,}000$), the number of distinct 5-word prefixes alone is $50{,}000^4 \approx 6 \times 10^{18}$. No corpus in existence is large enough to assign a reliable count to even a tiny fraction of those prefixes. Even if a specific prefix *was* seen, it may have appeared only once — not enough for a stable probability estimate.

**The consequence.** For any sentence of non-trivial length, the probability of having seen its exact prefix in training drops toward zero as the sentence grows. The chain rule's conditioning history grows with the sentence — so the longer the sentence, the more likely you are multiplying by $0/0$.

This is not a limitation you can fix by collecting more data. The required corpus size grows exponentially with sentence length, while real corpora grow linearly.

## The Markov Assumption

The n-gram solution: approximate the full history with just the last $N-1$ words.

$$P(w_n \mid w_{1:n-1}) \approx P(w_n \mid w_{n-N+1:n-1})$$

This is the **Markov assumption**: only the recent past matters.

| Model | Conditions on | Approximation |
|---|---|---|
| **Unigram** | nothing | $P(w_n)$ |
| **Bigram** | 1 previous word | $P(w_n \mid w_{n-1})$ |
| **Trigram** | 2 previous words | $P(w_n \mid w_{n-2}, w_{n-1})$ |
| **4-gram** | 3 previous words | $P(w_n \mid w_{n-3}, w_{n-2}, w_{n-1})$ |

### Unigram model

The most extreme approximation: drop all context entirely. Each word is treated as if it were drawn from a bag — its probability is fixed by its corpus frequency, and it does not depend on any of the words before it.

$$P(w_{1:n}) \approx \prod_{k=1}^{n} P(w_k)$$

Each word's probability is its relative frequency in the corpus:

$$P(w) = \frac{C(w)}{N}$$

where $N$ is the total number of tokens.

**What independence actually means.** In a bigram model, the probability of seeing "food" changes depending on the previous word — $P(\text{food} \mid \text{want})$ is high, $P(\text{food} \mid \text{the})$ is lower. The previous word carries information. In a unigram model there is no such conditioning: $P(\text{food})$ is the same fixed number regardless of whether the previous word was "want", "ate", or "sky". Knowing what came before tells you nothing about what comes next.

The most striking consequence: word order is invisible. Because multiplication is commutative, "I want food" and "food want I" get identical probabilities:

$$P(\text{I}) \times P(\text{want}) \times P(\text{food}) = P(\text{food}) \times P(\text{want}) \times P(\text{I})$$

The model cannot distinguish a grammatical sentence from a random shuffle of its words. Sampling from a unigram model produces exactly that — word salad at corpus frequencies (*"To him swallowed confess hear both"*). Unigram perplexity is high; it is the floor every better model must beat.

### Bigram model

Condition on exactly the previous word:

$$P(w_{1:n}) \approx \prod_{k=1}^{n} P(w_k \mid w_{k-1})$$

where $w_0 = \langle s \rangle$ (start-of-sentence marker). The model now knows *something* about order — "I want" is much more likely than "want I" — but it sees only one step of history.

### Trigrams, 4-grams, and beyond

The extension is mechanical: add one more word to the conditioning context at each step.

| Model | Formula |
|---|---|
| Trigram | $P(w_n \mid w_{n-2}, w_{n-1})$ |
| 4-gram | $P(w_n \mid w_{n-3}, w_{n-2}, w_{n-1})$ |
| 5-gram | $P(w_n \mid w_{n-4}, w_{n-3}, w_{n-2}, w_{n-1})$ |

Sampling quality improves noticeably with each step — more context means better predictions. But increasing N makes the sparsity problem worse and never escapes the finite-window problem; see [[#Limitations]].

## Maximum Likelihood Estimation

To estimate n-gram probabilities, count and normalize. For a **unigram**:

$$P(w) = \frac{C(w)}{N}$$

For a **bigram**:

$$P(w_n \mid w_{n-1}) = \frac{C(w_{n-1}\, w_n)}{C(w_{n-1})}$$

For a general **n-gram**:

$$P(w_n \mid w_{n-N+1:n-1}) = \frac{C(w_{n-N+1:n})}{C(w_{n-N+1:n-1})}$$

These are all [[maximum-likelihood-estimation-nlp|Maximum Likelihood Estimates]] (MLE) — the parameter settings that maximize the probability of the observed training data. The pattern is always the same: count the full n-gram in the numerator, count the $(N-1)$-gram prefix in the denominator. Count-and-normalize isn't a heuristic — it's the closed-form MLE of a categorical distribution.

### Worked example

Corpus (3 sentences, with sentence boundary markers):

```
<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>
```

To compute $P(\text{I} \mid \langle s \rangle)$: count how many times $\langle s \rangle$ is followed by "I" (twice), divide by the total count of $\langle s \rangle$ (three times, once per sentence).

$$P(\text{I} \mid \langle s \rangle) = \frac{2}{3} = 0.67 \qquad P(\text{Sam} \mid \langle s \rangle) = \frac{1}{3} = 0.33$$

$$P(\text{am} \mid \text{I}) = \frac{2}{3} = 0.67 \qquad P(\text{do} \mid \text{I}) = \frac{1}{3} = 0.33$$

$$P(\text{Sam} \mid \text{am}) = \frac{1}{2} = 0.5 \qquad P(\langle /s \rangle \mid \text{Sam}) = \frac{1}{2} = 0.5$$

Note that $P(\text{am} \mid \text{I}) + P(\text{do} \mid \text{I}) \neq 1$ — these are two entries from a full distribution over all words that can follow "I"; the remaining probability mass goes to other words.

## Log Probabilities

Sentence probabilities are products of many small numbers. For a 20-word sentence where each bigram has probability ~0.01, the product is $10^{-40}$ — too small for floating-point arithmetic (underflow).

The fix: work in **log space**.

$$\log P(w_{1:n}) = \sum_{k=1}^{n} \log P(w_k \mid w_{k-1})$$

Multiplication becomes addition. If you need the actual probability at the end: $P = \exp(\text{sum of log probs})$. In practice, you rarely need to leave log space.

## Worked Example: Berkeley Restaurant Project

The Berkeley Restaurant Project (BeRP) is a corpus of 9222 sentences about restaurant queries.

**Raw bigram counts** (subset of 8 words):

|  | i | want | to | eat | chinese | food | lunch | spend |
|---|---|---|---|---|---|---|---|---|
| **i** | 5 | 827 | 0 | 9 | 0 | 0 | 0 | 2 |
| **want** | 2 | 0 | 608 | 1 | 6 | 6 | 5 | 1 |
| **to** | 2 | 0 | 4 | 686 | 2 | 0 | 6 | 211 |
| **eat** | 0 | 0 | 2 | 0 | 16 | 2 | 42 | 0 |
| **chinese** | 1 | 0 | 0 | 0 | 0 | 82 | 1 | 0 |
| **food** | 15 | 0 | 15 | 0 | 1 | 4 | 0 | 0 |
| **lunch** | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| **spend** | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

**Unigram counts** (row totals from the full corpus):

| i | want | to | eat | chinese | food | lunch | spend |
|---|---|---|---|---|---|---|---|
| 2533 | 927 | 2417 | 746 | 158 | 1093 | 341 | 278 |

**Step 1 — Normalize.** Divide each bigram count by the row word's unigram count:

$$P(\text{to} \mid \text{want}) = \frac{C(\text{want, to})}{C(\text{want})} = \frac{608}{927} = 0.66$$

**Selected bigram probabilities** (full normalized table):

|  | i | want | to | eat | chinese | food | lunch | spend |
|---|---|---|---|---|---|---|---|---|
| **i** | 0.002 | 0.33 | 0 | 0.0036 | 0 | 0 | 0 | 0.00079 |
| **want** | 0.0022 | 0 | 0.66 | 0.0011 | 0.0065 | 0.0065 | 0.0054 | 0.0011 |
| **to** | 0.00083 | 0 | 0.0017 | 0.28 | 0.00083 | 0 | 0.0025 | 0.087 |
| **eat** | 0 | 0 | 0.0027 | 0 | 0.021 | 0.0027 | 0.056 | 0 |
| **chinese** | 0.0063 | 0 | 0 | 0 | 0 | 0.52 | 0.0063 | 0 |
| **food** | 0.014 | 0 | 0.014 | 0 | 0.00092 | 0.0037 | 0 | 0 |

**Step 2 — Sentence probability.** $P(\langle s \rangle$ I want english food $\langle /s \rangle)$:

$$P(\text{I} \mid \langle s \rangle) \times P(\text{want} \mid \text{I}) \times P(\text{english} \mid \text{want}) \times P(\text{food} \mid \text{english}) \times P(\langle /s \rangle \mid \text{food}) = 0.000031$$

Notice: a perfectly grammatical, meaningful sentence gets probability 0.000031. Each bigram multiplication makes the number smaller. This is why we use log probabilities — and why raw probability is a bad metric for comparing models (see [[perplexity-nlp|perplexity]]).

## Sampling: Generating Text from N-grams

You can visualize what a model has learned by **sampling** from it — generating text word by word.

**The Shannon (1948) method for bigrams:**

1. Sample a word $w_1$ from $P(w \mid \langle s \rangle)$.
2. Sample $w_2$ from $P(w \mid w_1)$.
3. Continue until $\langle /s \rangle$ is sampled.
4. String the words together.

From the BeRP bigram model: $\langle s \rangle \to \text{I} \to \text{want} \to \text{to} \to \text{eat} \to \text{Chinese} \to \text{food} \to \langle /s \rangle$

**Quality improves with model order.** Shakespeare corpus samples:

| Order | Sample |
|---|---|
| 1-gram | *To him swallowed confess hear both. Which. Of save on trail for are ay device and rote life have* |
| 2-gram | *Why dost stand forth thy canopy, forsooth; he is this palpable hit the King Henry. Live king. Follow.* |
| 3-gram | *Fly, and will rid me these news of price. Therefore the sadness of parting, as they say, 'tis done.* |
| 4-gram | *King Henry. What! I will go seek the traitor Gloucester. Exeunt some of the watch. A great banquet serv'd in;* |

The 4-gram output sounds eerily authentic — because with a vocabulary of 29,066 types, 99.96% of possible bigrams were never seen, so higher-order n-grams frequently reproduce verbatim training text rather than generating novel combinations.

> [!info] ASIDE — Other sampling methods
> Temperature, top-k, and top-p sampling are used with neural language models to control the trade-off between diversity and quality. They are not relevant to n-gram models and will be covered in later weeks.

## Limitations

**Long-distance dependencies.** Think of a bigram model as reading through a letterbox slot — it sees exactly two words at a time, sliding forward one word at a time, and forgets everything behind the slot.

Now consider: *"The soups that I made from that new cookbook I bought yesterday were amazingly delicious."*

The verb *were* must agree with the subject *soups* (plural). But watch what a bigram model actually sees when it reaches *were*:

| Position | Bigram context | What the model knows |
|---|---|---|
| ... | `yesterday were` | was yesterday singular or plural? no idea |

The word *soups* was nine positions ago — completely outside the window. All the model sees is "yesterday", which carries no information about number. It cannot distinguish *were* from *was*. A trigram sees "bought yesterday" — still no help. You would need a window of at least ten words, and even then the next sentence might need twenty.

The core issue: grammatical structure creates dependencies at arbitrary distances. A fixed-size window is always the wrong size for some sentence.

---

**Semantic similarity.** An n-gram model treats every word as an opaque token — a string of characters with no meaning attached. "Cat" and "feline" are as unrelated to it as "cat" and "refrigerator".

Suppose the model trains on *"The cat sat on the mat"* millions of times and learns $P(\text{sat} \mid \text{cat}) = 0.4$. Now it encounters *"The feline sat on the rug."* The bigrams "feline sat" and "sat on" may never have appeared in training. The model has zero evidence for them, even though any reader knows these sentences mean the same thing.

This matters in practice: any word the model has not seen in a specific context gets treated as a surprise, regardless of how many synonyms or paraphrases were in training. The model has memorized surface co-occurrences, not the meaning behind them.

---

**Sparsity.** Shakespeare wrote 884,647 tokens using a vocabulary of 29,066 word types. The number of possible bigrams is $29{,}066^2 \approx 844$ million. Only 300,000 of those were ever observed — that is 99.96% zeros.

Sparsity is the cause; the **zero-probability problem** is the consequence: MLE assigns $P = 0$ to any unseen n-gram, and since sentence probability is a product, one zero anywhere collapses the whole sentence.

The analogy: imagine a restaurant that will only cook a dish if a customer has ordered that *exact* combination before. Beef with rice? Sure, if someone ordered that last year. But beef with quinoa? Never been ordered — the kitchen refuses. An n-gram model behaves the same way: any bigram it has not seen in training gets probability zero, and any sentence containing it gets probability zero. One unseen word pair collapses the entire sentence.

This is why [[smoothing]] and [[interpolation-and-backoff]] exist — they are the mechanism for assigning non-zero probability to unseen combinations.

## Scaling to Web-Sized N-grams

When the training corpus is trillions of words (e.g. the Google Web 5-gram release: 1,024,908,267,229 words, 1,176,470,663 five-word sequences appearing ≥40 times, 13,588,391 unique words), the count tables do not fit in memory by default. Two classes of trick are used to shrink them.

**Pruning — drop counts you can afford to lose:**
- Only store n-grams with $\text{count} > \text{threshold}$.
- Remove singletons of higher-order n-grams (they dominate the table size but carry the least reliable counts).
- **Entropy-based pruning** — drop n-grams whose removal barely changes the model's probability distribution.

**Efficient representations — make the survivors cheaper:**
- **Tries** — prefix-tree data structure that shares storage between n-grams with common prefixes.
- **Bloom filters** — probabilistic set membership, used to build *approximate* LMs that answer "has this n-gram been seen?" with a controllable false-positive rate.
- Store words as **integer indexes**, not strings; use **Huffman coding** to pack frequent words into two bytes.
- **Quantize probabilities** to 4–8 bits per value instead of storing 8-byte floats.

For smoothing at this scale, the expensive Kneser-Ney normalization becomes impractical and **stupid backoff** (see [[interpolation-and-backoff]]) takes over — it drops the requirement of a valid probability distribution in exchange for constant-time lookup and trivial training.

**Infini-grams (Liu et al. 2024)** take this further: no precomputed n-gram table at all. Store 5 trillion words of web text in a **suffix array** and compute any n-gram probability on demand for any $n$ — arbitrary-length contexts, but at the cost of per-query disk access.

## Advanced Extensions

Beyond smoothing, several extensions exist within the n-gram family. They matter mainly as context for why neural LMs superseded them — each fixes a specific weakness of plain n-grams.

- **Discriminative LMs.** Instead of picking n-gram weights to fit the training corpus, pick them to improve performance on a downstream task (e.g. speech recognition word error rate). Decouples *what the LM scores highly* from *what the corpus happens to contain*.
- **Parsing-based LMs.** Condition on syntactic structure, not just a flat word window. Lets the model see "soups ... were" as subject–verb even across an intervening clause — the long-distance dependency problem that a fixed n-gram window cannot touch.
- **Caching LMs.** Recently used words are more likely to appear again. Interpolate the n-gram estimate with a frequency-of-word-in-history term:
  $$P_{\text{CACHE}}(w \mid \text{history}) = \lambda P(w_i \mid w_{i-2}, w_{i-1}) + (1 - \lambda) \frac{C(w \in \text{history})}{|\text{history}|}$$
  Turned out to perform poorly for speech recognition — misrecognized words in the cache become self-reinforcing errors.

## Related

- [[perplexity-nlp|perplexity]] — the standard intrinsic metric for evaluating language models
- [[evaluation-methodology]] — extrinsic/intrinsic evaluation and train/dev/test splits
- [[smoothing]] — handling the zero-probability problem in n-gram counts
- [[interpolation-and-backoff]] — mixing model orders when higher-order counts are zero
- [[tokenization]] — n-gram models operate over tokens produced by the tokenizer
- [[type-and-token]] — vocabulary size $V$ determines sparsity

## Active Recall

> [!question]- Write the chain rule decomposition for the sentence "I like cats" and then simplify it using the bigram assumption. What information is lost in the simplification?
> **Chain rule**: $P(\text{I like cats}) = P(\text{I} \mid \langle s \rangle) \times P(\text{like} \mid \langle s \rangle \text{ I}) \times P(\text{cats} \mid \langle s \rangle \text{ I like}) \times P(\langle /s \rangle \mid \langle s \rangle \text{ I like cats})$. **Bigram**: $P(\text{I} \mid \langle s \rangle) \times P(\text{like} \mid \text{I}) \times P(\text{cats} \mid \text{like}) \times P(\langle /s \rangle \mid \text{cats})$. The simplification drops all context beyond the immediately preceding word. For example, $P(\text{cats} \mid \text{like})$ is the same regardless of whether the subject was "I", "dogs", or "politicians" — the bigram model cannot condition on anything further back.

> [!question]- Given a training corpus where C(the) = 5000, C(the, cat) = 100, and C(the, dog) = 250, compute P(cat|the) and P(dog|the). What happens if C(the, iguana) = 0?
> $P(\text{cat} \mid \text{the}) = 100/5000 = 0.02$. $P(\text{dog} \mid \text{the}) = 250/5000 = 0.05$. If $C(\text{the, iguana}) = 0$, then $P(\text{iguana} \mid \text{the}) = 0$. Any sentence containing "the iguana" gets probability zero, and [[perplexity-nlp|perplexity]] becomes undefined. This is the zero-probability problem that motivates [[smoothing]].

> [!question]- Why does the Markov assumption make n-gram language modeling tractable, and what does it sacrifice?
> Without the Markov assumption, computing $P(w_n \mid w_{1:n-1})$ requires counting the exact prefix $w_1 \ldots w_{n-1}$ — which for long sentences will almost never appear in the training corpus. The Markov assumption truncates the conditioning context to the last $N-1$ words, making the counts dense enough to be useful. What it sacrifices: any dependency that spans more than $N-1$ words is invisible. Subject-verb agreement across a relative clause, for example, requires context that a bigram or trigram cannot reach.

> [!question]- Why are language model probabilities stored and computed in log space rather than as raw probabilities?
> Sentence probabilities are products of many small numbers. A 20-word sentence where each conditional probability is ~0.01 gives a product of $10^{-40}$, which underflows 64-bit floating point. In log space, the product becomes a sum of log probabilities, which stays in a numerically stable range. If the actual probability is needed, apply $\exp(\cdot)$ once at the end.

> [!question]- Write the MLE formula for trigram probability estimation. How does it generalize the bigram formula?
> Trigram: $P(w_n \mid w_{n-2}, w_{n-1}) = C(w_{n-2}, w_{n-1}, w_n) / C(w_{n-2}, w_{n-1})$. This generalizes the bigram formula $P(w_n \mid w_{n-1}) = C(w_{n-1}, w_n) / C(w_{n-1})$ by extending both the numerator and denominator by one word of context. The general pattern: an $N$-gram conditions on $N-1$ previous words, counts the $N$-word sequence in the numerator, and counts the $(N-1)$-word prefix in the denominator.

> [!question]- Why can't an n-gram model correctly predict that "The soups ... were amazingly delicious" requires "were" rather than "was"?
> The verb must agree with the subject "soups" (plural), but the intervening relative clause ("that I made from that new cookbook I bought yesterday") pushes "soups" far outside any practical n-gram window. A trigram model sees only the last two words — likely "yesterday" or similar — which carry no information about the subject's number. Capturing such long-distance dependencies requires a model architecture (like a recurrent or transformer network) that maintains state over arbitrarily long contexts.

> [!question]- Name three tricks used to fit web-scale n-gram tables into memory and briefly explain each.
> **Pruning** — drop n-grams below a count threshold, remove singletons at higher orders, or use entropy-based pruning to discard n-grams whose removal barely changes the distribution. **Compact representations** — tries to share storage across common prefixes; bloom filters for approximate membership; integer/Huffman indexing of words; quantize probabilities to 4–8 bits. **Cheaper smoothing** — stupid backoff instead of Kneser-Ney, trading a valid probability distribution for constant-time lookup.

> [!question]- What problem do caching language models try to solve, and why did they fail in speech recognition?
> Caching LMs exploit the fact that recently used words are more likely to reappear, by interpolating the n-gram probability with the frequency of the word in the recent history. This helps topic-coherence for well-transcribed text. In speech recognition it backfired: when the recognizer misheard a word and added the wrong word to the cache, the cache then inflated the probability of that wrong word for future predictions, reinforcing the error.

> [!question]- What is an infini-gram and how does it differ architecturally from classical n-gram LMs?
> An infini-gram (Liu et al. 2024) stores the training corpus (5 trillion words of web text) in a **suffix array** and computes any n-gram probability on demand at query time. Classical n-gram LMs precompute a count table for a fixed maximum $n$; infini-grams have no precomputed table and no fixed $n$, so context length is unbounded — trading query latency (disk access into the suffix array) for arbitrary context flexibility.
