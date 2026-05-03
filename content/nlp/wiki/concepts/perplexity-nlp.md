---
type: concept
sources:
  - raw/week-02/w02-slides.pdf
  - raw/week-02/w02-l05-transcript.txt
  - raw/week-02/w02-lab-01.pdf
  - raw/week-02/w02-lab-01-solutions.pdf
status: draft
updated: 2026-04-20
---

*Perplexity is the standard intrinsic metric for language models: the inverse probability of the test set, normalized per word, where lower means better.*

## Definition

The **perplexity** of a language model on a test set $W = w_1 w_2 \ldots w_N$ is:

$$\text{PP}(W) = P(w_1 w_2 \ldots w_N)^{-1/N} = \sqrt[N]{\frac{1}{P(w_1 w_2 \ldots w_N)}}$$

Expanding with the chain rule:

$$\text{PP}(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i \mid w_1 \ldots w_{i-1})}}$$

For a **bigram** model:

$$\text{PP}(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i \mid w_{i-1})}}$$

**Range**: probability is in $[0, 1]$; perplexity is in $[1, \infty)$. **Lower perplexity = better model.** Minimizing perplexity is the same as maximizing probability.

## Intuition

### The Shannon game

The clearest way to understand perplexity is through a guessing game. Imagine someone is about to say the next word, and you must keep guessing until you name it. How many guesses do you need on average?

- Very predictable text (*"Happy ___ to you"*) → you guess "birthday" on the first try.
- Very unpredictable text → you might need dozens of attempts.

The average number of guesses required is a direct measure of how surprised the model is by the text. **Perplexity is exactly that number.** A perplexity of 100 means the model is as confused at each step as if it were picking uniformly from 100 equally-likely words. A perplexity of 1 means the model always predicts the correct next word with certainty — no guessing needed.

Shannon (1951) ran a real version of this experiment: he had human subjects guess the next *letter* of English text, counted how many attempts they needed, and used this to estimate the entropy of English. The connection is direct: $\text{Perplexity} = 2^H$ where $H$ is the entropy (in bits) of the model's distribution. Higher entropy = more uncertainty = more guesses = higher perplexity.

### Branching factor

Perplexity has a second equivalent interpretation: it is the **weighted average branching factor** of the language as seen by the model — informally, "how many equally likely words could come next at each position."

**Example.** A language with three words: red, blue, green.

**Model A** — uniform: $P(\text{red}) = P(\text{blue}) = P(\text{green}) = 1/3$.

Test set: "red red red red blue" ($N = 5$).

$$\text{PP}_A = \left((1/3)^5\right)^{-1/5} = (1/3)^{-1} = 3$$

The model is always choosing among 3 equally likely options — branching factor is 3.

**Model B** — informed: $P(\text{red}) = 0.8$, $P(\text{green}) = 0.1$, $P(\text{blue}) = 0.1$.

$$\text{PP}_B = (0.8^4 \times 0.1)^{-1/5} = (0.04096)^{-1/5} = 0.527^{-1} = 1.89$$

Model B is less surprised by this red-heavy test set — its effective branching factor is only 1.89. It is a better model for this data.

## Why Perplexity Instead of Raw Probability

Raw probability $P(W)$ shrinks as the test set gets longer — a 1000-word test set will always have lower probability than a 10-word one, regardless of model quality. This makes raw probability useless for comparing models on different-length test sets.

Perplexity fixes this by normalizing per word (the $N$th root). It is a per-word metric that allows fair comparison.

> [!warning] COMMON MISCONCEPTION
> Comparing perplexity across different test sets is not meaningful. Model A with PP = 120 on WSJ and Model B with PP = 95 on Twitter tells you nothing about which model is better. The test set must be the same for the comparison to be valid.

## Perplexity and N-gram Order

More context = better predictions = lower perplexity. WSJ corpus (38M training words, 1.5M test words):

| N-gram Order | Unigram | Bigram | Trigram |
|---|---|---|---|
| **Perplexity** | 962 | 170 | 109 |

Going from unigram to bigram cuts perplexity by 5.7x. Bigram to trigram gives another 1.6x improvement. Higher-order models capture more context but face worse sparsity — eventually the gains plateau or reverse (see [[smoothing]]).

## Worked Example: Perplexity on a Number Corpus

**Training set**: 100 numbers — 91 zeros and 1 each of digits 1 through 9.

Unigram probabilities: $P(0) = 91/100 = 0.91$, $P(k) = 1/100 = 0.01$ for $k \in \{1, \ldots, 9\}$.

**Test set**: `0 0 0 0 0 3 0 0 0 0` ($N = 10$).

**Step 1** — Joint probability:

$$P(\text{test}) = P(0)^9 \times P(3)^1 = 0.91^9 \times 0.01$$

$0.91^9 \approx 0.4228$, so $P(\text{test}) \approx 0.004228$.

**Step 2** — Perplexity:

$$\text{PP} = P(\text{test})^{-1/N} = 0.004228^{-1/10} = 0.004228^{-0.1}$$

$\log(0.004228) = -2.374$, so $-0.1 \times (-2.374) = 0.2374$, so $\text{PP} = 10^{0.2374} \approx 1.73$.

**Interpretation**: the model's effective branching factor on this test set is ~1.73. Most of the time the model is very confident (predicting 0 with probability 0.91), but the single "3" forces it to use the low-probability estimate, pulling perplexity above 1.

> [!tip] TIP — Perplexity in log space
> In practice, compute $\log \text{PP} = -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid \text{context})$ and exponentiate at the end. This avoids underflow from multiplying small probabilities.

## Related

- [[n-gram-language-models]] — the model family evaluated by perplexity
- [[evaluation-methodology]] — perplexity is the standard intrinsic metric; see that page for extrinsic/intrinsic distinction and train/dev/test splits
- [[smoothing]] — zero probabilities make perplexity undefined (division by zero); smoothing fixes this

## Active Recall

> [!question]- Given a bigram model where P(the|⟨s⟩) = 0.5, P(cat|the) = 0.3, P(⟨/s⟩|cat) = 0.8, compute the perplexity of the sentence "⟨s⟩ the cat ⟨/s⟩".
> $N = 3$ (three predicted words: the, cat, ⟨/s⟩). $P = 0.5 \times 0.3 \times 0.8 = 0.12$. $\text{PP} = 0.12^{-1/3} = (1/0.12)^{1/3} = 8.333^{1/3} \approx 2.03$. The model's average branching factor on this sentence is about 2 — at each step it is choosing among roughly 2 equally likely options.

> [!question]- Explain the branching factor interpretation of perplexity. If a model achieves perplexity 100 on English text, what does that mean informally?
> Perplexity is the weighted average number of equally likely next words at each position. A perplexity of 100 means that, on average, the model is as uncertain as if it were choosing uniformly among 100 words at every position. A perfect model that always predicts the correct next word with probability 1 has perplexity 1 (no uncertainty). A model that treats all words as equally likely has perplexity equal to the vocabulary size.

> [!question]- Why does perplexity normalize by number of words rather than using raw probability to compare language models?
> Raw probability $P(W)$ decreases as the test set gets longer — a product of more terms, each $\leq 1$. A model evaluated on a 10-word test set will always have higher probability than the same model on a 1000-word test set, so raw probability is length-dependent and cannot be compared across test sets. The $N$th root in the perplexity formula normalizes to a per-word scale, making the metric independent of test set length.

> [!question]- Minimizing perplexity is equivalent to maximizing what? Explain why.
> Maximizing the probability $P(W)$ of the test set. Since $\text{PP}(W) = P(W)^{-1/N}$, perplexity is a monotonically decreasing function of probability (for fixed $N$). Higher probability → lower perplexity. The model that assigns the highest probability to the actual test data is the model with the lowest perplexity.

> [!question]- Model A has perplexity 120 on WSJ text. Model B has perplexity 95 on Twitter text. Can you conclude B is a better language model? Why or why not?
> No. Perplexity depends on the test set — Twitter text and WSJ text have different vocabularies, sentence structures, and entropy. A lower perplexity on an easier test set does not mean a better model. To compare A and B, you must evaluate both on the same test set. The WSJ benchmark (unigram 962, bigram 170, trigram 109) is only meaningful because all models are tested on the same 1.5M-word WSJ test set.
