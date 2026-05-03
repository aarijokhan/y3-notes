---
type: concept
sources:
  - raw/week-01/w01-slides.pdf
  - raw/week-01/w01-l01-transcript.txt
status: draft
updated: 2026-04-20
---

*Types are distinct word forms in a vocabulary; tokens are individual occurrences in running text. Their relationship follows a predictable sub-linear growth law.*

## Definition

- **Token**: a single instance of a word as it appears in a sequence of text. In *"the cat sat on the mat"*, the word `the` produces **two tokens** (it appears twice).
- **Type**: a distinct word form — each unique element of the vocabulary $V$. In the same sentence, `the` is only **one type** regardless of how many times it appears.

The sentence *"to be or not to be"* contains **6 tokens** but only **4 types**: `to`, `be`, `or`, `not` (`to` and `be` each appear twice, but they still count as one type each).

This distinction matters throughout NLP because model size, data requirements, and statistical estimates all depend on which count you mean.

## Heaps' Law (Herdan's Law)

As a corpus grows, new word types keep appearing — but the rate slows. This relationship is captured by **Heaps' Law**:

$$|V| = k N^{\beta}$$

where $N$ is the number of tokens, $|V|$ is the vocabulary size, and the constants are empirically:

- $k \approx 10$–$100$
- $0.67 \leq \beta \leq 0.75$ (typically around 0.67–0.75 for English)

The sub-linear exponent ($\beta < 1$) means vocabulary grows much more slowly than corpus size: doubling the corpus does not double the vocabulary. Church & Gale (1990) found approximately 44,000 types in the first million tokens of a Wall Street Journal corpus.

### Implications

1. **Open-vocabulary problem**: no matter how large your training corpus, you will encounter unseen words at test time. Heaps' Law tells you the rate at which this problem diminishes.
2. **Vocabulary cap decisions**: choosing a fixed vocabulary size $|V|$ for a model (e.g., 50,000 tokens in many LLMs) is a design tradeoff — larger $|V|$ captures more types but increases parameter count.
3. **Subword tokenization motivation**: rather than treating rare word types as unknown, modern systems decompose them into subword units (see [[subword-tokenization]]), which side-steps the open-vocabulary problem.

## Related

- [[corpora]] — corpus size $N$ is what drives Heaps' Law
- [[tokenization]] — tokenization determines what counts as a token
- [[subword-tokenization]] — one practical response to the open-vocabulary problem

## Active Recall

> [!question]- A sentence contains 100 tokens. How would you count its types, and why might the count differ depending on whether you apply case folding first?
> Count every unique string form — identical strings count as one type. Without case folding, `The` and `the` are two different types. After case folding to lowercase they collapse to one. So case folding reduces $|V|$ by merging case variants, which affects vocabulary size estimates and downstream model behavior.

> [!question]- What does the sub-linear exponent in Heaps' Law tell us about the open-vocabulary problem?
> Because $\beta < 1$, vocabulary grows much slower than corpus size. Doubling the corpus adds new types, but fewer and fewer per extra token. This means even an enormous training corpus will not have seen every word that appears at test time — rare and domain-specific types will always slip through. The law quantifies *how* the gap narrows, not that it closes.

> [!question]- Why does Heaps' Law motivate subword tokenization more directly than simply enlarging the vocabulary?
> Enlarging the vocabulary captures more types but only for forms seen in training data — truly novel forms (names, neologisms, typos) remain out-of-vocabulary regardless of vocabulary size. Subword tokenization decomposes any word into units that are almost certainly in the vocabulary (at worst, individual characters), so the open-vocabulary problem is eliminated by construction rather than mitigated by scale.
