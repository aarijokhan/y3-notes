---
type: week
week: 2
title: "Week 2 — Counting Words: N-gram Language Models"
dates: —
sources:
  - raw/week-02/w02-slides.pdf
  - raw/week-02/w02-l04-transcript.txt
  - raw/week-02/w02-l05-transcript.txt
  - raw/week-02/w02-l06-transcript.txt
  - raw/week-02/w02-lab-01.pdf
  - raw/week-02/w02-lab-02.pdf
  - raw/week-02/w02-lab-01-solutions.pdf
  - raw/week-02/w02-lab-02-solutions.pdf
concepts:
  - "[[n-gram-language-models]]"
  - "[[perplexity-nlp|perplexity]]"
  - "[[evaluation-methodology]]"
  - "[[smoothing]]"
  - "[[interpolation-and-backoff]]"
status: stable
updated: 2026-04-20
---

> [!question]+ THE CRUX: You have a tokenized corpus and want to predict the next word — how do you build the simplest possible model for that, how do you know whether it works, and what goes catastrophically wrong when it encounters something it has never seen?

*The simplest language model counts adjacent word pairs and normalizes — and that is enough to introduce nearly every problem that modern LLMs still face: evaluation, sparsity, smoothing, and the gap between memorization and generalization.*

---

## From Tokens to Prediction

Week 01 gave us tokens — a segmented, normalized text stream. The first real question of NLP is: *what can we do with it?*

The most fundamental task is **prediction**. Given a sequence of words, what comes next? "The water of Walden Pond is beautifully ___." Your brain instantly narrows the possibilities to a handful of adjectives — blue, clear, green. A good language model should do the same.

The probability of any sentence can be written exactly using the **chain rule** — a product of conditional probabilities, each conditioning on the full prior history. The problem is that estimating $P(w_n \mid w_1 \ldots w_{n-1})$ requires having seen that exact prefix in the training corpus. With a vocabulary of $V$ words there are $V^{n-1}$ possible prefixes of length $n-1$; for English-scale vocabularies this explodes to billions even for 4-word contexts. No corpus can supply reliable counts across all those prefixes. The required data grows exponentially with sentence length — this is not a problem more data solves, it is a structural impossibility.

The fix is the **Markov assumption**: throw away all but the last $N-1$ words of history. Suddenly the conditioning context is small and dense enough to count reliably. An [[n-gram-language-models|n-gram language model]] is the result: estimate $P(\text{next word} \mid \text{last } N-1 \text{ words})$ by counting how often that word followed that short context in a training corpus. A **bigram** model conditions on just the previous word; a **trigram** on the previous two. The probability of an entire sentence is the product of these conditional probabilities.

The estimation is embarrassingly simple. Count how many times "want to" appears in your corpus; divide by how many times "want" appears in any bigram. That ratio — $C(\text{want, to}) / C(\text{want}) = 608 / 927 = 0.66$ — is your maximum likelihood estimate of $P(\text{to} \mid \text{want})$. Do this for every bigram in the corpus and you have a complete model.

> [!question]- You have a corpus of 3 sentences. The bigram "green eggs" appears 2 times, and "green" appears 5 times total. What is P(eggs|green)? If "green ham" never appears, what is P(ham|green)?
> $P(\text{eggs} \mid \text{green}) = 2/5 = 0.4$. $P(\text{ham} \mid \text{green}) = 0/5 = 0$. That zero is the seed of the biggest problem in n-gram modeling — any sentence containing "green ham" now has probability zero, no matter how reasonable it is.

## What the Model Produces

You can visualize what a language model has learned by **sampling** from it — generating text word by word. Start from $\langle s \rangle$, sample the first word from $P(w \mid \langle s \rangle)$, then the second from $P(w \mid w_1)$, and so on until $\langle /s \rangle$.

The results are revealing. On Shakespeare, a unigram model produces gibberish: *"To him swallowed confess hear both."* A bigram model starts to sound vaguely English: *"Why dost stand forth thy canopy, forsooth."* A 4-gram model produces eerily authentic text: *"King Henry. What! I will go seek the traitor Gloucester."*

Why does quality improve so dramatically? More context means better predictions. But there is a dark side: with a vocabulary of 29,066 types, 99.96% of possible bigrams were never observed. By the time you reach 4-grams, the model is often reproducing verbatim training data — it is memorizing, not generalizing.

## Measuring Quality

How do we know whether one model is better than another? We could embed it in a real task — machine translation, speech recognition — and measure task performance. This is **extrinsic evaluation**, and it is the gold standard. But it is expensive and task-specific.

The faster alternative is **intrinsic evaluation**: [[perplexity-nlp|perplexity]], the inverse probability of a held-out test set, normalized per word. Lower perplexity means the model is less surprised by the test data — it assigns higher probability to what actually occurred. On WSJ text, a unigram model achieves perplexity 962; a bigram drops to 170; a trigram to 109. More context helps.

But computing perplexity requires a test set **separate** from training. And if you test repeatedly and tune accordingly, you are implicitly training on the test set. The solution: a three-way split — training, dev, and test — where the test set is touched exactly once (see [[evaluation-methodology]]).

> [!question]- A bigram model assigns perplexity 200 to a test set, and a trigram model assigns perplexity 150 to the same test set. Which model is better at predicting words? What if they were tested on different test sets?
> On the same test set, the trigram (PP 150) is better — lower perplexity means higher probability assigned to the actual words. If tested on different test sets, the comparison is meaningless: perplexity depends on both the model and the data. A PP of 150 on easy text (small vocabulary, predictable patterns) may be worse than PP 200 on hard text.

## The Fatal Flaw

Everything collapses the moment the test set contains a single bigram the model has never seen. If $P(\text{breakfast} \mid \text{ate}) = 0$, the entire sentence gets probability zero, and perplexity is undefined. This is the **zero-probability problem**, and it is devastating because language is creative — people constantly produce novel word combinations.

[[Smoothing]] is the first line of defense. Add-one (Laplace) smoothing pretends every bigram was seen one extra time. It is simple and guarantees no zeros — but it is too aggressive for n-grams. With a vocabulary of 1,446 types, adding 1 to every cell redistributes so much mass to the 99%+ zero entries that observed bigrams lose most of their probability. Smoothing can actually *lower* the probability of an observed sentence (from $8.48 \times 10^{-5}$ to $2.41 \times 10^{-6}$ in the BeRP example).

The real fix is [[interpolation-and-backoff]]: mix the trigram model with bigram and unigram models so that if the trigram count is zero, the lower-order models can still contribute. Linear interpolation — $\hat{P} = \lambda_1 P_{\text{tri}} + \lambda_2 P_{\text{bi}} + \lambda_3 P_{\text{uni}}$ — is the standard approach. The $\lambda$ weights are learned on held-out data. Interpolation generally outperforms backoff because it always uses evidence from all model orders rather than switching between them.

Stronger smoothing methods go further. **Absolute discounting** subtracts a fixed $d \approx 0.75$ from every non-zero count — justified by Church & Gale (1991), who found that bigram counts in a held-out set are consistently $c - 0.75$ relative to training. **Kneser-Ney smoothing** extends absolute discounting with a smarter lower-order model: instead of the raw unigram $P(w)$, use the **continuation probability** — how many distinct contexts $w$ has followed — so that a word like "Kong" (locked to "Hong Kong") correctly gets low probability after "reading", while "glasses" (appearing after many words) gets high probability. Extended interpolated Kneser-Ney is the production default for n-gram LMs. For web-scale training (trillions of words), **stupid backoff** — a 0.4-multiplier heuristic that does not produce a valid probability distribution — wins on speed. See [[smoothing]] for the full treatment.

> [!question]- Smoothing can actually lower the probability of an observed sentence compared to unsmoothed MLE. How is that possible?
> Adding 1 to every bigram count adds $V$ (the vocabulary size) to each row's denominator. When $V$ is large, the denominator grows substantially, pulling down the probability of every bigram — including observed ones. The total mass removed from observed events equals the mass redistributed to the thousands of previously-zero cells. If the table is 99%+ zeros (as it typically is for n-grams), most of the mass flows to unseen events.

> [!info] ASIDE — N-grams and LLMs
> N-gram models are studied not because anyone deploys bigram LMs in production, but because they introduce the foundational issues that large language models inherit: training vs test sets, perplexity, sampling, and the tension between memorization and generalization. Every concept from this week maps directly to an LLM concern — just at a scale where the mechanics are transparent.

> [!tip] TIP — Exam calculation problems
> The labs emphasize two calculation types: (1) computing bigram probabilities from a count table and multiplying for sentence probability, and (2) computing perplexity from probabilities. Always check whether `<s>` and `</s>` tokens are included in the counts — they affect the numbers. And always check whether smoothing is applied before computing.

---

## Concepts Introduced This Week

- [[n-gram-language-models]] — predict the next word from the previous N−1 words using counted corpus frequencies
- [[perplexity-nlp|perplexity]] — intrinsic metric: inverse probability of the test set, normalized per word; lower is better
- [[evaluation-methodology]] — extrinsic vs intrinsic evaluation, train/dev/test split, why three datasets
- [[smoothing]] — redistribute probability mass to unseen n-grams; add-one, add-k, and why add-one fails for LMs
- [[interpolation-and-backoff]] — mix model orders to handle unseen n-grams; lambdas learned on held-out data

## Connections

Builds on [[week-01]]: we need [[tokenization|tokens]] and a [[type-and-token|vocabulary]] to count n-grams. The [[corpora|corpus]] determines everything the model knows.

Sets up [[week-03]] — neural approaches to the same prediction task, or text classification, depending on the syllabus.

## Open Questions

- How do modern LLMs handle the zero-probability problem differently from smoothing?
- Is there a theoretical lower bound on perplexity for English?
- How does the choice of tokenization (word-level vs [[byte-pair-encoding|BPE]]) interact with n-gram model quality?
