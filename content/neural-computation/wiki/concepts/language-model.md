---
name: language-model
description: "A model that assigns probabilities to sequences of words — equivalently, predicts the probability of an upcoming word given context. Two flavours by which context is allowed: autoregressive (left context only, used for generation) and masked (left + right context, used for representation/classification). Underlies autocomplete, spelling/grammar correction, machine translation rescoring, and modern chatbots."
type: concept
sources:
  - raw/week-09/w09-l19-transcript.txt
  - raw/week-09/w09-l20-transcript.txt
  - raw/week-09/w09-slides.pdf
status: stable
updated: 2026-04-27
---

*A language model does exactly one thing: it assigns probabilities to sequences of words. Everything else — generation, autocomplete, spell-check, translation rescoring, ChatGPT — is built on top of that one primitive. Given a sequence $w_1 \ldots w_n$, an LM tells you $P(w_1, \ldots, w_n)$, or equivalently the conditional $P(w_n \mid w_1, \ldots, w_{n-1})$ from which the joint can be recovered by the chain rule. The remarkable thing is how much you get from such a narrow primitive.*

## The definition

A **language model** assigns probabilities to sequences. Given a vocabulary $V$ and a sequence $W = (w_1, w_2, \ldots, w_n)$ with each $w_i \in V$:

$$P(W) = P(w_1, w_2, \ldots, w_n)$$

The closely related (and more useful) task is the **next-word distribution**:

$$P(w_n \mid w_1, w_2, \ldots, w_{n-1})$$

These two are linked by the chain rule of probability:

$$P(w_1, \ldots, w_n) = \prod_{k=1}^{n} P(w_k \mid w_1, \ldots, w_{k-1})$$

So an LM that can do *either* task can do the other. Almost every concrete LM (n-gram, RNN, GPT) directly outputs the conditional next-word distribution; the joint is computed by multiplying them along the sequence.

> [!info] ASIDE — How big is the joint, naively?
> Suppose vocabulary size $|V| = 10^5$ and sequence length $n = 10$. The joint distribution $P(W)$ is a table over $|V|^n = 10^{50}$ entries. There aren't enough atoms in the observable universe to store one probability per entry. So we cannot just enumerate it — we *must* exploit structure (the chain rule) and impose simplifying assumptions (Markov, neural compression). This combinatorial blowup is *the* motivation for everything in the LM literature.

## Two flavours: autoregressive vs. masked

Language models split by what context they're allowed to use to predict a word.

| Flavour | Predicts | Context | Generation? | Examples |
|---|---|---|---|---|
| **Autoregressive (causal)** | $P(w_n \mid w_{1:n-1})$ | left only | yes — sample next token, append, repeat | n-gram, RNN, LSTM, GPT |
| **Masked** | $P(w_k \mid w_{1:k-1}, w_{k+1:n})$ | left + right | no (not directly) | BERT, ELMo |

- **Autoregressive** models predict the next word from past words only. This makes them naturally generative: pick a starting context, sample the next word, append it, and repeat. ChatGPT, autocomplete, machine translation decoders all use autoregressive LMs.
- **Masked** language models predict a word from words on *both* sides — left and right context. They cannot generate by left-to-right rollout because each prediction needs future context that doesn't exist yet during generation. But they produce excellent contextualized representations of words, which is what BERT-style models are used for: feed embeddings into a classifier, retrieval system, etc.

The Markov assumption underlying the n-gram model below applies to the autoregressive case.

## Where language models are used

The narrowness of the LM primitive (just probabilities over sequences) is deceptive — once you have one, an enormous range of applications opens up:

- **Autocomplete / next-word prediction.** Phone keyboards, IDE code completion. Sample (or argmax) from $P(w_n \mid \text{prefix})$.
- **Spelling and grammar correction.** Score competing candidate sentences and pick the highest-probability one. *"OpenAI is great. I love **there** products"* gets corrected because $P(\text{their products}) \gg P(\text{there products})$ in context.
- **Machine translation, speech recognition.** The acoustic/translation model proposes candidates; the LM rescores them by fluency in the target language. This is why training a *better LM* improves an *unrelated* task — the LM is reused as a fluency scorer.
- **Chatbots / question answering.** Modern LLMs reframe everything as text completion. Prompt: *"Q: What is the capital of UAE? A:"* — sampling the continuation gives *"Abu Dhabi"*. The chatbot is just an autoregressive LM continuing a structured prompt.
- **Author identification.** Train a separate LM on each candidate author's writings; for a disputed text, the LM with lowest perplexity is the most likely author.

> [!tip] TIP — The "general-purpose task solver" framing
> A modern LLM does not have a separate module for translation, summarisation, code generation, etc. They all reduce to: **"given this prompt as text, what is the most likely continuation as text?"** The training objective (next-word prediction) is unchanged from a 1990s n-gram model — what changed is the architecture (Transformer), the data scale (trillions of tokens), and post-training alignment (instruction tuning, RLHF). The probabilistic-sequence-modelling frame is the through-line.

## The taxonomy of language models in this module

The module covers a chronological progression of LMs. Each fixes a problem with the previous one:

| Era | Family | Example | Fixes |
|---|---|---|---|
| 1990s | Statistical | [[n-gram-language-model\|n-gram]] | first practical LM; counts in a big table |
| 2003 | Feed-forward neural | Bengio et al. NPLM | learned [[word-embedding\|word embeddings]] → handles unseen contexts |
| 2010 | Recurrent neural | [[recurrent-neural-network\|RNN]] | unbounded context, sequential processing |
| 2014 | Gated recurrent | [[lstm\|LSTM]], stacked LSTM | long-range dependencies via gating |
| 2018 | Bi-directional | ELMo | contextualized embeddings |
| 2017–now | Transformer-based | BERT, GPT family | parallel training, very long context, attention |
| 2020+ | Large language models | GPT-3, GPT-4, Claude | scale + instruction tuning + RLHF unlock general-purpose problem solving |

Each row inherits the same probabilistic framing — $P(W)$ or $P(w_n \mid \text{context})$ — and changes the parametric form used to compute it.

## Connections

- **Underpins** [[n-gram-language-model]] — the simplest concrete LM, derived from the chain rule plus a Markov assumption.
- **Evaluated by** [[perplexity-nc|perplexity]] — the standard intrinsic metric for any LM regardless of architecture.
- **Generation requires** [[decoding-strategies]] — sampling/beam/greedy methods to turn the next-word distribution into actual text.
- **Modern LMs use** [[word-embedding|word embeddings]] — dense vectors that solve the "synonymy / unseen context" problem n-grams suffer from.
- **Implemented by** [[recurrent-neural-network|RNNs]], [[lstm|LSTMs]], and Transformer-based architectures (covered in week 11).
