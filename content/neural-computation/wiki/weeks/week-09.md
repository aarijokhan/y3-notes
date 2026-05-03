---
type: week
week: 9
title: "Language Modeling: From Counting to ChatGPT"
sources:
  - raw/week-09/w09-l19-transcript.txt
  - raw/week-09/w09-l20-transcript.txt
  - raw/week-09/w09-l21-transcript.txt
  - raw/week-09/w09-slides.pdf
  - raw/week-09/w09-problem-set.pdf
  - raw/week-09/w09-problem-set-solution.pdf
concepts:
  - "[[language-model]]"
  - "[[n-gram-language-model]]"
  - "[[perplexity-nc|perplexity]]"
  - "[[decoding-strategies]]"
  - "[[word-embedding]]"
  - "[[recurrent-neural-network]]"
  - "[[lstm]]"
status: stable
updated: 2026-04-27
---

> [!question]+ THE CRUX: A "language model" assigns probabilities to sequences of words — that's literally all it does. From a 1990s n-gram counting in a table to ChatGPT, the *primitive is unchanged*. What changes between them, and why does each change matter?

*This week is the bridge between the module's "models for vision" half and the upcoming "Transformers and attention" half. The narrative arc is one of escalating context: count occurrences in a tiny window (n-grams) → represent words as dense vectors so similar words generalise (word embeddings) → use neural recurrence to capture longer history (RNNs, LSTMs) → use attention to handle arbitrary positions in parallel (Transformers, week 11). The training objective stays the same throughout — predict the next word — but the parametric form, scale, and post-training (instruction tuning, RLHF) compound into systems that solve almost any text-describable task.*

## Where we left off

Week 8 ended with diffusion models producing remarkable images by iterated denoising. Week 9 turns to a different modality — text — and a different task — modelling sequences rather than synthesising images. The connecting thread is generative modelling: in both cases we estimate $p_{data}(x)$ and sample from it. Diffusion does it for images via a noise-removal chain; an autoregressive [[language-model|language model]] does it for text via the [[n-gram-language-model#The chain rule + Markov assumption|chain rule of probability]] applied left-to-right.

This week is also a *survey* week — many architectures introduced briefly, several pointing forward to week 11's deep dive into Transformers and attention. The lecturer notes that the third lecture (l21) was largely lost to a national holiday and admin; some of the later content (BERT, GPT, RLHF, CLIP) is therefore covered slide-only and at survey depth.

## What is a language model?

A [[language-model|language model]] is a probability distribution over sequences of words. Given a vocabulary $V$ and a sequence $W = (w_1, \ldots, w_n)$:

$$P(W) = P(w_1, w_2, \ldots, w_n)$$

Two equivalent formulations:

$$P(W) \quad \text{(joint over the sequence)} \qquad\text{or}\qquad P(w_n \mid w_1, \ldots, w_{n-1}) \quad \text{(next-word given prefix)}$$

These are linked by the [[n-gram-language-model#The chain rule + Markov assumption|chain rule]]. An LM that gives you either gives you both.

> [!info] ASIDE — Just one primitive, an enormous range of uses
> The lecturer emphasises this is *all* a language model does — assign probabilities to sequences. Every application is built on top:
> - **Autocomplete** — sample from $P(w_n \mid \text{prefix})$ to suggest the next word.
> - **Spell/grammar correction** — score competing candidates ("I love **there** products" vs "I love **their** products") and pick the higher-probability one.
> - **Machine translation, speech recognition** — used as a fluency rescorer over candidates from another model.
> - **Chatbots** — frame the question as a prompt; the LM continues it. *"Q: What is the capital of UAE? A: ___"* sampled to *"Abu Dhabi"*.
> - **Author identification** — train one LM per candidate author; the lowest-perplexity model on disputed text wins.

Two flavours, distinguished by what context they're allowed:

| | Autoregressive (causal) | Masked |
|---|---|---|
| Predicts | $P(w_n \mid w_{1:n-1})$ | $P(w_k \mid w_{1:k-1}, w_{k+1:n})$ |
| Context | left only | left + right |
| Generation? | yes | not directly |
| Examples | n-gram, RNN, GPT | BERT, ELMo |

## N-gram language models — count and divide

The simplest concrete LM: assume each word depends only on the previous $N - 1$ ([[n-gram-language-model#The chain rule + Markov assumption|Markov assumption]]) and estimate the conditional by counting.

For a **bigram** model:

$$P(w_n \mid w_{1:n-1}) \approx P(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$$

That's it. No gradients, no parameters in the neural sense — a single pass over the corpus building count tables, then divide.

> [!example]+ A worked bigram example (Berkeley Restaurant corpus)
> Out of 9,222 sentences from people calling restaurants:
>
> | $w_{n-1}$ | $w_n$ | count | unigram of $w_{n-1}$ | $P(w_n \mid w_{n-1})$ |
> |---|---|---:|---:|---:|
> | want | to | 608 | 927 | 0.66 |
> | to | eat | 686 | 2417 | 0.28 |
> | eat | chinese | 16 | 746 | 0.021 |
> | i | want | 827 | 2533 | 0.33 |
> | to | food | 0 | 2417 | 0 |
>
> So $P(\text{<SOS> i want chinese food <EOS>}) = P(\text{i}|\text{<SOS>}) \cdot P(\text{want}|\text{i}) \cdot P(\text{chinese}|\text{want}) \cdot P(\text{food}|\text{chinese}) \cdot P(\text{<EOS>}|\text{food}) \approx 3.1 \times 10^{-5}$. Note that the model "knows" obvious patterns ("want to" is common) and rejects ungrammatical ones ("to food" is zero) just from counting — no syntax knowledge baked in.

The size of the n-gram table grows as $|V|^N$. For $|V| = 10^5$:

| $N$ | Table size |
|---|---|
| 2 (bigram) | $10^{10}$ |
| 3 (trigram) | $10^{15}$ |
| 5 (5-gram) | $10^{25}$ |

For comparison, GPT-4 is estimated at $\sim 2 \times 10^{12}$ parameters — *vastly* smaller than the naïve 5-gram table. The lecturer notes that production n-gram models at Microsoft Research circa 2008 maxed out at 5-grams for English (largest data); 4-grams for everything else. Storage and data sparsity become limiting before modelling power does.

## Two structural problems with n-grams

Two failures n-grams cannot escape, even with enormous corpora and clever smoothing:

1. **No long-distance dependencies.** *"The **soups** that I made from that new cookbook I bought yesterday **were** delicious"* — subject–verb agreement spans 12 words, invisible to any tractable n-gram.
2. **No notion of word similarity.** Each word is a discrete token. Seeing *"the dog ran"* in training tells the model nothing about *"the cat ran"* — they share zero bigrams. The model cannot generalise across semantically similar contexts.

Both problems collapse once you replace discrete tokens with [[word-embedding|dense word vectors]]. This is the bridge to the neural era.

## The zeros problem

Any combination not seen in training has count zero, hence probability zero. *"ate breakfast"* gets $P = 0$ if the corpus has *"ate lunch"*, *"ate dinner"* but never *"ate breakfast"*. A single zero in a sentence kills the entire sentence probability (chain-rule product); a zero in the test set sends [[perplexity-nc|perplexity]] to infinity.

Classical fix: smoothing (Laplace, Kneser-Ney, etc.) — redistribute small probability mass to unseen n-grams. The deeper fix: use [[word-embedding|word embeddings]] so the model never assigns hard zero to plausible continuations — semantically similar words have similar vectors and inherit each other's probability mass.

## Evaluating language models — perplexity

Two ways to evaluate:

- **Extrinsic (in-vivo).** Plug the LM into a downstream task (translation, ASR) and measure end-to-end. Honest but expensive.
- **Intrinsic (in-vitro).** Compare LMs directly on probability assigned to held-out text — [[perplexity-nc|perplexity]].

[[perplexity-nc|Perplexity]] is the inverse probability of a test set, geometric-meaned per word:

$$\text{PP}(W) = \sqrt[N]{\frac{1}{P(w_1, \ldots, w_N)}}$$

Lower is better. Range $[1, \infty)$. Intuition: the *average effective branching factor* — "if perplexity is 100, the model is, on average, as uncertain as if it were choosing uniformly among 100 candidate words." Holding test set constant, on the Wall Street Journal:

| Model | Perplexity |
|---|---|
| Unigram | 962 |
| Bigram | 170 |
| Trigram | 109 |
| Modern LLM | 70–80 |

Going from "any of $10^5$ words" to "one of ~70 plausible candidates" is a $\sim 1400\times$ reduction in uncertainty — *that* is what all the LM machinery buys.

## Generation — turning a distribution into text

Given an autoregressive LM, generation is sampling iteratively. The mechanics — turning a discrete distribution into an actual word — go via a CDF: cumulate the probabilities, draw a uniform random in $[0, 1]$, return the word whose CDF interval contains it. See [[decoding-strategies]] for the full menu of strategies:

| Strategy | What it does | When to use |
|---|---|---|
| Greedy / argmax | Pick highest-probability word | Reproducible; deterministic |
| Beam search (size $k$) | Maintain top-$k$ partial hypotheses | Translation, summarisation |
| Ancestral sampling | Sample from full distribution | Maximally diverse but noisy |
| Temperature | Sharpen ($T < 1$) or flatten ($T > 1$) | Tune diversity vs. fluency |
| Top-$K$ | Keep top-$K$ words, sample | GPT-2 default; cheap |
| Top-$P$ (nucleus) | Keep words covering $\ge p$ of mass | Adapts to local sharpness |

> [!example]+ Why beam search beats greedy
> Toy: starting from "The", greedy picks "nice" (P = 0.5) then "woman" (P = 0.4) for total 0.20. Beam search (size 2) also tracks "dog" (P = 0.4); from "The dog" it picks "has" (P = 0.9) for total 0.36. Greedy missed the better path because it locked in on a locally suboptimal first word. *Beam search trades compute for finding higher-probability sequences; pure greedy gets stuck in local optima.*

## The author-identification side use

Like all generative models, an LM can be *inverted* to attribute. Train one LM on Shakespeare, another on the Wall Street Journal, another on Jane Austen. Given a disputed text, compute its perplexity under each. Lowest perplexity wins. Three sample passages from 3-gram models:

1. *"They also point to ninety nine point six billion dollars from two hundred four oh six three percent of the rates of interest"* — Wall Street Journal.
2. *"This shall forbid it should be branded, if renown made it empty."* — Shakespeare.
3. *"You are uniformly charming! cried he, with a smile of associating and now and then I bowed and they perceived a chaise and four to wish for."* — Jane Austen.

The corpus's distinctive vocabulary and structure leave a perplexity signature.

## From counting to embeddings — the first revolution

The structural fix to n-grams is to give each word a **dense vector** in $\mathbb{R}^d$ instead of treating it as a discrete token (see [[word-embedding]]). Train the vectors so similar-meaning words have similar vectors. Then:

- *"the dog ran"* and *"the cat ran"* are similar inputs (their embeddings are close), so the model generalises across them automatically.
- $P(\text{breakfast} \mid \text{ate})$ inherits probability from $P(\text{lunch} \mid \text{ate})$ and $P(\text{dinner} \mid \text{ate})$ via the geometry — no hard zeros.

The canonical methods:
- **Word2vec** (Mikolov 2013) — train a small network to predict context words from a centre word (skip-gram) or vice versa.
- **GloVe** (Pennington, Socher, Manning 2014) — fit word vectors to log of global co-occurrence counts.

Both produce 100-300-dimensional vectors with remarkable emergent geometry: $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$. The "gender" direction, "royalty" direction, "country–capital" direction all show up as fixed offsets — emergent, not designed.

> [!warning] CAUTION — Word2vec is *not* a language model
> Word2vec produces embeddings; it does not assign probabilities to sentences. To get an LM, you embed words then feed them into an LM architecture (FFN, RNN, Transformer). The 2003 Bengio FFN LM is the first widely-used such combination — and it both improved perplexity over n-grams *and* produced useful embeddings as a byproduct.

## Neural language model architectures (a quick survey)

Each architecture below fixes a problem with the previous. The lecturer ran through these as a tour, noting that the *training objective* (predict the next word) stays the same throughout — only the parametric form changes.

### Feed-forward NLM (Bengio 2003)

Look up the previous $N-1$ words in an embedding table, concatenate, pass through a hidden layer with $\tanh$, softmax over vocabulary. First neural LM that beat n-gram perplexity. Limitations: fixed-size input window; doesn't handle variable-length context.

### [[recurrent-neural-network|RNN]] (Mikolov 2010)

Maintain a hidden state $h_t$ updated each step from the current input and previous hidden state — same weights $W_h, W_e$ reused at every step. *In principle* unbounded context. *In practice* limited by vanishing/exploding gradients, plus sequential (not parallelisable). Better perplexity than FFN.

### [[lstm|LSTM]] (Hochreiter & Schmidhuber 1997, popularised in LMs ~2014)

Add a cell state with explicit forget/input/output gates. Information can flow through the cell across many steps with minimal modification, so gradients propagate cleanly. Effective context: hundreds of tokens. Computationally expensive; still sequential.

### Stacked / Bidirectional LSTM (ELMo, 2018)

Stack multiple LSTM layers; run forward and backward LSTMs and concatenate. The architecture behind **ELMo** — first widely-used contextualized [[word-embedding|word embeddings]]. Each occurrence of a word gets a different vector depending on the surrounding sentence. State of the art for representation tasks until BERT.

### Transformer-based (2017+, week 11)

Replace recurrence entirely with self-attention. Operations parallelise across positions (huge GPU speedup), and information flows directly between any two positions in a single layer (no vanishing-gradient bottleneck). Three sub-flavours:

- **Encoder-only** (BERT family, 2018) — masked LM. Predicts a randomly-masked word from both left and right context. Bidirectional, *not* directly usable for generation, but excellent for representations / classification / retrieval. Trained on 15%-token-masking + next-sentence-prediction; outputs feed into downstream classifiers.
- **Decoder-only** (GPT family, 2018+) — autoregressive LM. Predicts the next token from left context only, using *masked* self-attention (each position attends only to earlier positions). Natural for generation. Architecture is largely the same across GPT-1 → GPT-2 → GPT-3 → GPT-4: more layers, longer context, more parameters, more data.
- **Encoder-decoder** (T5, Whisper) — both halves. Encoder processes the input; decoder autoregressively generates the output, attending to both its prefix and the encoder's representations. Standard for sequence-to-sequence tasks (translation, summarisation, ASR).

Detailed treatment of attention, multi-head attention, positional encodings, and the full Transformer block deferred to **week 11**. This week's takeaway: Transformers are the architecture that finally made language modelling scale.

## The LLM era — alignment

A pre-trained autoregressive LM produces fluent text but is not necessarily *helpful*. Asked *"Explain the moon landing to a 6 year old in a few sentences"*, GPT-3 (raw) might continue with *"Explain the theory of gravity to a 6 year old. Explain the theory of relativity to a 6 year old in a few sentences. Explain the big bang theory to a 6 year old."* — perfectly plausible text completion (same prompt template repeated) but not what the user wanted.

Two post-training stages address this:

- **Instruction tuning.** Collect (instruction, desired output) pairs across many tasks; fine-tune the LM on this dataset. The model learns to interpret the prompt as an instruction and produce a response, not just continue the text. After instruction tuning, the same prompt produces *"A giant rocket ship blasted off from Earth carrying astronauts to the moon..."*
- **RLHF — Reinforcement Learning from Human Feedback.** Step 1: train a separate **reward model** to predict human ratings of LM outputs. Step 2: fine-tune the LM with reinforcement learning to produce high-reward outputs. Used in ChatGPT, Claude, etc., to shape style, harmlessness, honesty, etc. — preferences that are hard to express as supervised labels.

Combined: pretraining (next-word prediction at scale) → instruction tuning → RLHF. Each stage moves the model from "text continuer" to "task solver."

## Multimodal models — CLIP

A bridge to non-text modalities. **CLIP** (Contrastive Language–Image Pre-training) trains an image encoder and a text encoder *jointly* with a [[contrastive-learning|contrastive loss]]: matched (image, caption) pairs should have aligned embeddings; unmatched pairs should be far apart.

The contrastive matrix: each batch of $N$ image-caption pairs produces an $N \times N$ similarity matrix. Diagonal entries (matched pairs) are pushed up; off-diagonal entries (unmatched pairs) are pushed down. After training, image and text live in a shared embedding space.

**Zero-shot classification** falls out for free: to classify an image among classes {"plane", "car", "dog", "bird"}, encode the prompts *"a photo of a plane"*, *"a photo of a car"*, etc. with the text encoder, encode the image with the image encoder, return the class whose text embedding has the highest similarity to the image embedding. No labelled training data needed for the new task — the joint embedding space transfers.

CLIP is the connective tissue for modern multimodal systems: text embeddings into Stable Diffusion, image-grounded reasoning in GPT-4V, etc.

## Problem-set lessons (week 9)

The problem set is a hand-computation drill, useful for confirming the n-gram mechanics:

- **Q1 — Unigram model on a small corpus.**
  - Probabilities: divide each word's count by the total count (60). $P(\text{the}) = 15/60 = 0.25$, $P(\text{<EOS>}) = 0.25$, etc.
  - Sentence *"the cat sat on the mat <EOS>"*: product of unigram probabilities, $\approx 2.6 \times 10^{-6}$.
  - Most probable sequence under a unigram model: tied between "the the the … the <EOS>" and "<EOS>" alone, both at probability $0.25$ per emitted token. *A unigram is structurally biased toward whichever word is most frequent — no syntactic structure.*
  - Most probable length-4 sequence (3 words + <EOS>): "the the the <EOS>" at $0.25^4 = 0.0039$.
- **Q2 — Bigram model, same corpus, with row-of-counts table.**
  - Conditional probabilities by row-normalising. $P(\text{the} \mid \text{<SOS>}) = 5/5 = 1$ (only one start in this toy corpus), $P(\text{cat} \mid \text{the}) = 4/12 \approx 0.33$, etc.
  - Sentence "the cat sat on the mat" with start/end tokens: $1 \times 0.33 \times 0.5 \times 0.67 \times 0.67 \times 0.17 \times 1 \approx 0.012$ — *much higher than the unigram estimate*, because the bigram model captures local structure (e.g., $P(\text{cat} \mid \text{the})$ is learned to be high).
  - Greedy search from `<SOS>`: `<SOS> the cat sat on the cat sat on the` — falls into a *cycle* (the → cat → sat → on → the → ...). A common failure mode: greedy decoding finds high-probability loops and gets stuck. Beam search or stochastic sampling would escape.

## Concepts introduced this week

- [[language-model]] — the framing: probability over sequences, autoregressive vs masked, applications.
- [[n-gram-language-model]] — the canonical statistical LM: chain rule + Markov assumption + MLE counting; zeros problem; structural limitations that motivate the neural transition.
- [[perplexity-nc|perplexity]] — the standard intrinsic evaluation metric; per-word geometric-mean inverse probability; intuition as average branching factor.
- [[decoding-strategies]] — greedy, beam, ancestral sampling, temperature, top-K, top-P; the menu for turning the next-word distribution into actual text.
- [[word-embedding]] — dense vectors per word; word2vec and GloVe; emergent geometry; the bridge from discrete-token n-grams to neural LMs.
- [[recurrent-neural-network]] — first formal introduction; weight sharing across time; vanishing/exploding gradients; sequential processing.
- [[lstm]] — gated recurrence with explicit cell state; addresses vanishing gradients via additive memory channel; state-of-the-art for sequence modelling pre-Transformer.

## Connections

- **Builds on** [[softmax]] — every neural LM ends in a softmax over vocabulary; temperature scaling for [[decoding-strategies|generation]] modifies that softmax.
- **Builds on** [[backpropagation]] — RNNs/LSTMs are trained by backpropagation through the unrolled time graph (BPTT).
- **Builds on** [[representation-learning]] and [[contrastive-learning]] — [[word-embedding|word embeddings]] are an instance of the same recipe used for image representations in week 6; CLIP extends contrastive learning to cross-modal text+image.
- **Sets up week 11 (Transformers and attention)** — this week introduces Transformer-based LMs at the survey level (BERT, GPT family, encoder-decoder); week 11 will derive self-attention, multi-head attention, positional encoding, and the full Transformer block.

## Open questions

- **How does perplexity translate across architectures?** A 109 perplexity for a trigram on WSJ vs 70 for a modern LLM on a different test set isn't directly comparable. Standard benchmarks (WikiText, Penn Treebank) help, but reported perplexity is now usually per *token* (subword), not per word, and the rescaling depends on tokenisation.
- **The recurrent renaissance.** Transformers won by replacing recurrence with attention — but their KV-cache memory grows linearly with sequence length, which is becoming a bottleneck. State-space models (S4, Mamba) revisit recurrent-style architectures with engineered cell-state dynamics. This is an active area; LSTMs may not be entirely retired.
- **Why instruction tuning works so well from so little data.** A few thousand (instruction, response) pairs are enough to dramatically reshape a 100B-parameter LM. The mechanism — and the limits — are not fully understood.
- **Emergent abilities.** Modern LLMs exhibit capabilities (reasoning, structured outputs, in-context learning) that don't appear at smaller scale. The "scale → emergent property" relationship is one of the open empirical surprises of the last few years.
