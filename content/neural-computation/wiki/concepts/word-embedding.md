---
name: word-embedding
description: A dense vector representation of words in $\mathbb{R}^d$ (typically $d = 100$–$300$) such that semantically similar words have geometrically close vectors. Solves the "discrete-token" weakness of n-gram LMs — once words live in a continuous space with similarity, learning about one word transfers to its neighbours, and unseen contexts no longer get probability zero. Word2vec (Mikolov 2013) and GloVe (Pennington 2014) are the canonical static embedding methods; modern contextualized embeddings (ELMo, BERT) are the next generation.
type: concept
sources:
  - raw/week-09/w09-l20-transcript.txt
  - raw/week-09/w09-slides.pdf
status: stable
updated: 2026-04-27
---

*The first major revolution in language modelling between n-grams and modern Transformers. The premise: instead of treating each word as an opaque, atomic symbol (the n-gram view, where "cat" and "dog" are unrelated tokens), assign each word a vector in $\mathbb{R}^d$. Train the vectors so that words appearing in similar contexts get close vectors. The geometry then encodes semantic structure — `king` minus `man` plus `woman` lands near `queen`, not because that's coded in but because it falls out of the training objective. Once words have continuous representations, neural language models can generalise across them: learning that `the dog ran` is plausible automatically makes `the cat ran` plausible, because cat and dog are nearby in vector space.*

## The problem word embeddings solve

[[n-gram-language-model|N-gram language models]] treat each word as a discrete token. The word index in the vocabulary is arbitrary — `cat` is index 752, `dog` is index 9101, `Wednesday` is index 7392 — and these numbers carry no meaning. Two consequences:

- **No transfer between similar words.** Seeing *"the dog ran"* in training tells the model nothing about *"the cat ran"* — they share zero bigrams. Each n-gram must be observed in its exact form to be assigned non-zero probability. Synonyms, near-synonyms, and grammatical variants are all *equally* unrelated to the model.
- **The zeros problem.** Any combination not seen in training has probability zero (see [[n-gram-language-model#The zeros problem]]). The training corpus, however large, will always be a vanishingly small subset of the syntactically valid sentences a fluent speaker could produce.

> [!example]+ The breakfast/dinner example
> Suppose the training corpus contains *"ate lunch"*, *"ate dinner"*, but never *"ate breakfast"*. An n-gram model says $P(\text{breakfast} \mid \text{ate}) = 0$. But intuitively, *breakfast*, *lunch*, and *dinner* are all "meals you eat", so $P(\text{breakfast} \mid \text{ate})$ *should* be similar to $P(\text{lunch} \mid \text{ate})$. To make that work, the model needs to *know* that breakfast and lunch are similar — which means representing them by vectors close to each other rather than by unrelated discrete tokens.

The fix: replace each word's discrete index with a **dense vector** in $\mathbb{R}^d$, learned so that similar words have similar vectors. Now the LM's output is computed by combining vectors via matrix multiplications and non-linearities, and the resulting probability distribution is *smooth* — never zero, smoothly varying with the input vectors. The breakfast bigram inherits probability from the lunch and dinner bigrams because their vectors are close.

## The distributional hypothesis

The construction principle behind every word embedding method is the **distributional hypothesis** (Harris 1954, popularised by Firth):

> *"You shall know a word by the company it keeps."*

If two words appear in similar contexts — surrounded by similar surrounding words — then they likely *mean* similar things. So if we represent each word by some summary of its typical contexts, similar-meaning words will get similar summaries.

A pre-neural way to make this concrete: for each word $w$, build a count vector indexed by vocabulary, where the $i$-th entry counts how many times word $i$ appeared in a window around $w$ in the corpus. Words with similar context-count vectors are similar in meaning. This co-occurrence vector is, however, $|V|$-dimensional and very sparse. Compress it (PCA, SVD) to a manageable dimension $d \approx 100$–$300$ and you have a usable dense representation.

Word2vec and GloVe automate this with neural-style training objectives that are far more sample-efficient than counting + SVD, but the hypothesis they're operationalising is identical.

## Word2vec (Mikolov et al., 2013)

The canonical neural method. The training objective is a small classifier:

> Given a word $w_t$, predict the words in its surrounding context window (skip-gram), or vice versa (CBOW — predict $w_t$ from its context).

The model has two embedding matrices, $E_{\text{in}}$ and $E_{\text{out}}$, each $|V| \times d$. Predicting a context word $c$ from a centre word $w$ is done by computing $E_{\text{in}}[w]^\top E_{\text{out}}[c]$ and softmax-normalising over the vocabulary. Train via maximum-likelihood gradient descent. The matrix $E_{\text{in}}$ rows are the final word embeddings.

The training objective never directly says "make synonyms have close vectors" — but because the same word $w$ is asked to predict the same kinds of context words across different occurrences, words that appear in similar contexts end up needing similar vectors to do their predictive job, and the result is the desired geometry.

## GloVe (Pennington, Socher, Manning, 2014)

A complementary method that fits embeddings to *global co-occurrence statistics*: build the matrix $X$ where $X_{ij}$ counts how often word $i$ appears in the context of word $j$, then learn vectors $v_i, v_j$ such that

$$v_i^\top v_j + b_i + b_j \approx \log X_{ij}$$

i.e., the inner product of two word vectors approximates the log co-occurrence count. Empirically very competitive with word2vec; both produce the same kind of geometry. Word2vec is "implicit, neural, predictive"; GloVe is "explicit, matrix-factorisation, count-based" — the field briefly debated which was conceptually better, the consensus settled on "both work fine."

## The geometry

The trained vector spaces have a remarkable structure beyond simple "similar words are close". Differences between embeddings encode *relations*:

$$v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$$

Direction "from man to woman" is approximately the same as "from king to queen", "from uncle to aunt", "from nephew to niece". This **emergent** linear analogy structure was not built in — it falls out of training on co-occurrence statistics. Similar regularities show up for verb tenses (*walking* − *walked* ≈ *running* − *ran*), country–capital pairs (*Paris* − *France* ≈ *Tokyo* − *Japan*), and many others.

The lecturer's example: $v_{\text{man}} - v_{\text{woman}}$ has a "gender" direction; adding it to *cooking* gives a different word — the geometry encodes more than just word similarity.

> [!info] ASIDE — These are emergent properties, not design choices
> Word2vec and GloVe both have very simple training objectives — predict a context word, or fit a co-occurrence count. Nobody told the model "encode gender as a fixed direction". The fact that gender, royalty, country, tense, etc. all show up as linear directions in the trained space is **emergent** — a consequence of the data structure plus the training objective. This is the first widely-discussed example of *emergent properties* in modern ML, predating LLM emergent properties by a decade.

## Static vs. contextualized embeddings

Word2vec and GloVe produce **static embeddings**: each word has *one* fixed vector. The word `bank` gets a single representation regardless of whether the sentence is about a river or a financial institution.

Modern alternatives — ELMo, BERT, GPT-internal embeddings — are **contextualized**: the embedding of `bank` depends on the surrounding sentence. Each occurrence in a corpus gets a different vector, computed by passing the full sentence through a deep network. This resolves polysemy (different meanings of `bank`) at the representation level. The lecturer notes:

> *"With BERT, we have contextualized embeddings — meaning each word has a representation depending on the context within the next sentence. The same word in a different sentence will have a different representation."*

For one-shot tasks (semantic similarity between word pairs, analogy benchmarks), static embeddings are often sufficient. For downstream NLP (classification, QA, NER), contextualized embeddings are dramatically better — they're essentially what every modern transformer-based system uses internally.

## Cross-domain generality of the recipe

The "represent as a dense vector trained by predicting context" recipe generalises far beyond words. The same construction underlies:

- **Sentence embeddings** — Sentence-BERT, Universal Sentence Encoder.
- **Image embeddings** — features from a pretrained CNN/ViT.
- **Cross-modal embeddings** — CLIP places images and text captions in the same space.
- **Recommendation systems** — user and item embeddings co-trained from interaction data.

The representation-learning pattern (week 6) is the same; only the input modality and context-prediction task change. Word embeddings were just the first place this recipe demonstrably worked at scale.

## How embeddings hook into a neural language model

In a feed-forward neural LM (Bengio et al. 2003, the first), the architecture is:

1. The previous $N-1$ words are looked up in an embedding table $E \in \mathbb{R}^{|V| \times d}$ to produce vectors $v_1, \ldots, v_{N-1} \in \mathbb{R}^d$.
2. The vectors are concatenated and passed through a hidden layer with $\tanh$.
3. A softmax head over the vocabulary produces $P(w_n \mid w_{1:n-1})$.

The embedding table $E$ is learned end-to-end with the rest of the network — same backprop, same loss (cross-entropy over the next-word distribution). At training time the embeddings co-evolve with the prediction task. This is why even from the very first neural LM (which beat n-gram perplexity), the embeddings learned were already meaningful — they had to be, to make the next-word task solvable.

Modern Transformer LMs do the same thing structurally — input tokens are looked up in an embedding table — except the embeddings then pass through self-attention layers that re-mix them according to context, producing the contextualized representations.

## Connections

- **Solves problems of** [[n-gram-language-model]] — zeros, no synonymy generalisation.
- **Used by** [[language-model|all neural LMs]] — feed-forward, RNN, LSTM, Transformer all rely on word embeddings as input.
- **An instance of** [[representation-learning]] — same recipe (predict context to learn good representations) used in week 6 for [[contrastive-learning|images]].
- **Static embeddings → contextualized embeddings** — ELMo (2018) and BERT (2018) are the next generation; week 11 covers Transformer-based contextualized representations.

## Common pitfalls

- **Confusing the embedding matrix with the model's predictions.** $E$ maps each word to a vector — it's a *representation*, not a probability distribution. The probability distribution $P(w_n \mid \text{context})$ comes from a *separate* output layer (the softmax over vocabulary).
- **Assuming the geometry is exact.** Analogies like $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}}$ land *near* $v_{\text{queen}}$, not exactly on it. The structure is approximate, sometimes brittle, and biased (gender stereotypes etc. are encoded too — a fairness concern).
- **Treating word2vec as a language model.** It's not. Word2vec produces embeddings; it does not assign probabilities to sentences. To get an LM, you embed words then feed them into an LM architecture.
