---
type: concept
sources:
  - raw/week-04/w04-slides.pdf
  - raw/week-04/w04-slides-updated.pdf
  - raw/week-04/w04-l12-transcript.txt
status: draft
updated: 2026-04-25
---

*Word2vec learns dense, short word embeddings by training a binary classifier to predict whether a pair of words co-occurred. The classifier is never used — the vectors it used as parameters are. It's a self-supervised, predict-don't-count alternative to [[tf-idf]] and [[pmi|PPMI]].*

## Sparse vs Dense, Count vs Predict

[[tf-idf]] and [[pmi|PPMI]] produce **sparse, long** vectors (20,000–50,000 dimensions, most entries zero). Word2vec (Mikolov et al., 2013) produces **dense, short** vectors (50–1000 dimensions, most entries non-zero). Why bother?

- **Fewer parameters downstream.** Feeding a 300-dimensional dense vector into a classifier is much cheaper than a 50,000-dimensional sparse one.
- **Better generalisation.** *Car* and *automobile* are synonyms but live on distinct sparse dimensions, so a feature like *"previous word was car"* won't trigger on *automobile*. Dense embeddings place both near each other, so anything learned about one generalises to the other.
- **Empirically better.** On most downstream tasks, dense embeddings simply work better than sparse ones. The mechanism is compression: reducing to a low-rank approximation forces the model to find the structure that matters.

Word2vec is the canonical dense embedding method. Alternatives include **GloVe** (Pennington et al., 2014), **SVD / LSA** (a classical dimensionality reduction of the co-occurrence matrix), and the modern **contextual embeddings** (ELMo, BERT) that replace "one vector per lemma" with "one vector per occurrence."

## Skip-Gram with Negative Sampling (SGNS)

Word2vec comes in two flavours — **skip-gram** and **CBOW**. The course focuses on skip-gram with negative sampling (SGNS):

**Idea: predict rather than count.** Instead of tallying how often *apricot* co-occurs with every context word, train a classifier on a binary prediction task: *"is word c likely to appear near apricot?"* The task itself is throwaway — nobody cares about the classifier's output at test time. What we keep is the **weights**, which become the word embeddings.

**Big idea: self-supervision.** A word that actually occurs near *apricot* in the corpus counts as the "gold correct answer" for supervised learning. No human labels needed. Any running text generates training data.

### Four-step algorithm

1. **Build positive examples** $(t, c)$: take a target word $t$ and each context word $c$ in a small $\pm k$ window around it.
2. **Build negative examples** $(t, c_{neg})$: for each positive example, sample $k$ random words from the vocabulary as non-neighbours.
3. **Train a logistic regression classifier** to distinguish positive from negative pairs, using vector dot product as the similarity input.
4. **Extract the learned weights** as the embeddings.

### Training data

Assume a $\pm 2$ word window around the target *apricot* in the sentence *"lemon, a tablespoon of apricot jam, a pinch…"*:

```
...lemon, a [tablespoon of apricot jam, a] pinch...
              c1          c2  target    c3    c4
```

Positive examples (one per context word in the window):

| t | c |
|---|---|
| apricot | tablespoon |
| apricot | of |
| apricot | jam |
| apricot | a |

For each positive example, draw $k$ negative examples by sampling by word frequency (actually unigram frequency raised to the 3/4 power in the original paper — but the course treatment uses plain frequency). With $k = 2$:

| t | c | label |
|---|---|-------|
| apricot | aardvark | − |
| apricot | my | − |
| apricot | where | − |
| apricot | coaxial | − |
| apricot | seven | − |
| apricot | forever | − |
| apricot | dear | − |
| apricot | if | − |

### Similarity via dot product

The classifier's core: two words are similar if their embeddings have a high dot product. [[cosine-similarity|Cosine]] is just normalised dot product. Define

$$\text{Sim}(w, c) \propto \vec{w} \cdot \vec{c}$$

and turn it into a probability via the sigmoid from logistic regression:

$$P(+ \mid w, c) = \sigma(\vec{c} \cdot \vec{w}) = \frac{1}{1 + \exp(-\vec{c} \cdot \vec{w})}$$

$$P(- \mid w, c) = 1 - P(+ \mid w, c) = \sigma(-\vec{c} \cdot \vec{w})$$

Assuming independence across the $L$ context words in the window:

$$P(+ \mid w, c_{1:L}) = \prod_{i=1}^{L} \sigma(\vec{c_i} \cdot \vec{w}), \qquad \log P(+ \mid w, c_{1:L}) = \sum_{i=1}^{L} \log \sigma(\vec{c_i} \cdot \vec{w})$$

### The loss function

For one positive example $c_{pos}$ and $k$ negatives $c_{neg_1}, \ldots, c_{neg_k}$ paired with target $w$, minimise the cross-entropy loss:

$$L_{CE} = -\left[ \log \sigma(\vec{c}_{pos} \cdot \vec{w}) + \sum_{i=1}^k \log \sigma(-\vec{c}_{neg_i} \cdot \vec{w}) \right]$$

In words: **maximise** the similarity of the target with the true context word, **minimise** the similarity of the target with each negative-sampled non-neighbour.

### Learning: SGD

Stochastic gradient descent walks each parameter towards the direction of steepest descent on this loss. The update rule per training example:

$$\vec{c}_{pos}^{t+1} = \vec{c}_{pos}^t - \eta\,[\sigma(\vec{c}_{pos}^t \cdot \vec{w}^t) - 1]\, \vec{w}^t$$

$$\vec{c}_{neg}^{t+1} = \vec{c}_{neg}^t - \eta\,[\sigma(\vec{c}_{neg}^t \cdot \vec{w}^t)]\, \vec{w}^t$$

$$\vec{w}^{t+1} = \vec{w}^t - \eta \left[ [\sigma(\vec{c}_{pos}^t \cdot \vec{w}^t) - 1] \vec{c}_{pos}^t + \sum_{i=1}^k [\sigma(\vec{c}_{neg_i}^t \cdot \vec{w}^t)] \vec{c}_{neg_i}^t \right]$$

where $\eta$ is the learning rate. The first term pulls *apricot* closer to *jam* (positive); the second pushes *apricot* away from *matrix*, *Tolstoy*, etc. (negative). After enough epochs, words that share many contexts end up near each other, words that don't end up apart.

Initialisation is random $d$-dimensional vectors. Convergence in practice takes a single pass or a few passes over a large corpus (SGNS is trained on billions of tokens).

## Two Sets of Embeddings

SGNS actually learns **two** embeddings per word: one as a *target* (matrix $W$, used when the word is the centre of its window) and one as a *context* (matrix $C$, used when the word is a neighbour). Final representations usually *sum* the two: $\vec{x}_i = \vec{w}_i + \vec{c}_i$.

Formally the full parameter vector is

$$\boldsymbol{\theta} = \begin{bmatrix} W\ (|V| \times d)\\ C\ (|V| \times d) \end{bmatrix}$$

with $2|V|$ rows total: a target-role and context-role vector for each word.

## Properties of Trained Embeddings

### Window size shapes what gets learned

Small windows $(\pm 2)$ capture **syntactic / taxonomic** similarity — words that could substitute grammatically. *Hogwarts*'s nearest neighbours become other fictional schools: *Sunnydale, Evernight, Blandings*.

Large windows $(\pm 5)$ capture **topical relatedness** — words in the same semantic field. *Hogwarts*'s nearest neighbours become the Harry Potter world: *Dumbledore, half-blood, Malfoy*.

This is the [[lexical-semantics#Word Relatedness (Association)|similarity vs relatedness]] distinction from lexical semantics, now tunable via a hyperparameter.

### Analogical relations (the parallelogram method)

A long-standing observation (Rumelhart & Abrahamson 1973 in psychology; Turney & Littman 2005 and Mikolov et al. 2013 in NLP): analogies can be solved by vector arithmetic. To solve *apple is to tree as grape is to* \_\_\_, compute

$$\hat{b}^* = \arg\max_x \text{cosine}(x, a^* - a + b)$$

With $a = \text{apple}, a^* = \text{tree}, b = \text{grape}$, the closest word to $\vec{\text{tree}} - \vec{\text{apple}} + \vec{\text{grape}}$ is *vine*. Similarly:

- $\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$
- $\vec{\text{Paris}} - \vec{\text{France}} + \vec{\text{Italy}} \approx \vec{\text{Rome}}$

The visual is a parallelogram in embedding space: opposite sides represent the same semantic relation (gender, capital-of-country).

**Caveats.** The parallelogram method only works for **frequent words**, **small distances**, and **certain relations** — capitals-of-countries and parts-of-speech do reliably; many others don't (Linzen 2016, Gladkova et al. 2016, Ethayarajh et al. 2019a). Understanding *when* and *why* analogies work remains an open research question.

### Diachronic embeddings: meaning change over time

Hamilton, Leskovec & Jurafsky (2016) trained separate embeddings on different decades of text (~30M Google Books, 1850–1990) to track **semantic shift**:

- *gay*: 1900s neighbours include *daft, tasteful, cheerful*; 1990s neighbours include *lesbian, homosexual, bisexual*.
- *broadcast*: 1850s sits near *sow, seed, scatter* (agricultural sense); 1990s sits near *television, radio, bbc*.
- *awful*: 1850s is near *majestic, solemn, awe*; 1990s near *wonderful, weird, terrible*.

Each word's embedding moves through space as its usage changes. A simple, striking use of vector semantics as a historical-linguistics tool.

### Embeddings reflect cultural bias

The same method that recovers *king − man + woman ≈ queen* also produces:

- *father : doctor :: mother : x* → **nurse**
- *man : computer programmer :: woman : x* → **homemaker** (Bolukbasi et al. 2016)

The embeddings are doing exactly what they're trained to do — reflect the statistical regularities of the training corpus. When the corpus is the web, those regularities encode gender, racial, and cultural stereotypes. Systems that consume embeddings — hiring searches, content ranking, translation — propagate those biases downstream.

Garg, Schiebinger, Jurafsky & Zou (2018) used diachronic embeddings to quantify historical bias, finding e.g. that **competence adjectives** (*smart, wise, brilliant*) were biased toward men in 1910s embeddings with the bias decreasing 1960–1990, and that **dehumanising adjectives** (*barbaric, monstrous, bizarre*) were biased toward Asians in 1930s embeddings — patterns that match independently-measured old surveys. Embeddings become a window into the cultural record.

This is the [[harms-in-classification|harms]] story again, one layer deeper: not "biased training labels" but "biased text itself." No classifier is in scope yet, but the bias is already baked into the representation that every classifier will use.

## Summary: How to Train Word2vec

1. Start with $|V|$ random $d$-dimensional vectors as initial embeddings.
2. Take a corpus, extract positive $(t, c)$ pairs from co-occurrence windows and negative $(t, c_{neg})$ pairs by frequency-sampling the vocabulary.
3. Train the skip-gram classifier (logistic regression over dot-product similarity) by SGD to distinguish the two classes.
4. Throw away the classifier weights you don't need; keep the word embeddings (optionally summing target and context matrices).

The classifier is a scaffold. The embeddings are the product.

## Related

- [[vector-semantics]] — the framework: words as points, meaning from distribution
- [[tf-idf]] — the sparse, count-based alternative word2vec replaces for most uses
- [[pmi]] — Levy & Goldberg (2014) showed skip-gram-with-negative-sampling implicitly factorises a shifted-PMI matrix
- [[cosine-similarity]] — the metric over trained embeddings
- [[lexical-semantics]] — the structure the embeddings are trying to recover
- [[harms-in-classification]] — embeddings propagate corpus bias into every downstream model

## Active Recall

> [!question]- What's the core idea of word2vec in one sentence, and why "self-supervision"?
> **Train a classifier to predict whether a word $c$ appears near a target $w$; throw away the classifier and keep the word vectors it used as parameters as the embeddings.** It's self-supervised because the training labels come from the text itself — a word that actually appeared in the window counts as a positive example, no human annotation needed.

> [!question]- What are the four steps of skip-gram with negative sampling?
> (1) Treat each target word $t$ with a neighbouring context word $c$ as a **positive example** $(t, c)$. (2) For each positive example, **randomly sample $k$ other words** from the vocabulary (by frequency) as **negative examples**. (3) Train a **logistic-regression classifier** that takes the dot product $\vec{w} \cdot \vec{c}$ through a sigmoid to distinguish positives from negatives. (4) Use the **learned weights** (the word vectors) as the embeddings.

> [!question]- Write down the SGNS loss function for one positive example and $k$ negatives, and explain what each term does.
> $L_{CE} = -[\log \sigma(\vec{c}_{pos} \cdot \vec{w}) + \sum_{i=1}^k \log \sigma(-\vec{c}_{neg_i} \cdot \vec{w})]$. **First term** — maximise similarity of the target with the true context word (pull them closer in vector space). **Second term** — minimise similarity with each of the $k$ negative-sampled non-neighbour words (push them apart). Both terms use the sigmoid to turn a dot-product similarity into a probability.

> [!question]- How does the **window size** affect what kind of similarity skip-gram learns?
> **Small windows** ($\pm 2$) emphasise **syntactic / taxonomic** similarity — words that could substitute grammatically. *Hogwarts*'s neighbours become other schools (*Sunnydale, Evernight*). **Large windows** ($\pm 5$) emphasise **topical relatedness** — words in the same semantic field. *Hogwarts*'s neighbours become the Harry Potter world (*Dumbledore, Malfoy*). This is a tunable hyperparameter that makes the [[lexical-semantics#Word Relatedness (Association)|similarity vs relatedness]] distinction operational.

> [!question]- Explain the parallelogram analogy method and give one example where it works.
> To solve $a : a^* :: b : \_$, compute $\hat{b}^* = \arg\max_x \cos(x, \vec{a}^* - \vec{a} + \vec{b})$. Example: *king − man + woman* is closest to *queen*. Example: *Paris − France + Italy* is closest to *Rome*. The idea: if a consistent semantic dimension (gender, capital-of-country) shows up as a direction in embedding space, analogy queries can ride that direction. **Caveat**: it only works reliably for frequent words and a few relation types.

> [!question]- Why does SGNS train **two** embedding matrices $W$ and $C$, and what's done with them at the end?
> SGNS distinguishes a word in its **target role** (centre of a window — matrix $W$) from its **context role** (neighbour in someone else's window — matrix $C$). During training, the two are updated by different gradients. At the end, it's common to either (a) keep only $W$ or (b) sum them: the final vector for word $i$ is $\vec{w}_i + \vec{c}_i$. Summing uses both training signals and is often more stable.

> [!question]- Name three distinct reasons why dense embeddings tend to outperform sparse count-based vectors.
> (1) **Fewer downstream parameters** — a 300-dim vector is ~100× cheaper than a 30,000-dim sparse one. (2) **Better generalisation** — synonyms like *car* and *automobile* live on distinct sparse dimensions but end up near each other in dense space, so anything learned about one transfers to the other. (3) **Implicit regularisation** — compressing to low dimension forces the model to discard noise and keep the dominant statistical structure.

> [!question]- What's the connection between SGNS and [[pmi|PMI]] that Levy & Goldberg (2014) identified?
> SGNS's training objective (with $k$ negative samples per positive) is equivalent to factorising a **shifted PMI matrix** — roughly $\text{PMI}(w, c) - \log k$ — into two low-rank factors $W$ and $C$. So dense word2vec embeddings and classical PMI-then-SVD pipelines are doing the same computation in different clothes: both factor a co-occurrence-association matrix to low rank. The practical difference is online stochastic training vs. batch matrix factorisation.

> [!question]- Name two ways embeddings can be used as diagnostic tools beyond NLP tasks.
> (1) **Diachronic semantic change** — train separate embeddings per decade and watch a word move (e.g. *gay* 1900s → 1990s, *broadcast* agricultural → media). (2) **Cultural bias quantification** — measure how close *woman* is to competence adjectives vs. to homemaking terms; track the metric over decades (Garg et al. 2018) to recover documented historical stereotypes. Both treat the embedding space as a compressed record of the corpus's cultural content.

> [!question]- Slide MCQ: Which statements about Word2Vec are correct? (a) Skip-Gram predicts context words from a target word, treating context order as irrelevant within the window; (b) Skip-Gram predicts target words from context words, so it doesn't capture word order; (c) Word2Vec embeddings are static and cannot capture polysemy dynamically; (d) Training on a larger corpus negatively impacts embedding quality; (e) Semantically similar words have nearby vectors, so distance in embedding space reflects semantic similarity.
> **Correct: (a), (c), and (e).** (a) Skip-gram's direction is **target → context**: given a centre word, predict each context word in the window, treating them as an unordered set (order-within-window is not a feature). (b) inverts the direction — that's **CBOW**, not skip-gram. (c) is the core limitation of static embeddings: one vector per lemma averages over all senses. (d) is false — more data generally *improves* embedding quality, subject to the usual caveats. (e) is a direct restatement of why embeddings are useful: [[cosine-similarity|cosine]] distance corresponds to semantic similarity.
