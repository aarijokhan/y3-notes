---
type: concept
sources:
  - raw/week-04/w04-slides.pdf
  - raw/week-04/w04-slides-updated.pdf
  - raw/week-04/w04-l10-transcript.txt
status: draft
updated: 2026-04-25
---

*Define the meaning of a word by its distribution in language use. Represent that distribution as a vector in a multi-dimensional space. Similar words become nearby points; every downstream NLP model computes on those points instead of strings.*

## Two Ideas, One Method

Vector semantics rests on two ideas that become one technique when combined:

**Idea 1 — Defining meaning by linguistic distribution.** From Wittgenstein's *Philosophical Investigations* §43: *"The meaning of a word is its use in the language."* Operationalised by Zellig Harris (1954): *"If A and B have almost identical environments we say that they are synonyms."* A word's meaning is a function of the company it keeps — the neighbouring words or grammatical environments it occurs in.

**Idea 2 — Meaning as a point in space.** The connotation sketch in [[lexical-semantics#Connotation (Sentiment)|lexical semantics]] already represented a word as a 3D vector over $\{$valence, arousal, dominance$\}$. Generalise from 3 to thousands of dimensions, and the whole vocabulary becomes a cloud of points where geometry recovers semantic relationships.

The payoff of combining them: **build the space automatically** by seeing which words are nearby in text. No hand-annotation required.

### The ongchoi intuition

Suppose you've never seen *ongchoi* before but encounter:

- *Ong choi is delicious sautéed with garlic.*
- *Ong choi is superb over rice.*
- *Ong choi leaves with salty sauces.*

And you've also seen:

- *...spinach sautéed with garlic over rice.*
- *Chard stems and leaves are delicious.*
- *Collard greens and other salty leafy greens.*

Without anyone telling you, the shared context (*leaves, delicious, sautéed, salty*) is enough to conclude that ongchoi is a leafy green like spinach, chard, or collard greens. The distribution did the work.

## Embeddings: Vectors That Know Things

A word's vector in this space is called an **embedding** — the word is *embedded* into a (typically short, typically learned) space whose geometry reflects meaning. The phrase *"every modern NLP algorithm uses embeddings as the representation of word meaning"* is not hyperbole — embeddings replace strings as the input to every classifier, parser, retriever, or generator that runs in 2026.

The practical payoff is **generalisation**. Consider a sentiment classifier:

- With **words** (strings / one-hot): *"the previous word was `terrible`"* is a feature that only fires on exactly *terrible*. A test document containing *awful* gets no credit.
- With **embeddings**: the previous word has a vector, say $[35, 22, 17, \ldots]$. A test document with *awful* might produce $[34, 21, 14, \ldots]$. The classifier can generalise to similar-but-unseen words.

In the rest of this module, computing moves from string representations to meaning representations. Zhuangzi put it best: *"words are for meaning; once you get the meaning, you can forget the words."*

## Two Kinds of Embedding

The course covers two construction methods, paired with their typical matrix layout:

| | [[tf-idf]] | [[word2vec]] |
|---|---|---|
| Vector length | 20,000–50,000 | 50–1000 |
| Density | **Sparse** (mostly zero) | **Dense** (mostly non-zero) |
| How made | Weighted counts of nearby words/documents | Classifier trained to predict whether words co-occur |
| Baseline use | Information retrieval workhorse | Default word representation for deep models |

Later in the module, **contextual embeddings** (ELMo, BERT) produce a different vector for each *occurrence* of a word, not one fixed vector per type. This handles polysemy — the two senses of *mouse* get distinct vectors in context.

## The Matrix Behind the Vectors

Both tf-idf and (conceptually) word2vec rest on **co-occurrence matrices** — tables counting how often word *A* appears with word *B* or in document *D*.

### Term-document matrix

Each column is a document, each row is a vocabulary word, each cell is the count of that word in that document. Jurafsky's recurring example:

|      | As You Like It | Twelfth Night | Julius Caesar | Henry V |
|------|----------------|---------------|---------------|---------|
| battle | 1 | 0 | 7 | 13 |
| good | 114 | 80 | 62 | 89 |
| fool | 36 | 58 | 1 | 4 |
| wit  | 20 | 15 | 2 | 3 |

Two ways to read this matrix:

- **Columns = document vectors.** *As You Like It* $= [1, 114, 36, 20]$. Documents with similar vectors are similar documents — this is the **vector space model** of information retrieval (Salton 1971). The two comedies look alike; the two histories look alike.
- **Rows = word vectors.** *battle* $= [1, 0, 7, 13]$ — *"the kind of word that occurs in Julius Caesar and Henry V."* *fool* $= [36, 58, 1, 4]$ — *"the kind of word that occurs in comedies, especially Twelfth Night."* Words with similar row vectors occur in similar documents.

### Term-context (word-word) matrix

More common in modern practice. Columns are **context words** (a $\pm k$ window around each occurrence), not documents. Cells count how often the row-word appeared with the column-word within a window.

|           | aardvark | … | computer | data | result | pie | sugar | … |
|-----------|----------|---|----------|------|--------|-----|-------|---|
| cherry      | 0 | … | 2 | 8 | 9 | 442 | 25 | … |
| strawberry  | 0 | … | 0 | 0 | 1 | 60 | 19 | … |
| digital     | 0 | … | 1670 | 1683 | 85 | 5 | 4 | … |
| information | 0 | … | 3325 | 3982 | 378 | 5 | 13 | … |

The row vectors of *digital* and *information* are nearly parallel (both heavy on *computer, data*); the row vectors of *cherry* and *strawberry* are similar on *pie*. **Two words are similar in meaning if their context vectors are similar.** This *is* Harris's distributional hypothesis, expressed in matrix form.

### What's a "document"?

For tf-idf, a document can be a play, a Wikipedia article, a paragraph, or a single paragraph-sized chunk of a corpus — whatever partition gives useful discriminative signal. For a word-word matrix, the "document" is a small window around each token.

## Similarity = Geometry

Once words are points, similarity becomes a geometric question. The next concepts answer it:

- [[cosine-similarity]] measures angular similarity between vectors — the standard metric.
- [[tf-idf]] weights raw counts to suppress uninformative frequent words and highlight distinctive ones.
- [[pmi]] does the same job through an information-theoretic lens.
- [[word2vec]] replaces counting with predicting — learn a compact dense vector whose dot products approximate co-occurrence probabilities.

## Related

- [[lexical-semantics]] — the linguistic structure this representation is trying to recover
- [[cosine-similarity]] — measuring distance between vectors
- [[tf-idf]] — sparse weighted-count embedding (first concrete instance)
- [[pmi]] — information-theoretic re-weighting
- [[word2vec]] — dense, learned, predictive embedding (the modern default)
- [[bag-of-words]] — the document-level analogue: a document is a multiset of word types
- [[text-classification]] — downstream consumer of whatever representation we build

## Active Recall

> [!question]- State the distributional hypothesis in one sentence and explain what it buys you.
> **Words that appear in similar contexts tend to have similar meanings** (Harris, 1954). What it buys: you can build a representation of word meaning **automatically** from unannotated text — no hand-curated sense inventories needed. The *ongchoi* example illustrates the mechanism: shared neighbours (*leaves, salty, delicious*) are enough to place a new word near *spinach* in semantic space.

> [!question]- In a term-document matrix, what do rows and columns mean, and what are the two natural reading directions?
> **Rows** = vocabulary words. **Columns** = documents. **Column-wise reading**: each column is a document's vector (list of word counts), used for information retrieval — similar documents have similar vectors. **Row-wise reading**: each row is a word's vector (list of counts across documents), used for meaning — words that occur in similar documents have similar row vectors, and so are hypothesised to have related meaning.

> [!question]- How is a term-context (word-word) matrix different from a term-document matrix, and why is it more common in modern embedding work?
> In a term-context matrix, columns are **context words** rather than documents — a cell counts how often the row-word co-occurred with the column-word inside a small window (typically $\pm 2$ to $\pm 5$). It's more granular than the term-document matrix (a word's neighbours carry stronger meaning signal than the topic of its document), and it's the direct ancestor of modern word embeddings: [[word2vec]] effectively does a low-rank factorisation of a (PMI-transformed) term-context matrix.

> [!question]- Why is it useful to call a word vector an "embedding" rather than just a "vector"?
> Any list of numbers is a vector — including a one-hot encoding. *Embedding* emphasises that the vector is **placed in a semantic space**: similar words sit near each other, and dot products or cosines over embeddings correspond to meaningful distances. A one-hot vector for *cat* has no geometric relationship to a one-hot for *dog*; an embedding does.

> [!question]- Give one sentiment-classifier example of why embeddings generalise better than word-identity features.
> With word-identity features, *"the previous word was `terrible`"* only fires on exactly that string. At test time, a review containing *awful* gets no credit — the feature is zero. With embeddings, the feature is the previous word's vector — *awful*'s vector sits near *terrible*'s, so the downstream weights that treat *terrible*'s vector as a negativity signal also fire (softly) on *awful*. The classifier generalises from training-set words to unseen similar words.

> [!question]- Slide MCQ: Which statements about vector semantics are correct? (a) Word2Vec/GloVe embeddings are static and cannot capture polysemy effectively; (b) Higher cosine similarity between two word vectors indicates *no* relationship; (c) Vector semantics makes TF-IDF obsolete by relying only on geometry; (d) Contextual embeddings (BERT/ELMo) offer dynamic representations that capture polysemy and context-dependent meaning; (e) Vector-space dimensionality is inversely proportional to a model's ability to capture fine-grained semantic distinctions.
> **Correct: (a) and (d).** Static embeddings like Word2Vec give one vector per lemma — they average across senses, so they don't capture polysemy. Contextual embeddings fix this by producing a per-occurrence vector. (b) is inverted — higher cosine means *more* related. (c) is wrong — tf-idf remains the default information-retrieval baseline; it's complementary, not obsolete. (e) is backwards — more dimensions generally capture more distinctions (up to the limits of training data), not fewer.
