---
title: "Week 4 Flashcards — Vector Semantics"
week: 4
topic: "Vector Semantics — TF-IDF, PMI, Word2Vec"
deck: NLP::Week-04
---

TARGET DECK
NLP::Week-04

# Week 4 Flashcards

## Lexical Semantics

> [!question]- What is a lemma and what is a sense?
> - **Lemma**: the dictionary headword form of a word (e.g., `mouse` is the lemma for `mouse` and `mice`).
> - **Sense**: a discrete meaning of a lemma. The lemma `mouse` has at least two senses: rodent, and pointing device.
> - A lemma can carry **multiple senses** (polysemy); a sense is the unit at which meaning is defined.
<!--ID: 1777386639948-->





> [!question]- Define synonymy, similarity, relatedness, antonymy, and connotation.
> - **Synonymy**: same meaning in some context (e.g., *couch* / *sofa*) — rarely perfect substitutability.
> - **Similarity**: words share core meaning (*cat* / *dog* — both mammals) without being interchangeable.
> - **Relatedness/association**: words share a topic without being similar (*coffee* / *cup*, *doctor* / *hospital*).
> - **Antonymy**: opposite along some dimension (*hot* / *cold*, *up* / *down*) — but otherwise highly similar.
> - **Connotation**: affective meaning along axes like valence (pleasantness), arousal (intensity), dominance (control).
<!--ID: 1777386639949-->





> [!question]- What is the distributional hypothesis?
> Wittgenstein: *"The meaning of a word is its use in the language."* Operationalised by Zellig Harris (1954): *"If A and B have almost identical environments we say that they are synonyms."* — words appearing in similar contexts have similar meanings. The basis of every modern word-embedding method.
<!--ID: 1777386639950-->





## Vector Semantics

> [!question]- What is vector semantics?
> The framework of representing word meaning as **vectors in a high-dimensional space**, where geometric proximity reflects semantic similarity. Each word becomes an **embedding** (a point in this space). The geometry is derived automatically from corpus distributions — no hand-coded ontology. *Every modern NLP algorithm uses embeddings as the representation of word meaning.*
<!--ID: 1777386639951-->





> [!question]- What is a term-document matrix?
> A matrix where rows are vocabulary words and columns are documents; cell $(w, d)$ records how often $w$ appears in $d$. Two readings:
> - **Columns** are document vectors → basis of vector-space information retrieval.
> - **Rows** are word vectors → words with similar row patterns occur in similar documents and have similar meanings.
> Sparse and high-dimensional (one column per document).
<!--ID: 1777386639952-->





> [!question]- What is a term-context (word-word) matrix and why is it preferred for word embeddings?
> Rows are vocabulary words; **columns are also vocabulary words** (context words). Cell $(w, c)$ counts how often $c$ appears in a small window (e.g., $\pm 4$ words) around $w$. Better for word-level semantics than term-document because it captures direct word-word co-occurrence rather than going via document membership — and the context window is a tunable hyperparameter that controls similarity vs relatedness.
<!--ID: 1777386639953-->





## Cosine Similarity

> [!question]- What is cosine similarity, and why use it instead of raw dot product?
> $$\cos(\vec{v}, \vec{w}) = \frac{\vec{v} \cdot \vec{w}}{\|\vec{v}\| \|\vec{w}\|}$$
> Measures the **angle** between vectors, normalising out their magnitudes. Raw dot product rewards length — frequent words like *the* have huge vectors and would appear "similar" to everything. Cosine asks "do these vectors point in the same direction?", which is what we mean by semantic similarity. Range: $[-1, 1]$ in general; $[0, 1]$ for non-negative count vectors.
<!--ID: 1777386639954-->





## TF-IDF

> [!question]- What is TF-IDF and why is it needed?
> Raw counts are bad signals because frequent words (*the*, *of*) dominate. TF-IDF reweights:
> $$w_{t,d} = \log_{10}(\text{count}(t, d) + 1) \times \log_{10}\left(\frac{N}{\text{df}_t}\right)$$
> - **TF (term frequency)**: how often $t$ appears in document $d$ — log-squashed so 1000 occurrences don't dominate 10.
> - **IDF (inverse document frequency)**: how rare $t$ is across the corpus — words in every document have $\text{df}_t = N$, so $\log(N/N) = 0$ and their weight collapses.
> Words that are frequent in *this* document but rare *elsewhere* get the highest weight.
<!--ID: 1777386639955-->





> [!question]- What does it mean if a word's IDF is 0?
> The word appears in **every document** in the corpus ($\text{df}_t = N$, so $\log(N/N) = 0$). Such words have no discriminative power — they can't help identify which document is which. Examples in the Shakespeare data: *good* appears in all 37 plays, so its TF-IDF weight is 0 everywhere despite high raw counts. *Romeo*, appearing in one play, gets a high IDF and becomes a strong fingerprint for that play.
<!--ID: 1777386639956-->





## PMI

> [!question]- What is PMI (pointwise mutual information)?
> $$\text{PMI}(w_1, w_2) = \log_2 \frac{P(w_1, w_2)}{P(w_1) P(w_2)}$$
> Measures how much more (or less) often a pair co-occurs than chance would predict. **Positive** → associated; **zero** → independent; **negative** → they avoid each other. A word-word weighting alternative to raw counts; downweights stopword-driven co-occurrences automatically.
<!--ID: 1777386639957-->





> [!question]- What is PPMI and why is it used instead of PMI?
> **Positive PMI**: clip negative values to zero — $\text{PPMI}(w_1, w_2) = \max(\text{PMI}(w_1, w_2), 0)$. Reasons:
> - Reliably estimating "this pair occurs *less* than chance" needs an astronomical corpus — small marginals make negative PMI extremely noisy.
> - Humans aren't good at "unrelatedness" judgments anyway; positive associations are what semantic similarity is built on.
> - The negative tail is mostly noise that hurts downstream tasks.
<!--ID: 1777386639958-->





> [!question]- Why is PMI biased toward rare words, and what is the standard fix?
> When $w_1$ or $w_2$ has small marginal probability, the denominator $P(w_1) P(w_2)$ becomes very small, blowing up the log. Rare words get artificially high PMI scores. **Fix**: smooth the context distribution by raising counts to $\alpha = 0.75$:
> $$P_\alpha(c) \propto \text{count}(c)^{0.75}$$
> This boosts rare contexts (0.01 → 0.03) while barely moving frequent ones (0.99 → 0.97), reducing the rare-word inflation. The same $\alpha = 0.75$ trick reappears in word2vec's negative sampling distribution.
<!--ID: 1777386639959-->





## Sparse vs Dense Embeddings

> [!question]- What is the difference between sparse and dense word embeddings?
> - **Sparse**: long ($|V| \approx 20{,}000$–$50{,}000$), with most entries zero. Examples: TF-IDF, PPMI rows. Each dimension corresponds to a specific context word or document.
> - **Dense**: short (50–1000), with most entries non-zero. Examples: Word2Vec, GloVe, BERT embeddings. Dimensions don't correspond to anything interpretable.
> Dense embeddings empirically work better downstream because (1) they have far fewer parameters, (2) they generalise across synonyms (sparse vectors put *car* and *automobile* on different dimensions), and (3) low-rank compression acts as implicit regularisation.
<!--ID: 1777386639960-->





## Word2Vec

> [!question]- What is the core idea of Word2Vec?
> **Predict, don't count.** Train a binary classifier on a self-supervised task — *"is word $c$ likely to appear near word $w$?"* — using corpus co-occurrences as positive examples. The trained model's parameters (the input/output embedding matrices) are what we keep; the classifier is thrown away. From Mikolov et al., 2013.
<!--ID: 1777386639961-->





> [!question]- What are the four steps of Skip-gram with Negative Sampling (SGNS)?
> 1. **Positive examples**: for each target word $t$ and each context word $c$ in a $\pm k$ window around it, treat $(t, c)$ as a positive pair.
> 2. **Negative examples**: for each positive, randomly sample $k$ "negative" words from the corpus (weighted by frequency$^{0.75}$) to form $(t, c_{\text{neg}})$ pairs that are presumed to *not* co-occur.
> 3. **Logistic regression on dot product**: $P(+ \mid w, c) = \sigma(\vec{c} \cdot \vec{w})$. Cross-entropy loss pulls the embeddings of true context words together and pushes negative-sample embeddings apart.
> 4. **SGD** over the corpus. Throw away the classifier; **keep the embedding vectors**.
<!--ID: 1777386639962-->





> [!question]- What is self-supervised learning, and how does Word2Vec fit?
> A learning paradigm where the **labels come from the data itself** — no human annotation. In Word2Vec, the "label" for $(t, c)$ is whether $c$ actually occurs near $t$ in the corpus, which we can read off directly. Any running text generates training data. This is why Word2Vec scales — there's no labelling bottleneck.
<!--ID: 1777386639963-->





> [!question]- Why are dense (Word2Vec) embeddings better than sparse (PPMI) ones in practice?
> - **Fewer downstream parameters**: a 300-dim vector is ~100× cheaper than a 30,000-dim sparse vector.
> - **Synonym generalisation**: *car* and *automobile* live on distinct sparse dimensions, so a feature trained for *car* doesn't trigger on *automobile*. Dense embeddings put both near each other in space, so features generalise.
> - **Implicit regularisation**: forcing data into a low-rank representation discards noise and keeps the dominant statistical structure.
> - **Empirical win**: virtually every downstream task improved when sparse → dense embeddings were swapped in.
<!--ID: 1777386639964-->





## Properties of Trained Embeddings

> [!question]- How does the context window size affect what trained embeddings capture?
> - **Small window** ($\pm 2$): nearest neighbours are **similar** words (same syntactic class, same semantic type). *Hogwarts* → *Sunnydale*, *Evernight* (other fictional schools).
> - **Large window** ($\pm 5$): nearest neighbours are **topically related** words. *Hogwarts* → *Dumbledore*, *Malfoy* (Harry Potter universe).
> Window size is a hyperparameter trading similarity vs relatedness.
<!--ID: 1777386639965-->





> [!question]- What is the parallelogram method for analogies?
> To solve $a : a^* :: b : ?$, compute:
> $$\arg\max_x \cos(\vec{x},\ \vec{a^*} - \vec{a} + \vec{b})$$
> Famous example: *king − man + woman* ≈ *queen*. *Paris − France + Italy* ≈ *Rome*. Works when the analogical relation corresponds to a **consistent direction** in embedding space. **Caveats**: only reliable for frequent words, small distances, and a handful of relation types (capitals, parts-of-speech, comparative/superlative).
<!--ID: 1777386639966-->





> [!question]- What is diachronic semantic shift, and how do embeddings reveal it?
> Train **separate embeddings per time period** (e.g., per decade) and observe how a word's nearest neighbours change. Examples (Hamilton, Leskovec & Jurafsky 2016):
> - *gay*: 1900s near *cheerful, daft, tasteful* → 1990s near *lesbian, homosexual, bisexual*
> - *broadcast*: agricultural (*sow, seed*) → media (*television, radio*)
> - *awful*: *majestic, solemn* → *terrible, weird*
> The embedding space functions as a quantitative tool for historical linguistics.
<!--ID: 1777386639967-->





## Bias in Embeddings

> [!question]- How do biased embeddings emerge from training data?
> Embeddings are trained to capture **statistical regularities of word usage**. When the corpus reflects cultural stereotypes, the embeddings encode them too. Bolukbasi et al. (2016):
> - *father : doctor :: mother : ?* → **nurse**
> - *man : computer programmer :: woman : ?* → **homemaker**
> The model is doing exactly what it's trained for; the bias is in the data, not the algorithm. Every downstream system using these vectors inherits the bias.
<!--ID: 1777386639968-->





> [!question]- How can embeddings be used as a measurement tool for societal bias (Garg et al., 2018)?
> Train **diachronic embeddings**, then for each decade quantify the cosine similarity between target word groups (e.g., *smart, wise, brilliant*) and group identity terms (e.g., *woman* synonyms vs *man* synonyms). Track this over time. Garg et al. found that the embedding-derived bias scores **track independently-measured 1930s survey data** on stereotype content, and the bias diminishes over time in both records. Embeddings are a compressed record of who a corpus thinks the world is — making them a useful (if imperfect) measurement instrument.
<!--ID: 1777386639969-->






