---
type: week
week: 4
title: "Week 4 — Vector Semantics: From Words as Strings to Words as Vectors"
dates: —
sources:
  - raw/week-04/w04-slides.pdf
  - raw/week-04/w04-slides-updated.pdf
  - raw/week-04/w04-l10-transcript.txt
  - raw/week-04/w04-l12-transcript.txt
  - raw/week-04/w04-worksheet.pdf
  - raw/week-04/w04-worksheet-solutions.pdf
concepts:
  - "[[lexical-semantics]]"
  - "[[vector-semantics]]"
  - "[[tf-idf]]"
  - "[[pmi]]"
  - "[[cosine-similarity]]"
  - "[[word2vec]]"
status: stable
updated: 2026-04-25
---

> [!question]+ THE CRUX: How do you get from words as arbitrary strings to a representation where *cat* and *dog* sit near each other, without being told so — and what does that representation enable?

*The trick: define a word's meaning by the company it keeps (distributional hypothesis), record that as a vector (meaning as a point in space), and measure similarity by the angle between vectors. Two concrete instantiations — sparse weighted counts (TF-IDF) and dense predicted associations (Word2Vec) — are the workhorses on which every modern NLP model is built.*

---

## From Strings to Meanings

Three weeks in, every word in this module has been treated as a **string**. [[regular-expressions|Regex]] matches literal characters. [[n-gram-language-models|N-gram LMs]] compute $P(w_i \mid w_{i-1})$ over string identities. [[naive-bayes|Naive Bayes]] counts word-string occurrences per class. That's enough for the tasks studied so far, but it breaks the moment you need a system to understand that *cat* and *dog* are both mammals, or that a review saying *"awful"* should be treated like one saying *"terrible"*.

The failed alternative is logic-class symbols: *the meaning of DOG is* $\text{DOG}(x)$*, and* $\forall x\,\text{DOG}(x) \to \text{MAMMAL}(x)$. Barbara Partee's old joke sums up the problem: *what's the meaning of life? A: LIFE*. Renaming strings as symbols doesn't teach a model anything new — every inference still has to be hand-coded.

[[lexical-semantics|Lexical semantics]] supplies the structure that any real representation has to recover: **lemmas** (the dictionary form, *mouse*) carry multiple **senses** (rodent, pointing device); senses stand in **relations** to each other — synonymy, similarity, relatedness, antonymy, connotation. A good representation should place *couch* near *sofa*, *cat* near *dog* but farther, *fast* near *slow* if we're generous about "near," and *car* far from *banana*. And it should do this without being hand-annotated — the dictionary approach of WordNet is valuable but incomplete and brittle.

## Idea 1 — The Distributional Hypothesis

The central insight is much older than NLP. Wittgenstein (*Philosophical Investigations* §43): *"The meaning of a word is its use in the language."* Zellig Harris (1954) made it operational: *"If A and B have almost identical environments we say that they are synonyms."*

The intuition is captured in the **ongchoi** thought experiment. Suppose you've never seen *ongchoi* before, but you read:

- *Ong choi is delicious sautéed with garlic.*
- *Ong choi leaves with salty sauces.*

And you've previously seen:

- *...spinach sautéed with garlic over rice.*
- *Chard stems and leaves are delicious.*

You can place *ongchoi* near *spinach, chard, collard greens* without anyone telling you. The shared neighbours (*leaves, sautéed, delicious, salty*) did all the work. This is the **distributional hypothesis** in action.

## Idea 2 — Meaning as a Point in Space

The second idea has a dry run in the connotation sketch. Osgood et al. (1957) posited that every word has three affective dimensions: **valence** (pleasantness), **arousal** (intensity), **dominance** (control). So *love* is $(1.0, 0.5, 0.7)$, *toxic* is $(0.0, 0.6, 0.4)$. A word is a point in 3-space.

Combine with Idea 1: rather than three dimensions picked by hand, use thousands of dimensions derived automatically from corpus distribution. Each word becomes a vector in high-dimensional space; similar words sit at similar points. [[vector-semantics|Vector semantics]] is the framework; an individual word's vector is an **embedding** because the word is *embedded* into a semantic space whose geometry reflects meaning.

This is not a minor methodological shift. *"Every modern NLP algorithm uses embeddings as the representation of word meaning."* Strings are gone. From this week onward, all computation happens on vectors.

## The Matrix Underneath

Both concrete embedding methods ([[tf-idf|TF-IDF]] and [[word2vec]]) are ultimately about co-occurrence matrices — tables counting how often words appear with other words (or documents).

The simplest is the **term-document matrix**. Each column is a document; each row is a vocabulary word; each cell is a count. The four Shakespeare plays yield:

|        | As You Like It | Twelfth Night | Julius Caesar | Henry V |
|--------|----------------|---------------|---------------|---------|
| battle | 1 | 0 | 7 | 13 |
| good   | 114 | 80 | 62 | 89 |
| fool   | 36 | 58 | 1 | 4 |
| wit    | 20 | 15 | 2 | 3 |

Two readings, both productive. **Column-wise**: each column is a document vector — comedies look like each other, histories look like each other, which makes the matrix the basis of the **vector space model of information retrieval** (Salton 1971). **Row-wise**: each row is a word vector — *battle* is *"the kind of word that occurs in histories,"* *fool* is *"the kind that occurs in comedies."* Words with similar row vectors have related meanings.

More useful for word embeddings is the **term-context (word-word) matrix** — columns are context words instead of documents, cells count co-occurrence within a small window. The row for *digital* is $[\ldots, 1670, 1683, 85, 5, 4, \ldots]$ on columns $(\textit{computer}, \textit{data}, \textit{result}, \textit{pie}, \textit{sugar})$. The row for *information* is $[\ldots, 3325, 3982, 378, 5, 13, \ldots]$ on the same columns. The two rows point in nearly the same direction, which is exactly what we want a similarity metric to reward.

## Measuring Similarity: Cosine

Raw dot product of row vectors almost works, but it rewards **length** — and frequent words like *the, of* have long row vectors with large coordinates in every dimension. A raw dot product would call *the* similar to every word, which defeats the purpose. [[cosine-similarity|Cosine similarity]] fixes this by dividing out the norms:

$$\cos(\vec{v}, \vec{w}) = \frac{\vec{v} \cdot \vec{w}}{|\vec{v}||\vec{w}|}$$

This asks about direction, not magnitude. For the *digital / information* example: $\cos \approx 0.996$, correctly reporting that the two vectors are nearly parallel despite *information* having much larger raw counts. For *cherry / information*: $\cos \approx 0.017$, correctly reporting that one lives in the food field and the other in the tech field.

> [!question]- Given word vectors *king* = [0.1, 0.3, 0.7] and *queen* = [0.2, 0.4, 0.6], compute their cosine similarity. (Worksheet Q5)
> Dot product $= 0.02 + 0.12 + 0.42 = 0.56$. $|king| = \sqrt{0.59} \approx 0.77$; $|queen| = \sqrt{0.56} \approx 0.75$. $\cos = 0.56 / (0.77 \cdot 0.75) \approx 0.97$. Very similar — the vectors nearly coincide in direction despite unequal lengths.

## Re-Weighting Raw Counts

Raw frequency is a bad signal because the loudest words are the least informative. Two classical fixes weight the matrix before any similarity computation.

[[tf-idf|**TF-IDF**]] multiplies term frequency (how much a word appears in *this* document) by inverse document frequency (how rare the word is *across the corpus*), using the course's log-squashed tf:

$$w_{t,d} = \log_{10}(\text{count}(t,d) + 1) \times \log_{10}(N / \text{df}_t)$$

A word in every document has $\text{df} = N$ and $\text{idf} = 0$ — its weight collapses. In the Shakespeare re-weighting, *good* (appearing in all 37 plays) goes from raw counts of 60–120 to tf-idf weight $0$ everywhere. *Romeo* (appearing in one play) gets $\text{idf} = 1.57$ and becomes a strong fingerprint for *Romeo and Juliet*. TF-IDF is the default weighting for sparse document retrieval and still the first thing any search system has to beat.

[[pmi|**PMI**]] takes the same insight from an information-theoretic angle:

$$\text{PMI}(w_1, w_2) = \log_2 \frac{P(w_1, w_2)}{P(w_1) P(w_2)}$$

How much more (or less) often does the pair co-occur than chance would predict? Positive means associated, zero means independent, negative means they avoid each other. Computed over a term-context matrix, it systematically downweights stopword-driven co-occurrences.

Two practical refinements matter. **PPMI** clips negative values to zero — removing the unreliable negative tail (humans aren't good at "unrelatedness" anyway, and distinguishing $10^{-12}$ from chance-level $10^{-12}$ needs an astronomical corpus). And because PMI inflates rare-word associations (small marginals produce large logs), the standard practice is to **smooth the context probability** by raising counts to $\alpha = 0.75$: $P_\alpha(c) \propto \text{count}(c)^{0.75}$. This boosts rare contexts (0.01 → 0.03) while barely moving frequent ones (0.99 → 0.97). The same $\alpha$ reappears in [[word2vec|word2vec's negative sampling distribution]].

> [!question]- Compute the PMI between *sunny* and *weather* given P(sunny) = 0.05, P(weather) = 0.1, P(sunny, weather) = 0.02, using natural log. What does the result mean? (Worksheet Q6)
> $\text{PMI} = \ln(0.02 / (0.05 \cdot 0.1)) = \ln(0.02 / 0.005) = \ln 4 \approx 1.39$. Positive PMI — the pair co-occurs about 4× more often than chance would predict, indicating a real corpus-level association.

> [!question]- Slide MCQ: Compute PPMI for *data* (50 occurrences) and *science* (30 occurrences) co-occurring 20 times in a 1000-word corpus, using $\log_2$. Pick closest of {1.5, 2.0, 3.0, 4.0}.
> $P(\text{data}) = 0.05$, $P(\text{science}) = 0.03$, $P(\text{data, science}) = 0.02$. $\text{PMI} = \log_2(0.02 / (0.05 \times 0.03)) = \log_2(13.33) \approx 3.74$. Since positive, PPMI = PMI. Closest option: **4.0**.

> [!question]- Slide MCQ: Compute tf-idf for "machine learning" appearing 4 times in a 100-word document, where 100 out of 1000 corpus documents contain it.
> Using the course formula $\text{tf} = \log_{10}(\text{count}+1)$: $\text{tf} = \log_{10}(5) \approx 0.699$, $\text{idf} = \log_{10}(1000/100) = 1$, tf-idf $\approx \mathbf{0.70}$. The document length is a red herring — it doesn't enter this tf formula.

## The Dense Alternative: Word2Vec

TF-IDF and PPMI vectors are **long** (|V| ≈ 20,000-50,000) and **sparse** (most entries zero). Word2vec (Mikolov et al., 2013) produces **short** (50-1000) **dense** (most entries non-zero) vectors that empirically work better downstream. Why?

- **Fewer parameters downstream** — a 300-dim vector is ~100× cheaper than a 30,000-dim sparse one.
- **Better synonym generalisation** — *car* and *automobile* live on distinct sparse dimensions, so a classifier feature built for *car* doesn't trigger on *automobile*. Dense embeddings place both near each other, so features generalise.
- **Implicit regularisation** — reducing to low rank forces the model to keep the dominant statistical structure and discard noise.

[[word2vec|Word2vec]]'s big idea: **predict rather than count**. Instead of tallying how often *apricot* co-occurs with *jam*, train a binary classifier to predict *"is `jam` likely to show up near `apricot`?"* The task itself is throwaway; the embeddings the classifier uses as its parameters are the product.

### Skip-gram with Negative Sampling (SGNS)

Four-step recipe:

1. **Positive examples**: for each target word $t$ and each context word $c$ in a $\pm k$ window, $(t, c)$ is a positive pair.
2. **Negative examples**: for each positive example, randomly sample $k$ other words by frequency to form $(t, c_{neg})$ pairs.
3. **Logistic regression** on dot-product similarity: $P(+ \mid w, c) = \sigma(\vec{c} \cdot \vec{w})$. Cross-entropy loss:

$$L_{CE} = -\left[ \log \sigma(\vec{c}_{pos} \cdot \vec{w}) + \sum_{i=1}^k \log \sigma(-\vec{c}_{neg_i} \cdot \vec{w}) \right]$$

4. **SGD**. One gradient step pulls *apricot* and *jam* closer, pushes *apricot* and *matrix* apart, adjusts everything by $\eta$. A single pass (or few) over a large corpus converges. **Throw away the classifier; keep the embedding vectors.**

This is **self-supervised learning** — no human labels needed. A word that actually occurred near the target in the corpus *is* the gold answer for the supervised task. Any running text generates training data.

> [!info] ASIDE — Word2Vec is implicit PMI factorisation
> Levy & Goldberg (2014) showed that SGNS's training objective is equivalent to factorising a shifted-PMI matrix, roughly $\text{PMI}(w, c) - \log k$, into two low-rank factors $W$ and $C$. So dense word2vec embeddings and classical PMI+SVD pipelines are computing the same thing in different clothes — the practical difference is online stochastic training vs. batch matrix factorisation.

## Properties of Trained Embeddings

Three striking things happen once you have dense embeddings trained on a real corpus.

**Window size tunes similarity vs. relatedness.** A small window $(\pm 2)$ makes *Hogwarts*'s nearest neighbours other fictional schools (*Sunnydale, Evernight*); a large window $(\pm 5)$ makes them the Harry Potter world (*Dumbledore, Malfoy*). The [[lexical-semantics#Word Relatedness (Association)|similarity vs relatedness]] distinction becomes a hyperparameter.

**Analogies via vector arithmetic.** The parallelogram method: to solve $a : a^* :: b : \_$, compute $\arg\max_x \cos(x, a^* - a + b)$. *king − man + woman ≈ queen*. *Paris − France + Italy ≈ Rome*. When a consistent semantic relation shows up as a direction in embedding space, analogy queries ride that direction. **Caveat**: the method only works reliably for frequent words, small distances, and certain relations (capitals-of-countries, parts-of-speech). The limits of this are an active research area.

**Diachronic semantic shift.** Train separate embeddings per decade of text (Hamilton, Leskovec & Jurafsky 2016) and watch words move. *gay* in the 1900s sits near *cheerful, daft, tasteful*; in the 1990s near *lesbian, homosexual, bisexual*. *broadcast* shifts from agricultural (*sow, seed*) to media (*television, radio*). *awful* inverts from *majestic, solemn* to *terrible, weird*. The embedding space doubles as a historical-linguistics instrument.

## What Goes Wrong

Embeddings trained on uncurated web text reflect the cultural content of that text — including its biases. Bolukbasi et al. (2016) showed:

- *father : doctor :: mother : x* → **nurse**
- *man : computer programmer :: woman : x* → **homemaker**

The embeddings are doing exactly what they're trained to do — capturing statistical regularities of usage. When those regularities encode gender or racial stereotypes, the embeddings encode them too. Every downstream system that consumes these vectors — hiring search, content ranking, translation — inherits the bias.

Garg, Schiebinger, Jurafsky & Zou (2018) used **diachronic embeddings as a measurement tool**: quantify how close *smart/wise/brilliant* is to *woman* vs. *man* synonyms, decade by decade, and compare to independently-measured 1930s surveys of stereotype content. The match is striking — the embedding-derived bias scores track the survey scores, and the bias diminishes over time in both records. Embeddings, it turns out, are a compressed record of who a corpus thinks the world is.

This is the [[harms-in-classification|harms]] story one layer deeper. Week 3 worried about biased training labels in a classifier. This week, the bias is already baked into the *representation* every classifier will use.

## The Full Stack

By the end of this week, a standard NLP pipeline looks like:

1. [[tokenization|Tokenize]] the text and handle [[byte-pair-encoding|OOV]] with subword units.
2. Look up [[word2vec|pre-trained dense embeddings]] for every token.
3. Build document or sentence representations from these embeddings (sum, average, or feed to a neural network).
4. Feed to a classifier (week 3), sequence model, or retrieval system.

Every stage operates on vectors. Strings survive only as keys into the embedding lookup table.

---

## Concepts Introduced This Week

- [[lexical-semantics]] — the linguistic structure (lemmas, senses, synonymy, similarity, relatedness, antonymy, connotation) every representation must recover
- [[vector-semantics]] — words as vectors, co-occurrence matrices (term-document, term-context), embeddings as the standard representation
- [[cosine-similarity]] — dot product normalised by vector lengths; the right metric for comparing word vectors
- [[tf-idf]] — term-frequency × inverse-document-frequency weighting; sparse embedding workhorse of information retrieval
- [[pmi]] — pointwise mutual information as an alternative weighting; related to PPMI and SGNS
- [[word2vec]] — skip-gram with negative sampling; dense predictive embeddings, analogies, bias, diachronic shift

## Connections

Builds directly on [[week-03]]: [[bag-of-words]] treated a document as a multiset of tokens with no meaning relations between them. Vector semantics gives tokens a geometry — *couch* near *sofa*, *battle* near *war*. The same Naive Bayes trick (log-space arithmetic, smoothing over sparse counts) transfers to tf-idf and PMI computations. The **sentiment analysis** motif from week 3 returns via the [[lexical-semantics#Connotation (Sentiment)|VAD connotation model]] — each word's affect is itself a 3-vector, a miniature version of embedding space.

Reaches back to [[week-02]]: the [[smoothing|smoothing]] intuition ($\log 0 = -\infty$ is a problem) recurs with PPMI. The **perplexity / predict-the-next-word** framing of n-gram LMs is the direct ancestor of word2vec's predict-the-nearby-word objective — self-supervision through prediction, now used to learn a different thing (embeddings, not P(word)).

Sets up later weeks: **contextual embeddings** (ELMo, BERT, transformers) will replace one-vector-per-lemma with one-vector-per-occurrence, solving the polysemy limitation of static embeddings. **Logistic regression** (the classifier underneath SGNS) becomes its own topic for general classification tasks. Embeddings are also the input to every neural architecture in later weeks — once dense vectors exist, RNNs, LSTMs, and transformers can consume them.

## Open Questions

- When does SGNS's implicit PMI-factorisation view break down? Real embeddings are trained with subsampling of frequent words and other tricks that don't appear in the clean theoretical equivalence.
- The parallelogram analogy method works for capitals and part-of-speech relations but not most others. What makes a relation "linear" in embedding space, and is there a way to predict which relations will work without trying them?
- Debiasing methods for embeddings (Bolukbasi et al. 2016 project out the "gender direction") partially repair demographic bias on benchmark tests but leave bias detectable by other probes. Can the bias really be removed, or only relocated?
- How much of the empirical advantage of dense over sparse vectors comes from **dimensionality compression** itself (implicit regularisation) versus **predictive training** (as opposed to counting)?
