---
type: concept
sources:
  - raw/week-04/w04-slides.pdf
  - raw/week-04/w04-slides-updated.pdf
  - raw/week-04/w04-l10-transcript.txt
status: draft
updated: 2026-04-25
---

*Raw word frequency is a bad representation because very frequent words like* the *and* it *are not discriminative. TF-IDF fixes this by multiplying each word's term frequency by its inverse document frequency — upweighting terms that are common inside a document but rare across the corpus.*

## The Problem with Raw Counts

The [[vector-semantics#The Matrix Behind the Vectors|term-document and term-context matrices]] represent each cell as a raw count. Counts are useful — if *sugar* appears a lot near *apricot*, that tells you something. But stop-words (*the, it, they, of*) appear a lot near **every** word, so they drown the signal in noise. A co-occurrence with *the* is not evidence of anything.

The paradox: frequent words carry the most co-occurrence mass but the least discriminative information. TF-IDF and [[pmi|PMI]] are two classical ways to resolve it.

## The Formula

$$w_{t,d} = \text{tf}_{t,d} \times \text{idf}_t$$

Each cell $w_{t,d}$ in the weighted matrix is the product of two terms. **TF** captures how important the term is *inside this document*; **IDF** captures how distinctive the term is *across the corpus*.

### Term frequency (tf)

The simplest choice is the raw count, $\text{tf}_{t,d} = \text{count}(t, d)$. In practice counts are **log-squashed** so that a word appearing 100 times isn't treated as 100× more important than one appearing once:

$$\text{tf}_{t,d} = \log_{10}(\text{count}(t, d) + 1)$$

The $+1$ inside the log keeps the value defined when a word doesn't occur (giving $\log_{10} 1 = 0$), and the log dampens the runaway effect of high-count words. This is the sentiment-classifier intuition restated in weighting form (see [[naive-bayes#Binary multinomial naive Bayes|binary multinomial NB]]): occurrence matters more than raw frequency.

> [!info] ALTERNATIVE CONVENTION
> Some texts use $\text{tf} = 1 + \log_{10} \text{count}$ (for $\text{count} > 0$, else 0) — a sublinear variant that gives a word appearing once a weight of $1$ rather than $\log_{10} 2 \approx 0.30$. These two conventions produce different numerical values (and tf-idf tables will disagree) but the same relative ordering. The course uses $\log_{10}(\text{count}+1)$ and we follow that here.

### Document frequency (df)

$\text{df}_t$ is the number of documents the term $t$ appears in — **not** the collection frequency (total count across all documents). The distinction matters:

| | Collection Frequency | Document Frequency |
|---|---|---|
| Romeo | 113 | 1 |
| action | 113 | 31 |

Both words appear 113 times in Shakespeare. But *Romeo* is concentrated in one play (it's highly distinctive) while *action* is spread across 31 plays (it's generic). Document frequency catches this; collection frequency doesn't.

### Inverse document frequency (idf)

$$\text{idf}_t = \log_{10}\!\left(\frac{N}{\text{df}_t}\right)$$

where $N$ is the total number of documents in the collection. The log compresses the dynamic range; $N/\text{df}$ on its own would vary over many orders of magnitude.

Illustrative values from the Shakespeare collection (37 plays):

| Word | df | idf |
|------|----|-----|
| Romeo | 1 | 1.57 |
| salad | 2 | 1.27 |
| Falstaff | 4 | 0.967 |
| forest | 12 | 0.489 |
| battle | 21 | 0.246 |
| wit | 34 | 0.037 |
| fool | 36 | 0.012 |
| good | 37 | 0 |
| sweet | 37 | 0 |

*Good* and *sweet* appear in every play, so their idf is exactly 0 — the product $\text{tf} \times \text{idf}$ zeroes out. TF-IDF says: *"a word that's everywhere tells you nothing about which document you're in."* *Romeo*, by contrast, gets a very high idf — it's a strong fingerprint for *Romeo and Juliet*.

## What "Document" Means

The definition is flexible: a play, a Wikipedia article, a paragraph, a tweet, or any fixed-size chunk of text. For word-similarity work, people often treat each paragraph as a document. The key requirement is that the **partition produces useful discriminative signal** — if every "document" is a single sentence, idf values saturate; if every "document" is an entire book, they under-discriminate.

## Worked Application: Shakespeare Plays

From raw counts to tf-idf weights, using the four-play term-document matrix:

**Raw counts:**

|      | As You Like It | Twelfth Night | Julius Caesar | Henry V |
|------|----------------|---------------|---------------|---------|
| battle | 1 | 0 | 7 | 13 |
| good | 114 | 80 | 62 | 89 |
| fool | 36 | 58 | 1 | 4 |
| wit  | 20 | 15 | 2 | 3 |

**TF-IDF weights (same data, re-weighted with $\text{tf} = \log_{10}(\text{count}+1)$):**

|      | As You Like It | Twelfth Night | Julius Caesar | Henry V |
|------|----------------|---------------|---------------|---------|
| battle | 0.074 | 0 | 0.22 | 0.28 |
| **good** | **0** | **0** | **0** | **0** |
| fool | 0.019 | 0.021 | 0.0036 | 0.0083 |
| wit  | 0.049 | 0.044 | 0.018 | 0.022 |

Observations:

- **`good` is zero everywhere.** idf = 0 because it appears in every play (df = 37, N = 37). Multiplying by log(1) = 0 wipes it out. This is the point: *good* was the loudest dimension in the raw matrix and contributed nothing distinctive.
- **`battle` dominates the history plays.** Its idf is non-trivial and its tf is high in *Julius Caesar* and *Henry V* — exactly the dimension that separates histories from comedies.
- **`fool` still discriminates comedies from histories**, but with much smaller absolute values because its idf is modest.

The net effect: tf-idf-weighted vectors separate documents *by what is distinctive to them*, not by what is frequent.

## Why TF-IDF Works as an Embedding

Each row of the tf-idf matrix is a word embedding (weighted counts of the documents the word appears in). Each column is a document embedding. These vectors are:

- **Long** — $|V|$ or $N$ dimensional, i.e. 20,000–50,000 if the columns are context words.
- **Sparse** — most entries are zero because most vocabulary words don't co-occur in any single window/document.
- **Interpretable** — every dimension is a named word or document; you can read off which contexts a word lives in.

Compared to [[word2vec|dense predictive embeddings]], tf-idf is cheaper (no training, just counts and logs), doesn't require a GPU, is deterministic, and remains the **default baseline in information retrieval** — the algorithm that underlies classical search engines and is the first thing any retrieval system should beat.

Its weaknesses are the mirror of word2vec's strengths. TF-IDF doesn't know that *car* and *automobile* are synonymous — they're different dimensions and their vectors don't overlap unless they happen to co-occur with the same words. Dense embeddings solve this by construction.

## Related

- [[vector-semantics]] — the framework tf-idf sits inside (weighted count vectors)
- [[pmi]] — the information-theoretic alternative to tf-idf for weighting co-occurrences
- [[cosine-similarity]] — the standard metric over tf-idf vectors (length normalisation is crucial for retrieval)
- [[word2vec]] — the dense predictive alternative that generalises across synonyms tf-idf can't equate

## Active Recall

> [!question]- Why does raw co-occurrence frequency fail as a similarity signal, and how does idf fix it?
> Frequent words like *the, it, of* dominate the counts but tell you nothing about the target word's meaning — every word co-occurs with *the*. Idf downweights terms that appear in many documents (logarithmically: $\log(N / \text{df}_t)$), so words that are *everywhere* get weight near zero and words that are *distinctive* get weight near $\log N$. Multiply by tf, and the surviving signal is "common in this document, rare in the corpus at large."

> [!question]- Walk through the tf-idf computation for a word *w* appearing 10 times in document *d*, in a collection of 1000 documents where *w* appears in 50 of them. Use $\text{tf} = \log_{10}(\text{count}+1)$ and $\log_{10}$ idf.
> $\text{tf} = \log_{10}(10+1) = \log_{10}(11) \approx 1.041$. $\text{idf} = \log_{10}(1000 / 50) = \log_{10}(20) \approx 1.301$. So $w_{t,d} \approx 1.041 \times 1.301 \approx 1.35$. For contrast: if *w* appeared in all 1000 documents, idf = 0 and the weight collapses, regardless of tf.

> [!question]- Slide MCQ: "machine learning" appears 4 times in document D (length 100). The corpus has 1000 documents, 100 of which contain "machine learning". Compute the TF-IDF.
> $\text{tf} = \log_{10}(4+1) = \log_{10}(5) \approx 0.699$. $\text{idf} = \log_{10}(1000/100) = \log_{10}(10) = 1$. $\text{tf-idf} = 0.699 \times 1 \approx \mathbf{0.70}$. Note: the document length (100 words) is a red herring — it doesn't enter the formula under the course's $\log_{10}(\text{count}+1)$ tf convention.

> [!question]- Why use document frequency rather than collection frequency in idf? Illustrate with the Shakespeare *Romeo* vs *action* example.
> Collection frequency is the total count across all documents; document frequency is the *number of documents* the word appears in. Both *Romeo* and *action* occur 113 times in Shakespeare's plays, so their collection frequencies are equal. But *Romeo* is concentrated in one play (df = 1 — highly distinctive) while *action* is spread across 31 plays (df = 31 — generic). Idf based on df gives *Romeo* a large weight and *action* a small weight; idf based on collection frequency would treat them identically, missing the distinctiveness that matters for search.

> [!question]- In the worked Shakespeare example, why is the tf-idf weight for *good* zero in every play, and what's the lesson?
> *Good* appears in all 37 plays, so $\text{df} = N = 37$ and $\text{idf} = \log_{10}(37/37) = \log_{10}(1) = 0$. Multiplying by tf gives 0 everywhere. The lesson: a word that appears in every document carries no discriminative information, regardless of how often it appears within a document. Raw counts would make *good* the loudest signal; tf-idf correctly demotes it to silence.

> [!question]- Why is log-squashing applied to tf rather than using raw counts?
> A word appearing 100 times is more important than a word appearing once, but not 100× more important — the marginal value of each additional occurrence shrinks. Using $\log_{10}(\text{count}+1)$ dampens that growth while keeping the monotonic *"more is more"* property. Without squashing, one highly repetitive word would dominate the vector and distort cosine similarities.

> [!question]- Why is tf-idf classified as a "sparse" embedding, and what does that mean in practice?
> Each tf-idf vector has one dimension per context word (for a term-context matrix) or per document (for a term-document matrix) — on the order of 20,000-50,000 dimensions. But any given word only co-occurs with a small fraction of the vocabulary, so most entries are exactly zero. In practice: vectors are stored as sparse arrays, dot products only sum over the non-zero terms, and the representation is cheap but high-dimensional. [[word2vec|Dense embeddings]] live in 50–1000 dimensions with almost every entry non-zero.
