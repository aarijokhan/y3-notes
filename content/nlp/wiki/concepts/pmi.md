---
type: concept
sources:
  - raw/week-04/w04-slides.pdf
  - raw/week-04/w04-slides-updated.pdf
  - raw/week-04/w04-worksheet.pdf
  - raw/week-04/w04-worksheet-solutions.pdf
status: draft
updated: 2026-04-25
---

*Pointwise mutual information measures how much more (or less) often two words co-occur than chance would predict. Positive PMI means the pair is associated; zero means independence; negative means they avoid each other.*

## Definition

For two words $w_1$ and $w_2$:

$$\text{PMI}(w_1, w_2) = \log_2 \frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)}$$

The denominator $P(w_1) P(w_2)$ is what the joint probability *would* be if $w_1$ and $w_2$ occurred independently. The numerator $P(w_1, w_2)$ is what we actually observe. Their ratio quantifies association; the log turns ratio into an additive quantity.

- $\text{PMI} > 0$: the pair co-occurs more than chance → **associated**.
- $\text{PMI} = 0$: independent.
- $\text{PMI} < 0$: co-occurs *less* than chance → **anti-associated**.

The base of the log is a convention: $\log_2$ gives an interpretation in bits; $\log_e$ gives nats. The worksheet uses natural log.

## Worked Example (Worksheet Q6)

From the lab:

- $P(\text{sunny}) = 0.05$
- $P(\text{weather}) = 0.1$
- $P(\text{sunny}, \text{weather}) = 0.02$

Compute:

$$\text{PMI}(\text{sunny}, \text{weather}) = \ln \frac{0.02}{0.05 \times 0.1} = \ln \frac{0.02}{0.005} = \ln 4 \approx 1.39$$

A positive PMI means *sunny* and *weather* co-occur more often than chance — there's a real association in the corpus. Useful: a basic sanity check on whether a word pair meaningfully clusters without needing labels.

## Estimating From a Corpus

$P(w_1)$ is estimated from the unigram count of $w_1$; $P(w_1, w_2)$ from the bigram/co-occurrence count within a chosen window. So PMI can be computed directly from the [[vector-semantics#The Matrix Behind the Vectors|term-context co-occurrence matrix]]:

$$\text{PMI}(w_1, w_2) = \log \frac{\hat{P}(w_1, w_2)}{\hat{P}(w_1)\hat{P}(w_2)} = \log \frac{\#(w_1, w_2)/N}{\#(w_1)/N \cdot \#(w_2)/N}$$

where $N$ is the total number of word pairs observed (or the total token count, depending on conventions).

## PMI vs TF-IDF

Both PMI and [[tf-idf]] solve the same underlying problem — raw frequency is a bad weighting, very common words carry no distinctive signal — but from different angles:

- **TF-IDF** corrects for *ubiquity across documents* (words that appear everywhere contribute nothing). Document-centred.
- **PMI** corrects for *baseline frequency* (a word co-occurring with *the* is less informative than one co-occurring with *apricot* because *the* co-occurs with everything). Pair-centred.

They often agree in practice: both downweight stopwords and upweight distinctive collocations. The practical choice comes from the matrix you have. If your matrix is term-by-document, tf-idf is natural. If it's term-by-context-word, PMI is natural.

## Positive PMI (PPMI)

PMI ranges from $-\infty$ to $+\infty$, but **negative values are problematic** for three reasons:

1. **Interpretation is weak.** A negative PMI means the pair co-occurs *less* than chance. That's a real signal in principle, but far less intuitive than the positive direction.
2. **Unreliable without enormous corpora.** If $P(w_1) = P(w_2) = 10^{-6}$, the chance-level joint is $10^{-12}$. Telling whether the observed joint is "significantly different" from $10^{-12}$ requires an astronomically large corpus. On anything short of that, the negative tail is pure noise.
3. **Humans aren't calibrated on "unrelatedness."** Similarity judgments correlate well with positive PMI; nobody reliably rates pairs as *anti*-related.

The standard fix is **Positive PMI (PPMI)**, which clips negative values at zero:

$$\text{PPMI}(w_1, w_2) = \max\!\left(\log_2 \frac{P(w_1, w_2)}{P(w_1) P(w_2)},\ 0\right)$$

PPMI also sidesteps the $\log 0 = -\infty$ problem for unobserved pairs (which become 0). PPMI-weighted co-occurrence matrices are the classical baseline that [[word2vec]]'s skip-gram with negative sampling was shown (Levy & Goldberg, 2014) to implicitly factorise.

## Computing PPMI on a Term-Context Matrix

Given a co-occurrence matrix $F$ with $W$ rows (words) and $C$ columns (contexts), where $f_{ij}$ is the number of times word $w_i$ occurs with context $c_j$, the joint and marginal probabilities are:

$$p_{ij} = \frac{f_{ij}}{\sum_{i=1}^W \sum_{j=1}^C f_{ij}}, \quad p_{i*} = \frac{\sum_{j=1}^C f_{ij}}{\sum_{i,j} f_{ij}}, \quad p_{*j} = \frac{\sum_{i=1}^W f_{ij}}{\sum_{i,j} f_{ij}}$$

$$\text{pmi}_{ij} = \log_2 \frac{p_{ij}}{p_{i*} p_{*j}}, \qquad \text{ppmi}_{ij} = \max(\text{pmi}_{ij}, 0)$$

### Worked example (Jurafsky's cherry/strawberry/digital/information)

Raw counts:

|               | computer | data | result | pie | sugar | count(w) |
|---------------|----------|------|--------|-----|-------|----------|
| cherry        | 2        | 8    | 9      | 442 | 25    | 486      |
| strawberry    | 0        | 0    | 1      | 60  | 19    | 80       |
| digital       | 1670     | 1683 | 85     | 5   | 4     | 3447     |
| information   | 3325     | 3982 | 378    | 5   | 13    | 7703     |
| **count(c)**  | **4997** | **5673** | **473** | **512** | **61** | **11716** |

Computing one cell — PMI(*information*, *data*):

- $p(\text{information}, \text{data}) = 3982 / 11716 \approx 0.3399$
- $p(\text{information}) = 7703 / 11716 \approx 0.6575$
- $p(\text{data}) = 5673 / 11716 \approx 0.4842$
- $\text{pmi} = \log_2 \frac{0.3399}{0.6575 \times 0.4842} = \log_2 \frac{0.3399}{0.3184} \approx 0.0944$

Resulting PPMI matrix (negatives clipped to 0):

|             | computer | data | result | pie  | sugar |
|-------------|----------|------|--------|------|-------|
| cherry      | 0        | 0    | 0      | 4.38 | 3.30  |
| strawberry  | 0        | 0    | 0      | 4.10 | 5.51  |
| digital     | 0.18     | 0.01 | 0      | 0    | 0     |
| information | 0.02     | 0.09 | 0.28   | 0    | 0     |

The matrix encodes the semantic split cleanly: *cherry* and *strawberry* light up on *pie* and *sugar* (food field); *digital* and *information* light up faintly on the tech columns. All the "stopword-ish" cells collapse to 0.

## Weighting PMI: The Rare-Word Bias

Even with the PPMI clip, PMI has one more bias worth knowing about: **it overweights rare events**. Very rare words can have very high PMI values on the (few) contexts where they co-occur — not because the association is strong, but because their marginal probability is tiny and dividing by a near-zero denominator inflates the log.

Two standard mitigations:

### 1. Raise context probabilities to $\alpha = 0.75$

Smooth the context marginal $p(c)$ by raising counts to a fractional power before normalising:

$$P_\alpha(c) = \frac{\text{count}(c)^\alpha}{\sum_c \text{count}(c)^\alpha}, \qquad \text{PPMI}_\alpha(w, c) = \max\!\left(\log_2 \frac{P(w, c)}{P(w)\, P_\alpha(c)},\, 0\right)$$

Because $\alpha < 1$, this *boosts* rare contexts' implied probability and *dampens* frequent contexts'. Concretely, if two events have $P(a) = 0.99$ and $P(b) = 0.01$:

- $P_{0.75}(a) = 0.99^{0.75} / (0.99^{0.75} + 0.01^{0.75}) \approx 0.97$
- $P_{0.75}(b) = 0.01^{0.75} / (\cdot) \approx 0.03$

The rare event's effective probability went from 0.01 to 0.03 — a 3× boost. This reduces the PMI inflation on rare contexts. (Incidentally, $\alpha = 0.75$ is the same smoothing that [[word2vec]]'s negative-sampling distribution uses — Mikolov et al. settled on it empirically.)

### 2. Add-one smoothing

Add 1 to every count in the co-occurrence matrix before computing probabilities. Same trick as [[smoothing|Laplace smoothing]] for n-gram models — it doesn't remove rare-word bias entirely but dampens the worst cases.

## Why PMI Matters

Two reasons it keeps appearing in the course:

1. **Interpretable baseline.** PPMI on a term-context matrix with a dimensionality reduction step (SVD → LSA) is a strong, fully explainable alternative to dense neural embeddings, and often within a few percentage points of word2vec on similarity benchmarks.
2. **Theoretical bridge.** Word2vec's training objective approximates PMI factorisation — understanding PMI is understanding what skip-gram is *really* computing underneath the loss function.

## Related

- [[tf-idf]] — document-weighted alternative; solves the same problem from a different angle
- [[vector-semantics]] — the matrix framework PMI computes over
- [[cosine-similarity]] — still the right metric for comparing PMI-weighted vectors
- [[word2vec]] — implicitly factorises a shifted-PMI matrix (Levy & Goldberg 2014)

## Active Recall

> [!question]- State the PMI formula and explain what each of the three cases (positive, zero, negative) mean.
> $\text{PMI}(w_1, w_2) = \log \frac{P(w_1, w_2)}{P(w_1) P(w_2)}$. The denominator is the joint probability *if* the words were independent. **Positive PMI**: the pair co-occurs more than chance — they're associated (e.g. *strong/coffee* or *sunny/weather*). **Zero**: independent. **Negative**: they co-occur less than chance — they avoid each other (e.g. *strong/day-of-week*).

> [!question]- Compute PMI for a pair where $P(w_1) = 0.05$, $P(w_2) = 0.1$, $P(w_1, w_2) = 0.02$, using natural log.
> $\text{PMI} = \ln \frac{0.02}{0.05 \cdot 0.1} = \ln \frac{0.02}{0.005} = \ln 4 \approx 1.39$. Positive, so the pair is associated — they co-occur about 4× more often than chance would predict.

> [!question]- What problem does Positive PMI (PPMI) solve that raw PMI does not?
> Raw PMI has two problems: (1) **unseen pairs** produce $\log 0 = -\infty$, which is uncomputable; (2) **negative values are unreliable** — with realistic corpus sizes you cannot distinguish "co-occurs less than chance" from sampling noise, and humans don't reason well about anti-association anyway. PPMI clips negative values at zero, treating unseen or chance-level pairs uniformly and discarding the unreliable negative tail. It's the standard weighting in practical sparse-vector pipelines.

> [!question]- Slide MCQ: In a corpus of 1000 words, "data" appears 50 times, "science" 30 times, and they co-occur 20 times. Compute PPMI($\log_2$) and pick the closest of {1.5, 2.0, 3.0, 4.0}.
> $P(\text{data}) = 0.05$, $P(\text{science}) = 0.03$, $P(\text{data}, \text{science}) = 0.02$. $\text{PMI} = \log_2 \frac{0.02}{0.05 \times 0.03} = \log_2 \frac{0.02}{0.0015} = \log_2(13.33) \approx 3.74$. Positive, so PPMI = PMI. Nearest option: **4.0**.

> [!question]- What does the $\alpha = 0.75$ smoothing trick do for PMI, and why does it help?
> It replaces $P(c)$ with $P_\alpha(c) = \text{count}(c)^\alpha / \sum_c \text{count}(c)^\alpha$ in the denominator. Because $\alpha < 1$, rare contexts' effective probability goes up (e.g. 0.01 becomes 0.03) while frequent contexts' barely move. That attenuates PMI's **rare-word bias** — the tendency of rare words to get astronomically high PMI on the few contexts where they happen to co-occur, because dividing by a near-zero marginal inflates the log. Same $\alpha$ value word2vec uses for negative sampling (coincidentally or not).

> [!question]- How does PMI differ conceptually from tf-idf, even though both re-weight raw counts?
> **TF-IDF** is document-centred: it asks "how rare is this word across documents?" and downweights ubiquitous words. **PMI** is pair-centred: it asks "how often do these two words co-occur compared to chance?" and downweights associations explained by a word's overall frequency. The two usually agree on stopwords (which are both ubiquitous *and* explain most of their co-occurrences by marginal frequency) but they're computed over different matrices — tf-idf over term-document, PMI over term-context.

> [!question]- Why might a practitioner prefer PMI over raw co-occurrence counts even without doing any dimensionality reduction?
> Raw counts overwhelmingly reflect word frequency: the cell for (*the*, *apricot*) is larger than the cell for (*sugar*, *apricot*) simply because *the* is common, not because *the* is semantically close to *apricot*. PMI normalises away the marginal frequencies, so the PMI cell for (*the*, *apricot*) is near zero (expected co-occurrence) while (*sugar*, *apricot*) is strongly positive. Dot products and cosines over PMI-weighted vectors therefore correlate better with semantic similarity than over raw-count vectors.
