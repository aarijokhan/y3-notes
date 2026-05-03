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

*Cosine similarity measures the angle between two vectors — the dot product divided by the product of their lengths. It's the standard way to compute word similarity because it separates direction (what the vectors mean) from magnitude (how often the words appeared).*

## Dot Product and Its Flaw

The **dot product** is the starting point:

$$\vec{v} \cdot \vec{w} = \sum_{i=1}^{N} v_i w_i$$

It's high when the two vectors have large values in the same dimensions — which intuitively means they co-occur with similar words — and low when their large-value dimensions disagree. So far so good.

The flaw: the dot product also grows with **vector length**. Define length as

$$|\vec{v}| = \sqrt{\sum_{i=1}^{N} v_i^2}$$

Frequent words like *of, the, you* have long vectors — they co-occur with many things many times. So their dot product with anything is high, simply because their coordinates are big. A raw dot product would say *the* is "similar" to every word in the vocabulary — a useless metric.

## The Cosine Fix

Divide by the lengths, and the magnitude dependence cancels:

$$\cos(\vec{v}, \vec{w}) = \frac{\vec{v} \cdot \vec{w}}{|\vec{v}|\, |\vec{w}|} = \frac{\sum_{i=1}^N v_i w_i}{\sqrt{\sum_i v_i^2} \sqrt{\sum_i w_i^2}}$$

This is just the cosine of the angle $\theta$ between the two vectors, following from $\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}| \cos\theta$. Geometrically it asks *"what direction do the vectors point, ignoring how long they are?"*

### Range

- $\cos = +1$: vectors point in the same direction (perfect similarity).
- $\cos = 0$: vectors are orthogonal (no similarity).
- $\cos = -1$: vectors point in opposite directions.

For **term-term matrices with raw counts**, all coordinates are non-negative, so the cosine ranges over $[0, 1]$ — you'll never see negative cosines. For [[pmi|PMI]] vectors (which can go negative when PMI is allowed to be negative) or for dense embeddings, the full $[-1, +1]$ range is possible.

## Worked Example (from the slides)

Using a small word-word submatrix over contexts $\{pie, data, computer\}$:

|             | pie | data | computer |
|-------------|-----|------|----------|
| cherry      | 442 | 8    | 2        |
| digital     | 5   | 1683 | 1670     |
| information | 5   | 3982 | 3325     |

Cosine between *cherry* and *information*:

$$\cos(\text{cherry}, \text{information}) = \frac{442 \cdot 5 + 8 \cdot 3982 + 2 \cdot 3325}{\sqrt{442^2 + 8^2 + 2^2}\sqrt{5^2 + 3982^2 + 3325^2}} \approx 0.017$$

Cosine between *digital* and *information*:

$$\cos(\text{digital}, \text{information}) = \frac{5 \cdot 5 + 1683 \cdot 3982 + 1670 \cdot 3325}{\sqrt{5^2 + 1683^2 + 1670^2}\sqrt{5^2 + 3982^2 + 3325^2}} \approx 0.996$$

The numbers match intuition: *digital* and *information* both live in the tech semantic field, while *cherry* lives in the food semantic field. Their cosine similarity (0.996 vs 0.017) captures that separation. Crucially, *information* is a much longer vector than *digital* (it appears more often), but the cosine doesn't care — it normalises length out.

## Worked Example (Worksheet Q5)

For word vectors *king* $= [0.1, 0.3, 0.7]$ and *queen* $= [0.2, 0.4, 0.6]$:

- Dot product: $0.1 \cdot 0.2 + 0.3 \cdot 0.4 + 0.7 \cdot 0.6 = 0.02 + 0.12 + 0.42 = 0.56$
- $|king| = \sqrt{0.01 + 0.09 + 0.49} = \sqrt{0.59} \approx 0.77$
- $|queen| = \sqrt{0.04 + 0.16 + 0.36} = \sqrt{0.56} \approx 0.75$
- $\cos = 0.56 / (0.77 \times 0.75) \approx 0.97$

Very similar — as expected from two words in the same semantic field of monarchy.

## Why Cosine Is the Default

Three practical reasons:

1. **Length independence.** Frequent words aren't automatically "similar to everything" just because their coordinates are large.
2. **Matches human similarity judgments.** On datasets like SimLex-999, cosine of [[word2vec|learned embeddings]] correlates well with humans' pairwise ratings.
3. **Cheap to compute.** One dot product and two norms — linear in vector dimension.

For **normalised** vectors (length = 1), cosine and dot product are identical — many systems pre-normalise embeddings so they can use the faster raw dot product during retrieval.

## Pitfall: Cosine Is Not a Metric

$1 - \cos(\vec{v}, \vec{w})$ is sometimes called cosine *distance*, but it isn't a proper distance metric — it doesn't satisfy the triangle inequality in general. For most NLP tasks this doesn't matter; for nearest-neighbour search with tree indexes it can. When it does matter, use Euclidean distance on length-normalised vectors — which is monotonically related to cosine and *is* a metric.

## Related

- [[vector-semantics]] — what the vectors being compared represent
- [[tf-idf]] — cosine is the standard metric over tf-idf vectors; length normalisation is critical for retrieval
- [[pmi]] — cosine also works over PPMI-weighted vectors
- [[word2vec]] — dense embeddings are typically compared via cosine; the skip-gram objective is itself dot-product-based

## Active Recall

> [!question]- Write out the cosine-similarity formula for two vectors $\vec{v}$ and $\vec{w}$.
> $\cos(\vec{v}, \vec{w}) = \frac{\vec{v} \cdot \vec{w}}{|\vec{v}||\vec{w}|} = \frac{\sum_i v_i w_i}{\sqrt{\sum_i v_i^2}\sqrt{\sum_i w_i^2}}$. Dot product on top; product of vector norms on the bottom. Equivalent to $\cos \theta$ between the vectors.

> [!question]- Why is raw dot product a bad similarity metric, and what does cosine fix?
> Raw dot product grows with vector **length**, which for count vectors grows with word frequency. Frequent words (*the, of*) have long vectors with large coordinates, so their dot product with anything is high — making them appear "similar" to every word. Cosine divides by the lengths, normalising magnitude away so the metric measures only **direction** — which word patterns the two vectors emphasise — not how often the words occurred.

> [!question]- Compute the cosine similarity between *king* = [0.1, 0.3, 0.7] and *queen* = [0.2, 0.4, 0.6].
> Dot product $= 0.02 + 0.12 + 0.42 = 0.56$. $|king| = \sqrt{0.59} \approx 0.77$. $|queen| = \sqrt{0.56} \approx 0.75$. $\cos = 0.56 / (0.77 \cdot 0.75) \approx 0.97$ — very similar, consistent with the two words being in the same semantic field.

> [!question]- For raw-count co-occurrence vectors, what range does cosine take, and why?
> $[0, 1]$. Raw counts are always non-negative, so every coordinate product $v_i w_i \geq 0$ and the numerator (dot product) can't be negative. The denominator is a product of norms — also non-negative. So the cosine is bounded above by 1 (achieved when $\vec{v}$ and $\vec{w}$ are parallel) and below by 0 (achieved when they share no non-zero dimensions). Negative cosines only arise when vectors have signed entries (e.g. PMI, dense embeddings).

> [!question]- Why does cosine similarity not punish *information* for appearing more often than *digital* in a term-context matrix?
> The row for *information* has larger magnitudes than *digital* because *information* occurs more in the corpus — its vector is *longer*. But the two vectors point in nearly the same direction (both large on *computer* and *data*, small on *pie*). Cosine divides by the norms, cancelling magnitude, so it reports the directional similarity (≈0.996) without being dominated by raw frequency. Euclidean distance would say they're *far* apart, but cosine correctly says they're *aligned*.

> [!question]- When would dot product and cosine give identical rankings?
> When all vectors are **length-normalised** (unit norm). If $|\vec{v}| = |\vec{w}| = 1$, the cosine formula reduces to $\vec{v} \cdot \vec{w}$ exactly. Many embedding libraries pre-normalise vectors once during indexing so they can use fast raw dot products at query time without losing the length-independence property.

> [!question]- Slide MCQ: Which statements about using the **dot product** for word similarity are correct? (a) It considers vector magnitude, so it cannot distinguish words with opposite meanings but similar magnitudes; (b) A high dot product signifies strong semantic relationship, removing the need for normalisation; (c) Inability to normalise for vector length makes it unsuitable for word-vector comparison; (d) Dot products don't account for the angle between vectors, so they can't capture similarity via direction alone; (e) Dot product's effectiveness negates cosine similarity's relevance.
> **Correct: (a) and (c).** Dot product magnitude depends on both direction *and* length, so antonyms with long vectors can score high simply because they're frequent (a). And without dividing by the norms, the metric is dominated by length, not direction — making it unsuitable for raw word-vector comparison (c). (b) is the opposite of the truth — high dot product can just mean "long vectors." (d) confuses the math — the dot product **is** $|\vec{v}||\vec{w}|\cos\theta$, so it does involve the angle; the problem is that the lengths multiply in too. (e) is wrong — cosine exists *precisely because* the dot product's length sensitivity is a problem.
