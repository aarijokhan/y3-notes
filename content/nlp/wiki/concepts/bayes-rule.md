---
type: concept
sources:
  - raw/week-03/w03-slides.pdf
  - raw/week-03/w03-l07-transcript.txt
status: draft
updated: 2026-04-24
---

*Bayes' rule inverts a conditional probability — it lets you compute $P(H \mid E)$ from $P(E \mid H)$, $P(H)$, and $P(E)$. In NLP it's the machinery behind classifiers, noisy-channel decoders, and any system that reasons about unseen causes from observed evidence.*

## The Rule

Start from the definition of conditional probability two ways:

$$P(H, E) = P(H \mid E) P(E) = P(E \mid H) P(H)$$

Rearrange:

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

Every term has a name:

| Term | Name | Meaning |
|---|---|---|
| $P(H \mid E)$ | **posterior** | updated belief in $H$ after seeing $E$ |
| $P(E \mid H)$ | **likelihood** | how probable $E$ is if $H$ is true |
| $P(H)$ | **prior** | belief in $H$ before seeing $E$ |
| $P(E)$ | **evidence** (or marginal) | total probability of seeing $E$ under any hypothesis |

## The Mantra

> New evidence shouldn't determine your belief *in a vacuum* — it should update a prior.

This is the whole point. The posterior is built from two ingredients: how well the evidence fits the hypothesis (likelihood) *and* how plausible the hypothesis was to begin with (prior). Dropping either gives the wrong answer.

## Base Rate Neglect: The Steve Problem

Kahneman & Tversky's canonical demonstration (adapted in 3Blue1Brown's [*Bayes theorem, the geometry of changing beliefs*](https://youtu.be/HZGCoVF3YvM)):

> Steve is a meek and tidy soul, with a need for order and structure and a passion for detail. Is Steve more likely to be a **librarian** or a **farmer**?

Most people say librarian — the description matches the stereotype. They're reasoning only from the **likelihood** $P(\text{description} \mid \text{librarian})$ and ignoring the **prior**.

But in the US there are roughly **20 farmers per librarian**. Work through the numbers with a representative sample of 210 people (10 librarians, 200 farmers), and suppose 40% of librarians fit Steve's description vs only 10% of farmers:

- Librarians matching the description: $10 \times 0.4 = 4$
- Farmers matching the description: $200 \times 0.1 = 20$

$$P(\text{librarian} \mid \text{description}) = \frac{4}{4 + 20} \approx 0.17$$

Steve is ~83% likely to be a farmer — despite the description screaming *librarian*. The likelihood ratio (4:1 in favour of librarian) is overwhelmed by the prior ratio (20:1 in favour of farmer). Ignoring the prior is **base rate neglect**, and it's the systematic error Bayes' rule corrects.

## Geometric Picture

Draw a 1×1 square representing the full space of possibilities.

1. **Partition by hypothesis.** A vertical strip of width $P(H)$ covers the cases where $H$ is true; the remaining strip covers $\neg H$. The strip widths are the **priors**.
2. **Carve out evidence within each strip.** Inside the $H$ strip, a sub-region of proportion $P(E \mid H)$ represents cases where $E$ is also observed. Same inside the $\neg H$ strip with proportion $P(E \mid \neg H)$.
3. **Restrict to the evidence.** Throw away everything where $E$ doesn't hold. You're left with the two evidence-sub-regions.
4. **The posterior is the fraction of the remaining area that sits in the $H$ strip.**

$$P(H \mid E) = \frac{\text{area of } (H \cap E)}{\text{area of } (H \cap E) + \text{area of } (\neg H \cap E)}$$

The key takeaway from this picture: **a small prior multiplies down the area**, and a large likelihood alone can't rescue a hypothesis whose strip is thin. The posterior is a proportion, not a score — a tiny slice times a generous likelihood can still be smaller than a fat slice times a mediocre likelihood.

## Where Bayes' Rule Shows Up in NLP

- **[[naive-bayes|Naive Bayes classification]]** — flip $P(\text{class} \mid \text{document})$ into $P(\text{document} \mid \text{class}) P(\text{class})$, which is estimable from counts. The formula from Naive Bayes is literally Bayes' rule with $H = c$ and $E = d$.
- **Noisy channel models** — for spelling correction, machine translation, and speech recognition: $P(\text{source} \mid \text{observed}) \propto P(\text{observed} \mid \text{source}) \cdot P(\text{source})$. A [[n-gram-language-models|language model]] supplies the prior; a channel model supplies the likelihood.
- **Any probabilistic decoder** — HMM decoding, probabilistic parsing, topic models. Anywhere you infer a latent cause from observed words, you are doing Bayesian inference.

The reason Bayes' rule is useful is almost always that $P(H \mid E)$ is hard to estimate directly but $P(E \mid H)$ and $P(H)$ can be estimated from data. You invert the conditioning to get at the quantity you actually want.

## Dropping the Denominator in $\arg\max$

When you only need the **most probable** hypothesis — not its actual probability — the denominator $P(E)$ is irrelevant. It's the same for every hypothesis, so it doesn't affect the ranking:

$$\underset{H}{\arg\max}\ P(H \mid E) = \underset{H}{\arg\max}\ P(E \mid H) \cdot P(H)$$

This is the step that turns Bayes' rule from "intractable marginalization problem" into "pick the hypothesis with the highest numerator" — and it's what makes [[naive-bayes|Naive Bayes]] practical.

## Related

- [[naive-bayes]] — Bayes' rule applied to (document, class) with conditional independence
- [[n-gram-language-models]] — supplies the prior $P(\text{text})$ in noisy-channel decoders
- [[evaluation-methodology]] — Bayesian reasoning underlies held-out validation and posterior predictive checks

## Active Recall

> [!question]- State Bayes' rule and name every term.
> $P(H \mid E) = P(E \mid H) P(H) / P(E)$. $P(H \mid E)$ is the **posterior** (updated belief after seeing evidence). $P(E \mid H)$ is the **likelihood** (probability of observing $E$ if $H$ were true). $P(H)$ is the **prior** (belief in $H$ before seeing evidence). $P(E)$ is the **evidence** or marginal (total probability of observing $E$ under any hypothesis). The formula inverts the conditional — useful whenever $P(H \mid E)$ is hard to estimate directly but the other terms aren't.

> [!question]- Why does the denominator $P(E)$ drop out when you use Bayes' rule in a classifier?
> The classifier picks $\arg\max_H P(H \mid E)$, not the actual probability. $P(E)$ is the same for every candidate $H$, so it scales every score identically and cannot change the ranking. Dropping it turns an intractable normalization problem (summing $P(E \mid H') P(H')$ over every possible $H'$) into a comparison of unnormalized scores — the step that makes Bayes' rule practical for NLP.

> [!question]- Explain the Steve librarian/farmer example and the error it exposes.
> Steve's description matches a librarian stereotype — $P(\text{description} \mid \text{librarian})$ is high. Most people stop there and conclude he's a librarian. They ignore the **prior**: there are ~20 farmers per librarian in the US. Even if librarians match the description 4× more often, the 20× larger pool of farmers dominates — Steve is ~83% likely to be a farmer. The error is **base rate neglect**: reasoning from likelihood without multiplying in the prior. Bayes' rule forces you to include both.

> [!question]- In the geometric 1×1 square picture, what do the vertical strip widths and the sub-region areas represent?
> Strip widths are **priors** — each hypothesis occupies a vertical strip of width $P(H)$ spanning the full square. Within each strip, a sub-region of proportion $P(E \mid H)$ represents the cases where the evidence is observed — those areas are the **joint probabilities** $P(H, E) = P(E \mid H) P(H)$. The **posterior** $P(H \mid E)$ is the fraction of the total evidence area (across all hypotheses) that sits in the $H$ strip.

> [!question]- Why is Bayes' rule useful in NLP — what does it let you swap?
> It lets you replace a hard-to-estimate conditional $P(H \mid E)$ with a combination of terms that *are* estimable from data: $P(E \mid H)$ (often countable — e.g. "how often does this word appear in documents of this class") and $P(H)$ (often just a corpus frequency). This inversion is the core trick in Naive Bayes, noisy-channel spelling correction, statistical MT, speech recognition, and HMM decoding.
