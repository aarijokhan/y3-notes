---
type: concept
sources:
  - raw/week-03/w03-slides.pdf
  - raw/week-03/w03-l08-transcript.txt
  - raw/week-03/w03-lab-01.pdf
  - raw/week-03/w03-lab-01-solutions.pdf
  - raw/week-03/3 NB and Sentiment Classification - Consolidation-1.pdf
status: draft
updated: 2026-04-24
---



*Sentiment analysis is the text classification task of labelling a document as positive, negative, or neutral — and the canonical worked example for [[naive-bayes|Naive Bayes]] text classification.*

## Task Definition

**Input**: a document (product review, tweet, movie review, earnings call transcript).
**Output**: a sentiment label — typically `positive`/`negative`, sometimes with `neutral` or a 1–5 star scale.

Sentiment is the prototypical example in the course because it exposes every weakness of bag-of-words classification — and the mitigations motivate several techniques ([[naive-bayes#Binary multinomial naive Bayes|binary NB]], negation handling, lexicon features).

## Where Sentiment Fits: Scherer's Typology of Affective States

Sentiment analysis is the detection of **attitudes**, which is one of five categories in Scherer's typology of affective states. The course focuses narrowly on attitudes; the other categories are mentioned to explain *what sentiment analysis is not*.

| Affective state | Description | Examples |
|---|---|---|
| **Emotion** | Brief, organically synchronized evaluation of a major event | angry, sad, joyful, fearful, ashamed, proud, elated |
| **Mood** | Diffuse, non-caused, low-intensity, long-duration change in subjective feeling | cheerful, gloomy, irritable, listless, depressed, buoyant |
| **Interpersonal stance** | Affective stance toward another person in a specific interaction | friendly, flirtatious, distant, cold, warm, supportive, contemptuous |
| **Attitudes** | Enduring, affectively coloured beliefs or dispositions toward objects or persons | liking, loving, hating, valuing, desiring |
| **Personality traits** | Stable personality dispositions and typical behaviour tendencies | nervous, anxious, reckless, morose, hostile, jealous |

The simple task this chapter focuses on is: "is the attitude of this text **positive** or **negative**?" General affect classification — emotion detection, mood tracking, stance analysis — is a richer and harder problem covered in later chapters.

## Worked Example: Small Sentiment Classifier

From the slides. Training set (5 documents):

| Class | Document |
|---|---|
| − | just plain boring |
| − | entirely predictable and lacks energy |
| − | no surprises and very few laughs |
| + | very powerful |
| + | the most fun film of the summer |

Test: *"predictable with no fun"*.

**Prior**: $P(-) = 3/5$, $P(+) = 2/5$.
**Drop `with`** (it's not in the training vocabulary; see [[naive-bayes#Handling unknown words and stop words|handling unknown words]]).

With Laplace smoothing and vocabulary size 20:

$$P(\text{predictable} \mid -) = \frac{1+1}{14+20}, \quad P(\text{predictable} \mid +) = \frac{0+1}{9+20}$$

And similarly for `no` and `fun`. Final scores:

$$P(-) \cdot P(S \mid -) = \frac{3}{5} \cdot \frac{2 \cdot 2 \cdot 1}{34^3} = 6.1 \times 10^{-5}$$

$$P(+) \cdot P(S \mid +) = \frac{2}{5} \cdot \frac{1 \cdot 1 \cdot 2}{29^3} = 3.2 \times 10^{-5}$$

Negative wins. The test document gets labelled **negative** — even though *fun* points positive, the two negative-leaning words (*predictable*, *no*) combined with the stronger negative prior dominate.

## Why Bag of Words Struggles with Sentiment

**Occurrence beats frequency.** Seeing the word *fantastic* once tells you a lot about sentiment; seeing it five times tells you little more. Raw counts weight repeated words too heavily. This is why [[naive-bayes#Binary multinomial naive Bayes|binary multinomial NB]] — clipping counts at 1 per document — often beats plain multinomial NB on sentiment.

**Negation flips polarity but not bag identity.** *"I like this movie"* and *"I don't like this movie"* are almost identical bags; the word `don't` is the only difference, and treating it as a single independent feature can't capture that it *inverts* the polarity of `like`. Mitigations:
- **`NOT_` prefixing** (Das & Chen 2001; Pang, Lee, Vaithyanathan 2002): add `NOT_` to every word between a negation and the next punctuation. *"didn't like this movie, but I"* becomes *"didn't NOT_like NOT_this NOT_movie but I"*. The model learns that `NOT_like` is a negative-class feature even though `like` is positive.

**Irony, sarcasm, domain shift** — bag-of-words has no hope for these. Modern work handles them with contextual embeddings; NB is a baseline that gets you most of the way on straightforward reviews.

## Handling Negation

The baseline method (simple and effective for NB):

```
didn't like this movie , but I
↓
didn't NOT_like NOT_this NOT_movie but I
```

Rule: after a negation word (`not`, `n't`, `no`, `never`), prefix `NOT_` to every subsequent word until the next punctuation mark. Treat `NOT_X` as a new vocabulary entry during training. This gives the classifier a chance to learn that `NOT_like` and `NOT_good` are negative-class features.

Approximate and over-aggressive in complex sentences, but it captures most of the signal for the same cost as tokenization.

## Sentiment Lexicons

When labelled training data is scarce, **pre-built word lists with polarity labels** (lexicons) can supply extra signal.

### MPQA Subjectivity Cues Lexicon

- **Source**: Wilson, Wiebe, Hoffmann (2005); Riloff & Wiebe (2003)
- **URL**: https://mpqa.cs.pitt.edu/lexicons/subj_lexicon/
- **Size**: 6,885 words from 8,221 lemmas, annotated for intensity (strong/weak)
- **Breakdown**: 2,718 positive, 4,912 negative
- **Positive examples**: *admirable, beautiful, confident, dazzling, ecstatic, favor, glee, great*
- **Negative examples**: *awful, bad, bias, catastrophe, cheat, deny, envious, foul, harsh, hate*

### The General Inquirer

- **Source**: Stone, Dunphy, Smith, Ogilvie (1966)
- **URL**: http://www.wjh.harvard.edu/~inquirer
- **Categories**: Positiv (1915 words), Negativ (2291 words); Strong/Weak; Active/Passive; Overstated/Understated; Pleasure, Pain, Virtue, Vice, Motivation, Cognitive Orientation
- Free for research use.

### How to Use Lexicons in Classification

**As a feature**: add "token occurs in positive lexicon" and "token occurs in negative lexicon" as two extra features that increment every time a matching word appears. Now all positive words (*good, great, beautiful, wonderful, ...*) count as a single dense feature, and similarly for negative.

Using just these two lexicon features is worse than using all the word features — individual words carry more information. But lexicon features help in two specific cases:

1. **Sparse training data** — when you have few labelled examples, each individual word has a noisy likelihood estimate; aggregating across the lexicon gives a more stable signal.
2. **Domain shift** — when test data is unlike training data, individual word counts may not transfer, but the positive-lexicon aggregate still fires on seen and unseen positive words alike.

The lexicon contains only isolated word labels — it does not provide sentence-level examples, so you can't use it to augment labelled training data directly. Use it as a feature source, not as extra supervision.

## Related

- [[naive-bayes]] — the classifier this task is taught with
- [[text-classification]] — sentiment is one of many text classification tasks
- [[bag-of-words]] — the representation sentiment exposes the weaknesses of
- [[classification-evaluation]] — how to measure sentiment classifier quality ([[harms-in-classification|accuracy isn't enough]])
- [[harms-in-classification]] — sentiment classifiers assign lower sentiment to sentences with African American names (Kiritchenko & Mohammad 2018)

## Active Recall

> [!question]- Why does the sentence *"I really don't like this movie"* present a fundamental problem for bag-of-words sentiment classification?
> Under BoW the sentence is almost identical to *"I really like this movie"* — the only difference is the single word `don't`, treated as an independent feature. The classifier cannot represent that `don't` **inverts the polarity** of `like`; it can only learn that `don't` tends to co-occur with negative documents. The baseline mitigation is `NOT_` prefixing, which adds `NOT_like` as a distinct feature, letting the model learn its polarity separately.

> [!question]- What is `NOT_` prefixing and what problem does it solve?
> A preprocessing rule for sentiment analysis: after each negation word, add `NOT_` to every subsequent word until the next punctuation mark. This turns *"didn't like this movie"* into *"didn't NOT_like NOT_this NOT_movie"*. The model can then learn that `NOT_like` is a negative-class signal even though `like` is positive-class. It solves the core bag-of-words limitation that a single negation word cannot propagate its effect to later tokens.

> [!question]- Why are binary multinomial NB and regular multinomial NB more likely to *disagree* on sentiment than on topic classification?
> For topic classification, repeated content words reinforce the topic signal — seeing "baseball" ten times makes the sports topic more confident. For sentiment, repetition mostly doesn't add evidence once you've seen the word — *fantastic* five times is barely more informative than *fantastic* once. Regular NB multiplies in the redundant evidence, while binary NB ignores repetitions. On sentiment they often pick different classes because the repeated-word weighting flips the balance.

> [!question]- When do sentiment lexicons like MPQA help, and when do they not?
> Lexicons help when **training data is sparse** (individual word likelihoods are noisy, so aggregating positive words into one feature stabilizes the signal) or when **test data differs from training data** (unseen positive words still fire the positive-lexicon feature). They help less when training data is abundant and representative, because individual word likelihoods are already well-estimated and more specific than a coarse lexicon flag. Lexicons are features, not substitutes for labelled data.
