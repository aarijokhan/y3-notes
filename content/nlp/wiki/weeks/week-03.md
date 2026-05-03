---
type: week
week: 3
title: "Week 3 — Text Classification: Naive Bayes, Sentiment, and Evaluation"
dates: —
sources:
  - raw/week-03/w03-slides.pdf
  - raw/week-03/w03-l07-transcript.txt
  - raw/week-03/w03-l08-transcript.txt
  - raw/week-03/w03-l09-transcript.txt
  - raw/week-03/w03-lab-01.pdf
  - raw/week-03/w03-lab-02.pdf
  - raw/week-03/w03-lab-01-solutions.pdf
  - raw/week-03/w03-lab-02-solutions.pdf
  - raw/week-03/3 NB and Sentiment Classification - Consolidation-1.pdf
concepts:
  - "[[text-classification]]"
  - "[[bag-of-words]]"
  - "[[bayes-rule]]"
  - "[[naive-bayes]]"
  - "[[sentiment-analysis]]"
  - "[[classification-evaluation]]"
  - "[[harms-in-classification]]"
status: stable
updated: 2026-04-24
---

> [!question]+ THE CRUX: Given a document and a fixed set of categories, how do you build the simplest classifier that works — how do you know whether it actually works — and what happens when you deploy it on people?

*A product of word probabilities, one per class, is enough to classify documents well. Making that work in practice means making a "naive" independence assumption that is obviously wrong, and learning to measure success with something more honest than accuracy.*

---

## From Probabilities over Sentences to Probabilities over Classes

[[week-02]] built [[n-gram-language-models|language models]] that assign probabilities to sequences — $P(w_1, w_2, \ldots, w_n)$. This week swaps the question. We have a document and want a *category label*: is this email spam? Is this review positive? Is this essay by Hamilton or Madison? The task is [[text-classification]], and it differs from language modelling in one structural way: the output is a *decision*, not a distribution.

The simplest approach is **hand-coded rules**. In narrow domains, rules can be very precise — but they miss paraphrases and edge cases, so recall is low. More importantly, rules are expensive to maintain. The alternative is **supervised machine learning**: collect hand-labelled documents and learn a classifier $\gamma: d \to c$ from them. This week introduces Naive Bayes — the simplest classifier that works for text, and the baseline everyone should try first.

## The Simplest Thing That Works

Before the classifier, the tool it's built on: [[bayes-rule|Bayes' rule]] inverts a conditional, $P(H \mid E) = P(E \mid H) P(H) / P(E)$. The payoff in NLP is always the same — $P(H \mid E)$ is hard to estimate directly, but the likelihood and prior on the right-hand side can be counted from data. The rule also forces you to include the **prior** $P(H)$: the classic Kahneman & Tversky *Steve* example (meek soul → librarian or farmer?) shows how judging only by likelihood and ignoring base rates produces systematically wrong answers.

[[naive-bayes|Naive Bayes]] applies Bayes' rule to (document, class). It scores each class by a product of word likelihoods times a prior, then picks the highest:

$$c_{NB} = \underset{c \in C}{\arg\max}\ P(c) \prod_{i} P(x_i \mid c)$$

Two "naive" assumptions make this tractable. The [[bag-of-words]] assumption says position doesn't matter — *"dog bites man"* and *"man bites dog"* have the same representation. The **conditional independence** assumption says that given the class, word features are independent. Both are obviously wrong — language is deeply sequential and words predict each other constantly — yet the classifier works anyway, because for classification you only need the *relative ranking* of class scores to be right.

Learning is a single pass of counting. Concatenate all documents of class $c$ into one mega-document, count each word's occurrences, and normalize to get per-class word likelihoods. Count documents per class to get the prior. Done. The algorithm is fast, needs little memory, and works surprisingly well on small training sets.

> [!question]- A training corpus has 3 negative reviews (total 14 tokens) and 2 positive reviews (total 9 tokens), with vocabulary size 20. The word "predictable" appears once in negative reviews and zero in positive. Using Laplace smoothing, compute P(predictable | −) and P(predictable | +).
> $P(\text{predictable} \mid -) = (1+1)/(14+20) = 2/34 \approx 0.059$. $P(\text{predictable} \mid +) = (0+1)/(9+20) = 1/29 \approx 0.034$. Without smoothing, the positive-class probability would be zero — and one zero in a product wipes out the entire class score, regardless of other evidence. Laplace rescues the math.

## The Two Traps: Zero Probabilities and Floating-Point Underflow

Maximum-likelihood estimation has the same flaw it has for n-gram LMs: any word not seen with a class in training gets probability zero, and one zero anywhere in a product zeroes the whole sentence. [[smoothing|Laplace smoothing]] — add 1 to every count — is the usual fix. Unlike in n-gram modelling, where add-one is too aggressive because the bigram space is huge, it works well for Naive Bayes because the vocabulary is smaller and the classifier only needs the *ranking* of classes, not calibrated probabilities.

The second trap is **floating-point underflow**. Multiplying many small probabilities produces numbers smaller than 64-bit floats can represent. The fix (same as language models): work in log space. $\log(ab) = \log a + \log b$, so multiplication becomes addition, and log is monotonic, so the $\arg\max$ is unchanged. A side effect worth noticing — in log space, the classifier's score is a *linear function* of the inputs, so Naive Bayes is a **linear classifier**.

> [!info] ASIDE — Naive Bayes is a unigram LM per class
> If you use only word features and all words in the document, each class's Naive Bayes likelihood $\prod_i P(w_i \mid c)$ is exactly a [[n-gram-language-models|unigram language model]] conditioned on the class. Classification = score the document under each class's unigram LM and pick the best. The same tricks (smoothing, log space) solve the same problems.

## Sentiment: Where Bag of Words Breaks

[[sentiment-analysis|Sentiment analysis]] is the canonical worked example for Naive Bayes because it exposes every weakness of bag-of-words representation and motivates several fixes.

**Occurrence matters more than frequency.** Seeing *fantastic* once tells you the review is positive; seeing it five times adds little more. Regular multinomial NB weights repetitions heavily. [[naive-bayes#Binary multinomial naive Bayes|Binary multinomial NB]] clips counts at 1 per document before training and often outperforms the regular version on sentiment. The two classifiers can pick different classes on the same document; the labs demonstrate a case where they do.

**Negation is a structural problem.** *"I like this movie"* and *"I don't like this movie"* differ only in the word `don't` — a bag-of-words representation cannot encode that `don't` *inverts* the polarity of `like`. The standard baseline fix (Das & Chen 2001; Pang, Lee, Vaithyanathan 2002) is `NOT_` prefixing: after a negation word, prepend `NOT_` to every subsequent word until the next punctuation mark. *"didn't like this movie"* becomes *"didn't NOT_like NOT_this NOT_movie"*, giving the model a chance to learn `NOT_like` as a distinct negative-class feature.

**When training data is scarce**, pre-built **sentiment lexicons** — the MPQA Subjectivity Lexicon (6,885 words) and the General Inquirer — provide a dense positive/negative feature. They don't replace labelled data but help when individual word likelihoods are noisy or the test domain differs from training.

> [!question]- Why do binary and multinomial Naive Bayes often *disagree* on sentiment classification but rarely on topic classification?
> For topic, repeated words reinforce the signal: seeing "baseball" ten times makes sports more confident. For sentiment, once you've seen *fantastic* you've learned what you need; repetition doesn't add evidence. Multinomial NB multiplies in the redundant signal and can be tipped by a single very-repeated word; binary NB ignores repetition. On borderline sentiment documents the repeated-word weighting can flip the decision.

## Measuring Quality: Accuracy Is a Lie

How do you know a classifier is good? The obvious metric is **accuracy** — fraction of correct predictions. It's fine on balanced data and useless on imbalanced data.

The slide example: 1,000,000 social media posts, 100 about Delicious Pie Co, 999,900 not. A "say no to everything" classifier predicts "not about pie" every time. It catches zero pie posts — useless at the actual job — but scores **99.99% accuracy**. Whenever one class dominates, accuracy stops meaning what you think it means.

[[classification-evaluation]] gives the correct tools. **Precision** (of items labelled positive, what fraction really are) and **recall** (of items that really are positive, what fraction were caught) are both defined per class. The "say no" classifier has tp = 0, so both precision and recall collapse to 0 — neither is fooled by imbalance. **F1** combines them as the harmonic mean, which punishes extremes: if either P or R is near zero, F1 is near zero.

For multi-class problems, precision and recall are defined per class — combining them requires choosing between **macro-averaging** (average per class, treats classes equally) and **micro-averaging** (pool all counts, treats examples equally). The two can give very different answers; in the lab's 3-class example, macro-precision is 0.57 while micro-precision is 0.75 — the gap exposes that the classifier does badly on a minority class that the pooled metric hides.

For comparing two classifiers rigorously, **statistical hypothesis testing** asks: is the observed effect size $\delta(x) = M(A, x) - M(B, x)$ large enough to reject the null hypothesis that A is not better than B? NLP metrics like F1 are non-linear and non-Gaussian, so parametric tests like the t-test misestimate significance. The standard tool is the **paired bootstrap**: resample the test set with replacement thousands of times, recompute $\delta(x^{(i)})$ on each resample, and count how often $\delta(x^{(i)}) \ge 2\delta(x)$ (the factor of 2 corrects for the fact that the bootstrap samples are drawn from a test set already biased by the observed effect). If fewer than 1% of resamples hit that threshold, the difference is significant at $p < 0.01$.

Before any of this, a **devset** (or $k$-fold cross-validation) is needed for tuning. Touching the test set during development leaks test information into the model and inflates the reported score.

> [!question]- A classifier on a cancer-screening task achieves 99% accuracy. Should you be impressed? What should you compute instead?
> Probably not — cancer is rare (say 1% prevalence), so a classifier that says "no cancer" for everyone achieves 99% accuracy while catching zero cases. Compute **recall** on the cancer class (what fraction of actual cancers are caught) and **precision** (what fraction of predicted cancers really are). A screening tool prefers high recall; follow-up tests can weed out false positives, but a missed cancer is untreated.

> [!tip] TIP — Exam calculation problems
> The labs emphasize three calculation types: (1) multinomial vs binary Naive Bayes scoring with Laplace smoothing — including the `9+7`-style denominator where 7 is vocab size, (2) accuracy/precision/recall/F1 from a confusion matrix, and (3) micro vs macro averaging for 3-class problems. Always check: is smoothing applied? Is the computation per class or pooled? Are counts binarized per document before training? These details change every number.

## What Goes Wrong in Deployment

Classifiers — including simple Naive Bayes ones — cause real harm when deployed. [[harms-in-classification]] covers three kinds:

- **Representational harms**: Kiritchenko & Mohammad (2018) found that 200 sentiment classifiers assigned lower sentiment to sentences with African American names (*Shaniqua*) than identical sentences with European American names (*Stephanie*). These systems are used in marketing and mental health research, so the bias propagates into how people are represented and treated.
- **Censorship**: toxicity classifiers over-flag sentences that merely *mention* identity terms like *"blind"* or *"gay"*, disproportionately censoring speech by women, disabled people, and LGBTQ+ people.
- **Performance disparities**: language identification — usually the first step in an NLP pipeline — performs worse on African American English and Indian English, so content from those writers is dropped from downstream processing entirely.

The causes are biased *data*, biased *labels*, and biased *optimization targets* (optimizing average accuracy gives the model no incentive to perform well on minorities). There is no general mitigation. The practical guidance: measure per-group performance directly — slicing the test set by demographic group — rather than trusting aggregate metrics that hide the disparity. One documentation practice that makes biases *visible* (though it doesn't fix them) is the **model card** (Mitchell et al., 2019) — a short structured description of the model's training data, intended use, and disaggregated per-group performance, released alongside every model.

---

## Concepts Introduced This Week

- [[text-classification]] — assign a document to one of a fixed set of classes; rules vs supervised ML
- [[bag-of-words]] — document-as-multiset-of-tokens representation; throws away order and structure
- [[bayes-rule]] — $P(H \mid E) = P(E \mid H) P(H) / P(E)$; the probabilistic identity Naive Bayes rests on
- [[naive-bayes]] — Bayes' rule + conditional independence gives a fast, robust linear classifier
- [[sentiment-analysis]] — the canonical worked example; binary NB, negation handling, lexicons
- [[classification-evaluation]] — confusion matrix, precision/recall/F1, micro vs macro, bootstrap
- [[harms-in-classification]] — representational harms, censorship, performance disparities

## Connections

Builds on [[week-02]]: Naive Bayes is effectively a [[n-gram-language-models|unigram LM]] per class, and reuses [[smoothing|Laplace smoothing]] and log-space arithmetic from there. Uses [[tokenization]] and [[type-and-token|vocabulary]] infrastructure from [[week-01]].

Sets up later weeks — logistic regression will replace the generative Naive Bayes with a discriminative linear classifier; eventually transformers replace the independence assumption with learned contextual representations. But the evaluation metrics (precision, recall, F1, macro/micro, bootstrap) carry forward unchanged to every classifier in the module.

## Open Questions

- When does the conditional independence assumption actually hurt classification accuracy, and when does it just hurt calibration?
- Why does binary multinomial NB typically beat regular multinomial NB on sentiment but not on topic classification? Is the cross-over point known?
- How much of the performance of large language models on classification tasks is explained by them being sophisticated per-class unigram (or n-gram) models?
