---
title: "Week 3 Flashcards — Text Classification"
week: 3
topic: "Naive Bayes, Sentiment Classification, and Evaluation"
deck: NLP::Week-03
---

TARGET DECK
NLP::Week-03

# Week 3 Flashcards

## Text Classification Basics

> [!question]- What is text classification?
> The task of assigning a document $d$ to one class $c$ from a fixed set of classes $C$. Examples: spam vs ham, positive vs negative sentiment, topic labels, language ID, authorship attribution. Output is a **decision** (or a distribution over classes), not a sequence — distinguishing it from language modelling.
<!--ID: 1777386639927-->





> [!question]- What are the tradeoffs between hand-coded rules and supervised ML for text classification?
> - **Rules**: high precision in narrow domains, no training data needed, interpretable. **But**: low recall (miss paraphrases), expensive to maintain, brittle when the data drifts.
> - **Supervised ML**: needs labelled data, but generalises better, scales to many classes, and recovers patterns humans wouldn't think to encode.
> - **Common practice**: rules for very narrow tasks or as features inside an ML system; supervised ML for everything else.
<!--ID: 1777386639928-->





> [!question]- What is the bag-of-words assumption?
> A document is represented as a **multiset of tokens** — counts of each word, with **position discarded**. *"Dog bites man"* and *"Man bites dog"* have the same representation. It throws away order, syntax, and structure but is empirically enough for many classification tasks because the **vocabulary distribution** is highly informative on its own.
<!--ID: 1777386639929-->





## Bayes' Rule and Naive Bayes

> [!question]- State Bayes' rule and explain each term.
> $$P(H \mid E) = \frac{P(E \mid H)\, P(H)}{P(E)}$$
> - $P(H \mid E)$: **posterior** — probability of hypothesis given evidence (what you want)
> - $P(E \mid H)$: **likelihood** — probability of evidence given hypothesis (countable from data)
> - $P(H)$: **prior** — base rate of the hypothesis
> - $P(E)$: marginal probability of the evidence (often ignored as a normalising constant when comparing classes)
<!--ID: 1777386639930-->





> [!question]- What is the Naive Bayes classifier formula?
> $$c_{NB} = \underset{c \in C}{\arg\max}\ P(c) \prod_{i=1}^{n} P(x_i \mid c)$$
> Pick the class that maximises the product of the **prior** $P(c)$ and the per-feature **likelihoods** $P(x_i \mid c)$. The marginal $P(d)$ in Bayes' rule cancels because it's the same across all classes — so we don't compute it.
<!--ID: 1777386639931-->





> [!question]- What are the two "naive" assumptions in Naive Bayes?
> 1. **Bag of words**: word position doesn't matter — only counts.
> 2. **Conditional independence**: given the class, all features (words) are independent: $P(x_1, \ldots, x_n \mid c) = \prod_i P(x_i \mid c)$.
>
> Both are obviously wrong (language is sequential and words predict each other), but the classifier still works well because for classification you only need the **relative ranking** of class scores to be correct, not the absolute probabilities.
<!--ID: 1777386639932-->





> [!question]- How do you train a multinomial Naive Bayes classifier?
> 1. **Prior**: $\hat{P}(c) = N_c / N_{\text{doc}}$ — fraction of training documents in class $c$.
> 2. **Likelihood**: concatenate all documents of class $c$ into one mega-document. For each word $w$ in the vocabulary:
> $$\hat{P}(w \mid c) = \frac{\text{count}(w, c) + 1}{\sum_{w' \in V} (\text{count}(w', c) + 1)} = \frac{\text{count}(w, c) + 1}{N_c^{\text{words}} + |V|}$$
> The $+1$ in the numerator and $+|V|$ in the denominator are **Laplace smoothing**, which keeps any zero-count word from wiping out the entire product.
<!--ID: 1777386639933-->





> [!question]- Why must we use Laplace smoothing in Naive Bayes?
> Without smoothing, any word in the test document that was never seen with class $c$ in training gets $P(w \mid c) = 0$. Since the score is a product, **one zero zeroes the entire class** — regardless of how strong the evidence from other words is. Laplace smoothing (add 1 to each count, add $|V|$ to each denominator) ensures no probability is exactly zero.
<!--ID: 1777386639934-->





> [!question]- Why does Naive Bayes work in log space, and what does that imply about the classifier?
> Multiplying many small probabilities causes **floating-point underflow**. In log space, products become sums:
> $$\log \hat{P}(c \mid d) = \log P(c) + \sum_i \log P(x_i \mid c)$$
> Log is monotonic, so the $\arg\max$ is unchanged. **Implication**: the score is a **linear function** of the input features (each word contributes additively), making Naive Bayes a **linear classifier**.
<!--ID: 1777386639935-->





> [!question]- Why is Naive Bayes equivalent to a unigram language model per class?
> If features are word occurrences, the class likelihood $\prod_i P(w_i \mid c)$ is exactly the probability assigned to the document by a unigram language model trained on class-$c$ documents. **Classification = score the document under each class's unigram LM and pick the highest** (after multiplying by the prior). The smoothing and log-space tricks transfer directly between LM and classifier work.
<!--ID: 1777386639936-->





## Binary Naive Bayes and Sentiment

> [!question]- What is binary multinomial Naive Bayes, and why does it help for sentiment?
> Before training, **clip every word count in every document to 1** — record only whether the word appeared, not how often. Reasons it helps for sentiment:
> - Seeing *fantastic* once is strong evidence of positive sentiment; seeing it five times adds little
> - Regular multinomial NB multiplies in repetition, which can let a single repeated word dominate
> - Binary NB is invariant to repetition — better matches the actual signal in sentiment data
>
> Often outperforms regular multinomial NB on sentiment but **not** on topic classification, where repeated words genuinely reinforce evidence.
<!--ID: 1777386639937-->





> [!question]- Why is negation a structural problem for bag-of-words sentiment classifiers?
> *"I like this movie"* (positive) and *"I don't like this movie"* (negative) differ only in `don't`. A bag-of-words representation has no way to encode that `don't` *inverts* the polarity of `like`. The model sees the same positive feature `like` in both documents, with `don't` as a tiny independent signal — too weak to flip the decision reliably.
<!--ID: 1777386639938-->





> [!question]- What is the standard `NOT_` prefixing fix for negation in sentiment classification?
> After encountering a negation word (`not`, `n't`, `no`), prepend `NOT_` to every subsequent word until the next punctuation mark. Example: *"didn't like this movie"* → *"didn't NOT_like NOT_this NOT_movie"*. This creates a distinct feature `NOT_like` that the classifier can learn as a negative-class signal. From Das & Chen (2001); Pang, Lee & Vaithyanathan (2002).
<!--ID: 1777386639939-->





> [!question]- What is a sentiment lexicon, and when is it useful?
> A **pre-built list of words tagged with positive/negative polarity**. Examples: MPQA Subjectivity Lexicon (~6,885 words), General Inquirer. Useful when:
> - Training data is scarce — the lexicon supplies dense polarity signal even when individual word likelihoods are noisy
> - Test domain differs from training — lexicons generalise better than corpus-specific likelihoods
>
> Used as a feature alongside (not replacing) the per-word likelihoods.
<!--ID: 1777386639940-->





## Classification Evaluation

> [!question]- Why is accuracy a misleading metric on imbalanced data?
> If 99.99% of examples are class A, a "predict A always" classifier scores 99.99% accuracy while doing zero useful work on class B. The slide example: 1M social posts, 100 about Delicious Pie Co — a "say no to everything" classifier catches zero pie posts but achieves 99.99% accuracy. **Whenever one class dominates, accuracy stops measuring what you care about.**
<!--ID: 1777386639941-->





> [!question]- Define precision and recall for a binary classifier.
> For class "positive":
> - **Precision** $= \frac{\text{TP}}{\text{TP} + \text{FP}}$ — of items predicted positive, what fraction really are?
> - **Recall** $= \frac{\text{TP}}{\text{TP} + \text{FN}}$ — of items that really are positive, what fraction did we catch?
>
> Both collapse to 0 when TP = 0 — neither is fooled by trivial classifiers on imbalanced data, unlike accuracy.
<!--ID: 1777386639942-->





> [!question]- What is F1 score, and why use it instead of averaging precision and recall?
> $$F_1 = \frac{2 P R}{P + R}$$
> The **harmonic mean** of precision and recall. Unlike the arithmetic mean, the harmonic mean is dominated by the **smaller** value — if either P or R is near zero, F1 is near zero. This punishes classifiers that are good on only one of the two metrics. F1 is the standard summary metric when you want both precision and recall to be reasonable.
<!--ID: 1777386639943-->





> [!question]- What is the difference between macro-averaging and micro-averaging?
> For multi-class metrics (precision, recall, F1):
> - **Macro-average**: compute the metric per class, then average — treats all classes equally regardless of size.
> - **Micro-average**: pool TP/FP/FN counts across all classes, then compute the metric — treats all examples equally; dominated by large classes.
>
> They can disagree dramatically. If a classifier does well on a large class but badly on a small class, micro looks good while macro looks bad. **Macro is the right metric when minority-class performance matters** (which it usually does).
<!--ID: 1777386639944-->





> [!question]- What is the paired bootstrap test, and why use it for comparing classifiers?
> A **resampling-based hypothesis test** for whether classifier A is significantly better than classifier B on a test set $x$:
> 1. Compute observed effect $\delta(x) = M(A, x) - M(B, x)$
> 2. Resample $x$ with replacement to make $x^{(i)}$ — repeat $b$ times (e.g., $b = 10{,}000$)
> 3. Count fraction of resamples where $\delta(x^{(i)}) \ge 2\delta(x)$ — this is the p-value
>
> Used because NLP metrics like F1 are **non-Gaussian and non-linear** — parametric tests like the t-test misestimate significance. The factor of 2 corrects for the fact that bootstrap samples come from a test set already biased by the observed effect.
<!--ID: 1777386639945-->





## Harms

> [!question]- What are the three types of harm classifiers can cause when deployed?
> 1. **Representational harms**: classifier outputs encode stereotypes about groups (e.g., 200 sentiment classifiers gave lower scores to sentences with African American names like *Shaniqua* than identical sentences with European American names like *Stephanie* — Kiritchenko & Mohammad 2018).
> 2. **Censorship**: toxicity classifiers over-flag sentences mentioning identity terms (*"blind"*, *"gay"*) as toxic — disproportionately silencing speech by women, disabled people, and LGBTQ+ people.
> 3. **Performance disparities**: pipelines that fail more often on minority dialects (e.g., language-ID misclassifies AAE and Indian English) drop content from those writers entirely.
<!--ID: 1777386639946-->





> [!question]- Why is a high aggregate metric not enough to deploy a classifier responsibly?
> Aggregate metrics hide **disparate per-group performance**. A classifier with 95% F1 overall might achieve 98% on the majority group and 60% on a minority — the average looks fine while the minority is failed systematically. Practical guidance: **slice the test set by demographic group and report per-group metrics**, not just the aggregate. **Model cards** (Mitchell et al., 2019) institutionalise this practice as a documentation standard.
<!--ID: 1777386639947-->






