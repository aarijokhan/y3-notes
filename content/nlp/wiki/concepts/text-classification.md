---
type: concept
sources:
  - raw/week-03/w03-slides.pdf
  - raw/week-03/w03-l07-transcript.txt
status: draft
updated: 2026-04-24
---

*Text classification assigns a predefined category to a document — the general pattern that covers spam detection, sentiment analysis, authorship identification, topic labelling, and language ID.*

## Definition

**Input**: a document $d$ and a fixed set of classes $C = \{c_1, c_2, \ldots, c_J\}$.
**Output**: a predicted class $c \in C$.

The set of classes is fixed in advance — text classification is a *closed-label* task. Choosing the labels, and the annotation process that produces the training examples, is a design decision that shapes every downstream decision about the model.

## Example Tasks

| Task | Classes |
|---|---|
| Spam detection | `spam` / `not spam` |
| Sentiment analysis | `positive` / `negative` (sometimes `neutral`) |
| Authorship identification | a fixed set of candidate authors (the *Federalist Papers* problem: Hamilton, Madison, Jay) |
| Topic categorization | MeSH subject hierarchy, newswire topics |
| Language ID | a fixed set of languages, often using character n-grams as features |

## Two Classification Strategies

### 1. Hand-coded rules

Rules over features: `spam := black-list-address OR ("dollars" AND "have been selected")`. When experts refine rules carefully in a narrow domain, **precision can be very high**. But maintaining rules is expensive, and rules are brittle — they miss paraphrases, so **recall is low**. This is a high-precision, low-recall regime.

### 2. Supervised machine learning

Given a training set $(d_1, c_1), \ldots, (d_m, c_m)$ of hand-labelled documents, learn a classifier $\gamma: d \to c$. Candidates include [[naive-bayes]], logistic regression, $k$-nearest neighbours, neural networks, and prompted/fine-tuned LLMs. This week introduces Naive Bayes as the simplest, most interpretable baseline — fast to train, robust on small datasets, and surprisingly hard to beat on many tasks.

## Where This Sits in the Module

Text classification is the **first real prediction task** in NLP after language modelling. Where [[n-gram-language-models|n-gram LMs]] asked "what is the probability of this sentence?", classification asks "what category does this document belong to?" — a decision rather than a distribution. The machinery overlaps: [[bag-of-words]] representations, MLE parameter estimation, Laplace smoothing, and log-space arithmetic all reappear.

## Related

- [[naive-bayes]] — the simplest classifier that works for text
- [[bag-of-words]] — the document representation used by most classical classifiers
- [[sentiment-analysis]] — the canonical worked example
- [[classification-evaluation]] — how to tell if a classifier is any good
- [[harms-in-classification]] — what goes wrong when classifiers are deployed

## Active Recall

> [!question]- Why is hand-coded rule-based classification high-precision but low-recall?
> Rules are literal and specific. A carefully written rule rarely fires on something it shouldn't (few false positives → high precision), but it also misses paraphrases, novel phrasings, and edge cases the rule-writer didn't anticipate (many false negatives → low recall). Supervised ML improves recall by generalizing from examples rather than enumerating conditions.

> [!question]- What are the inputs and outputs of a supervised text classifier?
> **Input at training time**: a document $d$, a fixed class set $C$, and a training set $(d_1, c_1), \ldots, (d_m, c_m)$ of labelled documents. **Output**: a learned classifier $\gamma: d \to c$ that maps any new document to one of the classes in $C$. At inference time, only $d$ is the input — $\gamma$ produces $c$.
