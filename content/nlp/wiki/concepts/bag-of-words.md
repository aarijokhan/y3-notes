---
type: concept
sources:
  - raw/week-03/w03-slides.pdf
  - raw/week-03/w03-l07-transcript.txt
status: draft
updated: 2026-04-24
---

*The bag-of-words representation treats a document as an unordered multiset of tokens — a vector of word counts, with word position discarded entirely.*

## Definition

Given a document $d$ tokenized into $w_1, w_2, \ldots, w_n$, the **bag of words** is the mapping from each type in the vocabulary to its count in $d$:

$$\text{BoW}(d) = \{w : C(w, d) \text{ for } w \in V\}$$

The representation loses **word order** and **syntactic structure**. The document *"dog bites man"* and *"man bites dog"* have the same bag-of-words representation.

## Why It's Useful

Despite the obvious information loss, bag of words is the default starting representation for text classification. Reasons:

1. **Simplicity** — the representation is a sparse integer vector over the vocabulary; no parsing, no feature engineering beyond tokenization.
2. **Classifier independence** — [[naive-bayes]], logistic regression, SVMs, and even early neural networks all consume it directly.
3. **Strong empirical baseline** — for topical tasks (spam, news categorization, MeSH labelling) the words that appear are already informative enough that order mostly doesn't help.
4. **Matches the Naive Bayes independence assumption** — if you're going to assume features are conditionally independent given the class, there is no reason to carry order information that the model cannot use.

## Variants

| Variant | What counts as a feature |
|---|---|
| **Count** | $C(w, d)$ — raw token frequency |
| **Binary** | $\mathbb{1}[C(w, d) > 0]$ — did $w$ occur at all? Used in [[naive-bayes#Binary multinomial naive Bayes|binary multinomial NB]] for sentiment |
| **TF-IDF** | frequency weighted by inverse document frequency (not covered this week) |

Binary counts often beat raw counts for [[sentiment-analysis|sentiment]] — whether the word *fantastic* occurs is more informative than how many times.

## What Bag of Words Discards

- **Word order**: *"not good"* and *"good not"* become identical. This is the reason [[sentiment-analysis#Handling negation|negation handling]] has to be bolted on as a preprocessing step — otherwise *"I don't like this movie"* and *"I like this movie"* differ only in the word `don't`, which the model treats as a single independent feature.
- **Syntactic structure**: any information carried by the position of a word relative to others — subject vs object, modifier vs head — is lost.
- **Phrasal semantics**: *"New York"* becomes two tokens; the phrase no longer functions as a single entity.

## Connection to Language Models

A bag-of-words classifier with all word features and no position is equivalent to a [[n-gram-language-models|unigram language model]] per class — Naive Bayes scores a document under class $c$ by $\prod_i P(w_i \mid c)$, exactly what a unigram LM does. This is covered in [[naive-bayes#Relationship to language modelling]].

## Related

- [[naive-bayes]] — the canonical consumer of bag-of-words features
- [[tokenization]] — bag of words is bag-of-*what* the tokenizer produces
- [[type-and-token]] — bag of words is a mapping from types to token counts within a document
- [[sentiment-analysis]] — where the lossiness of bag-of-words becomes a real problem (negation, irony)

## Active Recall

> [!question]- What information does the bag-of-words representation throw away, and which NLP problems does that make harder?
> It throws away **word order** and all **syntactic structure**. This makes negation handling hard (*"don't like"* looks like *"don't"* + *"like"* independently), multiword expressions disappear, and any task that depends on who-did-what-to-whom (relation extraction, question answering) is impossible from BoW alone. For topical classification, the loss is usually tolerable.

> [!question]- Why does binary bag-of-words often outperform count bag-of-words for sentiment analysis?
> For sentiment, *occurrence* carries most of the signal — seeing the word *fantastic* once tells you almost everything about its contribution; seeing it five times adds little. Raw counts let very repetitive documents dominate the likelihood calculation with redundant evidence. Binary clipping makes each informative word count exactly once per document, which matches the structure of the task better.
