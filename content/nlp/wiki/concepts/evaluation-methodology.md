---
type: concept
sources:
  - raw/week-02/w02-slides.pdf
  - raw/week-02/w02-l04-transcript.txt
  - raw/week-02/w02-l05-transcript.txt
status: draft
updated: 2026-04-20
---

*Evaluation methodology determines how we measure model quality; the core choices are extrinsic vs intrinsic evaluation and the three-way train/dev/test data split.*

## Definition

**Evaluation methodology** is the set of decisions about how to measure whether a model is any good. Two orthogonal choices govern it:

1. **What to measure** — extrinsic (task performance) vs intrinsic (model-level metric).
2. **What data to measure on** — and how to split available data to get honest answers.

These choices apply to every model in the course, not just [[n-gram-language-models]].

## Extrinsic vs Intrinsic Evaluation

### Extrinsic (in-vivo)

Put the model inside a real downstream task — machine translation, speech recognition, spam classification — run the task, measure a task-specific score (e.g., words translated correctly).

- **Advantage**: directly measures what you care about.
- **Disadvantage**: expensive, slow, task-specific. A model that excels at MT may fail at speech recognition.

### Intrinsic (in-vitro)

Measure the model's own quality directly, using a metric that does not require embedding it in an application. For language models, this metric is [[perplexity-nlp|perplexity]].

- **Advantage**: fast, cheap, general — one number that applies to any language model.
- **Disadvantage**: may not correlate perfectly with real-task performance. A model with low perplexity is not guaranteed to improve your downstream system.

In practice, intrinsic evaluation is used during development (fast iteration) and extrinsic evaluation is used for final validation (does it actually help?).

## The Data Split

### Train/test split

The most basic split:

- **Training set**: the data used to learn model parameters (e.g., count n-grams).
- **Test set**: a held-out dataset the model has never seen, used to measure generalisation.

The test set **must** be different from the training set. If they overlap, the model has already memorised the test data and will assign it artificially high probability — making itself look better than it really is. This is called **"training on the test set"** and it is bad science.

### The three-way split (train/dev/test)

A two-way split is not enough. If you test on the test set, notice the result, adjust your model, and test again, you are implicitly tuning to the test set's characteristics. After enough iterations you have effectively trained on it.

The fix: introduce a third dataset.

| Dataset | Purpose | When used | How often |
|---|---|---|---|
| **Training** | Learn model parameters | During training | Once |
| **Development (dev)** | Compare models, tune hyperparameters | During development | Many times |
| **Test** | Final evaluation of the chosen model | At the very end | Once |

The dev set absorbs all the iterative testing. The test set is reserved for one final evaluation so the reported number is honest.

> [!warning] COMMON MISCONCEPTION
> The dev set and the test set serve different purposes even though both are "held out." The dev set is your scratch paper — you can look at it as often as you like. The test set is the sealed exam — you open it once, report the score, and stop. Using the test set repeatedly during development turns it into a second dev set, which defeats its purpose.

## Choosing Splits

All three datasets should be drawn from the same distribution:

- **Task-specific model** (e.g., legal document classifier): train, dev, and test should all contain legal documents.
- **General-purpose model**: train and test should both be diverse — different domains, authors, time periods, language varieties. Otherwise the model looks good on one narrow slice and fails on everything else.

## Related

- [[perplexity-nlp|perplexity]] — the standard intrinsic metric introduced alongside this methodology
- [[n-gram-language-models]] — the first model family where this evaluation framework is applied
- [[corpora]] — corpus composition determines whether train/test splits are representative

## Active Recall

> [!question]- Why do we need three datasets (train/dev/test) rather than just two (train/test)? What specific problem does the dev set solve?
> Repeatedly evaluating on the test set implicitly tunes the model to the test set's characteristics — the researcher sees which changes improve the score and steers toward them. This is a form of overfitting to the test set. The dev set absorbs all this iterative evaluation so that the test set remains untouched. The final test-set evaluation is then an honest measure of generalization.

> [!question]- A researcher trains five different language models and evaluates each on the test set, then picks the best one and reports its test-set score. What methodological error has she committed?
> She has used the test set for model selection, which is the dev set's job. By choosing the model with the best test score, she has implicitly tuned to the test set. The reported score is now optimistic — it overestimates generalization to truly unseen data. The correct procedure: evaluate all five on the dev set, pick the best, then evaluate that single model on the test set once.

> [!question]- Explain the difference between extrinsic and intrinsic evaluation. When would you prefer each?
> Extrinsic evaluation measures model quality on a real downstream task (MT, speech recognition); it directly measures utility but is expensive and task-specific. Intrinsic evaluation measures the model itself with a general metric like perplexity; it is fast and general but may not correlate perfectly with task performance. Use intrinsic during development for rapid iteration, extrinsic for final validation that the model actually helps the application you care about.
