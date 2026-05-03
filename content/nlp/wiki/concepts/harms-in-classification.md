---
type: concept
sources:
  - raw/week-03/w03-slides.pdf
  - raw/week-03/w03-l09-transcript.txt
  - raw/week-03/3 NB and Sentiment Classification - Consolidation-1.pdf
status: draft
updated: 2026-04-24
---

*Text classifiers — including simple ones like [[naive-bayes|Naive Bayes]] — can produce systematic harm when they encode biases from their training data. Good aggregate metrics can coexist with serious per-group failures.*

## Three Kinds of Harm

### Representational harms

Harms caused by a system that **demeans a social group**, such as by perpetuating negative stereotypes.

**Kiritchenko and Mohammad (2018) study**: examined 200 sentiment analysis systems on pairs of sentences that were **identical except for names** — common African American names (*Shaniqua*) vs European American names (*Stephanie*). Example pair: *"I talked to Shaniqua yesterday"* vs *"I talked to Stephanie yesterday"*.

**Result**: systems systematically assigned **lower sentiment and more negative emotion** to sentences with African American names — despite the sentences being semantically identical.

**Downstream harm**: these sentiment tools are widely used in marketing research and mental health studies, so biased scores cause African Americans to be treated differently by the downstream systems consuming the outputs. The bias perpetuates stereotypes rather than creating new ones — but it amplifies and automates them.

### Harms of censorship

**Toxicity detection** is the text classification task of detecting hate speech, abuse, harassment, or other toxic language — widely deployed in online content moderation.

Toxicity classifiers have been documented to **incorrectly flag non-toxic sentences that simply mention minority identities** — sentences containing words like *"blind"* or *"gay"* are over-flagged as toxic regardless of context:
- Women (Park et al., 2018)
- Disabled people (Hutchinson et al., 2020)
- Gay people (Dixon et al., 2018; Oliva et al., 2021)

**Downstream harms**:
- Speech by these groups is censored disproportionately.
- Their speech becomes less visible online.
- Writers learn (explicitly or implicitly) to avoid identity-referring vocabulary, making them less likely to write about themselves.

### Performance disparities

Text classifiers perform **worse on many languages of the world** due to lack of data or labels — the long tail of low-resource languages.

They also perform worse on **varieties of even high-resource languages**. Example: **language identification**, typically a first step in an NLP pipeline ("is this post in English?"), **performs worse on English written by African Americans** (Blodgett & O'Connor 2017) or **by writers from India** (Jurgens et al., 2017) — leading to their content being dropped from downstream English-language pipelines entirely.

## Causes

Three classes of cause, each with examples:

1. **Issues in the data**: NLP systems *amplify* biases already present in training data. If a corpus underrepresents African American English, the model has no evidence for its patterns and falls back on majority-variety patterns. The biases come from who writes what, what gets published, and what gets scraped.
2. **Problems in the labels**: annotators bring their own biases and disagreements; some phenomena are labelled inconsistently; rare categories get conflated.
3. **Problems in the algorithm**: the choice of what the model is trained to optimize (accuracy) can itself encode a bias. Optimizing average accuracy gives a classifier no incentive to perform well on minority groups or rare classes.

## Prevalence and Solutions

**Prevalence**: the same problems occur throughout NLP — not just in Naive Bayes classifiers, not just in sentiment. They appear in search, translation, named entity recognition, and large language models. Bigger models amplify data biases, not remove them.

**Solutions**: there is **no general mitigation**. Harm mitigation is an active research area. What exists:
- Standard benchmarks for measuring disparate performance (e.g. across demographic slices).
- Bias-measurement tools that surface gaps aggregate metrics hide.
- Per-application mitigations — data augmentation for underrepresented varieties, fairness constraints at training time, post-hoc calibration.
- **Model cards** — a lightweight documentation practice (see below) that doesn't *fix* bias but makes it visible, so downstream users can make informed decisions.

None of these solve the problem in general. Each requires thinking carefully about the specific task, who is affected, and what kind of harm is at stake.

## Model Cards (Mitchell et al., 2019)

**Model cards** are a documentation standard: for each algorithm released, publish a short, structured card describing how the model was trained, what it's intended for, and how it performs across different groups. The idea is borrowed from nutrition labels — you can't taste what's in the food, so the label makes the ingredients explicit.

Each model card should document:

- **Training algorithms and parameters** — what was actually trained, what hyperparameters, what optimizer, what budget.
- **Training data sources, motivation, and preprocessing** — where the data came from, why this data was chosen, what filtering/cleaning was applied before training. This is where [[corpora#Corpus datasheets|corpus datasheets]] feed in.
- **Evaluation data sources, motivation, and preprocessing** — what the model was benchmarked against, and the provenance of those benchmarks.
- **Intended use and users** — the scenarios the model was designed for, and the scenarios it was explicitly *not* designed for.
- **Model performance across different demographic or other groups and environmental situations** — disaggregated metrics, not just aggregate numbers. This is the critical fairness-audit piece: slice performance by gender, race, age, language variety, dialect, etc., and report per-slice precision/recall/F1.

Model cards don't fix any of the underlying biases — biased training data produces biased models regardless of how well you document them. What they do is make the biases **legible**: a downstream practitioner picking a sentiment classifier for an app can read the model card, see that it underperforms on African American English, and choose differently (or at least build in compensating checks). Documentation shifts harm from invisible-and-automatic to visible-and-decidable.

## Connection to Other Weeks

- [[corpora]] (week 1): corpus datasheets / data statements (Gebru et al. 2020; Bender & Friedman 2018) exist specifically to make the assumptions embedded in a corpus explicit — a prerequisite for reasoning about what biases a model might inherit.
- [[classification-evaluation]]: aggregate metrics like accuracy or macro-$F_1$ **hide per-group disparities**. Evaluation for fairness requires slicing the test set by demographic group and reporting metrics separately.

## Related

- [[text-classification]] — the tasks being discussed
- [[sentiment-analysis]] — the setting of the Kiritchenko/Mohammad study
- [[classification-evaluation]] — why aggregate metrics aren't enough
- [[corpora]] — where the biases come from and how to document them

## Active Recall

> [!question]- What is a representational harm and what is the canonical sentiment-analysis example?
> A **representational harm** is a harm caused by a system demeaning a social group — e.g. by perpetuating negative stereotypes. **Kiritchenko & Mohammad 2018**: sentiment classifiers assigned lower sentiment and more negative emotion to sentences with African American names (*Shaniqua*) than the same sentences with European American names (*Stephanie*). Downstream, this distorts how African Americans appear in marketing and mental health analyses that use these tools — amplifying existing stereotypes via automation.

> [!question]- How do toxicity detection classifiers cause censorship harm, and which groups are affected?
> Toxicity classifiers over-flag sentences that simply *mention* minority identity terms (e.g. *"blind"*, *"gay"*) as toxic, regardless of whether the sentence is actually abusive. Documented effects on women (Park et al., 2018), disabled people (Hutchinson et al., 2020), and gay people (Dixon et al., 2018; Oliva et al., 2021). Consequences: speech by these groups is censored disproportionately; their content becomes less visible; writers may avoid identity-related vocabulary, making them less likely to write about themselves at all.

> [!question]- Why does language identification perform worse on African American English and Indian English, and why is that especially consequential?
> Training data for language ID is biased toward majority varieties of each language. The model has less evidence for minority varieties, so classification performance drops. It is especially consequential because language ID is **a first step in most NLP pipelines** — if a post is mis-identified as not-English, the rest of the pipeline (sentiment analysis, content moderation, search indexing) never sees it. A performance gap at step 1 cascades into total invisibility downstream.

> [!question]- Why isn't there a general solution to harms in NLP classifiers, despite growing research?
> The causes vary by task: biased *data* (training corpus underrepresents a variety), biased *labels* (annotators bring assumptions), or biased *optimization target* (optimizing average accuracy has no incentive to help minority groups). Each cause needs task-specific mitigation — data augmentation, fairness constraints, post-hoc calibration. No single intervention fixes all three. Harm mitigation is active research; the practical guidance is to measure disparities directly (slicing the test set by group) rather than trust aggregate metrics.

> [!question]- What is a model card and what five things should it document? Does it actually fix bias?
> A **model card** (Mitchell et al., 2019) is a short structured document released with a model, analogous to a nutrition label. It documents: (1) **training algorithms and parameters**, (2) **training data sources, motivation, and preprocessing**, (3) **evaluation data sources, motivation, and preprocessing**, (4) **intended use and users**, (5) **model performance across different demographic or other groups and environmental situations** (disaggregated metrics, not just aggregate). It does **not** fix bias — biased training data still produces biased models. What it does is make the bias **visible**, so downstream users can decide whether the model is appropriate for their use case or whether compensating checks are needed.
