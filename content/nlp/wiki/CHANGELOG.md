---
type: changelog
title: Changelog
---

# Changelog

Append-only log of wiki ingests.

## 2026-04-20
- **Initialized vault structure.** Created raw/ and wiki/ folders, index.md, and CHANGELOG.md.
- **Ingested week 01.** Sources: w01-slides.pdf (109 pp), w01-l01-transcript.txt, w01-l02-transcript.txt, w01-l03-transcript.txt, w01-lab-01.pdf, w01-lab-01-solutions.pdf, w01-lab-02-solutions.pdf, w01-lab-02-solutions-pt2.pdf.
  - Created: `wiki/weeks/week-01.md`
  - Created: `wiki/concepts/regular-expressions.md`
  - Created: `wiki/concepts/eliza.md`
  - Created: `wiki/concepts/type-and-token.md`
  - Created: `wiki/concepts/corpora.md`
  - Created: `wiki/concepts/tokenization.md`
  - Created: `wiki/concepts/subword-tokenization.md`
  - Created: `wiki/concepts/byte-pair-encoding.md`
  - Created: `wiki/concepts/text-normalization.md`
  - Created: `wiki/concepts/edit-distance.md`
  - Updated: `wiki/index.md` (week-01 gloss and concept links)
- **Ingested week 02.** Sources: w02-slides.pdf (75 pp), 3 lecture transcripts (lectures 4–6), NLP_Lab_2-1-1.pdf, NLP_Lab_2_2.pdf, NLP_Worksheet_2_Solutions-1-1.pdf, NLP_Worksheet_2_1_Solutions-1.pdf.
  - Created: `wiki/weeks/week-02.md`
  - Created: `wiki/concepts/n-gram-language-models.md`
  - Created: `wiki/concepts/perplexity.md`
  - Created: `wiki/concepts/evaluation-methodology.md`
  - Created: `wiki/concepts/smoothing.md`
  - Created: `wiki/concepts/interpolation-and-backoff.md`
  - Updated: `wiki/index.md` (week-02 gloss and concept links)
  - Renamed 7 raw files to follow naming convention (w02-l04-transcript.txt, etc.)

## 2026-04-25
- **Consolidated week 4 from updated slide deck.** Cross-referenced `raw/week-04/w04-slides-updated.pdf` (118 pp) against existing notes. Changes:
  - `tf-idf.md`: **corrected tf formula** to match course convention $\text{tf} = \log_{10}(\text{count}+1)$ (was $1 + \log_{10}\text{count}$ from the original deck; the updated deck changed this). Updated the Shakespeare tf-idf worked table with recomputed values. Rewrote the walkthrough recall question to use the new formula.
  - `pmi.md`: expanded PPMI section significantly — added full matrix-based PPMI worked computation (cherry/strawberry/digital/information with count(w), count(c), joint and marginal probability tables, full PPMI output matrix), plus a new **Weighting PMI** section covering the $\alpha=0.75$ context-probability smoothing trick (same $\alpha$ that word2vec's negative sampling uses) and add-one smoothing. Three reasons now given for why negative PMI is problematic (reliability on small corpora, human calibration).
  - Added 5 new MCQ-derived active-recall Q&As across `lexical-semantics.md`, `vector-semantics.md`, `cosine-similarity.md`, `tf-idf.md`, `pmi.md`, `word2vec.md` — each with the full MCQ statement and reasoning for which options are correct.
  - `week-04.md`: updated tf-idf formula in the narrative, expanded the PMI paragraph to mention PPMI's reliability rationale and the $\alpha=0.75$ smoothing, added 2 new MCQ recall callouts (TF-IDF = 0.70, PPMI = 4.0).

- **Ingested week 04.** Sources: w04-slides.pdf (98 pp), 2 lecture transcripts (lectures 10 and 12; lecture 11 transcript not available — covered tf-idf), NLP_Worksheet_4.pdf, NLP_Worksheet_4_Solutions-1.pdf.
  - Created: `wiki/weeks/week-04.md`
  - Created: `wiki/concepts/lexical-semantics.md`
  - Created: `wiki/concepts/vector-semantics.md`
  - Created: `wiki/concepts/tf-idf.md`
  - Created: `wiki/concepts/pmi.md`
  - Created: `wiki/concepts/cosine-similarity.md`
  - Created: `wiki/concepts/word2vec.md`
  - Updated: `wiki/index.md` (week-04 gloss and concept links)
  - Renamed 4 raw files to follow naming convention (w04-l10-transcript.txt, w04-l12-transcript.txt, w04-worksheet.pdf, w04-worksheet-solutions.pdf).

## 2026-04-24
- **Ingested week 03.** Sources: w03-slides.pdf (81 pp), 3 lecture transcripts (lectures 7–9), NLP_Lab_3.pdf, NLP_Lab_3_1.pdf, NLP_Worksheet_3_Solutions-1.pdf, NLP_Worksheet_3_1_Solutions-1.pdf.
  - Created: `wiki/weeks/week-03.md`
  - Created: `wiki/concepts/text-classification.md`
  - Created: `wiki/concepts/bag-of-words.md`
  - Created: `wiki/concepts/bayes-rule.md`
  - Created: `wiki/concepts/naive-bayes.md`
  - Created: `wiki/concepts/sentiment-analysis.md`
  - Created: `wiki/concepts/classification-evaluation.md`
  - Created: `wiki/concepts/harms-in-classification.md`
  - Updated: `wiki/index.md` (week-03 gloss and concept links)
  - Renamed 7 raw files to follow naming convention (w03-l07-transcript.txt, w03-lab-01.pdf, w03-lab-02-solutions.pdf, etc.)
- **Extracted cross-cutting concept `maximum-likelihood-estimation`.** MLE was referenced by `n-gram-language-models`, `naive-bayes`, and `smoothing` but had no standalone page. Created one covering the principle, the categorical closed-form derivation sketch, the zero-probability problem, and the MLE/MAP/Bayesian spectrum. Retrofitted links from the three referring pages. Added to the week-02 concept line in `wiki/index.md` (MLE is first taught there).
- **Consolidated week 3 from updated slide deck.** Cross-referenced `raw/week-03/3 NB and Sentiment Classification - Consolidation-1.pdf` (117 pp) against existing notes. Added missing content:
  - `sentiment-analysis.md`: Scherer's Typology of Affective States (emotion / mood / interpersonal stance / attitudes / personality traits) — frames sentiment as the detection of *attitudes* specifically.
  - `classification-evaluation.md`: expanded bootstrap section into full **Statistical Significance Testing** treatment — effect size $\delta(x)$, null/alternative hypotheses, p-value definition, parametric vs non-parametric tests, full paired bootstrap algorithm (after Berg-Kirkpatrick et al., 2012), and the $2\delta(x)$ shift correction. Added new **Devsets and $k$-fold Cross-Validation** section.
  - `harms-in-classification.md`: added **Model Cards** (Mitchell et al., 2019) — five-field documentation standard that makes biases visible without fixing them.
  - `week-03.md`: tightened statistical significance and model-cards threads to match the expanded concept pages.
  - Added 5 new active-recall Q&As across the three updated concept pages (devset vs cross-validation, null/alternative hypotheses + p-value, $2\delta(x)$ shift, paired vs unpaired tests, model card structure).
