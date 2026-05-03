---
type: concept
sources:
  - raw/week-01/w01-slides.pdf
  - raw/week-01/w01-l02-transcript.txt
status: draft
updated: 2026-04-20
---

*Text normalization transforms raw text into a canonical form that downstream NLP components can reason over consistently; the three core steps are tokenization, word normalization, and sentence segmentation.*

## Definition

**Text normalization** is the process of converting raw text into a standard representation. It is not a single operation but a pipeline. The three canonical steps (Jurafsky & Martin) are:

1. **Tokenizing words** — segmenting a character stream into tokens (see [[tokenization]])
2. **Normalizing word formats** — mapping token variants to a canonical form
3. **Segmenting sentences** — identifying sentence boundaries

This page covers steps 2 and 3.

## Word Normalization

### Case Folding

Converting all characters to lowercase. Merges `Woodchuck` and `woodchuck` into the same type.

**When to do it**: information retrieval, where `the` and `The` should match the same documents.

**When not to**: sentiment analysis (`US` vs `us`), machine translation (case carries information), named entity recognition (`Apple` the company vs `apple` the fruit). Case folding is lossy and the loss sometimes matters.

### Lemmatization

Reducing each word form to its **dictionary headword** (the *lemma*). This requires a full morphological analysis.

| Surface form                     | Lemma  |
| -------------------------------- | ------ |
| `am`, `are`, `is`, `was`, `were` | `be`   |
| `runs`, `running`, `ran`         | `run`  |
| `better`                         | `good` |

Lemmatization correctly identifies that `sang` is the past tense of `sing` — it requires knowing the morphology. The result is always a real word that appears in a dictionary.

### Stemming

A cruder, faster alternative: strip common affixes using heuristic rules, without consulting a lexicon. The result may not be a real word.

**Porter Stemmer** (Porter, 1980) is the canonical English stemmer. It applies rules in cascading phases. Example rules from Phase 1:

| Rule            | Condition             | Example               |
| --------------- | --------------------- | --------------------- |
| `ATIONAL → ATE` | —                     | `relational → relate` |
| `ING → ε`       | stem contains a vowel | `motoring → motor`    |
| `SSES → SS`     | —                     | `grasses → grass`     |

Stemming is fast and sufficient for IR but too noisy for applications where word meaning matters (e.g., `universe` and `university` both stem to `univers`).

> [!warning] COMMON MISCONCEPTION
> Stemming and lemmatization are not interchangeable. Lemmatization is linguistically correct and produces real words; stemming is a heuristic that may conflate unrelated words (`wand` and `wander` both stem to `wand` under some stemmers). For IR, stemming's recall gain usually outweighs the precision loss. For classification or translation, lemmatization (or no normalization at all) is usually better.

## Morphological Parsing

Both lemmatization and stemming require understanding **morphology** — the internal structure of words.

- **Morpheme**: the smallest meaning-bearing unit. `cats` = `cat` (stem) + `s` (affix).
- **Stems** carry the core lexical meaning.
- **Affixes** are **prefixes** (before the stem: `un-`, `pre-`) or **suffixes** (after: `-ing`, `-ed`, `-ness`).

A full morphological parser can decompose any word into its morphemes and tag their grammatical function. This is needed for lemmatization and for languages with rich morphology (Finnish, Arabic, Turkish).

## Sentence Segmentation

Identifying sentence boundaries is harder than it looks because the most common boundary marker — the period `.` — is also used for abbreviations, numbers, and URLs (see [[tokenization]]).

**Standard approach**: tokenize first, then classify each period-containing token.

A classifier for each potential boundary token (`Dr.`, `45.`, `end.`) considers:
- Is the token in an abbreviation list?
- Is the next token capitalized?
- Is the next token a function word (suggesting a continuation)?

In practice, sentence segmentation is often handled by the same component as tokenization, since both require the same contextual reasoning about periods.

## Related

- [[tokenization]] — step 1 of the normalization pipeline; prerequisite for the steps described here
- [[regular-expressions]] — normalization rules are often implemented as substitutions
- [[edit-distance]] — edit distance is used downstream to correct normalized text (spell checking)

## Active Recall

> [!question]- When should you use case folding, and when would it hurt? Give a concrete example of each.
> **Use it**: information retrieval — "the" and "The" should match the same documents; folding removes irrelevant case variation. **Avoid it**: named entity recognition — `Apple` (the company) and `apple` (the fruit) are different entities; folding would conflate them. Similarly, in sentiment analysis, `US` (United States, likely neutral) and `us` (first-person plural, likely in subjective text) carry different signals.

> [!question]- How does stemming differ from lemmatization, and in what situation would each be preferred?
> Lemmatization maps a word to its dictionary headword using full morphological analysis — the result is always a real word with the correct meaning (`better → good`, `sang → sing`). Stemming strips affixes heuristically without a lexicon — the result may not be a real word (`universe → univers`) and may conflate unrelated words. Stemming is faster and adequate for IR where crude conflation raises recall at acceptable precision cost. Lemmatization is preferred for applications where word meaning matters: translation, classification, question answering.

> [!question]- Why must sentence segmentation typically be done after tokenization rather than before?
> Sentence segmentation classifies potential sentence boundaries, which are almost always periods attached to tokens. To reason about whether a period ends a sentence or an abbreviation, you need to know what token it belongs to (`Dr.` vs `leave.`). Without tokenization, you only have a raw character stream where this context is not structured. Tokenization also resolves many ambiguous cases by annotating tokens with their type (abbreviation, number, word), making the segmentation classifier's job easier.

> [!question]- What is a significant challenge in sentence segmentation within NLP? A) Ambiguity in punctuation usage B) Lack of capitalization C) Uniform sentence length D) Cross-language consistency
> **A) Ambiguity in punctuation usage.** The period `.` can mark a sentence boundary (`She left.`), an abbreviation (`Dr.`), or a decimal (`3.14`). This ambiguity forces segmentation algorithms to classify each period in context rather than applying a simple rule. **B** misrepresents the issue — capitalization is actually a *helpful* cue, not a problem. **C** is wrong — NLP algorithms do not assume uniform sentence length. **D** is about multilingual model adaptation, not sentence segmentation itself.
