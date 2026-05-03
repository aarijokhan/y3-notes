---
type: concept
sources:
  - raw/week-01/w01-slides.pdf
  - raw/week-01/w01-l02-transcript.txt
status: draft
updated: 2026-04-20
---

*Tokenization is the task of segmenting a stream of text into meaningful units (tokens); it is the first step in virtually every NLP pipeline and harder than it appears.*

## Definition

**Tokenization** takes a raw character sequence as input and produces a sequence of **tokens** — the atomic units the downstream system will reason over. What counts as a token depends on the application: in most Western-language NLP, tokens are roughly words; in other settings they may be subwords, characters, or morphemes.

## Space-Based Tokenization

The naive approach: split on whitespace.

```bash
tr -sc 'A-Za-z' '\n' < file.txt | sort | uniq -c | sort -rn
```

This works well for clean, formal English but fails immediately on nearly everything else.

## Hard Cases

### Abbreviations and punctuation

A period can end a sentence, mark an abbreviation, or appear in a number:

| Token | Period role |
|---|---|
| `Dr.` | Abbreviation marker |
| `$45.55` | Decimal separator |
| `google.com` | Part of URL |
| `She left.` | Sentence boundary |

A simple split-on-period rule will mangle all of these. Handling abbreviations requires either a lookup list (`Mr., Dr., Ph.D., m.p.h.`) or a trained classifier.

### Clitics

In English, `we're`, `I'd`, `can't` should often be split into component morphemes (`we + 're`, `I + 'd`, `can + n't`). In French this is even more complex: `j'ai` ("I have"), `l'homme` ("the man"), `du` (`de + le`).

### Multiword expressions

`New York`, `ad hoc`, `rock 'n' roll`, and `part-time` function as single semantic units but span multiple whitespace-delimited tokens. Deciding whether to treat them as one token or several depends on the application.

### Chinese and Japanese

These languages use no spaces between words. The character `字` (*zì*) is a morpheme-sized unit, but "words" are sequences of characters, and segmentation is genuinely ambiguous:

> 姚明进入总决赛  
> (Yao Ming enters the finals)

Can be parsed as 3 words, 5 words, or 7 characters depending on the segmenter. The correct segmentation is 3 words: `姚明 / 进入 / 总决赛`.

## NLTK Regex Tokenizer

A practical step up: define a regex pattern for the shapes of tokens you want, and collect them rather than splitting on delimiters.

```python
import nltk

pattern = r'''(?x)           # verbose mode
    (?:[A-Z]\.)+             # abbreviations: U.S.A.
  | \w+(?:-\w+)*             # words with optional hyphens
  | \$?\d+(?:\.\d+)?%?       # currency and percentages
  | \.\.\.                   # ellipsis
  | [][.,;"'?():-_`]         # punctuation marks
'''

tokens = nltk.regexp_tokenize(text, pattern)
```

The pattern is a cascade of alternatives, tried in order. This handles many edge cases that whitespace splitting cannot.

> [!warning] COMMON MISCONCEPTION
> Tokenization is not a solved problem that can be ignored. The errors introduced here propagate forward — a mis-tokenized word is an incorrect feature for every downstream component. In low-resource languages or noisy text (tweets, clinical notes, code), tokenization quality is often the binding constraint on overall system performance.

## Related

- [[type-and-token]] — tokenization defines what counts as a token, which determines $N$ and $|V|$
- [[subword-tokenization]] — modern alternative to word-level tokenization
- [[text-normalization]] — tokenization is step 1 of the normalization pipeline
- [[regular-expressions]] — regex cascades are the classical tokenization implementation

## Active Recall

> [!question]- Give three ways a period character can appear in English text that require different tokenization decisions. Why does this make simple split-on-punctuation rules insufficient?
> (1) Sentence boundary: `She left.` — period ends the token stream. (2) Abbreviation: `Dr.` — period is part of the token. (3) Decimal: `$45.55` — period is within a numeric token. A split-on-period rule would break all three incorrectly: it would isolate `Dr` and `.`, split `45` from `55`, and would still need to mark sentence boundaries correctly. Correct handling requires knowing *which role* the period plays in context, which requires either a lookup table or a classifier.

> [!question]- Why is tokenizing Chinese fundamentally different from tokenizing English, and what does this imply for building a Chinese NLP system?
> Chinese text has no whitespace between words; word boundaries must be inferred from meaning and context, not orthography. The character (字/zi) is the smallest orthographic unit, but words are sequences of characters, and the correct segmentation is often ambiguous without syntactic or semantic information. A Chinese NLP system therefore cannot use whitespace splitting at all — it needs a dedicated word segmentation model that has learned Chinese word boundary patterns, typically from annotated data.

> [!question]- Why does the NLTK regex tokenizer use a "collect matches" strategy rather than "split on delimiters"?
> Collecting matches lets you define *what a token looks like* (its shape) rather than *what separates tokens* (a delimiter). This is more expressive: you can have patterns for abbreviations, hyphenated words, currencies, and ellipsis all in one list, tried in priority order. A delimiter approach would need to enumerate every possible separator — and some token types (like `U.S.A.`) contain characters that would otherwise be treated as delimiters.
