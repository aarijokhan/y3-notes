---
type: concept
sources:
  - raw/week-01/w01-slides.pdf
  - raw/week-01/w01-l02-transcript.txt
status: draft
updated: 2026-04-20
---

*Subword tokenization learns a vocabulary of word fragments from data, allowing any string to be represented without an unknown-word token.*

## Definition

**Subword tokenization** is a family of data-driven tokenization methods that operate between the character level and the word level. Instead of treating each word form as an atomic token (which fails on rare and unseen words) or each character as a token (which produces very long sequences), subword methods learn a vocabulary of *subword units* — pieces of words — from training data.

Every modern large language model uses subword tokenization.

## Architecture: Two Components

All subword tokenizers share the same two-stage architecture:

1. **Token learner**: runs offline on a training corpus and produces a vocabulary of subword types.
2. **Token segmenter**: runs at inference time and applies the learned vocabulary to segment new text.

The learner is run once; the segmenter runs on every input.

## Three Algorithms

| Algorithm | Token learner | Used by |
|---|---|---|
| **BPE** (Sennrich et al., 2016) | Iteratively merge most-frequent adjacent pair | GPT-2/3/4, RoBERTa |
| **WordPiece** (Schuster & Nakajima, 2012) | Merge pair that maximises language model likelihood | BERT, DistilBERT |
| **Unigram LM** (Kudo, 2018) | Start from large vocab, iteratively prune subwords that least reduce LM likelihood | SentencePiece (T5, mBERT) |

All three produce similar vocabularies in practice. The differences are in the objective function used to select merges/prunings.

### BPE vs WordPiece

BPE merges the pair with the highest **raw frequency**. WordPiece merges the pair that most improves a **language model score** — loosely, it prefers merges that reduce perplexity. WordPiece is slightly more principled but slower to train.

### Unigram LM

Takes the opposite direction from BPE: start with a large over-complete vocabulary (all substrings up to some length), then iteratively remove the subwords whose removal has the least effect on LM likelihood. This produces a probabilistic segmenter — at inference time, you decode the most likely segmentation rather than applying greedy merges.

## Why Subword, Not Whole-Word?

Word-level vocabularies fail in two ways:
1. **Out-of-vocabulary (OOV)**: unseen words at test time collapse to `<UNK>`, losing all morphological information.
2. **Vocabulary explosion**: for morphologically rich languages (Finnish, Turkish, Arabic), every inflected form is a separate type, producing enormous vocabularies.

Subword tokenization solves both: any string decomposes into known subword pieces (worst case: individual characters), and related morphological forms share subword units.

## Related

- [[byte-pair-encoding]] — the most widely used subword token learner, in detail
- [[tokenization]] — the classical alternative that subword methods replace
- [[type-and-token]] — subword tokenization is motivated by the open-vocabulary problem quantified by Heaps' Law

## Active Recall

> [!question]- What are the two components of every subword tokenizer, and why are they separated?
> The **token learner** runs offline on a training corpus to build the vocabulary; the **token segmenter** applies that vocabulary at inference time to segment new text. They are separated because learning is expensive (it processes an entire training corpus) while segmentation must be fast (it runs on every input). Running the learner once and reusing the segmenter amortizes the training cost.

> [!question]- BPE and WordPiece both iteratively merge pairs. What is the key difference in how they choose which pair to merge?
> BPE merges the pair with the highest raw co-occurrence frequency in the corpus. WordPiece merges the pair that maximises the language model likelihood of the corpus — it asks "does merging this pair make the corpus more probable under a unigram LM?" WordPiece is therefore more principled (it optimises a meaningful objective) but slower to compute.

> [!question]- Why does subword tokenization improve over word-level tokenization for morphologically rich languages like Finnish?
> Morphologically rich languages have very high type-to-token ratios: each root word can appear in dozens of inflected forms, each counted as a separate type under word-level tokenization. This leads to enormous vocabularies and severe OOV problems. Subword tokenization shares units across inflections — `running`, `runner`, `ran` all contain the subword `run` — dramatically reducing vocabulary size while preserving morphological information.
