---
type: concept
sources:
  - raw/week-01/w01-slides.pdf
  - raw/week-01/w01-l02-transcript.txt
  - raw/week-01/w01-lab-02-solutions.pdf
status: draft
updated: 2026-04-20
---

*BPE learns a vocabulary of subword units by iteratively merging the most frequent adjacent pair in a corpus; it is the token-learning algorithm behind GPT, RoBERTa, and many other large models.*

## Definition

**Byte Pair Encoding (BPE)** (Sennrich et al., 2016, adapted from a data-compression algorithm) is a **token learner** that builds a vocabulary of subword symbols by starting from individual characters and repeatedly merging the most frequent adjacent pair. The result is a vocabulary that represents frequent words as single units and rare or unseen words as sequences of subword pieces.

## Algorithm

### Token learner

**Input**: a training corpus, a target vocabulary size $k$ (a hyperparameter; typically 8K–32K for word-level, larger for modern LLMs).

**Initialization**: split every word into characters, appending a special end-of-word symbol `_` to each word. Start with a vocabulary of all characters plus `_`.

**Repeat $k$ times**:
1. Count all adjacent symbol pairs across the corpus.
2. Find the most frequent pair (e.g., `e` + `r`).
3. Merge that pair into a new symbol `er`.
4. Replace all occurrences of the pair with the new symbol.
5. Add the new symbol to the vocabulary.

Each iteration reduces the corpus length by one (two symbols become one) and adds one entry to the vocabulary.

### Worked example — token learner

Corpus (4 word types, split into characters; `_` = end-of-word):

```
l o w _          ×5    (low)
l o w e r _      ×2    (lower)
n e w e s t _    ×6    (newest)
w i d e s t _    ×3    (widest)
```

Initial vocabulary: `{l, o, w, e, r, n, s, t, i, d, _}`

**Initial pair counts (top entries):**

| Pair | Count | Source |
|---|---|---|
| `e`+`s` | **9** | newest(×6), widest(×3) |
| `s`+`t` | **9** | newest(×6), widest(×3) |
| `t`+`_` | **9** | newest(×6), widest(×3) |
| `w`+`e` | 8 | lower(×2), newest(×6) |
| `l`+`o` | 7 | low(×5), lower(×2) |
| `o`+`w` | 7 | low(×5), lower(×2) |

**Merge 1: `e`+`s` → `es`** — replace every adjacent `e s` in the corpus:

```
l o w _          ×5    (unchanged — no e s)
l o w e r _      ×2    (unchanged — no e s)
n e w es t _     ×6    ← merged
w i d es t _     ×3    ← merged
```

Recount. New top pairs: `es`+`t` = 9, `t`+`_` = 9.

**Merge 2: `es`+`t` → `est`**

```
n e w est _      ×6
w i d est _      ×3
```

Top pair: `est`+`_` = 9.

**Merge 3: `est`+`_` → `est_`**

```
l o w _          ×5
l o w e r _      ×2
n e w est_       ×6
w i d est_       ×3
```

Top pairs: `l`+`o` = 7, `o`+`w` = 7.

**Merge 4: `l`+`o` → `lo`** → **Merge 5: `lo`+`w` → `low`**

```
low _            ×5
low e r _        ×2
n e w est_       ×6
w i d est_       ×3
```

**Merges 6–8:** `n`+`e` → `ne`, `ne`+`w` → `new`, `new`+`est_` → `newest_`

| Merge | Pair | Count | New symbol |
|---|---|---|---|
| 1 | `e`+`s` | 9 | `es` |
| 2 | `es`+`t` | 9 | `est` |
| 3 | `est`+`_` | 9 | `est_` |
| 4 | `l`+`o` | 7 | `lo` |
| 5 | `lo`+`w` | 7 | `low` |
| 6 | `n`+`e` | 6 | `ne` |
| 7 | `ne`+`w` | 6 | `new` |
| 8 | `new`+`est_` | 6 | `newest_` |

After enough merges, frequent words like `low` and `newest` become single tokens; rare words are represented by sequences of frequent subword pieces.

### Worked example — token segmenter

New word: **`lowest`** (never seen in training). Split into characters: `l o w e s t _`

Apply every learned merge **in the order they were learned** — skip any merge that has no matching pair:

| Step | Merge checked | Match? | Sequence after |
|---|---|---|---|
| Start | — | — | `l o w e s t _` |
| Merge 1 | `e`+`s` → `es` | ✓ | `l o w es t _` |
| Merge 2 | `es`+`t` → `est` | ✓ | `l o w est _` |
| Merge 3 | `est`+`_` → `est_` | ✓ | `l o w est_` |
| Merge 4 | `l`+`o` → `lo` | ✓ | `lo w est_` |
| Merge 5 | `lo`+`w` → `low` | ✓ | `low est_` |
| Merge 6 | `n`+`e` → `ne` | ✗ | `low est_` |
| Merge 7 | `ne`+`w` → `new` | ✗ | `low est_` |
| Merge 8 | `new`+`est_` → `newest_` | ✗ | `low est_` |

**Result: `["low", "est_"]`** — a word never seen in training decomposes into two meaningful pieces.

### End-of-word marker: `_` vs `</w>`

The lecture notes use `_` as the end-of-word marker. In code (including the original Sennrich et al. implementation and most Python tutorials) you will see `</w>` instead:

```python
vocab = {
    'l o w </w>'     : 5,
    'l o w e r </w>' : 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3,
}
```

The choice of symbol does not affect the algorithm — `</w>` is just less likely to collide with actual characters in the corpus. The conceptual role is identical: mark the word boundary so that merged tokens do not span across word boundaries.

### Token segmenter

At inference time, given a new word, apply the learned merges **greedily in training order**:

1. Split the word into characters (+ `_`).
2. Apply the first learned merge wherever it applies.
3. Apply the second learned merge wherever it applies.
4. Continue through all merges in order.

This is deterministic and runs in $O(n \cdot k)$ where $n$ is word length and $k$ is the number of merges.

**Crucial property**: a word never seen in training will be decomposed into its constituent subword pieces. In the worst case, it decomposes to individual characters. The open-vocabulary problem is eliminated by construction.

> [!tip] TIP — Choosing vocabulary size $k$
> $k$ trades off two things: large $k$ means more words are single tokens (faster inference, more parameters); small $k$ means more subword splitting (slower inference, handles rare words better). Common values: 32K–50K for English-only models, larger for multilingual models where the character inventory is much bigger.

## Relationship to Other Subword Algorithms

BPE is the most widely used but not the only option. See [[subword-tokenization]] for a comparison of BPE, WordPiece, and Unigram LM.

## Related

- [[subword-tokenization]] — the broader family of data-driven subword approaches
- [[type-and-token]] — BPE directly addresses the open-vocabulary problem diagnosed by Heaps' Law
- [[tokenization]] — BPE replaces classical word tokenization in modern systems

## Active Recall

> [!question]- Walk through the first two steps of BPE on a small corpus. What is being counted and what determines which pair gets merged?
> BPE counts all adjacent symbol pairs in the current representation of the corpus (where initially each word is split into characters). The pair with the highest total frequency across all positions in the corpus is selected for merging. The merged symbol replaces every instance of that pair, and the new symbol is added to the vocabulary. Then counting restarts on the updated corpus.

> [!question]- How does the BPE token segmenter handle a word it has never seen during training?
> The segmenter applies all learned merges in their training order, greedily. Starting from individual characters, it combines wherever a learned merge applies. Since the initial vocabulary always includes individual characters, any word — no matter how rare or novel — can be represented as a sequence of characters. In practice, frequent subword patterns get merged early, so most words are represented by a short sequence of meaningful subword units rather than individual characters.

> [!question]- Why is vocabulary size $k$ described as a trade-off rather than something you should always maximize?
> Larger $k$: more words become single tokens (frequent words are represented compactly, reducing sequence length and speeding up attention computation), but the embedding matrix grows, increasing parameter count and memory use. Smaller $k$: rare and unseen words are split into more pieces (richer compositional representation, better generalization to unknown words), but frequent words become longer sequences. The right $k$ depends on the language (multilingual models need larger $k$ to cover diverse scripts), domain, and hardware constraints.

> [!question]- What property of BPE makes it strictly better than a fixed word vocabulary for handling rare words?
> BPE guarantees that any string can be represented — it degrades gracefully from whole-word tokens to subword pieces to individual characters. A fixed word vocabulary must assign unknown words to a single `<UNK>` token, losing all information about their internal structure. BPE preserves structure: `unbelievability` might not be in the vocabulary, but `un`, `believe`, `ability` likely are, and their composition carries meaningful information.

> [!question]- What is the key property of BPE tokens that makes the algorithm effective for NLP tasks? A) Uniform Token Size B) Fixed Vocabulary Size C) Subword Tokenization D) Lossless Compression
> **C) Subword Tokenization.** BPE splits words into smaller subword units, which lets the model handle rare words, morphological variations, and out-of-vocabulary words by recognising patterns at the subword level. **A** is wrong — BPE tokens vary in length depending on corpus frequency. **B** is misleading — BPE does produce a controlled vocabulary size, but that's not the *key* property; it's the subword decomposition. **D** is misleading — while BPE reduces redundancy, its purpose in NLP is not lossless compression but handling vocabulary coverage.

> [!question]- Given the corpus `"low low low low low lowest newest newest newest newest newest newest wider wider wider new new"`, which pair would BPE merge first? A) `er` B) `ew` C) `lo` D) `ne`
> **A) `er`.** Count adjacent pairs: `er` appears in `newer` (×6) and `wider` (×3) = 9 occurrences. `ew` appears in `newest` (×6) = 6. `lo` appears in `low` (×5) and `lowest` (×1) = 6. `ne` appears in `newest` (×6) and `new` (×2) = 8. So `er` has the highest frequency at 9 and is merged first.
