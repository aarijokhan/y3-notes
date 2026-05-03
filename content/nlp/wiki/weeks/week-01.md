---
type: week
week: 1
title: "Week 1 — Text as Data: Building the Processing Pipeline"
dates: —
sources:
  - raw/week-01/w01-slides.pdf
  - raw/week-01/w01-slides-updated.pdf
  - raw/week-01/w01-l01-transcript.txt
  - raw/week-01/w01-l02-transcript.txt
  - raw/week-01/w01-l03-transcript.txt
  - raw/week-01/w01-lab-01.pdf
  - raw/week-01/w01-lab-01-solutions.pdf
  - raw/week-01/w01-lab-02-solutions.pdf
  - raw/week-01/w01-lab-02-solutions-pt2.pdf
concepts:
  - "[[regular-expressions]]"
  - "[[eliza]]"
  - "[[type-and-token]]"
  - "[[corpora]]"
  - "[[tokenization]]"
  - "[[subword-tokenization]]"
  - "[[byte-pair-encoding]]"
  - "[[text-normalization]]"
  - "[[edit-distance]]"
status: stable
updated: 2026-04-20
---

> [!question]+ THE CRUX: What does it take to turn raw text into units a computer can work with — and why does every step in that pipeline turn out harder than it first appears?

*Text starts as a stream of characters with no inherent structure. The first job of NLP is to impose structure — and the tools for doing that, from regex to BPE to dynamic programming, each reveal a different facet of how much is packed into what looks like "just words".*

---

## Starting Point: Text Is Just Characters

Before a language model can attend to tokens, before a parser can build a syntax tree, before any of the impressive downstream machinery kicks in — there's a raw character sequence. No words, no sentences, no meaning. Just bytes.

The first question is deceptively simple: *how do we get from that to something a computer can reason about?*

The first tool in the box is [[regular-expressions]]. A regex is a compact way to describe a *pattern* over strings — and patterns are all you need to pull off a surprising amount of text processing. Want every capitalized word? `[A-Z][a-z]+`. Want every phone number? `\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}`. Want to pull out prices from a product catalog? `\$\d+(?:\.\d{2})?`.

Regex sounds low-tech, but it's the Swiss army knife of NLP. It appears inside tokenizers, named entity recognizers, information extractors, and even (as we'll see) one of the most famous programs in the history of AI.

> [!question]- You want to find all occurrences of either "colour" or "color" in a corpus using a single regex. What pattern would you write, and what does the `?` quantifier do?
> `colou?r` — the `?` makes the preceding character optional (zero or one occurrence), so it matches both `color` (no `u`) and `colour` (with `u`).

## ELIZA and the Limits of Rules

In 1966, Joseph Weizenbaum demonstrated that a startlingly small ruleset could fool people into thinking they were talking to a therapist. His program, [[eliza]], worked entirely through regex substitution:

```
s/.* I'M (depressed|sad) .*/I AM SORRY TO HEAR YOU ARE \1/
```

Match the user's statement. Extract the feeling word. Reflect it back in therapist-speak. That's it. No understanding, no memory, no world model. Yet users formed genuine emotional attachments to ELIZA — a phenomenon Weizenbaum called the **ELIZA effect**, and which disturbed him deeply.

ELIZA is worth studying not because it's impressive (by modern standards, it isn't) but because it makes the stakes clear. Linguistic surface form is enough to trigger human social projection. The gap between "produces plausible text" and "understands language" is real and important.

> [!info] ASIDE — ELIZA's cleverness was in domain choice
> Rogerian psychotherapy is a style where the therapist mostly reflects the patient's words back and asks open questions — almost never asserting facts about the world. Weizenbaum picked it deliberately. The domain choice did much of the work that the rules themselves couldn't. A substitution-rule-based doctor or lawyer would have failed immediately.

Practically, ELIZA also illustrates the central tension in regex work: **precision vs recall**. A broad pattern catches more (higher recall) but makes more false-positive errors. A tight pattern is precise but misses things. Real regex engineering is iterating on that tradeoff.

## The Vocabulary Problem

Before we can do anything useful with text, we need to know what our vocabulary is. This is where [[type-and-token]] comes in.

A **token** is one occurrence of a word in text. A **type** is one distinct word form. The sentence *"to be or not to be"* has 6 tokens but only 5 types. This distinction permeates NLP: model parameters are indexed by type; data statistics are computed over tokens.

How many types exist? **Heaps' Law** tells us: $|V| \approx k N^\beta$ with $\beta \approx 0.67$. As you read more text, you keep encountering new words — but at a decelerating rate. The kicker: it *never stops*. No matter how large your training corpus, you will see unseen words at test time. This is the **open-vocabulary problem**, and it motivates a huge amount of engineering.

But before vocabulary size, we need to ask about the text itself. What corpus are we working with? [[Corpora]] vary enormously — by language, by variety (AAE, Nigerian English, Singaporean English), by genre, by time period, by who wrote them. A model trained on 1990s Wall Street Journal text will have blind spots that a model trained on Twitter would not, and vice versa. The emerging practice of **corpus datasheets** (Gebru et al. 2020) tries to make these assumptions explicit so downstream users know what they're inheriting.

## Tokenization: Harder Than It Looks

The most natural first step in processing text is [[tokenization]] — splitting the character stream into words. And for clean, formal English, naive whitespace splitting works fine. Until it doesn't.

The period `.` is the canonical villain. It ends sentences, marks abbreviations (`Dr.`, `m.p.h.`), appears in numbers (`$45.55`), and shows up in URLs (`google.com`). A split-on-period rule mangles all of them.

Clitics complicate things further: `we're` should usually become `we + 're`, `can't` becomes `can + n't`, and French `j'ai` is `j + ai`. Then there are multiword expressions: `New York` and `rock 'n' roll` are semantically single units that happen to contain spaces.

And then there's Chinese. No spaces, no word boundaries marked in the script, and genuine ambiguity in where words start and end. The string 姚明进入总决赛 (*Yao Ming enters the finals*) can be parsed as 3 words or 5 words or 7 characters, depending on your segmenter. The right answer is 3.

A practical upgrade from whitespace splitting is the NLTK regex tokenizer: instead of splitting on delimiters, you define what tokens *look like* and collect matches. This lets you handle abbreviations, hyphenated words, currency, and ellipsis all in one ordered cascade.

> [!question]- Why does the NLTK regex tokenizer use a "collect matches" strategy rather than "split on delimiters"? What does this buy you?
> Collecting matches lets you define positive patterns for each token type — abbreviations, numbers, hyphenated words — tried in priority order. A delimiter strategy can only define what *separates* tokens, which forces you to enumerate every possible separator. Some token shapes (like `U.S.A.`) contain potential delimiters, making them impossible to handle correctly with a pure split approach.

## Subwords: Killing the Unknown-Word Problem

Word-level vocabularies have a fatal flaw: any word not seen in training becomes `<UNK>`, throwing away all morphological information. Doubling the training corpus narrows the problem but never eliminates it (Heaps' Law again).

The modern solution is [[subword-tokenization]]: learn a vocabulary of *word fragments* from data, so that any novel word can be represented as a sequence of known pieces. Every major language model — GPT, BERT, T5 — uses a subword tokenizer.

[[Byte-pair-encoding]] (BPE, Sennrich et al. 2016) is the most widely used algorithm. It starts with a character vocabulary, then iteratively merges the most frequent adjacent pair into a new symbol, repeating for $k$ rounds. After training, the segmenter applies all learned merges greedily in training order. The result: common words get single tokens; rare words decompose into frequent subword pieces. A word the model has never seen can still be represented — worst case, as individual characters.

> [!info] ASIDE — What counts as a "subword"?
> After BPE runs, the vocabulary might contain entries like `un`, `##believ`, `##able` (WordPiece notation) or `un`, `believe`, `ability` (depending on the implementation). These aren't linguistic morphemes — they're statistically frequent fragments. They often *happen* to align with morphological structure because morphology correlates with frequency, but that's a side effect, not the design.

## From Words to Canonical Forms

Even once we have tokens, we often want to normalize them. [[Text-normalization]] covers three sub-problems:

**Case folding** — lowercasing everything — is appropriate for information retrieval (where `The` and `the` should match) but harmful for tasks where case carries information (named entities, sentiment, machine translation).

**Lemmatization** maps surface forms to their dictionary headwords: `ran → run`, `better → good`. It requires full morphological analysis and a lexicon.

**Stemming** is the faster, cruder alternative: strip affixes with heuristic rules, accepting that the result may not be a real word. The Porter Stemmer's rules (e.g., `ATIONAL → ATE`, `ING → ε` if the stem contains a vowel) produce stems good enough for most IR tasks.

Finally, **sentence segmentation** — which typically runs after tokenization — uses the same contextual reasoning about periods that tokenization needs. The difference is that tokenization asks "is this period part of the token?", while segmentation asks "does this period end a sentence?". In practice both are often handled together.

## Measuring String Similarity: Edit Distance

The week closes with a conceptually elegant tool: [[edit-distance]], the minimum number of insertions, deletions, and substitutions needed to transform one string into another. It shows up everywhere: spell checking, sequence alignment in biology, deduplication, OCR post-processing.

The key insight is that this minimum can be computed exactly — not by trying all possible edit sequences (exponentially many), but by dynamic programming. The recurrence is:

$$D(i,j) = \min\{D(i-1,j)+1,\ D(i,j-1)+1,\ D(i-1,j-1)+c_{ij}\}$$

where $c_{ij}=0$ if characters match and $2$ (Levenshtein) if they differ. Fill a table of size $(n+1) \times (m+1)$, read off the answer at $D(n,m)$, and optionally backtrace to recover the actual edit sequence. Time: $O(nm)$. Space: $O(nm)$, reducible to $O(\min(n,m))$ if you only need the distance.

The standard costs assume all errors are equally likely. **Weighted edit distance** relaxes this: assign lower costs to character pairs that are commonly confused (adjacent keyboard keys, visually similar characters, frequently misspelled pairs learned from corpus data). The algorithm is unchanged; only the cost functions differ.

> [!tip] TIP — Edit distance on the exam
> The most common exam question on this topic asks you to fill in the DP table or recover the alignment via backtrace. Practise filling the table by hand on short strings (5–7 characters). The base cases ($D(i,0)=i$ and $D(0,j)=j$) are where most errors happen. **Check the cost convention before computing**: some problems (including lab exercises) use uniform costs (ins = del = sub = 1), while the textbook uses Levenshtein costs (sub = 2). The same pair of strings gives a different distance under each. See [[edit-distance]] for worked examples of both.

---

## Concepts Introduced This Week

- [[regular-expressions]] — pattern language for text search and substitution
- [[eliza]] — first chatbot; demonstrates the power and ceiling of regex rules
- [[type-and-token]] — vocabulary items vs. running occurrences; Heaps' Law
- [[corpora]] — text collections and why their composition matters
- [[tokenization]] — segmenting text into tokens; harder than whitespace splitting
- [[subword-tokenization]] — data-driven word-fragment vocabularies; solves open-vocabulary problem
- [[byte-pair-encoding]] — iterative pair-merging algorithm; basis of GPT/BERT tokenizers
- [[text-normalization]] — case folding, lemmatization, stemming, sentence segmentation
- [[edit-distance]] — minimum-cost string transformation via dynamic programming

## Connections

Sets up [[week-02]] (language models will need a tokenized, normalized corpus and a vocabulary to work over).

## Open Questions

- How do different subword vocabulary sizes affect downstream task performance empirically?
- What are the failure modes of BPE for morphologically rich languages vs. character-based models?
- Are there cases where edit distance alignment gives the wrong intuitive answer? (E.g., aligning names across transliteration systems.)
