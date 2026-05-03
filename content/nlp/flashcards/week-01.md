---
title: "Week 1 Flashcards — Text as Data"
week: 1
topic: "Text as Data — Regex, Tokenization, Text Normalization"
deck: NLP::Week-01
---

TARGET DECK
NLP::Week-01

# Week 1 Flashcards

Write your own flashcards here using the `[!question]-` callout format. Each callout becomes one Anki card (front = question, back = answer).

> [!tip] How to write a good flashcard
> - One concept per card — don't bundle multiple ideas
> - Prefer "write a regex that…" or "what happens when…" over "define X"
> - Include a concrete example in the answer when possible
> - Keep answers short enough to review in under 30 seconds

---

## Regex

> [!question]- What is a regex?
> A **regular expression** (regex) is a sequence of characters that defines a search pattern over strings. A regex engine takes the pattern and an input string, and returns all substrings that match. In NLP, regex is used for tokenization, text normalization, feature extraction, and rule-based systems like ELIZA.
<!--ID: 1777386639862-->




> [!question]- What are the three meanings of `^` in regex, and what determines which one applies?
> 1. **First character inside `[]`** → negation: `[^abc]` matches anything except a, b, c
> 2. **Outside `[]`** → start-of-line anchor: `^The` matches `The` only at line start
> 3. **Non-first character inside `[]`** → literal caret: `[e^]` matches `e` or `^`
>
> Position alone determines the meaning — the same symbol, three unrelated jobs.
<!--ID: 1777386639864-->





> [!question]- What does `\b` match, and how is it different from constructs like `\d` or `.`?
> `\b` is a **zero-width assertion** — it matches a *position* between a word character (`\w`) and a non-word character (`\W`), not an actual character. Unlike `\d` or `.`, it consumes nothing. `\bcat\b` matches `cat` as a standalone word but not inside `concatenate`. Without `\b`, `cat` matches anywhere — including inside other words.
<!--ID: 1777386639865-->





> [!question]- What goes wrong with `re.findall('\bcat\b', text)` in Python, and why is there no error?
> Python interprets `'\b'` as the **backspace character** (ASCII 8) before the regex engine ever sees it. The regex engine receives a backspace instead of a word boundary — so it silently matches the wrong thing with no error. Fix: use a raw string `r'\bcat\b'`, which passes the literal `\b` to the regex engine.
<!--ID: 1777386639866-->





> [!question]- What does `<.+>` match on the string `<a>hello</a>`, and why?
> It matches `<a>hello</a>` — the **entire string**, not just `<a>`. By default, `+` is **greedy**: it consumes as many characters as possible. The `.+` grabs everything, then backtracks just enough to find the *last* `>`. Use `<.+?>` (non-greedy) to get `<a>` and `</a>` as separate matches.
<!--ID: 1777386639867-->





> [!question]- `\1` appears in both patterns and replacements. What does it mean in each, and how are they different?
> - **In a pattern**: `\1` is a **constraint** — "this position must match exactly what group 1 already captured." E.g., `([a-z]+)\s+\1` matches `the the` but not `the cat`.
> - **In a replacement**: `\1` is an **insertion** — "paste whatever group 1 captured here." E.g., `re.sub(r'(\w+)', r'<\1>', text)` wraps each word in angle brackets.
>
> Same syntax, different semantics.
<!--ID: 1777386639868-->





> [!question]- What does `cat+` match, and why doesn't it mean "one or more `cat`"?
> `cat+` means "`ca` followed by one or more `t`" — it matches `cat`, `catt`, `cattt`, etc. Quantifiers attach to the **single token immediately before them**, not the whole preceding word. To match one or more repetitions of the full word, you need a group: `(cat)+` → `cat`, `catcat`, `catcatcat`.
<!--ID: 1777386639869-->





> [!question]- How does `a|bc*` parse, and why is it not `(a|b)` followed by `c*`?
> It parses as `a | (b(c*))` — either `a`, or `b` followed by zero or more `c`. The `|` operator has the **lowest precedence** in regex (below quantifiers and concatenation), so it splits the entire expression. To get "either `a` or `b`, followed by `c*`", you must write `(a|b)c*`.
<!--ID: 1777386639870-->





> [!question]- What are the two directions a regex pattern can make mistakes in, and how do you address each?
> - **False positives** (Type I): the pattern matches strings it shouldn't → lowers **precision**. Fix: **tighten** the pattern by adding constraints (e.g., add `\b` to stop matching substrings).
> - **False negatives** (Type II): the pattern misses strings it should catch → lowers **recall**. Fix: **broaden** the pattern by adding alternatives (e.g., `colou?r` to handle both spellings).
>
> There is no free lunch — tightening raises precision but risks lowering recall, and vice versa. Practical regex engineering is iterating between the two deliberately.
<!--ID: 1777386639871-->





> [!question]- What is the role of regex in NLP?
> - Regex is often the **first model** tried for any text processing task — fast to write, interpretable, no training data needed.
> - For harder tasks, **ML classifiers take over**, but regex is still used for **pre-processing** text before it reaches a classifier, and as **features** fed into classifiers (e.g., "does this token match `\d+`?").
> - Regex is well-suited for **capturing generalizations** — a single pattern like `\$\d+(?:\.\d{2})?` covers every price format without labelled examples.
<!--ID: 1777386639872-->





> [!question]- What is a lookahead assertion, and give an example of each type?
> A lookahead matches a **position** without consuming any characters — it checks what comes next but doesn't include it in the match.
> - **Positive lookahead** `(?=...)`: matches if the pattern *is* ahead. `\d+(?= dollars)` matches `100` in "100 dollars" but not in "100 cats".
> - **Negative lookahead** `(?!...)`: matches if the pattern is *not* ahead. `Windows(?! NT)` matches "Windows" in "Windows 10" but not in "Windows NT".
<!--ID: 1777386906111-->

## ELIZA

> [!question]- What is ELIZA?
> ELIZA (Weizenbaum, 1966) is the **first chatbot** — a program that simulates a Rogerian psychotherapist. It operates entirely through regex pattern matching: each user utterance is scanned against an ordered list of substitution rules, and the first match fires a template response that echoes and reframes the input. It has no world model, no memory, and no understanding — only a list of rules.
<!--ID: 1777386906203-->


> [!question]- How does ELIZA work? Trace through an example input.
> ELIZA uses **regex substitution rules** in the form `s/pattern/replacement/`. Rules are applied sequentially — each substitution feeds the next.
>
> For input `"He says I'm depressed much of the time"`:
> 1. Pronoun inversion rules fire first: `I'm` → `YOU ARE`
> 2. The string becomes `"He says YOU ARE depressed much of the time"`
> 3. Rule `s/.*YOU ARE (depressed|sad) .*/I AM SORRY TO HEAR YOU ARE \1/` matches
> 4. `.*` absorbs the surrounding text, `(depressed|sad)` captures `"depressed"` as `\1`
> 5. Output: `"I AM SORRY TO HEAR YOU ARE depressed"`
>
> The `.*` at the start and end is essential — without it, the rule would only fire on the exact string `YOU ARE depressed` with nothing else around it.
<!--ID: 1777386906262-->


> [!question]- What is the significance of ELIZA, and what are its limitations?
> **Significance**: ELIZA produced the **ELIZA effect** — users attributed understanding and empathy to the program despite knowing it was rule-based. This raised early questions about the relationship between linguistic behaviour and actual cognition.
>
> **Limitations**:
> - **No memory** — each utterance is processed in isolation; it cannot refer back to earlier exchanges
> - **No semantic understanding** — it only matches surface patterns, not meaning
> - **Domain-dependent** — Weizenbaum chose Rogerian therapy deliberately because the therapist mostly reflects words back, hiding the system's shallowness. Any other domain would break it
<!--ID: 1777386906300-->


> [!question]- Why did Weizenbaum choose Rogerian psychotherapy as ELIZA's domain?
> - Rogerian therapists mostly **reflect the patient's words back** and ask open-ended questions
> - They rarely assert facts about the world
> - This maps directly onto **regex substitution rules** that echo and reframe input
> - Most other domains (doctor, lawyer, tutor) require **factually correct, contextually specific statements** — which pattern matching cannot generate
> - ELIZA doesn't overcome the limits of regex — it picks a domain where those limits don't show
<!--ID: 1777386906328-->

## Types and Tokens

> [!question]- What is the difference between a type and a token? Give an example.
> - **Token**: an instance of a word in running text
> - **Type**: an element of the vocabulary — each unique word form
> - Example 1 (slides): *"they lay back on the San Francisco grass and looked at the stars and their"* → **15 tokens**, **13 types** — `the` and `and` each appear twice but count as one type each
> - Example 2: *"to be or not to be"* → **6 tokens**, **4 types** (`to`, `be`, `or`, `not`) — `to` and `be` are repeated but count once each
<!--ID: 1777386906342-->


> [!question]- What is Heaps' Law and what does each variable represent?
> $$|V| = k N^{\beta}$$
> - $N$ = number of tokens (corpus size)
> - $|V|$ = vocabulary size (number of types)
> - $k \approx 10$–$100$ (constant)
> - $\beta \approx 0.67$–$0.75$ (sub-linear exponent)
>
> It describes how vocabulary size grows as a corpus gets larger.
<!--ID: 1777386639873-->





> [!question]- What does $\beta < 1$ in Heaps' Law tell you intuitively?
> - Vocabulary grows **sub-linearly** — much slower than corpus size
> - Doubling the corpus does **not** double the vocabulary
> - New types keep appearing but at a **decreasing rate**
> - Even an enormous corpus will not have seen every word that appears at test time
<!--ID: 1777386639874-->





> [!question]- What is the open-vocabulary problem, and why is it unavoidable?
> - The **open-vocabulary problem**: no matter how large your training corpus, you will encounter unseen words at test time
> - Heaps' Law proves it's unavoidable — $\beta < 1$ means new types keep appearing indefinitely, just at a slower rate
> - Rare words, names, neologisms, typos, and domain-specific terms will always slip through
<!--ID: 1777386639875-->





> [!question]- Why does Heaps' Law motivate subword tokenization rather than just enlarging the vocabulary?
> - A bigger vocabulary only captures types **seen in training** — truly novel forms (names, neologisms, typos) remain out-of-vocabulary regardless of size
> - **Subword tokenization** decomposes any word into units that are almost certainly in the vocabulary (at worst, individual characters)
> - This **eliminates** the open-vocabulary problem by construction rather than mitigating it by scale
<!--ID: 1777386639877-->





## Corpora

> [!question]- What is a corpus?
> - A **corpus** (pl. *corpora*) is a collection of text or speech assembled for analysis or model training
> - Every NLP system is shaped by the corpora it was trained on — corpus selection is a design decision with real consequences
> - A corpus is not a neutral sample of "language" — it is a sample of specific people's language use under specific conditions
<!--ID: 1777386639878-->





> [!question]- Along what dimensions do corpora vary, and why does this matter for NLP?
> - **Language** — 7097 languages in the world; most NLP defaults to English
> - **Variety** — e.g., African American English, Singaporean English — distinct grammar, not just accent
> - **Code-switching** — multilingual speakers mix languages within sentences
> - **Genre** — newswire, fiction, social media, legal, medical — very different statistical properties
> - **Author demographics** — age, gender, ethnicity, SES of writers
>
> Each dimension affects model performance: a model trained on one slice will generalise poorly to another.
<!--ID: 1777386639879-->





> [!question]- What is a corpus datasheet and why is it needed?
> - A **datasheet** (Gebru et al., 2020) is structured documentation recording how a corpus was collected, what it contains, its intended uses, and known limitations
> - Also called a **data statement** (Bender & Friedman, 2018)
> - Covers: motivation, situation, collection process, annotation process, language variety, demographics
> - The goal is to make the assumptions embedded in a corpus **explicit**, so downstream users can reason about what biases and gaps they are inheriting
<!--ID: 1777386639880-->





> [!question]- Why does training on a biased corpus produce systematic errors, not random ones?
> - The model learns statistical patterns from whatever text it sees
> - If certain language varieties are absent, the model has **no evidence** for their patterns and falls back on the majority variety
> - This produces errors **correlated with speaker identity** — the model performs worse on some groups by construction
> - These are not random mistakes — they are systematic bias built into the training data
<!--ID: 1777386639881-->





## Lemmas and Wordforms

> [!question]- What is the difference between a lemma and a wordform?
> - **Lemma**: same stem, same part of speech, same rough word sense — `cat` and `cats` share the lemma `cat`
> - **Wordform**: the full inflected surface form — `cat` and `cats` are **different wordforms**
> - A lemma groups together inflected variants; a wordform distinguishes them
> - Whether you count lemmas or wordforms affects your type count: *"the cats sat on the mats"* has 6 wordform types but only 5 lemmas (`the`, `cat`, `sit`, `on`, `mat`)
<!--ID: 1777386639882-->





## Tokenization

> [!question]- What is tokenization?
> - The task of segmenting a raw character stream into **tokens** — the atomic units the downstream system reasons over
> - It is the **first step** in virtually every NLP pipeline
> - What counts as a token depends on the application: words, subwords, characters, or morphemes
<!--ID: 1777386639883-->





> [!question]- Why is tokenization harder than it appears? Give examples.
> - **Periods** have multiple roles: abbreviation (`Dr.`), decimal (`$45.55`), URL (`google.com`), sentence boundary (`She left.`) — a split-on-period rule mangles all of these
> - **Clitics**: `we're` → `we + 're`, `can't` → `can + n't` — need to split within a "word"
> - **Multiword expressions**: `New York`, `rock 'n' roll` — multiple whitespace tokens that function as one semantic unit
> - **Languages without spaces**: Chinese, Japanese — word boundaries must be inferred, not read off the text
<!--ID: 1777386639884-->





> [!question]- Why is Chinese tokenization fundamentally different from English?
> - Chinese text has **no spaces between words** — word boundaries must be inferred from meaning and context
> - Segmentation is genuinely **ambiguous**: 姚明进入总决赛 can be parsed as 3 words, 5 words, or 7 characters
> - Needs a **dedicated word segmentation model** trained on annotated data — whitespace splitting is not an option
<!--ID: 1777386906364-->


> [!question]- What is the difference between "split on delimiters" and "collect matches" tokenization?
> - **Split on delimiters**: define what *separates* tokens (e.g., whitespace, punctuation) and split there
> - **Collect matches**: define what a token *looks like* (its shape) using regex patterns and collect matches
> - Collect matches is more expressive — you can have patterns for abbreviations (`(?:[A-Z]\.)+`), hyphenated words, currencies, all tried in priority order
> - Some tokens like `U.S.A.` contain characters that would be treated as delimiters under a split approach
<!--ID: 1777386906396-->

## Text Normalization

> [!question]- What is text normalization?
> **Text normalization** is the process of transforming raw text into a canonical, consistent form before it is fed into an NLP system.
> - Raw text is messy: inconsistent casing, spelling variants (`colour` vs `color`), inflected word forms (`running`, `ran`, `runs`), punctuation attached to words, multiple sentence boundaries
> - Normalization reduces this surface variation so the downstream model sees equivalent inputs as the same
> - It is a **pre-processing stage** — applied once to the corpus before any learning or inference
> - The three canonical steps are: **tokenization → word-format normalization → sentence segmentation**
> - What counts as "normal" is task-dependent: IR folds case, NER preserves it; chatbots keep contractions, some classifiers split them
> Tags: text-normalization preprocessing
>
> [!question]- What are the 3 steps of text normalization?
> 1. **Tokenizing words** — segmenting a character stream into tokens
> 2. **Normalizing word formats** — mapping token variants to a canonical form (case folding, lemmatization, stemming)
> 3. **Segmenting sentences** — identifying sentence boundaries
>
> Every NLP task requires this pipeline. The steps are ordered — tokenization must happen first because the other two depend on having tokens.
<!--ID: 1777386906430-->


> [!question]- What is the difference between stemming and lemmatization?
> - **Lemmatization**: reduces a word to its **dictionary headword** (lemma) using full morphological analysis — result is always a real word (`better` → `good`, `sang` → `sing`)
> - **Stemming**: strips affixes using **heuristic rules** without a lexicon — result may not be a real word (`universe` → `univers`) and may conflate unrelated words (`wand` / `wander` → `wand`)
> - **Use stemming** for IR — crude conflation raises recall at acceptable precision cost
> - **Use lemmatization** when word meaning matters — translation, classification, QA
<!--ID: 1777386906461-->


> [!question]- When should and shouldn't you use case folding? Give examples.
> - **Use it**: information retrieval — `The` and `the` should match the same documents; folding removes irrelevant variation
> - **Avoid it**: named entity recognition — `Apple` (company) vs `apple` (fruit) are different entities
> - **Avoid it**: sentiment analysis — `US` (United States) vs `us` (first-person pronoun) carry different signals
> - Case folding is **lossy** — the information lost sometimes matters
<!--ID: 1777386906477-->

## Subword Tokenization

> [!question]- What is subword tokenization and why is it needed?
> - Operates **between character-level and word-level** — learns a vocabulary of word fragments from data
> - **Word-level fails** in two ways: unseen words collapse to `<UNK>` (OOV problem), and morphologically rich languages produce enormous vocabularies
> - Subword solves both: any string decomposes into known subword pieces (worst case: individual characters), and related forms share units (`running`, `runner` both contain `run`)
> - Every modern large language model uses subword tokenization
<!--ID: 1777386906513-->


> [!question]- What are the two components of every subword tokenizer?
> 1. **Token learner**: runs **offline** on a training corpus → produces a vocabulary of subword types
> 2. **Token segmenter**: runs at **inference time** → applies the learned vocabulary to segment new text
>
> They are separated because learning is expensive (processes entire corpus) while segmentation must be fast (runs on every input). Learn once, segment many times.
<!--ID: 1777386906527-->


> [!question]- How does BPE differ from WordPiece?
> - **BPE** (Sennrich et al., 2016): merges the pair with the highest **raw frequency** — used by GPT-2/3/4, RoBERTa
> - **WordPiece** (Schuster & Nakajima, 2012): merges the pair that most improves **language model likelihood** — used by BERT, DistilBERT
> - WordPiece is more principled (optimises a meaningful objective) but slower to train
> - Both produce similar vocabularies in practice
<!--ID: 1777386906563-->

## Edit Distance

> [!question]- What is edit distance, and what are the two cost conventions?
> - The **minimum number of single-character operations** (insertion, deletion, substitution) needed to transform one string into another
> - Two conventions exist — always check which one a question uses:
>   - **Levenshtein** (Jurafsky & Martin): ins = 1, del = 1, **sub = 2**
>   - **Uniform**: ins = 1, del = 1, **sub = 1**
> - The same pair of strings produces a **different distance** under the two conventions
<!--ID: 1777386906577-->


> [!question]- What is the recurrence relation for edit distance?
> $$D(i, j) = \min \begin{cases} D(i-1, j) + 1 & \text{(deletion)} \\ D(i, j-1) + 1 & \text{(insertion)} \\ D(i-1, j-1) + c & \text{(match/sub)} \end{cases}$$
> - $c = 0$ if $X[i] = Y[j]$ (characters match), $c = 2$ (Levenshtein) or $1$ (uniform) if they differ
> - **Base cases**: $D(i,0) = i$, $D(0,j) = j$
> - **Answer**: $D(n,m)$ (bottom-right cell)
> - **Complexity**: $O(nm)$ time and space
<!--ID: 1777386639885-->





> [!question]- Walk through the edit distance DP table for "SIT" → "SAT" (Levenshtein: sub = 2).
> Base cases: row 0 = 0 1 2 3, col 0 = 0 1 2 3. Bold = backtrace path.
>
> |       | **""** | **S** | **A** | **T** |
> |-------|--------|-------|-------|-------|
> | **""**| **0**  | 1     | 2     | 3     |
> | **S** | 1      | **0** | 1     | 2     |
> | **I** | 2      | 1     | **2** | 2     |
> | **T** | 3      | 2     | 2     | **2** |
>
> **Reading the path (top-left → bottom-right):**
> - (0,0)→(1,1): S = S → match, cost **0**
> - (1,1)→(2,2): I ≠ A → substitution, cost **+2**
> - (2,2)→(3,3): T = T → match, cost **0**
>
> **Edit distance = 2**
>
> **Alignment:**
> ```
> S   I   T
>     ↕
> S   A   T
> ```
> Tags: edit-distance dynamic-programming worked-example
<!--ID: 1777549264704-->


> [!question]- What is backtrace in edit distance, and how does it recover the alignment?
> - Each cell stores a **pointer** to which predecessor gave the minimum: ← (insertion), ↓ (deletion), ↖ (match/substitution)
> - Start at $D(n,m)$, follow pointers back to $D(0,0)$
> - Each step reads off one edit operation; reversing gives the alignment in forward order
> - Complexity: $O(n+m)$
<!--ID: 1777386639886-->





> [!question]- When would you use weighted edit distance instead of uniform costs?
> - Uniform costs assume all errors are equally likely — in reality some are far more common
> - **Spelling correction**: confused character pairs (`m`/`n`, `b`/`d`) should have lower substitution cost
> - **Keyboard proximity**: adjacent keys (`s`/`d`) are more likely to be confused by a typist
> - **OCR correction**: visually similar characters (`0`/`O`, `l`/`1`) should cost less to substitute
> - Requires a **confusion matrix** — empirical frequencies of which characters get confused, from error logs or misspelling corpora
<!--ID: 1777386639887-->





## Byte Pair Encoding (BPE)

> [!question]- How does the BPE token learner algorithm work?
> 1. Split every word in the corpus into **individual characters** + an end-of-word marker (`_`)
> 2. Start with a vocabulary of all characters + `_`
> 3. **Repeat** $k$ times:
>    - Count all **adjacent symbol pairs** across the corpus
>    - Find the **most frequent pair** (e.g., `e` + `s` → `es`)
>    - Merge that pair into a new symbol everywhere in the corpus
>    - Add the new symbol to the vocabulary
> 4. Each merge reduces corpus length by one and adds one vocabulary entry
> - Frequent words eventually become single tokens; rare words stay as sequences of subword pieces
<!--ID: 1777386639888-->





> [!question]- How does the BPE segmenter handle a word it has never seen?
> - Split the new word into characters + end-of-word marker
> - Apply every learned merge **in the order it was learned**, greedily
> - Skip any merge whose pair doesn't appear in the current sequence
> - Example: unseen word `lowest` → `l o w e s t _` → merges produce `["low", "est_"]`
> - **Key property**: degrades gracefully — whole words → subword pieces → individual characters. Never produces `<UNK>`
<!--ID: 1777386639889-->





> [!question]- What is the precision/recall tradeoff in regex pattern design?
> - **Precision** = of the strings the pattern matched, what fraction were correct (true positives / (true positives + false positives))
> - **Recall** = of the strings that should have matched, what fraction did (true positives / (true positives + false negatives))
> - Tightening a pattern (more constraints) → raises precision, risks lowering recall
> - Broadening a pattern (more alternatives) → raises recall, risks lowering precision
> - There is no free lunch — practical regex engineering iterates between the two
<!--ID: 1777386639890-->





> [!question]- What does the Porter Stemmer do, and give an example rule?
> - A widely-used **rule-based stemmer** for English — strips affixes via a cascade of heuristic rewrite rules
> - Example rules: `ATIONAL → ATE` (e.g., `relational → relate`), `ING → ε` if the stem contains a vowel (e.g., `motoring → motor`), `SSES → SS` (e.g., `caresses → caress`)
> - Output may not be a real word (`computer → comput`)
> - Used heavily in information retrieval, where crude conflation raises recall at acceptable precision cost
<!--ID: 1777386639891-->





> [!question]- What is sentence segmentation, and why is it harder than splitting on `.!?`?
> - The task of identifying **sentence boundaries** in a token stream
> - `!` and `?` are relatively unambiguous; `.` is the troublemaker
> - A period can mark: sentence end, abbreviation (`Dr.`, `Inc.`), decimal (`3.14`), URL (`google.com`), or ellipsis (`...`)
> - The standard approach is a **binary classifier** at every period: "end of sentence" vs "not", using features like word shape, capitalization of next token, and abbreviation lists
> - Often handled jointly with tokenization — both involve the same period-disambiguation decision
<!--ID: 1777386639892-->







