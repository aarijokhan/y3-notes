---
type: concept
sources:
  - raw/week-01/w01-slides.pdf
  - raw/week-01/w01-l01-transcript.txt
  - raw/week-01/w01-lab-01.pdf
  - raw/week-01/w01-lab-01-solutions.pdf
status: draft
updated: 2026-04-20
---

*A formal language for specifying text search patterns, built from a small set of operators that compose into arbitrarily complex matchers.*

## Definition

A **regular expression** (regex) is a sequence of characters that defines a pattern over strings. A regex engine takes a pattern and an input string and returns all substrings (matches) that conform to the pattern. In NLP they appear at every layer: corpus preprocessing, tokenization, feature extraction, and rule-based systems like [[eliza]].

---

## Syntax

### Basic building blocks

| Construct | Meaning                                                                   | Example        | Matches                   | Test string                        |
| --------- | ------------------------------------------------------------------------- | -------------- | ------------------------- | ---------------------------------- |
| `[abc]`   | Any one of a, b, c                                                        | `[wW]oodchuck` | `woodchuck`, `Woodchuck`  | "The **Woodchuck** ate his dinner" |
| `[a-z]`   | Any character in range                                                    | `[A-Z]`        | any uppercase letter      | "**D**renched **B**lossoms"        |
| `[^abc]`  | Anything *except* a, b, c (negation — **only when `^` is first in `[]`**) | `[^A-Z]`       | any non-uppercase char    | "**O**yfn **p**ripetchik"          |
| `.`       | Any single character                                                      | `beg.n`        | `begin`, `began`, `begun` | "We must **begin** now"            |
| `^`       | Start of line (outside `[]`)                                              | `^The`         | `The` only at line start  | "**The** quick brown fox"          |
| `$`       | End of line                                                               | `end$`         | `end` only at line end    | "reach the **end**"                |

> [!warning] `^` HAS THREE DISTINCT MEANINGS
> - **First character inside `[]`** → **negation**: `[^abc]` matches any character that is *not* `a`, `b`, or `c`
> - **Outside `[]`** → **start-of-line anchor**: `^The` matches `The` only at the beginning of a line
> - **Non-first character inside `[]`** → **literal caret**: `[e^]` matches `e` or `^`; `[^e^]` matches any character that is neither `e` nor `^`
>
> The same symbol, three unrelated jobs — which one applies is determined entirely by position.

---

### Shorthand character classes

These shorthands expand to character classes and appear constantly in real patterns:

| Shorthand | Equivalent | Meaning |
|---|---|---|
| `\d` | `[0-9]` | Any digit |
| `\D` | `[^0-9]` | Any non-digit |
| `\w` | `[a-zA-Z0-9_]` | Any word character (letter, digit, underscore) |
| `\W` | `[^a-zA-Z0-9_]` | Any non-word character |
| `\s` | `[ \t\n\r\f]` | Any whitespace |
| `\S` | `[^ \t\n\r\f]` | Any non-whitespace |

Examples: `\d+` matches one or more digits. `\w+` matches a whole word token. `\s+` matches any run of whitespace and is the basis of whitespace tokenization.

---

### Word boundary `\b`

`\b` is a **zero-width assertion** — it matches a *position* (between a `\w` character and a `\W` character) rather than consuming any characters.

```
\bthe\b
```

This matches `the` as a standalone word, but **not** `the` inside `there`, `other`, or `these`.

Without `\b`:

```
the     →   matches "the", "there", "other", "hypothesis", …
```

With `\b`:

```
\bthe\b →   matches "the"   in "I went to the store"
             no match        in "there" or "other"
```

This is the standard fix for the false-positive problem when searching for whole words. You almost always want `\b` when matching specific words in NLP.

---

### Quantifiers

Quantifiers attach to the immediately preceding element and control how many times it can appear.

| Quantifier | Meaning | Example | Test string |
|---|---|---|---|
| `*` | Zero or more | `oo*h!` | `oh!`, `ooh!`, `oooh!`, `ooooh!` |
| `+` | One or more | `o+h!` | `ooh!`, `oooh!` (not `oh!`) |
| `?` | Zero or one (optional) | `colou?r` | `color`, `colour` |
| `{n}` | Exactly *n* | `[0-9]{4}` | `2026` in "Call by 2026" |
| `{n,m}` | Between *n* and *m* | `[a-z]{2,5}` | `be`, `began`, `begin` |
| `{n,}` | *n* or more | `\d{3,}` | `1800` in "Call 1800 today" |

Examples:
- `colou?r` — matches `color` and `colour` (the `u` is optional)
- `[0-9]{4}` — matches exactly four digits (e.g. a year)
- `\d+\.\d{2}` — matches a price like `45.99`

#### Greedy vs non-greedy

By default, quantifiers are **greedy** — they consume as many characters as possible while still allowing the overall match to succeed.

```python
re.findall(r'<.+>', '<a>hello</a>')   # → ['<a>hello</a>']   (greedy)
re.findall(r'<.+?>', '<a>hello</a>')  # → ['<a>', '</a>']    (non-greedy)
```

Add `?` after a quantifier (`*?`, `+?`, `??`) to make it **non-greedy**: it will match as *few* characters as possible. Non-greedy is usually what you want when parsing structured text like HTML tags.

---

### Grouping and alternation

**How is a group different from just writing a pattern?**

Without groups, quantifiers (`*`, `+`, `?`, `{n}`) attach to the **single token immediately before them** — never more. So `cat+` does not mean "one or more repetitions of `cat`"; it means "`ca` followed by one or more `t`s". The `+` only sees the `t`.

This is the same problem as operator precedence in arithmetic. In `2 + 3 × 4`, the `×` only applies to `3` and `4`. Parentheses let you override that: `(2 + 3) × 4`. Regex parentheses work identically — they bundle multiple tokens into a single unit so that a quantifier or alternation applies to all of them together:

```
cat+        →   ca, catt, cattt, …        (+ applies only to t)
(cat)+      →   cat, catcat, catcatcat, … (+ applies to the whole word)

the cat|dog →   "the cat"  or  "dog"      (| splits at low precedence)
the (cat|dog) → "the cat"  or  "the dog"  (| applies only inside the group)
```

**Grouping is the primary job.** The capturing side-effect is a bonus.

**Capturing**: a *capturing* group also stores whatever text matched inside it into a numbered slot — `\1` for the first group, `\2` for the second, left to right. You can then reference that slot in two places:

- **In a replacement string** — reuse or rearrange the captured text:  
  `re.sub(r'(\w+) (\w+)', r'\2 \1', 'Smith John')` → `"John Smith"`
- **Inside the same pattern** — enforce that a later part must equal what was already captured:  
  `([a-z]+)\s+\1` matches `"the the"` (second word must repeat the first) but not `"the cat"`

If you only need grouping for scope — and have no use for the captured text — use `(?:...)`, a **non-capturing group**, to avoid allocating an unnecessary slot.

| Construct | Meaning |
|---|---|
| `(abc)` | Capturing group: groups and saves match as `\1`, `\2`, … |
| `(?:abc)` | Non-capturing group: groups without saving |
| `a\|b` | Alternation: match `a` or `b` |

Use `(?:...)` when you need grouping for a quantifier but don't want to create a capture group:

```
(?:cat|dog)s?   →   cat, cats, dog, dogs   (no capture created)
(cat|dog)s?     →   same matches, but \1 holds 'cat' or 'dog'
```

Multiple capture groups are indexed left to right:

```
(\w+)\s(\w+)    →   \1 = first word,  \2 = second word
```

#### Backreferences in patterns

A backreference (`\1`, `\2`, …) can appear inside the pattern itself — not just in the replacement. This matches text that is identical to what was captured earlier in the same match.

```python
re.findall(r'([a-zA-Z]+)\s+\1', text)
```

This matches any word that appears twice consecutively, separated by whitespace: `"the the"`, `"Humbert Humbert"`, but not `"the cat"`. The capture group matches the first word; `\1` requires the second word to be identical.

Practical uses:
- Detecting duplicate words in text
- Finding repeated tokens in corpora
- Validating that two parts of a pattern match the same value

> [!warning] COMMON MISCONCEPTION
> Backreferences in patterns (`\1` inside the regex) are different from backreferences in substitution strings (`\1` in the replacement). In a pattern, `\1` is a constraint — it says "this position must match whatever group 1 already captured." In a replacement, `\1` is an interpolation — it inserts the captured text. The syntax looks the same; the meaning is different.

---

### Operator precedence (high → low)

1. `()` — parentheses
2. `* + ? {}` — quantifiers
3. Sequences and anchors (concatenation)
4. `|` — alternation

So `a|bc*` parses as `a | (b(c*))` — either `a`, or `b` followed by any number of `c`. To mean "zero or more of either `a` or `b` followed by `c`" you would write `(a|b)c*`.

---

### Escaping special characters

The characters `. * + ? ^ $ { } [ ] ( ) | \` all have special meaning in regex. To match them literally, prefix with `\`:

| You want to match | Write |
|---|---|
| A literal period | `\.` |
| A literal asterisk | `\*` |
| A literal parenthesis | `\(` or `\)` |
| A literal backslash | `\\` |

Example: `\$\d+\.\d{2}` matches a price like `$45.99` — the `\$` and `\.` match literal symbols, `\d+` and `\d{2}` match digit runs.

---

### Python raw strings

Python processes escape sequences (`\n`, `\t`, etc.) in strings before the regex engine ever sees the pattern. This causes double-escaping problems:

```python
# BAD: Python converts \b to a backspace character; regex never sees \b
re.findall('\bcat\b', text)

# GOOD: raw string r'...' passes the characters literally to the regex engine
re.findall(r'\bcat\b', text)
```

**Always use `r'...'` for regex patterns in Python.** This is not optional — patterns containing `\d`, `\w`, `\b`, `\s` will silently misbehave without it.

---

## Lookahead Assertions

Lookaheads match a **position** without consuming characters.

| Construct | Meaning |
|---|---|
| `(?=...)` | Positive lookahead: position is followed by the pattern |
| `(?!...)` | Negative lookahead: position is *not* followed by the pattern |

```python
re.findall(r'Windows(?! NT)', text)   # "Windows" only when NOT followed by " NT"
re.findall(r'\d+(?= dollars)', text)  # digits only when followed by " dollars"
```

Lookaheads are useful when you want to match something based on what comes after it, without including that context in the match itself.

---

## Substitutions and Capture Groups

The substitution `s/pattern/replacement/` replaces matches. Capture groups let you reuse matched text in the replacement:

```python
import re

# Wrap every number in angle brackets: "34 items" → "<34> items"
re.sub(r'([0-9]+)', r'<\1>', '34 items in 2 boxes')
# → '<34> items in <2> boxes'

# Swap first and last name: "Smith John" → "John Smith"
re.sub(r'(\w+) (\w+)', r'\2 \1', 'Smith John')
# → 'John Smith'
```

ELIZA's core mechanism is exactly this:

```python
re.sub(r".* I'M (depressed|sad) .*", r"I AM SORRY TO HEAR YOU ARE \1", text)
```

Multiple capture groups are referenced as `\1`, `\2`, … in the replacement string.

---

## Building a Pattern Iteratively

Regex engineering is not "write once, it works". It is a loop:

1. **Write a first-pass pattern** — simple, covering the obvious cases.
2. **Test against real text.** Find:
   - **False positives**: strings that match but shouldn't.
   - **False negatives**: strings that should match but don't.
3. **Tighten** (add constraints) to reduce false positives.
4. **Broaden** (add alternatives) to reduce false negatives.
5. Repeat.

**Worked example**: find the word *the* in text.

```
Attempt 1:   [tT]he
```

False positives: matches `the` inside `there`, `other`, `theology`.

```
Attempt 2:   [tT]he[^a-zA-Z]
```

False negatives: misses `the` at end of line (no character follows).

```
Attempt 3:   \b[tT]he\b
```

This correctly matches `the` and `The` as standalone words, and nothing else. Word boundaries handle both the "followed by non-letter" and "end of string" edge cases simultaneously.

---

## Precision and Recall

Any regex pattern makes errors in two directions:

- **False positives** (Type I): the pattern fires when it shouldn't → lower **precision**.
- **False negatives** (Type II): the pattern misses what it should catch → lower **recall**.

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

Tightening a pattern raises precision but risks lowering recall. Broadening raises recall but risks lowering precision. There is no free lunch — practical regex engineering is managing this tradeoff deliberately.

---

## Python `re` Module

```python
import re

re.findall(r'\b[A-Z]\w+', text)        # all capitalized words (as whole words)
re.search(r'\bcat\b', text)            # first match object (or None)
re.match(r'\d+', text)                 # match only at START of string
re.sub(r'\bcolou?r\b', 'color', text)  # normalize spelling
re.split(r'\s+', text)                 # split on any whitespace run
re.compile(r'\d{4}', re.IGNORECASE)    # pre-compile for reuse
```

**Key flags:**

| Flag | Effect |
|---|---|
| `re.IGNORECASE` (`re.I`) | Case-insensitive matching |
| `re.MULTILINE` (`re.M`) | `^` and `$` match start/end of each *line*, not just the string |
| `re.DOTALL` (`re.S`) | `.` matches newline characters too |
| `re.VERBOSE` (`re.X`) | Allows whitespace and `#` comments inside the pattern |

`re.VERBOSE` is useful for complex patterns:

```python
pattern = re.compile(r'''
    (?:[A-Z]\.)+       # abbreviations: U.S.A.
  | \w+(?:-\w+)*       # hyphenated words
  | \$?\d+(?:\.\d+)?   # prices and numbers
  | \.\.\.             # ellipsis
''', re.VERBOSE)
```

---

## Role of Regex in NLP

Regular expressions play a surprisingly large role in NLP:

- **Sophisticated sequences of regular expressions are often the first model** tried for any text processing task — before any machine learning is involved. They are fast to write, fully interpretable, and require no training data.
- **For harder tasks, machine learning classifiers take over** — when patterns become too complex or too numerous to enumerate by hand, learned models are more practical.
- Even then, **regex doesn't disappear**: it is used for pre-processing text before it reaches a classifier, and regex-derived features (e.g. "does this token match `\d+`?") are fed directly into classifiers as input signals.
- Regex is also well-suited for **capturing generalizations** — a single pattern like `\$\d+(?:\.\d{2})?` covers every price format without needing labelled examples.

The practical takeaway: when starting any new NLP task, write a regex first. It sets a baseline, reveals edge cases, and often turns out to be good enough.

---

## Related

- [[eliza]] — uses regex substitution as its sole reasoning mechanism
- [[tokenization]] — early tokenizers are regex cascades; NLTK uses `re.VERBOSE` patterns
- [[text-normalization]] — normalization rules are implemented as substitutions

---

## Active Recall

> [!question]- What is the difference between `[^abc]` and `^abc` in a regular expression?
> Inside a character class `[...]`, the caret `^` is a negation operator: `[^abc]` matches any character that is not `a`, `b`, or `c`. Outside a character class, `^` is a line-start anchor: `^abc` matches `abc` only at the beginning of a line. Same symbol, two unrelated meanings — context (inside vs outside `[]`) determines which.

> [!question]- Why does `\bcat\b` behave differently from `cat`, and when does `\b` matter in NLP?
> `\b` is a zero-width assertion matching the position between a word character (`\w`) and a non-word character (`\W`). `cat` matches `cat` anywhere — inside `concatenate`, `bobcat`, or `category`. `\bcat\b` matches `cat` only as a standalone word. In NLP, searching for content words without word boundaries produces systematic false positives on substrings.

> [!question]- What goes wrong if you write `re.findall('\bcat\b', text)` in Python, and how do you fix it?
> Python processes escape sequences before the regex engine sees the string. `'\b'` in a Python string literal is the backspace character (ASCII 8), not the regex word boundary. The regex engine never receives `\b`. Fix: use a raw string `r'\bcat\b'` — raw strings pass the literal characters `\`, `b` to the regex engine, which then interprets `\b` as a word boundary.

> [!question]- What is the difference between a greedy and a non-greedy quantifier? Give an example where the choice matters.
> Greedy (`*`, `+`): matches as many characters as possible. Non-greedy (`*?`, `+?`): matches as few as possible. On `<a>hello</a>`, the pattern `<.+>` greedily matches the whole string `<a>hello</a>` as one match; `<.+?>` non-greedily matches `<a>` and `</a>` as two separate matches. Non-greedy is necessary whenever you want to match the shortest possible span between delimiters.

> [!question]- What is the operator precedence in regular expressions, and why does it matter for reading `a|bc*`?
> Precedence (high to low): parentheses, then quantifiers (`* + ? {}`), then sequences/anchors, then alternation `|`. So `a|bc*` parses as `a | (b(c*))` — either `a`, or `b` followed by any number of `c`. To mean "zero or more of `a` or `b` after `c`" you need `(a|b)c*`. Misreading precedence is a common source of incorrect patterns.

> [!question]- How does the substitution operator use capture groups, and what is a concrete NLP use case?
> Capture groups `(...)` save the matched substring as `\1`, `\2`, etc. The replacement string can reference them. Example: `re.sub(r'([0-9]+)', r'<\1>', text)` wraps every number in angle brackets. In ELIZA, `s/.* I'M (depressed|sad) .*/I AM SORRY TO HEAR YOU ARE \1/` echoes back the captured feeling word to produce therapist-like output.

> [!question]- Which of the following regular expressions will correctly match email addresses in the format `username@domain.extension`? A) `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` B) `\[a-zA-Z]+@[a-zA-Z]+\.[a-zA-Z]{2,4}$` C) `[a-zA-Z0-9._%+-]+#[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$` D) `^[a-zA-Z0-9._%+-]@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
> **A** is correct. It anchors with `^...$`, matches one or more username characters from the set `[a-zA-Z0-9._%+-]+`, then `@`, then one or more domain characters, then `\.` (literal dot), then 2+ letters for the extension. **B** uses `\[` which is a literal bracket, not a character class, and restricts extensions to 2–4 chars (excluding newer TLDs). **C** uses `#` instead of `@`. **D** lacks the `+` after the username class, so it matches only a single character before `@`.

> [!question]- Write a single regex that matches the set of all strings where each `a` is immediately preceded and followed by a `b`, over the alphabet {a, b}. What is the key insight?
> `/(b+(b|ab)*b+)?/` — The pattern allows the empty string (the outer `?`) and any string over {a,b} where every `a` is wrapped with `b`s. The core idea: either there are no `a`s at all (just `b`s), or `a` appears only inside a `b...b` sandwich. The alternation `(b|ab)*` handles runs of `b`s and `ab` units in the middle, while the outer `b+` anchors enforce leading and trailing `b`s. Verifying edge cases: `ba` — fails correctly (trailing `a` is not followed by `b`); `bab` — matches correctly; `bb` — matches correctly; `a` — fails correctly.

> [!question]- Write a regex that matches any string containing both the whole word `grotto` and the whole word `raven`, in either order.
> `/\bgrotto\b.*\braven\b|\braven\b.*\bgrotto\b/` — Two alternatives connected by `|`, one for each ordering. Both use `\b` to enforce whole-word matching (so `grottos` does not match). The `.*` between them allows any content between the two words. This pattern illustrates a recurring idiom: to require two independent patterns both appear in a string, write both orderings explicitly with `.*` between them.

> [!question]- Write a regex that captures the first word of an English sentence into group 1, handling any leading punctuation.
> `^[^a-zA-Z]*([a-zA-Z]+)` — The `^` anchors to line start. `[^a-zA-Z]*` skips any non-letter characters at the beginning (e.g., leading quotes `"`, dashes `—`, or whitespace). `([a-zA-Z]+)` captures the first run of letters as group 1. The key insight is using a negated character class to skip punctuation rather than trying to enumerate all possible punctuation characters.
