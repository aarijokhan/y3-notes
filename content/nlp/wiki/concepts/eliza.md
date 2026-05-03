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

*ELIZA (Weizenbaum, 1966) is the first chatbot: a program that simulates a Rogerian therapist by transforming user input through a cascade of regular-expression substitution rules.*

## Definition

**ELIZA** is a natural language processing program written by Joseph Weizenbaum at MIT in 1966. It operates entirely through pattern matching: each user utterance is scanned against an ordered list of regex patterns, and the first matching pattern fires a template-based response that echoes and reframes the input. ELIZA carries no world model, no memory, and no understanding — only a list of rules.

## How It Works

The central mechanism is a **substitution rule** of the form:

```
s/<pattern>/<response template>/
```

A typical rule:

```
s/.* I'M (depressed|sad) .*/I AM SORRY TO HEAR YOU ARE \1/
```

The `s/` notation just means "substitute" — it is find-and-replace. Everything between the first and second `/` is the pattern to find; everything between the second and third `/` is what to replace it with.

**Annotated breakdown of the pattern half:**

```
.*         → match anything before (the lead-up: "He says", "My doctor told me", …)
 I'M       → match the literal text " I'M " (spaces included)
(depressed|sad)  → capture group: match either word and save it as \1
.*         → match anything after (the tail: "much of the time", "today", …)
```

**Step-by-step trace** for the input `"He says I'm depressed much of the time"`:

| Step | Part of pattern | What it matches |
|---|---|---|
| 1 | `.*` | `"He says"` |
| 2 | ` I'M ` | `" I'M "` |
| 3 | `(depressed\|sad)` | `"depressed"` — saved as `\1` |
| 4 | `.*` | `"much of the time"` |

The entire input is consumed by the pattern, so the whole string is replaced with the response template: `I AM SORRY TO HEAR YOU ARE \1` → `"I AM SORRY TO HEAR YOU ARE depressed"`.

The `.*` at the start and end are essential — without them, the pattern would only fire if the input were *exactly* `I'M depressed` with nothing else. The wildcards let ELIZA ignore everything except the key phrase in the middle.

The user says *"I'm very sad today"*; ELIZA replies *"I AM SORRY TO HEAR YOU ARE sad"*.

Rules are prioritized. If no specific rule matches, a default rule fires:

```
s/.*(.+).*/PLEASE TELL ME MORE ABOUT \1/
```

This fallback gives the illusion of engagement regardless of input.

## Python Implementation

A minimal ELIZA loop in Python shows the full architecture clearly:

```python
import re, string

patterns = [
    (r"\b(i'm|i am)\b",            "YOU ARE"),
    (r"\b(i|me)\b",                "YOU"),
    (r"\b(my)\b",                  "YOUR"),
    (r"\b(well,?) ",               ""),
    (r".*YOU ARE (depressed|sad) .*",  r"I AM SORRY TO HEAR YOU ARE \1"),
    (r".*YOU ARE (depressed|sad) .*",  r"WHY DO YOU THINK YOU ARE \1"),
    (r".*all .*",                  "IN WHAT WAY"),
    (r".*always .*",               "CAN YOU THINK OF A SPECIFIC EXAMPLE"),
    (r"[%s]" % re.escape(string.punctuation), ""),
]

while True:
    comment = input("User: ")
    response = comment.lower()
    for pat, sub in patterns:
        response = re.sub(pat, sub, response)
    print(response.upper())
```

Three design details worth noting:

1. **Sequential application**: patterns are applied one after another to the same string — each substitution feeds the next. The first few rules normalise pronouns (`I'm → YOU ARE`, `I → YOU`) so that later rules can match on the normalised form.
2. **Pronoun inversion happens first**: the user says "I'm depressed"; after the first rule, the string becomes "YOU ARE depressed"; the later rule then matches `YOU ARE (depressed|sad)` and fires the response. Without the ordering, the emotion rules would need to handle all first-person variants.
3. **Punctuation stripping**: the final rule removes all punctuation, which prevents noise from breaking matches.

## Significance

ELIZA produced the **ELIZA effect**: users attributed understanding, empathy, and even genuine intelligence to the program despite knowing it was rule-based. Weizenbaum himself was disturbed by how readily people formed emotional attachments to it. This raised early questions — still unresolved — about the relationship between linguistic behaviour and cognition.

From an NLP perspective, ELIZA demonstrates both the power and the ceiling of pure regex pattern matching:

- **Power**: a small, well-chosen ruleset can produce surprisingly plausible output for a constrained domain.
- **Ceiling**: ELIZA has no memory (it cannot refer back to earlier exchanges), no semantic understanding, and breaks immediately outside its scripted domain.

> [!info] ASIDE — ELIZA's Script
> ELIZA's rules were organised into a "script" (the most famous being DOCTOR, simulating a Rogerian therapist). Weizenbaum chose psychotherapy deliberately: in that style of therapy, the therapist mostly asks questions and reflects the patient's words back, which maps naturally onto substitution rules. A different domain would have required a fundamentally different script — and would likely have been much harder.

## Related

- [[regular-expressions]] — ELIZA is a direct application of regex substitution with capture groups
- [[tokenization]] — ELIZA scans raw text; later systems would tokenize first

## Active Recall

> [!question]- Why did Weizenbaum choose Rogerian psychotherapy as ELIZA's domain, and what does this reveal about the limits of the regex-substitution approach?
> Rogerian therapy is largely non-directive: the therapist reflects the patient's words back, asks open questions, and avoids asserting facts about the world. This maps directly onto substitution rules that echo user input. The choice was strategic — most other domains require the system to produce factually correct, contextually specific statements, which regex rules cannot generate. The domain choice hides the system's limitations rather than overcoming them.

> [!question]- What is the ELIZA effect, and why is it significant beyond just being an interesting demo?
> The ELIZA effect is the tendency of users to attribute understanding, emotion, and genuine intelligence to ELIZA despite knowing it is a rule-based program. Its significance: it showed that linguistic surface behaviour — even very shallow simulation — is enough to trigger human social projection. This is both a practical concern for AI systems today (users over-trust chatbots) and a conceptual puzzle for AI and cognitive science (what, if anything, does human-like language require beyond behavioural output?).

> [!question]- How does ELIZA's lack of memory constrain what it can do compared to even a simple modern dialogue system?
> ELIZA processes each utterance in isolation, with no state carried between turns. It cannot refer back to anything said earlier ("You mentioned earlier that…"), track commitments, update beliefs, or maintain topic coherence across more than one exchange. A simple modern system with even a conversation buffer can do all of these. ELIZA's statelessness means every turn is effectively a fresh start, making sustained coherent dialogue impossible.
