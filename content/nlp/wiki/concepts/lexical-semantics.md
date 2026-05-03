---
type: concept
sources:
  - raw/week-04/w04-slides.pdf
  - raw/week-04/w04-slides-updated.pdf
  - raw/week-04/w04-l10-transcript.txt
  - raw/week-04/w04-worksheet.pdf
  - raw/week-04/w04-worksheet-solutions.pdf
status: draft
updated: 2026-04-25
---

*Lexical semantics is the linguistic study of word meaning — the vocabulary every computational model of meaning has to recover. Words have senses, senses have relations to each other, and both are needed before any representation can be called adequate.*

## Why a Theory of Word Meaning

Everything before this week treated words as **strings** (or indices $w_i$ in a vocabulary). That's fine for [[n-gram-language-models|n-gram LMs]] and [[text-classification|text classification]] because the task doesn't demand that the model understand *cat* and *dog* are both mammals. But any downstream task that depends on *similarity* — paraphrase, retrieval, translation, question answering — needs a representation that knows these things.

The alternative that logic classes offer — *DOG* as a symbol, $\forall x\ \text{DOG}(x) \to \text{MAMMAL}(x)$ — is equally bad: it just renames strings as symbols and still requires every inference to be hand-coded. Barbara Partee's 1967 joke sums up the failure mode: *what's the meaning of life? A: LIFE.* Listing symbols isn't a theory of meaning.

So what do we want from a theory? Some **desiderata**, drawn from lexical semantics:

1. Distinguish **senses** from **lemmas** (one word, many meanings).
2. Capture **relations between senses** — synonymy, similarity, antonymy, relatedness.
3. Capture the **affective content** of words — sentiment, connotation.
4. Generalise: near-synonyms should look near each other without being told so.

The pages on [[vector-semantics]] and [[word2vec]] answer this with vectors. This page is about the linguistic structure those vectors have to recover.

## Lemmas, Senses, and Polysemy

A **lemma** is the canonical form of a word — the entry in a dictionary. *Mouse* is a lemma; *mice* is one of its inflected forms.

A **sense** (or **concept**) is a unit of meaning. One lemma can have many senses — this is **polysemy**:

> mouse (N)
> 1. any of numerous small rodents…
> 2. a hand-operated device that controls a cursor…

The two senses of *mouse* are completely unrelated — the relation between them is called **homonymy** when the meanings have separate histories that happen to share a form. Polysemy proper is when the senses are related (e.g. *bank* as financial institution vs. *bank* as building), but in NLP the distinction usually doesn't matter — what matters is that the representation of *mouse* has to somehow accommodate both meanings.

Any honest theory of word meaning is at least a **many-to-many mapping** between words and senses. [[word2vec|Static embeddings]] collapse this to a single vector per lemma (a known limitation); contextual embeddings like BERT recover per-occurrence senses.

## Relations Between Senses

### Synonymy

**Synonyms** have the same meaning in some or all contexts: *filbert/hazelnut, couch/sofa, big/large, automobile/car, vomit/throw up, water/H₂O*.

**There are probably no examples of perfect synonymy.** Even when the denotation is identical, words differ in politeness, slang, register, genre:

- *"H₂O"* in a surfing guide is wrong — register mismatch.
- *"my big sister"* ≠ *"my large sister"* — *big* has a sense (*elder*) that *large* lacks.

This is the **Linguistic Principle of Contrast**: difference in form tends to produce difference in meaning. Abbé Gabriel Girard (1718) put it as *"I do not believe that there is a synonymous word in any language."*

### Similarity

Most pairs of words aren't synonyms — they're just **similar**, sharing *some* element of meaning: *car/bicycle, cow/horse*. Humans can rate similarity on a scale: the **SimLex-999** dataset (Hill et al., 2015) has ratings like *vanish/disappear* = 9.8, *muscle/bone* = 3.65, *hole/agreement* = 0.3. Vector-semantic models are evaluated by how well their [[cosine-similarity|cosine similarities]] correlate with these human ratings.

### Word Relatedness (Association)

Words can be **related** without being similar — they appear in the same **semantic frame** or **semantic field**:

- *coffee, tea* — **similar** (both beverages)
- *coffee, cup* — **related, not similar** (different kinds of things, but they co-occur in the same scene)

A **semantic field** is a set of words that cover a particular semantic domain and bear structured relations:

- **hospitals**: surgeon, scalpel, nurse, anaesthetic, hospital
- **restaurants**: waiter, menu, plate, food, chef
- **houses**: door, roof, kitchen, family, bed

This distinction matters for embedding evaluation: [[word2vec]] models with **large windows** tend to learn *relatedness* (Harry Potter characters near *Hogwarts*), while **small windows** learn *similarity* (other fictional schools near *Hogwarts*).

### Antonymy

**Antonyms** are senses that differ with respect to only one feature of meaning — otherwise they are very similar: *dark/light, short/long, fast/slow, rise/fall, hot/cold, up/down, in/out*.

Two formal patterns:

- **Binary opposition or scale endpoints**: *long/short, fast/slow*.
- **Reversives**: *rise/fall, up/down*.

Antonyms are an empirical headache for distributional methods: *hot* and *cold* occur in nearly identical contexts ("the X coffee," "a X day") and cosine-similarity tends to put them *close*, not far apart, which is the opposite of what semantics wants.

### Connotation (Sentiment)

Words have **affective meaning** beyond their denotation:

- *happy* — positive connotation, *sad* — negative connotation.
- *copy, replica, reproduction* — positive; *fake, knockoff, forgery* — negative. Same referent, different evaluation.

Osgood et al. (1957) proposed three affective dimensions for any word — the **VAD** model:

- **valence**: pleasantness of the stimulus.
- **arousal**: intensity of emotion the stimulus provokes.
- **dominance**: the degree of control the stimulus exerts.

So the connotation of a word is a **vector in 3-space**. Each dimension can be read off a lexicon like the **NRC VAD Lexicon** (Mohammad 2018):

| | Word (high) | Score | Word (low) | Score |
|---|---|---|---|---|
| Valence | love | 1.000 | toxic | 0.008 |
| Arousal | elated | 0.960 | mellow | 0.069 |
| Dominance | powerful | 0.991 | weak | 0.045 |

This matters beyond this page: it's the first appearance of the "meaning as point in space" idea that motivates [[vector-semantics]], and it connects directly to [[sentiment-analysis|sentiment analysis]]'s concern with *attitudes*.

## Summary

The objects:

- **Concepts / word senses** — meaning units, many-to-many with words, supporting homonymy and polysemy.
- **Relations between senses** — synonymy, antonymy, similarity, relatedness, connotation.

Every method from now on is judged by how well it recovers this structure automatically — without being told *couch ≈ sofa* or *car ≠ bicycle*. Computing on strings loses it; computing on [[word2vec|embeddings]] can preserve a lot of it.

## Related

- [[vector-semantics]] — the computational answer: words as vectors defined by their distribution
- [[word2vec]] — dense embeddings that learn many of these relations without supervision
- [[cosine-similarity]] — the standard way to measure vector similarity matches human similarity ratings
- [[sentiment-analysis]] — connotation / affective meaning is the feature that sentiment models exploit
- [[tf-idf]] — sparse vectors as a first attempt at representing word meaning by distribution

## Active Recall

> [!question]- What's the difference between a lemma and a sense, and why does it matter?
> A **lemma** is a lexical form (the dictionary entry, e.g. *mouse*); a **sense** is a unit of meaning. One lemma can carry multiple senses (*mouse* = rodent, *mouse* = pointing device) — this is polysemy / homonymy. It matters because a representation that assigns one vector per lemma silently averages over unrelated meanings, which is a known limitation of [[word2vec|static embeddings]] and the motivation for contextual embeddings (BERT, ELMo).

> [!question]- Why does the Linguistic Principle of Contrast mean "perfect synonymy" essentially doesn't exist?
> Any time two word forms persist in a language, they tend to specialise — differ in register, politeness, slang, connotation, or subtle denotation. *Big sister* and *large sister* have identical denotations but *big* has a non-size sense (*elder*) that *large* lacks. *Water/H₂O* are semantic equivalents but register-mismatched outside chemistry. The robust lesson: embedding similarity should be high for such pairs but need not be 1.

> [!question]- Give an example pair that is related but not similar, and explain the difference.
> *coffee/cup*. They are **related** (co-occur in the restaurant/breakfast scene, same semantic frame) but not **similar** (one is a liquid, one is a container). Similarity asks about shared meaning features; relatedness asks whether words inhabit the same semantic field. [[word2vec|Word2vec]] with large windows captures relatedness; with small windows it captures similarity.

> [!question]- The worksheet asks how NLP systems identify synonyms. List the three main approaches.
> (1) **Lexical databases** like WordNet — hand-curated sense inventories with synonym (synset) groupings. (2) **Distributional similarity from corpora** — find words that appear in nearly identical contexts, following Harris's distributional hypothesis; see [[vector-semantics]]. (3) **Contextual embeddings** from pre-trained models like BERT, which produce occurrence-specific vectors and can distinguish sense-specific synonyms in context.

> [!question]- What are the three VAD dimensions, and what does "meaning as a point in space" mean for a word like *happy*?
> **Valence** (pleasantness), **arousal** (intensity of emotion provoked), **dominance** (degree of control). A word like *happy* gets a score on each dimension and is represented as a 3-vector (e.g. roughly $[1.0, 0.7, 0.6]$). This is the simplest possible "meaning as point in space" model — later [[word2vec|embedding methods]] generalise this to 50-1000 dimensions learned from corpus distribution rather than human annotation.

> [!question]- Why are antonyms particularly hard for distributional models?
> Antonyms share nearly all features except one (*hot* and *cold* both describe temperature, modify similar nouns, occur with similar verbs). Their **distributions** therefore look almost identical, so any method that equates "similar distribution" with "similar meaning" will put antonyms close together — exactly the opposite of what lexical semantics wants. Fixing this typically requires auxiliary signal (sentiment lexicons, pattern-based rules, or constraints during training).

> [!question]- Slide MCQ: Which statements about lexical semantics are correct? (a) Polysemy is one word with multiple *related* meanings (*bank* = river/finance), but does not extend to completely unrelated meanings; (b) Synonymy denotes the absence of any relationship between word meanings; (c) Antonymy involves opposites (*hot*/*cold*) and does not preclude scalar antonyms on a continuum; (d) Lexical semantics negates the importance of syntax, focusing solely on meaning.
> **Correct: (a) and (c).** (a) Polysemy strictly requires *related* senses — the completely-unrelated case (e.g. river *bank* and *bank* the institution, which have separate etymologies) is **homonymy**, not polysemy. (c) Antonymy accommodates both binary opposites (*in/out*) and scalar endpoints (*short/long*), so scalar antonyms on a continuum are still antonyms. (b) is the exact opposite of the truth — synonymy is *shared* meaning. (d) is false — lexical semantics sits alongside syntax, not in opposition to it.
