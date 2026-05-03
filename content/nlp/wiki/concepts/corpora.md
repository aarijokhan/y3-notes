---
type: concept
sources:
  - raw/week-01/w01-slides.pdf
  - raw/week-01/w01-l01-transcript.txt
status: draft
updated: 2026-04-20
---

*A corpus is a collection of text or speech used to train, evaluate, or study NLP systems; its composition determines what those systems can and cannot do.*

## Definition

A **corpus** (pl. *corpora*) is any structured collection of linguistic data — written text, transcribed speech, or both — assembled for analysis or model training. Every NLP system is shaped by the corpora it has been exposed to, which makes corpus selection a design decision with real consequences.

## Dimensions of Variation

Corpora differ along several axes, each of which affects the models trained on them:

**Language and variety**
English NLP often defaults to Standard American English, but language is not monolithic. African American English (AAE), Nigerian English, Singaporean English, and dozens of other varieties have distinct phonology, morphology, and syntax. A model trained only on formal written Standard English will generalise poorly — or produce systematically biased errors — when applied to other varieties.

**Code-switching**
Multilingual speakers routinely mix languages within a conversation or even a sentence (*"Por favor, pass the salt"*). Corpora that strip code-switching out, or that are monolingual by construction, leave models unable to handle real-world multilingual text.

**Genre and register**
News wire, social media, legal documents, medical records, and spoken transcripts have very different statistical properties. A sentiment analyser trained on movie reviews will not straightforwardly transfer to clinical notes.

**Time period**
Language changes. A corpus from the 1990s will underrepresent contemporary slang, new named entities, and shifting word meanings.

**Author demographics**
Who wrote the texts matters. If a corpus over-represents certain demographics (e.g., male, Western, highly educated authors), models trained on it will encode those perspectives and may perform worse on text from other groups.

## Corpus Datasheets

Gebru et al. (2020) proposed **datasheets for datasets**: structured documentation recording how a corpus was collected, what it contains, its intended uses, and its known limitations. Bender & Friedman (2018) similarly argued for **data statements**, emphasising the importance of documenting speaker demographics and annotation practices.

The goal in both cases is to make the assumptions embedded in a corpus explicit, so that downstream users can reason about what biases and gaps they are inheriting.

## Related

- [[type-and-token]] — corpus size $N$ drives vocabulary growth via Heaps' Law
- [[tokenization]] — how raw corpus text is segmented into usable units
- [[text-normalization]] — normalization choices interact with corpus conventions

## Active Recall

> [!question]- Why does training on a corpus that lacks certain language varieties lead to systematically biased model performance rather than just random errors?
> The model learns statistical patterns from whatever text it sees. If certain varieties are absent or underrepresented, the model has no evidence for their patterns and must fall back on the majority variety — producing errors that are correlated with speaker identity, not random. This is systematic bias: the model performs worse on some groups by construction.

> [!question]- What problem do corpus datasheets (Gebru et al. 2020) address, and why is it an NLP-specific concern rather than a general ML concern?
> Datasheets make corpus provenance explicit: who produced the text, when, in what variety, under what annotation scheme, with what known gaps. This matters especially in NLP because text inherently encodes social and cultural context — a corpus is not a neutral sample of "language" but a sample of specific people's language use under specific conditions. Without documentation, downstream users inherit hidden assumptions they cannot reason about.

> [!question]- How does code-switching expose a limitation in standard monolingual corpus design?
> Monolingual corpora assume each document belongs to one language. But real multilingual speakers routinely mix languages at clause or even word boundaries. A model trained on monolingual data has never seen these patterns and has no mechanism to handle them — it may assign the wrong language, fail to tokenize correctly, or produce incoherent output. The corpus design enforces a monolingual fiction that does not match how many speakers actually communicate.
