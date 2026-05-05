---
name: cross-attention
description: Self-attention's twin, with one critical asymmetry — queries come from one sequence (typically the decoder's running output), while keys and values come from a different sequence (typically the encoder's latent code). The bridge that lets a decoder *consult* an encoder's representation. Same scaled dot-product machinery, two sources.
type: concept
sources:
  - raw/week-11/w11-slides.pdf
status: stable
updated: 2026-05-04
---

*Self-attention has every token query every other token in **the same sequence**. Cross-attention lets the decoder query a **different** sequence — the encoder's latent code. Each decoder position computes "which parts of the input do I need to look at to produce my next output?" — and the answer is read straight out of the encoder.*

## The one-line difference from self-attention

In [[self-attention]], queries, keys, and values all come from the same input $X$:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

In **cross-attention**, the query comes from one source and the keys/values come from another:

$$Q = X_{\text{decoder}}\, W^Q, \quad K = X_{\text{encoder}}\, W^K, \quad V = X_{\text{encoder}}\, W^V$$

After that, the formula is identical to self-attention's:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Same softmax, same scaling, same weighted sum of values. The mechanism is unchanged; only the sourcing of $Q$ vs $(K, V)$ differs. Most introductions to attention compress these together and just say "attention". When the distinction matters — and in the transformer's encoder-decoder bridge, it matters a lot — we call it **cross-attention** to flag the two-sequence setup.

## Where it lives in the transformer

The transformer decoder block has *three* sub-layers in sequence:

1. **Masked self-attention** over the decoder's own running outputs (causal, can't see the future).
2. **Cross-attention** with $Q$ from the decoder, $K, V$ from the encoder's latent code.
3. **Feed-forward MLP**, applied position-wise.

The cross-attention sub-layer is the *only* place information flows from the encoder into the decoder. Without it, the decoder is a standalone language model with no idea what the input was. With it, the decoder can ask "given what I've generated so far, which input tokens are most relevant to my next output?" — and pull the relevant content directly from the encoder's representation.

## Reading the architecture

Schematically, the decoder block at one position looks like:

```
                 from encoder (shared across all decoder positions)
                                  │
                                  ▼
                              [K, V]  ← projected with W^K, W^V from latent code
decoder hidden state ──► [Q] ──► cross-attention ──► output
                          ▲
                          └── projected with W^Q from decoder hidden state
```

Crucially:

- The **encoder is run once**, producing a fixed latent code (one vector per input token). $K$ and $V$ are computed once from this code and then *reused at every decoder time step*.
- The **decoder is run repeatedly**, once per output token. At each step, the latest decoder hidden states produce fresh $Q$s, but the encoder's $K, V$ stay constant.

This asymmetry is what the slides label "Encoder: executed once / Decoder: executed repeatedly". Caching $K, V$ from the encoder is a major efficiency win — you don't redo encoder work per decoder step.

## What cross-attention buys

For each output position the decoder is generating, cross-attention computes:

> *Which input positions are relevant to me right now, and what content should I pull from them?*

In machine translation ("Je suis étudiant" → "I am a student"):

- When the decoder is generating "I", cross-attention attends most strongly to the encoded "Je".
- When generating "am", cross-attention attends most strongly to "suis".
- When generating "student", cross-attention attends most strongly to "étudiant".

The attention pattern is *learned*, not hardcoded — and not strictly aligned word-by-word in general. For non-English-French pairs (e.g. English-Japanese), cross-attention learns the correct, often non-monotonic alignment automatically.

## Cross-attention beyond translation

The same machinery shows up wherever a network needs to **condition** one sequence's processing on another:

- **Encoder-decoder transformers (Vaswani et al. 2017)** — translation, summarisation, T5, Whisper. The decoder cross-attends to the encoded source.
- **Latent diffusion models (Stable Diffusion)** — the U-Net's image-side features cross-attend to a text encoder's output, conditioning image generation on the text prompt. See [[latent-diffusion-model]].
- **Visual question answering, image captioning** — the language decoder cross-attends to an image encoder's feature map.
- **Multimodal models (CLIP-conditioned generators, audio-visual transformers)** — one modality queries another via cross-attention.

Anywhere you see "conditioned on $X$" in a modern generative architecture, cross-attention is usually how the conditioning is implemented.

## Comparison: self-attention vs. cross-attention

| | Self-attention | Cross-attention |
|---|---|---|
| Query source | Same as keys/values (one input $X$) | Different from keys/values |
| Key/value source | Same as query | "Memory" — typically encoder output |
| Mechanism | Scaled dot-product attention | Scaled dot-product attention (identical formula) |
| Output sequence length | Equals input length | Equals query length (decoder length) |
| Where it appears | Encoder layers; decoder's first sub-layer (masked) | Decoder's second sub-layer |
| Role | Mix info *within* one sequence | Pull info *from* another sequence |

Both are "attention" in the generic sense. The label "cross" emphasises that two distinct sequences are involved.

## A note on shapes

If the encoder produced a latent code with $n_e$ tokens and the decoder is currently at position $n_d$, cross-attention's matrices have shapes:

- $Q$ from decoder: $n_d \times d_k$
- $K$ from encoder: $n_e \times d_k$
- $V$ from encoder: $n_e \times d_v$
- Attention scores $Q K^\top$: $n_d \times n_e$ — every decoder query against every encoder key.
- Output: $n_d \times d_v$ — one output per decoder query.

So the *output sequence length matches the decoder*, not the encoder. The encoder's content is mixed in via the values, but the *number* of cross-attention outputs is set by how many queries the decoder has.

## Multi-head cross-attention

Like self-attention, cross-attention is run with multiple heads in parallel. Each head has its own $(W^Q, W^K, W^V)$ — different heads can attend to different aspects of the encoder's latent code. The architectural mechanics are identical to [[multi-head-attention]]; the only twist is that $Q$ comes from one sequence and $(K, V)$ from another.

## Related

- [[self-attention]] — the same operation but with $Q, K, V$ all from one sequence
- [[multi-head-attention]] — running cross-attention in parallel with multiple heads
- [[transformer]] — the encoder-decoder architecture cross-attention bridges
- [[latent-diffusion-model]] — uses cross-attention to condition diffusion on text
- [[conditional-generative-model]] — the broader framing; cross-attention is the mechanism

## Active Recall

> [!question]- What is the one structural difference between self-attention and cross-attention?
> The source of the query versus the sources of the keys and values. In self-attention, $Q, K, V$ are all linear projections of the *same* input sequence. In cross-attention, $Q$ is projected from one sequence (typically the decoder) and $K, V$ are projected from a *different* sequence (typically the encoder's latent code). The scaled-dot-product formula $\text{softmax}(QK^\top/\sqrt{d_k})V$ is identical; only the sourcing differs.

> [!question]- Why is cross-attention the bridge between a transformer's encoder and decoder?
> It's the *only* layer where information from the encoder enters the decoder. The decoder's masked self-attention sees only its own running output; the feed-forward layer is position-wise and sees only one position. Cross-attention is what lets each decoder position "look back" at the encoded input to find relevant content. Removing it would leave the decoder as a stand-alone language model with no knowledge of what was input.

> [!question]- The encoder runs once but the decoder runs many times. How does cross-attention exploit this?
> Cross-attention's keys $K$ and values $V$ depend only on the encoder's output, which is computed once at the start. They can be **cached** and reused at every decoder time step. Only the queries $Q$ — which depend on the decoder's running state — need to be recomputed each step. This caching is a major efficiency win during autoregressive generation: the encoder cost is paid once, the decoder iterates against fixed $K, V$.

> [!question]- A decoder is generating "am" in the translation "I am a student" from input "Je suis étudiant". What does cross-attention compute at this position?
> The decoder's hidden state at the "am" position is projected to a query $\mathbf{q}$. The encoder's representations of "Je", "suis", and "étudiant" each give keys and values. The query $\mathbf{q}$ is dot-producted against each key, scaled and softmaxed to produce a distribution that should peak strongly on "suis" (after training). The output for the "am" position is then a weighted sum of the encoder values, dominated by the value vector of "suis" — telling the decoder "the relevant input word for what I'm generating now is *suis*, here is its content".

> [!question]- Beyond machine translation, where else does cross-attention appear?
> Anywhere a network needs to *condition* one sequence's processing on another sequence. Latent diffusion models (Stable Diffusion) use cross-attention from the U-Net's image features to a text encoder's output, conditioning image generation on the text prompt. Vision-language models use cross-attention from a language decoder to an image encoder's feature map. Audio-visual and other multimodal architectures use cross-attention to fuse modalities. The pattern is "queries from this stream, keys/values from that stream" whenever conditional/multimodal information flow is required.
