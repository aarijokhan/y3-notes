---
name: positional-encoding
description: A vector added to each token's embedding that carries its position in the sequence. Required because self-attention is *permutation-invariant* — without explicit position information, "the cat sat on the mat" and "mat the on sat cat the" would produce identical attention patterns and identical outputs. The encoding can be a fixed sinusoidal pattern (Vaswani et al. 2017) or a set of learned vectors.
type: concept
sources:
  - raw/week-11/w11-slides.pdf
status: stable
updated: 2026-05-04
---

*[[self-attention|Self-attention]] treats its input as a **set** of tokens, not a sequence. The dot products $\mathbf{q}_i^\top \mathbf{k}_j$ depend only on the contents of tokens $i$ and $j$, never on their positions. So the transformer needs to be told positions explicitly — by adding a position-encoding vector to each token's embedding before the first attention layer.*

## Why this is needed

Take a sequence of tokens "the cat sat on the mat" and feed it through a transformer encoder. Because self-attention's output for token $i$ is

$$\sum_j \text{softmax}_j\!\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}\right) \mathbf{v}_j$$

— a function only of dot products between token *contents* — the encoder is **permutation-invariant**: shuffle the tokens and the output set is identically shuffled, but the *information* per token is unchanged. Reorder "the cat sat" into "sat cat the" and the encoder produces the same set of output vectors, just in a different order.

This is fine for a *bag of words* but catastrophic for language. "Dog bites man" and "Man bites dog" are different sentences. The model must know order.

Convolutional layers and RNNs encode order implicitly — convolutions through their fixed receptive field, RNNs through sequential processing. Self-attention does neither. So order has to be injected explicitly into the input.

## The fix: add a position vector to every token

Before the first encoder layer, modify each token embedding $\mathbf{x}_i$ by adding a vector that depends only on $i$:

$$\tilde{\mathbf{x}}_i = \mathbf{x}_i + \mathbf{p}_i$$

where $\mathbf{p}_i \in \mathbb{R}^{d_{\text{model}}}$ is the **positional encoding** for position $i$. Now identity *and* position both contribute to the input. Two tokens with the same word but at different positions get *different* embeddings; subsequent attention layers can pick up on positional differences via the dot product.

This is added — not concatenated — partly because addition keeps the dimension fixed (no extra width to pay for) and partly because the network can learn to ignore the additive position component in subspaces where it shouldn't matter.

## Two common implementations

### 1. Sinusoidal (Vaswani et al. 2017)

The original transformer uses *fixed*, non-learnable sinusoids of geometrically-spaced frequencies. For position $\text{pos}$ and feature dimension $i$:

$$\mathbf{p}_{\text{pos}, 2i} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right), \qquad \mathbf{p}_{\text{pos}, 2i+1} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

Even-indexed dimensions get sines, odd-indexed dimensions get cosines, with frequencies geometrically spread from $1$ down to $1/10000$. The intuition:

- **High-frequency dimensions** alternate quickly across positions → encode fine-grained position differences.
- **Low-frequency dimensions** vary slowly → encode coarse positional regions.

Together they form a unique fingerprint per position. Two big advantages:

- **No learned parameters.** Pure deterministic encoding.
- **Generalises to longer sequences** than seen in training: the formula gives a defined encoding for any position, even ones the model never trained on.

### 2. Learned positional embeddings

The alternative used in BERT, GPT, and most modern LLMs: treat each position as a "vocabulary" item and learn a separate embedding vector per position. So position $0$ has its own $d_{\text{model}}$-dimensional learned vector, position $1$ has another, …, up to a maximum sequence length.

| | Sinusoidal | Learned |
|---|---|---|
| Parameters | 0 | $L_{\max} \cdot d_{\text{model}}$ |
| Generalises to longer sequences? | Yes (extrapolation) | No (positions beyond $L_{\max}$ are undefined) |
| Empirical performance | Comparable | Comparable to slightly better in practice |

Modern variants extend this further (relative position encodings, ALiBi, RoPE) but the basic idea — make the input position-aware — is the same.

## On the slide

The transformer architecture diagram shows the positional encoding being added to the input embedding *immediately* before the first attention layer (the symbol "$\oplus$" with the spiral). Once added, attention layers and feed-forward layers see only the combined embedding; they never receive position information separately.

Both encoder and decoder add their own positional encodings — the encoder to the input sequence, the decoder to its output-shifted-right sequence. Without them, the transformer would produce the same output for every permutation of its input.

## Why "fixed sinusoids of varied frequency"?

A useful intuition: positional encodings should be **distinguishable** between positions and **capable of carrying relative-distance signals**. Sinusoids of varied frequencies do both:

- Distinguishability: the combined sine/cosine pattern across all dimensions is unique per position, so two positions never collide.
- Relative position via dot product: $\mathbf{p}_{\text{pos}+\Delta} \cdot \mathbf{p}_{\text{pos}}$ depends mostly on $\Delta$, not on the absolute positions. This gives attention a way to learn "how far apart are these two tokens?" via dot products of their position encodings, which is exactly what attention needs to track relative order.

The geometrically-spaced frequencies ensure the encoding has multi-scale resolution: high-frequency dimensions distinguish neighbouring positions; low-frequency ones distinguish distant ones.

## Without positional encoding: what breaks

A simple thought experiment: feed the encoder a permuted version of its input. Without positional encoding, every output position is a function only of a *set* of token contents — the per-position outputs would be the same set, just permuted. The decoder cross-attending to such a set would have no way to distinguish word order.

Concretely:

- "Dog bites man" → outputs in some order encoding the set $\{$dog, bites, man$\}$.
- "Man bites dog" → identical outputs in a *different* order — but encoding the *same* set of facts.

Adding positional encoding breaks this symmetry. After the addition, the embedding for "dog" at position 0 differs from "dog" at position 2 — and downstream attention learns to use that difference.

## Related

- [[self-attention]] — the layer that needs positional encoding to function on ordered sequences
- [[transformer]] — uses positional encoding immediately after token embedding
- [[word-embedding]] — what's added to the positional encoding
- [[recurrent-neural-network]] — encodes order implicitly via sequential processing, which is why RNNs *don't* need positional encoding

## Active Recall

> [!question]- Why does self-attention need positional encoding when convolutions and RNNs don't?
> Self-attention is **permutation-invariant**: the dot product $\mathbf{q}_i^\top \mathbf{k}_j$ depends only on the contents of tokens $i$ and $j$, never on their positions. Shuffle the input tokens and you get a shuffled output containing the same information per token. Convolutions encode order through their fixed receptive field; RNNs encode order through sequential processing. Self-attention has no such mechanism, so position must be injected explicitly into the input embeddings.

> [!question]- How is positional encoding combined with the token embedding, and why is addition (rather than concatenation) used?
> The position vector $\mathbf{p}_i$ is **added** elementwise to the token embedding $\mathbf{x}_i$ before the first attention layer: $\tilde{\mathbf{x}}_i = \mathbf{x}_i + \mathbf{p}_i$. Addition keeps the dimension at $d_{\text{model}}$ — no extra parameters or width downstream. The network can also learn to ignore the position contribution in subspaces where it shouldn't matter, by having the corresponding rows of subsequent weight matrices be small.

> [!question]- Compare sinusoidal vs learned positional encodings. What does each give up?
> Sinusoidal encodings have **no learned parameters** and **generalise to longer sequences** than seen in training (the formula gives a defined value for any position). Learned encodings cost $L_{\max} \cdot d_{\text{model}}$ parameters and are **undefined** beyond the max length seen at training. Empirically they perform similarly. Most modern LLMs use learned (or rotary/relative) variants because the parameter cost is negligible at scale; the original transformer used sinusoids precisely because they were free.

> [!question]- A transformer is trained on sentences of length up to 512 tokens with learned positional embeddings. At inference you feed it a 1000-token sentence. What happens?
> Positions 0 to 511 work normally. Positions 512 to 999 have **no learned embedding** — the embedding table only has entries up to position 511. The model has no way to handle them, so it either crashes (out-of-bounds indexing) or the user has to clip / truncate the input. This is a real practical limitation of learned positional encodings; sinusoidal encodings (and modern alternatives like rotary or ALiBi) avoid it because they extrapolate to any position.
