---
type: week
week: 11
title: "Transformers and Autoregressive Generation"
sources:
  - raw/week-11/w11-slides.pdf
concepts:
  - "[[self-attention]]"
  - "[[multi-head-attention]]"
  - "[[cross-attention]]"
  - "[[positional-encoding]]"
  - "[[autoregressive-model]]"
  - "[[transformer]]"
status: stable
updated: 2026-05-04
---

> [!question]+ THE CRUX: Sequences in language have arbitrarily long-range dependencies — the subject and verb of a sentence can be many words apart. RNNs forget across long distances; CNNs need exponentially deep stacks to bridge them. What architecture can let *every* token directly attend to every other token, and how do we use it to **generate** sequences?

*Replace recurrence and convolution with **self-attention**: every token forms a query, key, and value, and each output is a softmaxed similarity-weighted sum over the whole sequence. Stack this in an encoder-decoder shape, add **positional encoding** (because attention is order-agnostic) and **masked attention** (so the decoder can't peek at future tokens), and you get the **transformer** — a conditional generative model that powers ChatGPT, Stable Diffusion's text branch, Whisper, and most modern sequence and vision models. Generation is **autoregressive**: one token at a time, each conditioned on everything before it.*

## Where we left off

Week 9 built the language-modelling stack: $P(W) = \prod P(w_i \mid w_{<i})$, the [[n-gram-language-model|N-gram approximation]], and the move to neural language models — [[recurrent-neural-network|RNNs]] and [[lstm|LSTMs]] for sequence processing.

Two structural problems pushed the field beyond LSTMs:

1. **Sequential bottleneck.** Each step depends on the previous, so training cannot be parallelised across the sequence. GPUs sit half-idle.
2. **Long-range forgetting.** Information has to survive many recurrent updates to influence a distant downstream token; gradients vanish along the chain. LSTMs help but don't fully solve this.

> [!info] ASIDE — Why CNNs aren't the answer either
> One natural thought: replace recurrence with [[convolution|convolution]]. CNN-based language models (PixelCNN-style or WaveNet-style) avoid the sequential bottleneck — every position is computed in parallel. But each layer's receptive field is bounded by its kernel size, so to cover a sentence of length $n$ you need $\log n$ stacked layers (with dilation) or $n / k$ layers (without). For "The black cat that sleepily sat on the mat that its owner had bought yesterday" the verb-subject link spans the entire clause; a shallow CNN simply can't see across that gap. Self-attention does it in one layer, regardless of distance.

## Self-attention as soft dictionary lookup

Self-attention starts from a familiar idea: a **key-value lookup**. A Python dictionary or JSON object stores key-value pairs; a query is matched against keys, and the corresponding value is returned:

```
{ "Date of birth": "May 5th 2000",  ... }
query "Date of birth" → exact-match key → return "May 5th 2000"
```

Mathematically, $z = \sum_j \mathbb{1}(q = k_j)\, v_j$ — a hard exact match. But indicator functions are not differentiable, so this can't be trained with gradient descent.

The fix: **relax** the hard match into a soft, similarity-based one. Replace the indicator with a softmaxed dot-product score:

$$z = \sum_j \text{softmax}_j\!\left(\frac{q^\top k_j}{\sqrt{d_k}}\right) v_j$$

Now every key contributes, weighted by how aligned it is with the query. Differentiable, content-addressed, fully neural-network-compatible.

In **self-attention**, every token in a sequence makes its own query, key, and value via three learned projections — $Q = X W^Q$, $K = X W^K$, $V = X W^V$. Then the canonical formula is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Every token gets to "look at" every other token in one step, weighted by content similarity. See [[self-attention]] for the full derivation, the worked example, and the comparison to a standard MLP layer (key takeaway: self-attention's weight matrix is *dynamic*, computed from the input).

> [!tip] TIP — Multi-head attention as ensemble
> A single attention layer learns one attention pattern per query — but real sentences have many simultaneous relational structures (syntactic, coreferential, positional). [[multi-head-attention|Multi-head attention]] runs $h$ attention computations in parallel, each with its own learned projections, then concatenates and projects. Each head specialises in a different pattern. Same total parameter budget; far more expressive.

> [!question]- Why is the dot product divided by $\sqrt{d_k}$ before the softmax?
> Without scaling, dot-product magnitudes grow with $d_k$ — large dot products push softmax into saturation regions where the gradient is nearly zero, and one position ends up dominating the attention with all others getting zero. Dividing by $\sqrt{d_k}$ keeps the score magnitudes roughly constant regardless of model dimension, so gradient signal flows properly during training. (The exact $\sqrt{d_k}$ comes from assuming approximately unit-variance random vectors — their dot product has variance $d_k$.)

## The transformer architecture

The [[transformer]] (Vaswani et al. 2017, *"Attention Is All You Need"*) is the encoder-decoder neural architecture built around stacks of attention. Two halves with subtly different connectivity:

**Encoder** — stack of $N$ identical blocks, each with multi-head self-attention + position-wise feed-forward. Runs *once* on the input, produces a sequence of contextualised vectors called the **latent code**.

**Decoder** — stack of $N$ identical blocks, each with three sub-layers: *masked* multi-head self-attention (over the decoder's own running output), *cross-attention* with $Q$ from the decoder and $K, V$ from the encoder's latent code, and a position-wise feed-forward. Runs *repeatedly*, generating one token per step.

[[cross-attention|Cross-attention]] is the bridge. It's the *only* layer where information from the encoder enters the decoder. Same scaled-dot-product machinery as self-attention — but with the queries coming from one sequence and the keys/values from another.

> [!info] ASIDE — Why "executed once" and "executed repeatedly"
> The encoder runs once at the start and its output is cached. The decoder iterates: at each step, the decoder produces $Q$ from its growing hidden state, then does cross-attention against the *fixed* encoder $K, V$. This asymmetry is a core efficiency win — the encoder cost is paid once per input, regardless of how long the output is.

Because attention has no built-in notion of order, the transformer **adds a position vector to every token's embedding** before the first attention layer. See [[positional-encoding]]. Without this, the encoder would treat its input as a *set* and produce identical outputs for "Dog bites man" and "Man bites dog".

> [!question]- A friend builds a transformer encoder-decoder for English-to-French translation but forgets the positional encoding. Training loss looks fine for a while, but evaluation on test sentences is bizarre — the model produces output sentences that contain the right words but in nonsensical orders. What's happening?
> Without positional encoding, self-attention treats both the input and the decoder's outputs as sets, not sequences. The encoder produces the same latent code for any permutation of the input, and the decoder has no way to track which output position it's currently generating. The model can still pick up word identity (and therefore translate the *vocabulary* roughly correctly) but has no information about word order. Adding sinusoidal or learned positional encodings to the input embeddings fixes this immediately.

## Generation as autoregression

The decoder produces output sequences via [[autoregressive-model|autoregressive]] sampling. Mathematically, this is just the chain rule of probability applied to the joint distribution over a sequence:

$$p_\theta(\mathbf{x}) = \prod_{i=1}^{n} p_\theta(x_i \mid x_1, \ldots, x_{i-1})$$

Each conditional $p_\theta(x_i \mid x_{<i})$ is parameterised by the network. To generate, sample $x_1$, feed it back in, sample $x_2$, repeat — one token per network forward pass.

This factorisation isn't unique to language: WaveNet uses it for raw audio (one sample at a time, with dilated causal convolutions), PixelRNN/PixelCNN uses it for images (one pixel at a time in raster scan). Same idea, different domain.

> [!tip] TIP — The fundamental asymmetry: parallel train, sequential generate
> At training, the *full* target sequence is known, so all $n$ next-token predictions can be computed in **one** parallel forward pass (this is "teacher forcing", enforced by masked attention preventing future leakage). At generation, each next-token sample depends on the previous sample, so generation requires $n$ **sequential** forward passes. This is why training a 7B-parameter LLM is feasible but generating 1000 tokens from it takes seconds rather than milliseconds.

> [!question]- Why is masked attention required in the decoder during training?
> During training, the entire ground-truth target sequence is fed into the decoder in parallel, so the model can predict all $n$ next tokens in one forward pass. Without masking, position $i$'s attention could see positions $j > i$ — the *answer* — and the model would just copy the next token from its input rather than predict it. Masked attention sets the attention scores for $j > i$ to $-\infty$ before the softmax, structurally preventing future leakage. At training the model genuinely has to predict each next token from only the past, mirroring what it'll have to do at generation.

## Training and generation: putting it together

For machine translation ("Je suis étudiant" → "I am a student"):

**Training step**:

1. Encoder forward pass on "Je suis étudiant" → latent code (3 vectors).
2. Decoder forward pass on `<start> I am a student` (target shifted right by one) — masked self-attention prevents looking ahead; cross-attention pulls relevant content from the encoder.
3. Linear + softmax at every output position → distribution over vocabulary.
4. Cross-entropy loss between each predicted distribution and the true next token: predict "I" at position 0, "am" at position 1, "a" at position 2, "student" at position 3.
5. Backprop, update weights.

**Generation**:

1. Encoder forward pass on "Je suis étudiant" → cached latent code.
2. Decoder runs at position 0 with input `<start>` → softmax → sample → "I".
3. Decoder runs at position 1 with input `<start> I` → softmax → sample → "am".
4. Decoder runs at position 2 with input `<start> I am` → softmax → sample → "a".
5. Decoder runs at position 3 with input `<start> I am a` → softmax → sample → "student".
6. Continue until end-of-sequence sampled or max length reached.

Each generation step is one full decoder forward pass. The encoder is never re-run. The decoder's self-attention $K, V$ from previous positions can be cached (the famous "KV cache") so each step costs $O(n)$ rather than recomputing all $n^2$ attention scores from scratch.

## How transformers became universal

The original transformer was an encoder-decoder for translation. The architecture proved astonishingly modular — splitting it into halves gives two important variants:

| Variant | Use | Examples |
|---|---|---|
| **Encoder-only** | Build representations of input | BERT, RoBERTa, sentence encoders |
| **Decoder-only** | Autoregressive generation | GPT, LLaMA, ChatGPT, Claude |
| **Encoder-decoder** | Sequence-to-sequence | T5, BART, Whisper, original transformer |

The decoder-only variant — pure causal language modelling — is the dominant LLM architecture today. No encoder, no cross-attention, just a stack of masked-self-attention + feed-forward, trained on next-token prediction over web-scale text.

The same architecture also works for images (Vision Transformer / ViT — split image into patches, treat each patch as a token), audio (Whisper), code, and multimodal data. *One primitive, many domains.*

## Concepts introduced this week

- [[self-attention]] — the core operation: scaled dot-product attention with query/key/value triples per token
- [[multi-head-attention]] — running self-attention with $h$ parallel heads for relational specialisation
- [[cross-attention]] — bridge between encoder and decoder ($Q$ from one, $K, V$ from another)
- [[positional-encoding]] — required addition because attention is permutation-invariant
- [[autoregressive-model]] — chain-rule factorisation $p(\mathbf{x}) = \prod p(x_i \mid x_{<i})$; basis for transformer LMs, WaveNet, PixelRNN
- [[transformer]] — the architecture itself: encoder + decoder + latent-code bridge

## Connections

- **Builds on** [[week-09]]: language modelling as $P(W) = \prod P(w_i \mid w_{<i})$ — the autoregressive framing was already there in the n-gram and RNN context. Transformers are the natural successor that fixes RNNs' bottlenecks.
- **Builds on** [[recurrent-neural-network]] and [[lstm]]: transformers replace the recurrent computation with self-attention, removing the sequential bottleneck and the long-range forgetting.
- **Builds on** [[convolution]]: causal convolutions in WaveNet are conceptually similar to masked self-attention — both are ways to enforce "only attend to the past".
- **Builds on** [[residual-connection]] and [[normalization]]: every transformer sub-layer is wrapped in `Add & Norm` — residual connection plus [[normalization|layer normalisation]]. Without these, training the deep stack would be unstable.
- **Builds on** [[word-embedding]]: transformer inputs are token embeddings, learned end-to-end as part of the model. Positional encoding is *added* to the embedding before the first attention layer.
- **Builds on** [[conditional-generative-model]]: an encoder-decoder transformer is the canonical conditional generative model for sequences — $p_\theta(y \mid x)$ is the decoder's distribution over output sequences given the encoded input.
- **Connects to** [[latent-diffusion-model]]: cross-attention is what conditions Stable Diffusion's U-Net on the text encoder — a transformer block embedded in a diffusion model.

## Open questions

- The transformer's $O(n^2)$ attention complexity is a fundamental scaling limit. Many approximate-attention variants exist (Longformer, Performer, FlashAttention's IO-aware optimisations) — none was covered in lecture. Worth knowing they exist, especially for long-document or long-context settings.
- Training tricks that make modern LLMs work — RMSNorm vs LayerNorm, rotary positional embeddings (RoPE), grouped-query attention, mixture-of-experts — are extensions of the basic architecture covered here. The fundamentals don't change; the engineering improves.
- The decoder-only architecture (GPT family) is structurally simpler than the encoder-decoder original — it's just a stack of masked self-attention + feed-forward. Why has decoder-only "won" for general LLMs while encoder-decoder remains dominant for translation and speech-to-text? The standard answer is "decoder-only is more flexible because any task can be cast as text-to-text generation"; the more nuanced answer involves training data, scaling efficiency, and instruction-tuning effectiveness — beyond this module's scope.
