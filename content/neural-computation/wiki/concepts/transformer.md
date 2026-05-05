---
name: transformer
description: The encoder-decoder neural architecture introduced in "Attention Is All You Need" (Vaswani et al. 2017). Replaces recurrence and convolution entirely with stacks of multi-head self-attention and position-wise feed-forward layers. Encoder runs once on the input, producing a latent code; decoder runs autoregressively, attending to its own running output (masked self-attention) and to the encoder's latent code (cross-attention). Foundation of GPT, BERT, T5, ChatGPT, Stable Diffusion's text branch, and most modern sequence and vision models.
type: concept
sources:
  - raw/week-11/w11-slides.pdf
status: stable
updated: 2026-05-04
---

*One architecture, two modes. The encoder ingests an input sequence in parallel and produces a latent code. The decoder generates an output sequence one token at a time, each step querying both its own past outputs and the encoder's code. Both halves are stacks of the same primitive — multi-head attention plus a position-wise MLP — held together by residual connections and layer norms. No recurrence, no convolution, just attention.*

## What problem the transformer solves

[[recurrent-neural-network|RNNs]] and [[lstm|LSTMs]] handle sequences by stepping through them one token at a time, propagating a hidden state. Two structural problems:

1. **Sequential bottleneck.** Each step depends on the previous, so training cannot be parallelised across the sequence.
2. **Long-range forgetting.** Information has to survive many recurrent updates to influence a distant downstream token; gradients vanish or explode along the chain.

[[convolution|CNNs]] over text avoid the sequential bottleneck but limit each token's view to a fixed receptive field — bridging long-range dependencies (subject and verb separated by a clause) needs deep stacks.

The transformer rejects both crutches. Every output position can directly attend to every input position in *one* layer, regardless of distance. Training is fully parallel across positions. Long-range dependencies are no harder than short-range ones.

## High-level structure

A transformer consists of two stacks:

- **Encoder** — a stack of $N$ identical blocks, each containing multi-head self-attention and a position-wise feed-forward network. Processes the input sequence in parallel and produces a sequence of hidden vectors of the same length, called the **latent code**.
- **Decoder** — a stack of $N$ identical blocks, each containing *masked* multi-head self-attention, *cross-attention* over the encoder's latent code, and a position-wise feed-forward network. Generates the output sequence autoregressively.

Each block is wrapped in residual connections and layer normalisation. The architecture diagram on the slides shows this clearly: the encoder is a single column of "Multi-Head Attention → Feed Forward" repeated $N$ times; the decoder is a column of "Masked Multi-Head Attention → Multi-Head Attention (cross) → Feed Forward" repeated $N$ times, with arrows from encoder to decoder feeding the cross-attention's keys and values.

> [!info] ASIDE — "Attention Is All You Need"
> Vaswani et al. titled their 2017 paper *Attention Is All You Need* — a deliberately punchy claim that *no* recurrence and *no* convolution are necessary, just stacks of attention and feed-forward. The architecture won state-of-the-art on machine translation immediately, then proved astonishingly general: language modelling (GPT), masked language modelling (BERT), text-to-text (T5), image classification (ViT), audio (Whisper), and image generation (Stable Diffusion's text branch + DiT) all use the same primitive. It is, by some distance, the most influential neural-network architecture of the last decade.

## The encoder

Inputs a sequence of tokens. Pipeline at each position $i$:

1. **Token embedding** — look up an embedding vector for the token (see [[word-embedding]]).
2. **Add positional encoding** — add a position-dependent vector so the layer can use order ([[positional-encoding]]). Without this, attention is permutation-invariant.
3. **$N$ encoder blocks** in sequence. Each block:
   - Multi-head self-attention over the full sequence (every token attends to every other).
   - Add & Norm: residual connection ($+\text{input}$) and layer normalisation.
   - Position-wise feed-forward MLP applied independently to each token.
   - Add & Norm again.

After $N$ blocks, the encoder outputs a sequence of hidden vectors of the same length as the input. Each vector is a contextualised representation of one token, having mixed in information from every other token via the attention layers. This sequence is the **latent code**.

> [!tip] TIP — Why "executed once"
> The encoder runs *once* per input sequence. Its latent code is then reused by the decoder at every output time step. This caching is one of the practical efficiency wins of the architecture: the encoder cost is paid once, the decoder iterates against fixed encoder output.

## The decoder

Generates the output sequence autoregressively. At each output time step:

1. **Output token embedding** — embed the previously generated token (or the start token at step 0).
2. **Add positional encoding** — same machinery as the encoder, with positions starting from 0 in the decoder's own sequence.
3. **$N$ decoder blocks** in sequence. Each block has *three* sub-layers:
   - **Masked multi-head self-attention** over the decoder's own running output. The mask is lower-triangular: position $i$ can only attend to positions $\leq i$, so the model can never peek at future tokens. This is what makes the decoder autoregressive.
   - **Cross-attention** with $Q$ from the decoder, $K, V$ from the encoder's latent code ([[cross-attention]]). This is the *only* layer where information from the input sequence enters the decoder.
   - Position-wise feed-forward MLP.
   - Each sub-layer wrapped in Add & Norm.

4. After the $N$ decoder blocks, project each output position through a final **linear layer** to vocabulary logits, then **softmax** to produce a probability distribution over the next token.

5. **Sample** from this distribution (or take argmax for greedy decoding) to produce the next token. Append it, advance one step, repeat.

Generation continues until an end-of-sequence token is sampled or a maximum length is reached.

## The latent-code bridge

The encoder and decoder communicate through exactly one channel: the encoder's output sequence enters every decoder block's cross-attention layer as the source of $K$ and $V$. The decoder's $Q$ comes from its own hidden state.

This means:

- The **size of the latent code** equals the input sequence length (one vector per encoder input position).
- The **content** of the latent code is whatever the encoder learned to extract.
- The decoder can attend to *any subset* of the encoder's positions per output step — and in fact attends to different parts at different output steps. For translation, when generating the German word "Katze", cross-attention typically peaks on the encoder's representation of the English word "cat".

The slide labels this clearly: the encoder is "executed once" producing the latent code, and the decoder is "executed repeatedly", each iteration consulting the same fixed latent code via cross-attention.

## Training: parallel forward, masked attention, teacher forcing

During training, both the input (e.g. "Je suis étudiant") *and* the target output (e.g. "I am a student") are available. The training step:

1. Encoder forward pass on the input — produces the latent code in parallel across all input positions.
2. Decoder forward pass on the target output, **shifted right by one** (so the decoder's input at position $i$ is the target's token at position $i-1$). Masked self-attention prevents the decoder from peeking at positions $\geq i$ — the model genuinely has to predict each next token from only the previous ones.
3. Linear + softmax produces a distribution over the vocabulary at each output position.
4. Compute cross-entropy loss between each predicted distribution and the corresponding *true* next token.
5. Sum the per-position losses, backpropagate, update.

All output positions are predicted **in parallel** in one decoder forward pass. This is **teacher forcing**: the decoder is given the ground-truth previous tokens at each position, regardless of what it would have generated. Combined with masked attention, this lets the entire training pass happen in $O(N \cdot n^2)$ time on a GPU rather than $n$ sequential steps.

This is the key efficiency advantage over training an RNN — every position in every layer gets its gradient signal in one forward/backward pass.

## Generation: sequential, one token at a time

At inference, the target output is *not* available — that's what we're trying to generate. So the parallel teacher-forcing trick doesn't apply. Generation proceeds:

1. Run encoder once on the input → cached latent code.
2. Start decoder with the special `<start>` token at position 0.
3. Run decoder forward → softmax → distribution over next token at position 0.
4. Sample (or argmax) → first output token (e.g. "I").
5. Append it. Run decoder again with `[<start>, I]` → distribution at position 1 → sample → "am".
6. Continue until end-of-sequence or max length.

Each step requires one full decoder forward pass. The encoder is run once at the start; cross-attention's $K, V$ are cached. The decoder's *self-attention* keys and values from previous positions can also be cached (the famous "KV cache") so that the per-step cost stays $O(n)$ rather than $O(n^2)$.

This is the [[autoregressive-model|autoregressive]] generation pattern. It is fundamentally serial — you cannot parallelise across output positions because each one depends on what was sampled at the previous step.

## Why the transformer dominated

A short list of properties that put it ahead of RNNs and CNNs:

- **Parallelism at training time.** All input positions processed at once; all decoder positions predicted at once via teacher forcing. RNNs are strictly sequential.
- **Constant path length.** Information flows from any token to any other in a single attention hop. RNNs have $O(n)$ path length; CNNs have $O(\log n)$ at best with stacked dilations.
- **Scaling laws.** Empirically, transformer performance scales smoothly with parameters, data, and compute over many orders of magnitude. RNNs and CNNs hit walls earlier.
- **Architectural reusability.** Same primitive works for text (GPT, BERT, T5), images (ViT, DiT), audio (Whisper, AudioLM), code, and multimodal data. RNNs and CNNs are domain-specific.
- **Dynamic weights.** Self-attention's input-dependent weighting is more expressive per parameter than a fixed MLP weight matrix. See the discussion in [[self-attention#self-attention-vs-feed-forward]].

## Variants and lineage

The original transformer is encoder-decoder. Two simpler variants split it:

- **Encoder-only** (BERT, RoBERTa) — keeps the encoder, drops the decoder. Useful for classification, embedding, retrieval tasks where you need a representation, not a generation. Trained with masked language modelling (predict randomly-masked input tokens given the rest).
- **Decoder-only** (GPT, LLaMA, Claude, ChatGPT) — keeps the decoder, drops the encoder. The decoder's self-attention is masked, and there's no cross-attention to anything. Trained with next-token prediction. The dominant architecture for modern LLMs.
- **Encoder-decoder** (original transformer, T5, BART, Whisper) — both halves. Useful for sequence-to-sequence tasks (translation, summarisation, speech-to-text) where input and output are different modalities or sequences.

For all three, the building blocks — multi-head attention, position-wise feed-forward, residual connections, layer norm, positional encodings — are the same. Only the connectivity differs.

## Related

- [[self-attention]] — the core operation of every transformer block
- [[multi-head-attention]] — how each block runs $h$ parallel attention computations
- [[cross-attention]] — the encoder-decoder bridge, used only in encoder-decoder transformers
- [[positional-encoding]] — required because attention is order-agnostic
- [[autoregressive-model]] — the framing for transformer language models and decoders
- [[residual-connection]] — the "Add" in "Add & Norm"
- [[normalization]] — the "Norm" in "Add & Norm" (specifically layer norm)
- [[language-model]] — the dominant application of decoder-only transformers
- [[conditional-generative-model]] — encoder-decoder transformers are the canonical conditional sequence-to-sequence GMs
- [[latent-diffusion-model]] — uses transformer blocks with cross-attention for text conditioning
- [[recurrent-neural-network]] / [[lstm]] — what transformers replaced

## Active Recall

> [!question]- Sketch the encoder-decoder transformer at a high level. What are the main sub-layers in an encoder block? In a decoder block?
> An **encoder block** has two sub-layers, each followed by Add & Norm: (1) multi-head self-attention over the full input sequence, (2) position-wise feed-forward MLP. Stack $N$ of these. A **decoder block** has three sub-layers, each followed by Add & Norm: (1) masked multi-head self-attention over the decoder's own previous outputs, (2) multi-head cross-attention with $Q$ from the decoder and $K, V$ from the encoder's latent code, (3) position-wise feed-forward MLP. Stack $N$ of these. Plus token embedding + positional encoding at the start of each stack and a final linear+softmax over vocabulary at the decoder's output.

> [!question]- What does it mean that the encoder is "executed once" and the decoder is "executed repeatedly"?
> The encoder processes the entire input sequence in *one* forward pass, producing a fixed latent code. The decoder then runs autoregressively — one forward pass per output token — and consults the encoder's latent code via cross-attention at every step. The encoder's $K$ and $V$ projections of the latent code can be cached and reused across all decoder steps; only the decoder's $Q$ depends on the running output and is recomputed each step.

> [!question]- During training, the decoder predicts an entire output sequence in a single forward pass. How is this possible if generation at inference is strictly sequential?
> Because at training, the ground-truth target sequence is known. The decoder is given the target sequence shifted right (each position $i$ sees the true token at $i-1$) and uses **masked self-attention** to ensure position $i$ never peeks at $j \geq i$. So all $n$ positions can be evaluated in parallel in one pass — each predicting its next token from the structurally-enforced "past". This is **teacher forcing**. At inference, the target isn't known, so each step depends on the previous sample and must run sequentially.

> [!question]- Where does the encoder's information enter the decoder, exactly?
> Through the **cross-attention** sub-layer of every decoder block. In cross-attention, the decoder's hidden state provides queries; the encoder's latent code provides keys and values. This is the *only* path by which information from the input reaches the decoder. Without it, the decoder is a stand-alone language model with no knowledge of what was input. Removing cross-attention turns the encoder-decoder transformer into a decoder-only language model (which is exactly the GPT family).

> [!question]- Compare an encoder-only (BERT-style), decoder-only (GPT-style), and encoder-decoder (T5-style) transformer. What is each best suited to?
>
> - **Encoder-only (BERT)**: produces contextualised representations of the input. Trained with masked language modelling. Best for classification, retrieval, embedding, NER — anywhere you need to *understand* but not *generate*.
> - **Decoder-only (GPT)**: autoregressive next-token predictor. No encoder, no cross-attention. Trained with causal language modelling. Best for free-form generation, instruction-following, chat — currently the dominant LLM architecture.
> - **Encoder-decoder (T5, original transformer)**: sequence-to-sequence with cross-attention bridging the two halves. Best for tasks with a clear distinct input and output sequence — translation, summarisation, speech-to-text, structured output.
>
> Same primitives in all three; only the connectivity (encoder/decoder/both, mask or no mask) differs.

> [!question]- Why does positional encoding need to be added to the inputs of *both* the encoder and the decoder?
> Because attention is permutation-invariant in both halves. Without positional encoding, the encoder would produce identical outputs for "Dog bites man" and "Man bites dog" (just in a different order), and the decoder would produce identical predictions regardless of the order of its previous outputs. Both halves need explicit position information injected into their input embeddings before the first attention layer to be able to use word order at all.
