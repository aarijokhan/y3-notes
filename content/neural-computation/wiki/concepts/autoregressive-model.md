---
name: autoregressive-model
description: A generative model that factorises a high-dimensional distribution into a product of one-step conditional distributions over its components, $p_\theta(\mathbf{x}) = \prod_{i=1}^{n} p_\theta(x_i \mid x_1, \ldots, x_{i-1})$. Every component is generated *one at a time*, conditioned on all previous ones. WaveNet (audio), PixelRNN/PixelCNN (images), and GPT-style language models are all autoregressive. The factorisation makes training fast (parallel teacher forcing) but generation slow (one component at a time, sequentially).
type: concept
sources:
  - raw/week-11/w11-slides.pdf
status: stable
updated: 2026-05-04
---

*Modelling a high-dimensional joint distribution is hard. Modelling a single one-dimensional conditional distribution given the past is easy. Autoregressive models exploit the chain rule of probability to turn the hard problem into a sequence of easy ones — and recover the joint by multiplying.*

## The factorisation

For any random vector $\mathbf{x} = (x_1, x_2, \ldots, x_n)$, the chain rule of probability gives:

$$p(\mathbf{x}) = p(x_1) \cdot p(x_2 \mid x_1) \cdot p(x_3 \mid x_1, x_2) \cdot \ldots \cdot p(x_n \mid x_1, \ldots, x_{n-1})$$

$$= \prod_{i=1}^{n} p(x_i \mid x_1, \ldots, x_{i-1})$$

This identity holds for *any* joint distribution, regardless of the dimensionality or structure. It's not an approximation. The autoregressive idea is to **parameterise each conditional with a neural network**:

$$p_\theta(\mathbf{x}) = \prod_{i=1}^{n} p_\theta(x_i \mid x_1, \ldots, x_{i-1})$$

A single shared network with parameters $\theta$ takes the previous components as input and outputs the distribution over the next component $x_i$. Train by maximum likelihood, and you have a tractable density estimate of the full joint.

> [!info] ASIDE — Why this is a *huge* simplification
> Directly modelling $p_\theta(\mathbf{x})$ for high-dimensional $\mathbf{x}$ (e.g. a $1024 \times 1024$ image, or a 10-second audio waveform sampled at 16 kHz) is intractable — you'd have to specify a probability density over $10^6$- or $10^5$-dimensional space. Autoregressive factorisation reduces this to predicting a *one-dimensional* distribution at a time (just $x_i$), conditioned on what came before. Each step is something a network can plausibly do; the chain rule guarantees the product equals the joint.

## What the network outputs

For each step, the network outputs a *distribution* over $x_i$ (not just a point estimate). How that distribution is parameterised depends on the data:

| Data type | Output distribution | Typical implementation |
|---|---|---|
| Discrete (text, classes) | Categorical over vocab | Softmax over vocab logits |
| Discrete-valued audio (8-bit) | Categorical over 256 levels | Softmax over 256 logits |
| Continuous | Gaussian, mixture, or histogram | Mean+variance heads, or discretised bins |

Slide 57 shows the histogram view: the model can output a discretised distribution over possible values — a histogram-like vector over the discrete bin index — or a parametric form like a Gaussian. Either works. Sampling at generation time is then drawing from that output distribution.

## Examples

### WaveNet — audio (van den Oord et al. 2016)

Raw audio is a 1D sequence of pressure samples (16 kHz means 16{,}000 numbers per second). WaveNet models $p_\theta(x_i \mid x_{1:i-1})$ for each sample with a stack of **dilated causal convolutions**:

- *Causal*: filter only sees past samples, never future ones (just like masked attention).
- *Dilated*: each successive layer skips an exponentially growing number of timesteps (dilation 1, 2, 4, 8, …) so the receptive field grows exponentially with depth.

A WaveNet with $L$ layers of dilation $1, 2, \ldots, 2^{L-1}$ has receptive field $2^L$ samples — covers seconds of audio with a modest stack. Output is a softmax over discretised audio levels (typically 256 via $\mu$-law companding).

### PixelRNN / PixelCNN — images (van den Oord et al. 2016)

Raster-scan an image into a sequence: $x_1$ = top-left pixel, $x_2$ = next pixel along, …, $x_{n^2}$ = bottom-right pixel. Then model

$$p_\theta(\text{image}) = \prod_i p_\theta(x_i \mid x_1, \ldots, x_{i-1})$$

with a network that produces a categorical distribution over 256 intensities for each pixel given everything that came before in raster order. PixelRNN uses an RNN; PixelCNN uses *masked* convolutions that only see pixels above and to the left.

### Transformer-based language models (GPT family)

Same factorisation, with the conditional $p_\theta(x_i \mid x_{1:i-1})$ implemented by a transformer with [[self-attention#masked-self-attention-causal-attention|masked self-attention]]. Each token attends only to previous tokens; the final hidden state at each position is projected to a softmax over the vocabulary, giving the distribution over the next token.

GPT, the decoder side of the original transformer, and any "causal language model" you've heard of are all autoregressive in this sense. The architecture changes; the factorisation doesn't.

## Training: parallel and fast

Autoregressive models are trained by **maximum likelihood**: maximise

$$\sum_{\mathbf{x} \in \text{data}} \sum_{i=1}^{n} \log p_\theta(x_i \mid x_1, \ldots, x_{i-1})$$

Crucially, during training **the entire ground-truth sequence is known**. So all $n$ conditional predictions can be computed *in parallel* in a single forward pass — feed the full target sequence in, get all $n$ predictions out, compute the per-position cross-entropy losses, sum, backpropagate. This is **teacher forcing**: each position is given the ground-truth previous tokens as input, regardless of what the model would have generated.

For a transformer, masked self-attention enforces the "only look at previous tokens" constraint structurally — the lower-triangular mask ensures position $i$ never attends to positions $j > i$. So training is one forward pass: $n$ predictions and $n$ loss terms in $O(n^2)$ time (the attention cost), all parallelisable on GPUs.

## Generation: sequential and slow

At inference, the ground truth isn't available. To generate a sample:

1. Sample $x_1 \sim p_\theta(x_1)$.
2. Run the network with input $(x_1)$ to get $p_\theta(x_2 \mid x_1)$. Sample $x_2$.
3. Run the network with input $(x_1, x_2)$ to get $p_\theta(x_3 \mid x_1, x_2)$. Sample $x_3$.
4. ...
5. Run the network with input $(x_1, \ldots, x_{n-1})$ to get $p_\theta(x_n \mid \cdot)$. Sample $x_n$.

$n$ sequential network passes — *cannot* be parallelised, because each step depends on the previous sample. This is the **fundamental cost** of autoregressive models: training is fast, generation is slow.

For a 1024-token GPT generation, you do 1024 forward passes back-to-back. For a 10-second WaveNet sample at 16 kHz, you do 160{,}000 forward passes. This is why generative LLMs are heavily optimised at inference (KV-caching, speculative decoding, etc.) — the per-step cost dominates.

## The asymmetry, in a table

| | Training | Generation |
|---|---|---|
| Inputs available | Full ground-truth sequence | Only what's been generated so far |
| Parallelism | All $n$ positions in one pass | Strictly sequential, $n$ passes |
| Cost | $O(n^2)$ for transformer (one pass) | $O(n^2 \cdot n) = O(n^3)$ if naively recomputed; $O(n^2)$ with caching |
| What's being computed | Likelihood of the ground truth under the model | A new sample by composition |

The mismatch between training and generation modes is sometimes called *exposure bias* — at training the model always sees correct previous tokens; at generation it sees its own (potentially wrong) previous samples. Various decoding strategies and training tricks (scheduled sampling, etc.) address this.

## Why "ensure correct receptive field"

Slide 57 emphasises that autoregressive models must ensure the network only sees the *past*. Get this wrong and the model can cheat by peeking at the future during training, then fail catastrophically at generation when the future isn't available.

Three structural ways to enforce causality:

- **Masked attention** (transformers) — set future-position attention scores to $-\infty$.
- **Causal convolutions** (WaveNet) — pad asymmetrically so the filter's output at time $i$ depends only on times $\leq i$.
- **Masked convolutions** (PixelCNN) — zero out the kernel weights corresponding to future pixels in raster order.

All three implement the same constraint in different architectural shapes.

## Related

- [[generative-model]] — the broader family; autoregressive is one approach among GANs, VAEs, diffusion, normalising flows
- [[language-model]] — most modern language models are autoregressive (next-token prediction)
- [[transformer]] — the dominant architecture for autoregressive language modelling
- [[self-attention]] — masked self-attention enforces causality structurally
- [[convolution]] — causal/masked convolutions enforce causality in WaveNet and PixelCNN
- [[decoding-strategies]] — how to actually sample from the per-step distributions (greedy, beam, top-k, top-p, temperature)
- [[diffusion-model]] — a non-autoregressive generative alternative; trades sequential generation for iterative denoising

## Active Recall

> [!question]- Write the autoregressive factorisation of a joint distribution $p_\theta(\mathbf{x})$ for $\mathbf{x} = (x_1, \ldots, x_n)$ and explain why this is a useful framing.
> $p_\theta(\mathbf{x}) = \prod_{i=1}^{n} p_\theta(x_i \mid x_1, \ldots, x_{i-1})$. Each factor is a *one-dimensional* conditional distribution given the past — far easier to model than the high-dimensional joint directly. The chain rule of probability guarantees the product equals the true joint exactly (this is not an approximation). One neural network with parameters $\theta$ implements all conditionals.

> [!question]- Why are autoregressive models *fast to train but slow to generate*?
> At training, the full ground-truth sequence is available, so all $n$ conditional predictions can be computed in a single parallel forward pass (teacher forcing). At generation, each next-token prediction depends on the previously sampled token, which can only be computed *after* the previous step finishes. So generation requires $n$ sequential forward passes, fundamentally serial. This is why LLM inference is the bottleneck and KV-caching / speculative decoding exist.

> [!question]- A friend trains a transformer language model and notices that during training the model gets near-perfect accuracy by attending to position $i+1$ when predicting position $i$. What's gone wrong, and how should the architecture fix it?
> The friend forgot to apply causal **masking** to the self-attention. Without the lower-triangular mask, position $i$ can attend to *future* positions including $i+1$, which trivially gives away the answer. At training the model exploits this; at generation, future positions don't exist yet, so the model collapses. The fix is masked self-attention: set the attention scores for $j > i$ to $-\infty$ before softmax, so they contribute zero weight.

> [!question]- WaveNet uses dilated causal convolutions. What does each adjective mean and what does each contribute?
> **Causal**: the filter at output time $i$ only sees inputs at times $\leq i$ (no peeking into the future). Implemented by asymmetric padding so the filter is left-aligned. Necessary for autoregressive sampling. **Dilated**: each successive layer skips an exponentially growing number of timesteps (dilation 1, 2, 4, 8, …). The receptive field grows exponentially with depth, so a stack of $L$ layers covers $2^L$ timesteps without needing $L$ layers of size $2^L$. Together: long context window, no future leakage, modest compute.

> [!question]- Compare the autoregressive approach to a [[diffusion-model|diffusion model]] for image generation. What's traded off?
> Autoregressive (PixelRNN/CNN): sequential pixel-by-pixel generation; tractable likelihood; very slow generation ($O(\text{pixels})$ network passes); modest sample quality. Diffusion: iterative denoising over $T$ steps for the *whole* image at once; sample quality state-of-the-art; generation cost is $T$ network passes regardless of resolution; likelihood not directly tractable. Autoregressive trades *quality* and *parallelism* for *exact tractability*; diffusion trades *exactness* for *quality and parallelism per step*.
