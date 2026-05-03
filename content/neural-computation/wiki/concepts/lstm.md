---
name: lstm
description: A gated recurrent architecture (Hochreiter & Schmidhuber 1997) that addresses the vanishing-gradient problem of vanilla RNNs by adding a separate "cell state" alongside the hidden state and explicit input/forget/output gates that decide what to write to, retain in, and read from the cell. Information can flow through the cell with minimal transformation across many time steps, so gradients propagate cleanly to long-range dependencies. State-of-the-art for sequence modelling between 2014 and the rise of Transformers; still used where the recurrent structure or low-memory inference is needed.
type: concept
sources:
  - raw/week-09/w09-l20-transcript.txt
  - raw/week-09/w09-slides.pdf
status: stable
updated: 2026-04-27
---

*The fix for vanilla [[recurrent-neural-network|RNNs]]' vanishing-gradient problem. Where an RNN's hidden state is overwritten at every step (so signals from far back get diluted by the chain of multiplications), an LSTM maintains a separate "cell state" with explicit gates that decide what to write, what to keep, and what to read out. The crucial property: information can pass through the cell across many steps with minimal modification, so gradients can flow back through long chains without vanishing. This unlocked sequence modelling at lengths where vanilla RNNs failed — translation of long sentences, language modelling with paragraph-scale context — and became the dominant architecture for sequence tasks until Transformers replaced it around 2017.*

## The vanilla RNN failure mode (recap)

A [[recurrent-neural-network|vanilla RNN]] updates a single hidden state at each step:

$$h_t = \sigma(W_h h_{t-1} + W_e e_t + b)$$

The same matrix $W_h$ is multiplied at every step. After $T$ steps, the gradient with respect to early inputs has been multiplied by $W_h^T$ — exponential decay (vanishing) or growth (exploding). Effective context is ~10 steps. Long-range dependencies (subject-verb agreement across a clause, plot threads in a paragraph, etc.) cannot be learned.

## The LSTM idea: a separate, lightly-touched memory channel

Add a **cell state** $c_t$ that flows alongside the hidden state $h_t$. The cell state is updated by a *small* additive correction at each step rather than being completely rewritten:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

where $f_t, i_t \in [0, 1]^d$ are gates (element-wise) deciding what to forget and what new content to add. If $f_t \approx 1$ and $i_t \approx 0$ at a given dimension, the cell state at that dimension passes through unchanged: $c_t \approx c_{t-1}$. This is the "highway" through which information can travel many steps without distortion.

Three gates control the flow:

- **Forget gate** $f_t = \sigma(W_f [h_{t-1}, e_t] + b_f)$ — what to keep from the previous cell state. Element-wise sigmoid; near 1 = keep, near 0 = forget.
- **Input gate** $i_t = \sigma(W_i [h_{t-1}, e_t] + b_i)$ — how much of the new candidate content to write into the cell.
- **Output gate** $o_t = \sigma(W_o [h_{t-1}, e_t] + b_o)$ — how much of the (updated) cell state to expose as the new hidden state.

The candidate content is a fresh $\tanh$-transformed combination of input and previous hidden state:

$$\tilde{c}_t = \tanh(W_c [h_{t-1}, e_t] + b_c)$$

And the new hidden state — what's exposed to the rest of the network — is:

$$h_t = o_t \odot \tanh(c_t)$$

So the model learns *separately*: how much to forget, how much new information to add, what to expose to downstream layers. All gates are differentiable functions of the current input and previous hidden state.

> [!info] ASIDE — "You're not providing instructions, you're creating capacity"
> The lecturer's framing of LSTMs as design philosophy: *"When you create the architecture, you are not providing instruction. You are creating capacity. You're providing tools for your internal network to use, and you hope that it would use that to maximise the accuracy."* The gates aren't told what to remember; they're given the *ability* to remember selectively, and trained end-to-end. The lecturer's example: an LSTM trained on Arabic might learn that a particular cell dimension stores grammatical gender, because that information is useful several words later when conjugating verbs. Nobody tells it to do this; the structure permits it and the loss rewards it.

## Why this fixes vanishing gradients

Backpropagation through the cell-state update encounters mostly identity-like flows when $f_t \approx 1$. The gradient at dimension $i$ at time $t$ flowing back to time $t-1$ goes through multiplication by $f_t^{(i)}$, which is close to 1 if the model has learned to remember at this position. So gradients don't decay to zero across time — they pass through cleanly via the "carry" mechanism.

This is the same architectural trick that makes [[residual-connection|residual connections]] work in deep CNNs: provide an additive shortcut so gradients have a path that bypasses many layers of multiplication. LSTMs' cell state is the residual connection of the RNN era.

## Stacked and bidirectional LSTMs

LSTMs gain capacity (and quality) from depth and direction:

- **Stacked LSTM** — multiple LSTM layers, each consuming the hidden states of the layer below. Standard for sequence-to-sequence learning (Sutskever, Vinyals, Le 2014).
- **Bidirectional LSTM** — run one LSTM forwards and another backwards, concatenate their hidden states at each position. Each position's representation now depends on both left and right context. Cannot be used for autoregressive generation (it'd need future tokens at training time) but excellent for representation/classification.
- **Stacked Bidirectional LSTM** — the architecture behind **ELMo** (Peters et al. 2018), which produced the first widely-used contextualized [[word-embedding|word embeddings]] and demonstrated that representations from a deep recurrent LM transfer to many downstream NLP tasks.

## Practical limitations (the path to Transformers)

LSTMs solved vanishing gradients but inherited two problems from RNNs and added one of their own:

- **Sequential — not parallelisable.** Like any recurrent model, computing $h_t$ requires $h_{t-1}$. Training and inference are linear in sequence length, with no GPU-friendly parallelism across positions. This is the dominant practical complaint.
- **Computationally expensive per step.** Four gate matrices instead of one, plus several non-linearities, plus the cell-state update. Per-token cost is several times that of a vanilla RNN.
- **Long-range dependencies are *better* but not *unlimited*.** Empirically, LSTMs handle context lengths in the hundreds. Past that, even gating doesn't fully prevent dilution — gradients can still attenuate over very long chains, just much more slowly than vanilla RNNs.

These limitations motivated the [[language-model|Transformer architecture]] (week 11): replace recurrence entirely with self-attention, which is parallelisable across positions and propagates information directly between any two positions in a single layer. After ~2017, Transformers progressively displaced LSTMs across NLP tasks.

## Where LSTMs still appear

- **Memory-constrained inference.** A Transformer's KV cache scales linearly with sequence length; an LSTM's hidden state is fixed-size. For streaming or low-memory scenarios, LSTMs can still be competitive.
- **Speech recognition pipelines** still often use LSTMs in front-end feature extractors.
- **Conceptual ancestor of state-space models** (S4, Mamba, etc.) — modern research on sub-quadratic alternatives to attention often returns to recurrent-style architectures with engineered cell-state dynamics.

## Connections

- **Refines** [[recurrent-neural-network|RNNs]] — same recurrent skeleton with gated additive memory replacing the simple recurrence.
- **Solves the same problem as** [[residual-connection|ResNet's residual connections]] — both add identity-like shortcuts so gradients can flow across many transformations.
- **Provides architecture for** [[language-model|stacked-LSTM and BiLSTM language models]] (e.g., ELMo) — state-of-the-art for sequence modelling between roughly 2014 and 2018.
- **Largely superseded by Transformers** (week 11) — attention removes the recurrence, enabling parallel training and longer effective context.
