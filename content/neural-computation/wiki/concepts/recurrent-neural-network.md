---
name: recurrent-neural-network
description: A neural architecture for sequence data that maintains a hidden state $h_t$ updated at each step from the current input and the previous hidden state — $h_t = \sigma(W_h h_{t-1} + W_e e_t + b)$. Output at each step is a function of the hidden state. Crucially, the same weight matrices $W_h, W_e$ are reused at every step, so a single set of parameters handles inputs of any length. Standard architecture for autoregressive language modelling before Transformers; suffers from vanishing/exploding gradients on long sequences and processes one token at a time (unparallelisable).
type: concept
sources:
  - raw/week-09/w09-l20-transcript.txt
  - raw/week-09/w09-slides.pdf
status: stable
updated: 2026-04-27
---

*An RNN is the architectural answer to "what if the input is a sequence of arbitrary length?" Feed-forward networks need a fixed-size input; convolutional networks can handle variable spatial extent but not variable depth in the temporal sense. The RNN trick: maintain a hidden state vector that summarises everything seen so far, and update it at each step using the same weights. The hidden state is the model's running memory; the weight reuse is what lets a single network process inputs of any length. This was the standard architecture for language modelling between feed-forward LMs (2003) and Transformers (2017) — and it shows up again as the conceptual ancestor of state-space models in modern research.*

## The setup

The input is a sequence $x_1, x_2, \ldots, x_T$ (e.g., a sentence's word embeddings — see [[word-embedding]]). The RNN maintains a **hidden state** $h_t \in \mathbb{R}^d$ that's updated at each step:

$$h_t = \sigma\!\left(W_h\, h_{t-1} + W_e\, e_t + b_1\right)$$

where:
- $e_t = E\, x_t$ is the embedding of the input token (lookup in embedding matrix $E$).
- $W_h \in \mathbb{R}^{d \times d}$ updates the hidden state from its previous value.
- $W_e \in \mathbb{R}^{d \times d_e}$ projects the new input.
- $\sigma$ is a non-linearity (tanh, sigmoid, ReLU).
- $h_0$ is an initial hidden state (often the zero vector or a learned parameter).

For each step (e.g., language modelling) the output distribution is computed from the hidden state:

$$\hat{y}_t = \mathrm{softmax}\!\left(U\, h_t + b_2\right) \in \mathbb{R}^{|V|}$$

where $U \in \mathbb{R}^{|V| \times d}$ is the output projection. For an autoregressive [[language-model|LM]] this is $\hat{y}_t = P(w_{t+1} \mid w_{1:t})$ — a distribution over the next word.

## The crucial point: weight sharing across time

A feed-forward network with $T$ inputs would need $T$ separate hidden layers, each with its own weights. Total parameters scale as $T \times d^2$. The RNN uses the **same** $W_h, W_e$ at every step:

$$h_1 = \sigma(W_h h_0 + W_e e_1 + b_1)$$
$$h_2 = \sigma(W_h h_1 + W_e e_2 + b_1)$$
$$h_3 = \sigma(W_h h_2 + W_e e_3 + b_1)$$
$$\ldots$$

A *single* set of parameters $\{W_h, W_e, b_1, U, b_2, E\}$ handles sequences of any length. This is what makes RNNs unbounded-context language models in principle (each step's hidden state is a function of all previous inputs through the chain of updates).

Compared to [[n-gram-language-model|n-gram LMs]] limited to context $N \le 5$, an RNN can in principle condition on the entire prefix — this is the reason RNNs improved perplexity dramatically when introduced.

> [!tip] TIP — Same weights = same operator
> The weight-sharing across time is the recurrent analogue of weight-sharing across spatial positions in [[convolutional-neural-network|CNNs]]. A CNN slides one filter across the image; an RNN slides one update operator across the sequence. The benefit in both cases: parameters don't scale with input size, and patterns learned at one position automatically transfer to others (translation/time-shift equivariance).

## Training: backpropagation through time

To train, unroll the RNN over the sequence (drawing each step explicitly), compute losses at each output step, and run standard backpropagation through the unrolled graph. This is called **backpropagation through time** (BPTT). The gradient with respect to a single hidden state $h_t$ flows back through every later step that depended on it:

$$\frac{\partial L}{\partial h_t} = \sum_{s \ge t} (\text{contribution from step } s)$$

Each "contribution from step $s$" multiplies through $W_h$ once per intervening step. After 50 steps, you've multiplied by $W_h$ fifty times — and this is the source of the next problem.

## The vanishing/exploding gradient problem

If $W_h$'s singular values are slightly less than 1, the gradient shrinks geometrically as it flows backwards in time — eventually to numerical zero. The model can't learn from inputs more than ~10 steps in the past. If singular values are slightly greater than 1, the gradient *grows* geometrically and overflows. Either way, **long-range dependencies become unlearnable** — a paradox, since the architectural promise of RNNs is unbounded context.

Symptoms:
- The hidden state $h_t$ near the *start* of a long sequence has essentially no influence on outputs near the *end*. Whatever was important about the start has been "forgotten" because gradient signal couldn't propagate that far.
- Training becomes unstable on long sequences — exploding gradients cause loss to spike unpredictably; vanishing gradients cause it to plateau.

The standard fixes:

- **Gradient clipping** — bound $\|\nabla\|$ at a maximum value to prevent explosions.
- **Gating mechanisms** — replace the simple recurrence with a gated one. This is the move from RNN to [[lstm|LSTM]]: explicit "remember" and "forget" gates allow gradients to flow through long-range "shortcuts" rather than chains of multiplications.

The lecturer summarises:

> *"The gradient is going to slowly move around to zero. Then there's the similarity is to have an exploding gradient problems, because if you have some large numbers, it will be multiplied many times. The advantage of RNN was still it's an elegant solution where you can have an input of any size."*

## The other major problem: not parallelisable

To compute $h_t$, you need $h_{t-1}$. To compute $h_{t-1}$, you need $h_{t-2}$. The recurrence is *sequential* — you cannot compute $h_t$ without first computing all $h_s$ for $s < t$. Both training and inference therefore process one token at a time.

For long sequences this is slow on GPU hardware that thrives on parallelism. A 1000-token sequence requires 1000 sequential network applications — not 1000 parallel ones. This is what would later motivate [[language-model|Transformers]]: the attention mechanism replaces the recurrence with an operation that *can* be parallelised across positions, unlocking GPU efficiency.

## RNNs as language models

For language modelling, the input $x_t$ is the one-hot encoding (or embedding) of word $w_t$, and the output $\hat{y}_t$ is the predicted distribution for $w_{t+1}$:

$$\hat{y}_t = P(w_{t+1} \mid w_1, \ldots, w_t)$$

Training maximises the log-likelihood of the next word given the prefix — equivalently, minimises cross-entropy. Generation works by sampling $w_{t+1} \sim \hat{y}_t$, feeding it back as the next input, and continuing (see [[decoding-strategies]]). This is *the* canonical autoregressive LM training and inference loop, identical in spirit to GPT despite the very different architecture.

Advantages over feed-forward NLMs (Bengio 2003):
- Unbounded context (can in principle use the entire prefix, not just last $N - 1$ words).
- Better perplexity on standard benchmarks.

Disadvantages:
- Vanishing gradients limit *effective* context to ~tens of words despite unbounded *theoretical* context.
- Sequential processing → slow training, poor GPU utilisation.
- Unidirectional — natural for autoregressive generation, but means each hidden state only sees left context. (Bidirectional RNNs/LSTMs fix this for representation tasks.)

## Stacking and bidirectionality

Two enhancements that improve quality:

- **Stacked / deep RNN.** Use the outputs of one RNN layer as inputs to a second, then a third. Same way feed-forward networks gain capacity via depth. Each layer learns to summarise progressively higher-level features.
- **Bidirectional RNN.** Run one RNN forwards and another backwards, then concatenate (or sum) their hidden states at each position. Result: each position's representation depends on both left *and* right context. Used in tasks where the entire input is available (classification, sequence labelling) — but not directly usable for autoregressive generation, since it'd need the future to predict the present.

A **stacked bidirectional LSTM** is the architecture behind ELMo (2018), which produced the first widely-used contextualized [[word-embedding|word embeddings]] before BERT.

## Connections

- **Sequence model used by** [[language-model|RNN-based LMs]] — the dominant LM architecture between Bengio 2003 and the Transformer era.
- **Generalised by** [[lstm|LSTMs]] — gating mechanisms address vanishing gradients while keeping the recurrent structure.
- **Superseded by Transformers** (week 11) — attention replaces recurrence with an operation that can be parallelised across positions and that propagates information without the multiplication-chain bottleneck.
- **Trained by backpropagation through time**, which is just standard [[backpropagation]] applied to the unrolled graph.
