---
name: self-attention
description: The core operation of the transformer. Every token in a sequence projects itself into a query, key, and value; each position then aggregates values from every other position weighted by the softmax of query-key similarity. Replaces the fixed weight matrix of an MLP with a *dynamic*, input-dependent one — and lets every token directly attend to every other, regardless of distance.
type: concept
sources:
  - raw/week-11/w11-slides.pdf
status: stable
updated: 2026-05-04
---

*A learnable, content-aware soft dictionary lookup. Each token forms a query and asks every other token (via their keys) "how relevant are you to me?", then collects a weighted sum of their values. The weights are softmaxed dot-product similarities — fully differentiable, fully parallel, and unbounded in range.*

## What problem it solves

[[recurrent-neural-network|RNNs]] process tokens sequentially and *forget* across long distances; gradients vanish through long chains. [[convolution|CNNs]] only see a local receptive field; bridging long-range dependencies requires deep stacks of layers. Yet language is full of arbitrarily long-range dependencies — the subject and verb of a sentence can be separated by an unbounded relative clause.

Self-attention removes the bottleneck. Every token can directly look at every other token in the sequence, in a single layer, with a learnable importance weighting. Distance no longer matters; only relevance does.

## Intuition: a soft dictionary lookup

A Python dictionary or JSON object stores **key–value pairs** and answers **queries** by exact-matching the query against the keys:

```
{
  "Name":          "Jane Doe",
  "Address":       "37 Coronation street",
  "Date of birth": "May 5th 2000",
  "Place of birth":"Hull",
}
```

Query `"Date of birth"` → exact-match the key `"Date of birth"` → return the value `"May 5th 2000"`. Mathematically:

$$z = \sum_{j=1}^{n} \mathbb{1}(q = k_j)\, v_j$$

The indicator $\mathbb{1}(q = k_j)$ is 1 for exactly one matching key, 0 elsewhere. Hard match. **Not differentiable**, so it can't be inserted into a neural network and trained with gradient descent.

### Relaxing the match

Replace the hard indicator with a soft, continuous similarity. Instead of "are these equal?", ask "how aligned are these vectors?" — a dot product. Push the dot products through softmax to turn them into a probability distribution over keys:

$$z = \sum_{j=1}^{n} \text{score}_j \, v_j, \qquad \text{score}_j = \frac{\exp(q^\top k_j / \sqrt{d_k})}{\sum_{j'} \exp(q^\top k_{j'} / \sqrt{d_k})}$$

Now every key contributes to the result, but the highest-similarity key contributes the most. The result $z$ is a *weighted average* of values, not a single value. **Fully differentiable.**

This soft, content-addressed lookup is the entire substance of attention. The rest of the page is just "where do $q$, $k$, $v$ come from, and how do we apply this across a sequence?"

## Self-attention: every token makes its own Q, K, V

In **self-attention**, the queries, keys, and values are all derived from the *same* input sequence — every token contributes one query, one key, and one value. (When the queries come from a different source than the keys/values, you get [[cross-attention]] instead.)

For a sequence of token embeddings $\mathbf{x}_1, \dots, \mathbf{x}_n$, the layer holds **three learnable weight matrices**:

| Matrix | Shape | Job |
|---|---|---|
| $W^Q$ | $d_{\text{model}} \times d_k$ | Project token embedding into the query space |
| $W^K$ | $d_{\text{model}} \times d_k$ | Project token embedding into the key space |
| $W^V$ | $d_{\text{model}} \times d_v$ | Project token embedding into the value space |

For each token $i$:

$$\mathbf{q}_i = W^Q \mathbf{x}_i, \qquad \mathbf{k}_i = W^K \mathbf{x}_i, \qquad \mathbf{v}_i = W^V \mathbf{x}_i$$

Stacking all $n$ tokens row-wise gives matrices $X \in \mathbb{R}^{n \times d_{\text{model}}}$ and:

$$Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V$$

each in $\mathbb{R}^{n \times d}$ (with $d = d_k$ or $d_v$). Three projections of the same sequence into three different "roles" — interrogator, label, content.

> [!tip] TIP — Why three projections of the same input?
> If $Q = K = V = X$, the attention pattern becomes "every token attends most strongly to itself" (because a vector is most similar to itself). The three separate learned projections let the network decouple "what is this token about?" (key) from "what does this token want to know about?" (query) from "what information does this token contribute when attended to?" (value). All three roles emerge from the same word, but live in different learned subspaces.

## Scaled dot-product attention

Combine the three projections into the canonical formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Reading it left to right:

1. **$Q K^\top$** ($n \times n$) — every token's query dotted against every other token's key. Entry $(i, j)$ is the raw similarity score between token $i$ as a querier and token $j$ as a target.
2. **$/\sqrt{d_k}$** — divide by the square root of the key dimension. Without scaling, large $d_k$ makes dot products grow large in magnitude, pushing softmax into saturation regions where gradients vanish.
3. **$\text{softmax}(\cdot)$** (applied row-wise) — turn each row of similarities into a probability distribution. Row $i$ now says "how much does token $i$ care about each other token?" — sums to 1.
4. **$\cdot V$** — multiply the attention matrix by the value matrix. Each output row is a weighted sum of value vectors, weighted by token $i$'s attention distribution. Output is $n \times d_v$ — one new vector per input token.

The output is a sequence of the same length as the input, but every output position is now a *content-mixed* combination of every input position.

## A worked picture

Consider the input "the animal didn't cross the street because it was too tired". For the token `it`, the model needs to figure out what `it` refers to. After training, `it`'s query strongly aligns with the key of `animal`, weakly with `street`, weakly with the others. The attention distribution for `it` looks roughly like:

| Token | Attention weight from `it` |
|---|---|
| `the` | 0.02 |
| `animal` | **0.45** |
| `didn't` | 0.05 |
| ... | ... |
| `street` | 0.10 |
| ... | ... |
| `tired` | 0.08 |

The output vector at the position of `it` is now a weighted blend dominated by the value vector of `animal` — the network has *contextually rewritten* `it` to mean "it (referring to animal)". Coreference resolution as a side effect of training the right attention distribution.

## Self-attention vs. feed-forward

A standard MLP layer applies a fixed weight matrix to every input: $\mathbf{y} = W \mathbf{x}$. The same $W$ is used regardless of the input.

Self-attention applies a **dynamic, input-dependent** weight matrix: the matrix $\text{softmax}(QK^\top / \sqrt{d_k})$ is *computed from the inputs themselves*. Different inputs produce different attention patterns, and therefore different effective weight matrices.

| Layer | Weight matrix | What it depends on |
|---|---|---|
| MLP / feed-forward | $W$ — fixed after training | Nothing per-input |
| Self-attention | $\text{softmax}(QK^\top / \sqrt{d_k})$ — computed at every forward pass | The input sequence itself |

This is the deeper reason transformers are so expressive: the *connection pattern* between tokens is itself learned and varies with input. The MLP analogue is a network whose connections rewire on the fly to fit each new sentence.

## Masked self-attention (causal attention)

When self-attention is used in a decoder for autoregressive generation, every token must only see *previous* tokens — never future ones. Otherwise, predicting token $i+1$ would be trivial: the model could just look at the answer.

**Masked attention** enforces causality by zeroing out the attention scores for positions $j > i$ before the softmax:

$$\text{score}_{i,j} = \begin{cases} q_i^\top k_j / \sqrt{d_k} & j \leq i \\ -\infty & j > i \end{cases}$$

After softmax, $-\infty$ becomes $0$, so future positions contribute nothing. The attention matrix is **lower-triangular**: each row $i$ has non-zero weights only on columns $\leq i$.

In the encoder, no masking is applied (the encoder can freely attend to the entire input). The decoder uses *masked* self-attention on its own outputs and *unmasked* [[cross-attention]] over the encoder's latent code.

## Time and memory cost

Self-attention computes an $n \times n$ matrix for a sequence of length $n$. Time and memory both scale as $O(n^2 \cdot d)$. For $n = 1000$ this is fine; for $n = 100{,}000$ it becomes prohibitive. This is the **quadratic complexity bottleneck** that motivated countless approximate-attention variants (sparse attention, linear attention, Longformer, Performer, FlashAttention's IO-aware optimisations). For the canonical "Attention Is All You Need" transformer, the quadratic cost is just the price you pay.

## Worked example: tiny self-attention by hand

For a single token's query $\mathbf{q}_i$ and three other tokens' keys $\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3$ and values $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$, all 3-dimensional, with $d_k = 3$:

**Query:** $\mathbf{q}_i = [0.3, 1.2, 2.2]$.

**Keys and values:**
- $\mathbf{k}_1 = [0.4, 1.4, 0],\ \mathbf{v}_1 = [2.3, 1.1, 3.5]$
- $\mathbf{k}_2 = [0.2, -1.1, 0],\ \mathbf{v}_2 = [3.0, 1.0, 0.5]$
- $\mathbf{k}_3 = [0.3, 1.2, 4],\ \mathbf{v}_3 = [-2.2, 4.1, 2.5]$

**Step 1 — raw similarities** $\mathbf{q}_i^\top \mathbf{k}_j$:

- $\mathbf{q}_i^\top \mathbf{k}_1 = 0.3(0.4) + 1.2(1.4) + 2.2(0) = 1.80$
- $\mathbf{q}_i^\top \mathbf{k}_2 = 0.3(0.2) + 1.2(-1.1) + 2.2(0) = -1.26$
- $\mathbf{q}_i^\top \mathbf{k}_3 = 0.3(0.3) + 1.2(1.2) + 2.2(4) = 10.33$

**Step 2 — scale by $\sqrt{d_k} = \sqrt{3} \approx 1.732$:**

$(1.04, -0.73, 5.97)$.

**Step 3 — softmax** (numerically: $e^{1.04} \approx 2.83$, $e^{-0.73} \approx 0.48$, $e^{5.97} \approx 391.5$; sum $\approx 394.8$):

$(0.0072,\; 0.0012,\; 0.992)$.

Token 3 dominates the attention.

**Step 4 — weighted sum of values:**

$\mathbf{z}_i \approx 0.0072 \cdot [2.3, 1.1, 3.5] + 0.0012 \cdot [3.0, 1.0, 0.5] + 0.992 \cdot [-2.2, 4.1, 2.5]$
$\approx [-2.16,\; 4.08,\; 2.51]$.

The output for token $i$ is essentially $\mathbf{v}_3$, lightly perturbed by tiny contributions from $\mathbf{v}_1$ and $\mathbf{v}_2$. Token $i$ "looked at" token 3 and copied its value, because $\mathbf{q}_i$ was much more aligned with $\mathbf{k}_3$ than the other keys.

## Related

- [[transformer]] — the architecture built around stacked self-attention layers
- [[multi-head-attention]] — running self-attention with $h$ parallel heads
- [[cross-attention]] — same machinery, but $Q$ comes from a different sequence than $K, V$
- [[positional-encoding]] — required because self-attention is permutation-invariant
- [[softmax]] — the row-wise normalisation that turns scores into a distribution
- [[dot-product]] — the per-pair similarity measure
- [[autoregressive-model]] — masked self-attention enables autoregressive decoders

## Active Recall

> [!question]- What is the formula for scaled dot-product attention, and what does each piece do?
> $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d_k}) V$. **$QK^\top$** computes pairwise query-key similarities; **$/\sqrt{d_k}$** scales the magnitudes so the softmax doesn't saturate at large $d_k$; **softmax** (row-wise) turns the similarity scores into a probability distribution per query; **$\cdot V$** computes a weighted sum of value vectors using those probabilities. Output is one new vector per query position.

> [!question]- Why three separate projections $W^Q, W^K, W^V$ instead of using the input directly as queries, keys, and values?
> If $Q = K = V = X$, every token's query is most similar to its own key (a vector is most similar to itself), so attention degenerates to "every token attends to itself". The three learned projections decouple "what does this token want to ask?" (query), "what is this token labelled as?" (key), and "what content does this token contribute when attended to?" (value), all from the same input embedding but in different learned subspaces. This is what makes attention expressive.

> [!question]- How is self-attention different from a feed-forward / MLP layer in terms of the *weights*?
> A feed-forward layer applies a *fixed* weight matrix $W$ — the same matrix is used for every input, set at training and frozen at inference. Self-attention applies a **dynamic** weight matrix $\text{softmax}(QK^\top / \sqrt{d_k})$ that is *computed from the input itself* every forward pass. Different inputs produce different attention patterns. This input-dependence is why transformers are so much more expressive per parameter than MLPs.

> [!question]- Why is masked self-attention needed in the decoder, and how is it implemented?
> An autoregressive decoder predicts token $i+1$ given tokens $1, \ldots, i$. If the decoder's self-attention layer could see future tokens, prediction would be trivial — the answer is in the input. Masked attention prevents this by setting attention scores for positions $j > i$ to $-\infty$ before softmax, so they become $0$ after softmax and contribute nothing. The attention matrix is lower-triangular: each row $i$ only has non-zero weights on columns $\leq i$.

> [!question]- Why does self-attention scale quadratically with sequence length, and why does that matter?
> The attention score matrix $QK^\top$ is $n \times n$ — one entry per (query, key) pair. Computing and storing this matrix costs $O(n^2 \cdot d_k)$ in time and $O(n^2)$ in memory. For $n = 1000$ tokens this is fine; for $n = 100{,}000$ (long documents, high-resolution image sequences) it becomes prohibitive. This is the *quadratic bottleneck* that motivates approximate-attention variants (Longformer, Performer, FlashAttention).

> [!question]- A query $\mathbf{q} = [1, 0, 0]$ is compared against keys $\mathbf{k}_1 = [1, 0, 0]$, $\mathbf{k}_2 = [0, 1, 0]$, $\mathbf{k}_3 = [-1, 0, 0]$. Without softmax-scaling, what are the raw dot-product similarities?
> $\mathbf{q}^\top \mathbf{k}_1 = 1$, $\mathbf{q}^\top \mathbf{k}_2 = 0$, $\mathbf{q}^\top \mathbf{k}_3 = -1$. Token 1 is most similar (highest dot product), token 2 is orthogonal (zero similarity), token 3 is anti-aligned (negative similarity). After softmax, almost all weight goes to token 1, very little to token 2, and even less to token 3.
