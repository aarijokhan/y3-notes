---
name: multi-head-attention
description: Run self-attention $h$ times in parallel, each with its own learned $W^Q, W^K, W^V$ projections. Concatenate the $h$ output sequences along the feature axis and project through one final learned matrix $W^O$. Each head can specialise in a different relational pattern (syntax, coreference, position-relative, etc.) — the layer learns several attention "lenses" simultaneously.
type: concept
sources:
  - raw/week-11/w11-slides.pdf
status: stable
updated: 2026-05-04
---

*One attention layer learns one attention pattern per query — but a sentence has many simultaneous relational structures (subject-verb agreement, coreference, prepositional attachment, …). Multi-head attention runs $h$ parallel attention computations, each with independent Q/K/V projections, lets each head specialise, then merges the results.*

## Why one head isn't enough

A single self-attention layer with one set of $(W^Q, W^K, W^V)$ produces *one* attention pattern per query. Whatever subspace the projections happen to land in determines what kind of similarity gets measured.

But natural sentences contain many *simultaneous* relational structures:

- "The animal didn't cross the street **because it** was too tired" — `it` should attend to `animal` (coreference).
- "The black cat that sleepily **sat** on the mat" — `sat` should attend to `cat` (subject-verb agreement) *and* to `sleepily` (adverbial modifier).
- Long-range syntactic dependencies, local positional ones, semantic affinities — all happening at once.

A single head has to compromise — its weights settle on the most loss-reducing average of all these patterns. **Multi-head attention** sidesteps the compromise by running $h$ attention computations in parallel, each with its own independent projections. Each head ends up specialising in a different relational structure.

## The mechanism

Pick the number of heads $h$ (typically 8 in the original transformer). For each head $i \in \{0, 1, \ldots, h-1\}$, instantiate three independent learnable matrices:

$$W_i^Q,\ W_i^K,\ W_i^V$$

Each maps from the model dimension $d_{\text{model}}$ to a smaller per-head dimension — typically $d_k = d_v = d_{\text{model}} / h$. So in the original transformer with $d_{\text{model}} = 512$ and $h = 8$, each head has $d_k = d_v = 64$.

For each head, run [[self-attention|scaled dot-product attention]] independently:

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i$$

where $Q_i = X W_i^Q$, $K_i = X W_i^K$, $V_i = X W_i^V$.

Each head produces an $n \times d_v$ output. **Concatenate** all $h$ heads' outputs along the feature axis:

$$\text{Concat}(\text{head}_0, \ldots, \text{head}_{h-1}) \in \mathbb{R}^{n \times h d_v}$$

Then project through one final learnable matrix $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ to merge them into one output sequence of the original dimension:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_0, \ldots, \text{head}_{h-1})\, W^O$$

The output is again $n \times d_{\text{model}}$, ready to feed into the next layer.

## Why split into smaller heads instead of running $h$ full-sized heads?

The per-head dimension $d_k = d_{\text{model}} / h$ is chosen so the *total* compute and parameter cost of multi-head attention matches that of a single full-sized head. Each head is "cheaper" but there are $h$ of them; the bookkeeping cancels.

This is a deliberate design choice. You could instead run $h$ heads each of full size $d_{\text{model}}$ — but that would multiply the parameter and compute cost by $h$. The split keeps the total budget fixed while gaining the specialisation benefit.

> [!tip] TIP — Multi-head as ensemble
> Conceptually, multi-head attention is something like running an *ensemble* of small attention layers and learning how to weight their outputs. Different heads end up looking at different parts of the input — some attend to neighbouring tokens (positional), some to syntactically related tokens (long-range), some to coreferent mentions, some to repeated patterns. The final $W^O$ projection learns how to combine the heads' findings into a single representation that downstream layers can use.

## What each head actually learns

Empirically, when you visualise the attention patterns of trained transformers, different heads exhibit distinct, interpretable behaviours:

- **Positional heads** — attend to the previous or next token regardless of content (basically a learned shift).
- **Syntactic heads** — attend to syntactically related tokens (subject of the verb, object of the preposition).
- **Coreference heads** — link pronouns to their antecedents.
- **Rare-token heads** — focus on less frequent words that carry more information.
- **Diffuse heads** — spread attention broadly, contributing average context.

The figure on the slides shows two heads on the same sentence "The Law will never be perfect, but its application should be just" — one attending sharply along a few specific edges (likely a syntactic head), the other diffusely attending across many edges (likely a content-mixing head). Same input, two different views.

This division of labour is *emergent* — nobody tells head 3 to be the coreference head. The training signal pushes the heads to specialise because diversifying their patterns reduces loss more than redundantly fitting the same pattern.

## In code (compact form)

The full multi-head attention layer fits on a few lines if you batch the heads as an extra tensor dimension:

1. Project $X$ into $Q, K, V$ of shape $(n, h \cdot d_k)$.
2. Reshape to $(h, n, d_k)$ — heads become a batch dimension.
3. Run scaled dot-product attention in parallel across the head axis.
4. Concatenate heads (reshape back to $(n, h \cdot d_v)$).
5. Project through $W^O$ to $(n, d_{\text{model}})$.

The "multi-head" label refers to logical specialisation; computationally it's a single batched matrix operation, which is why GPUs handle it efficiently.

## Counting parameters

A multi-head attention layer has four sets of learned matrices:

- $h$ copies of $W^Q$, each $d_{\text{model}} \times d_k$ → total $d_{\text{model}} \cdot h \cdot d_k = d_{\text{model}}^2$ (since $h d_k = d_{\text{model}}$).
- Similarly for $W^K$ and $W^V$ → $d_{\text{model}}^2$ each.
- One $W^O$ of shape $h d_v \times d_{\text{model}} = d_{\text{model}} \times d_{\text{model}}$ → $d_{\text{model}}^2$.

**Total: $4 d_{\text{model}}^2$ parameters per layer**, regardless of $h$. The number of heads is a design choice that trades width-per-head against number-of-heads without changing the total budget.

## Related

- [[self-attention]] — the underlying single-head operation
- [[cross-attention]] — multi-head attention with $Q$ from one sequence and $K, V$ from another
- [[transformer]] — the architecture that stacks multi-head attention with feed-forward and normalisation
- [[softmax]] — used inside each head's attention
- [[dropout]] — typically applied to the attention weights and after the $W^O$ projection

## Active Recall

> [!question]- Multi-head attention runs $h$ parallel attention heads each with its own $(W^Q, W^K, W^V)$. What is the *total* parameter cost compared to a single full-sized head, and why?
> Each head has projections of size $d_{\text{model}} \times d_k$ where $d_k = d_{\text{model}}/h$. Across all $h$ heads the projections sum to $d_{\text{model}}^2$ — exactly the same as one full-sized head with $d_k = d_{\text{model}}$. Plus one final $W^O$ of $d_{\text{model}}^2$. The total $4 d_{\text{model}}^2$ parameters is independent of $h$. Multi-head doesn't cost more parameters; it just splits the same budget across $h$ specialists.

> [!question]- Why is multi-head attention more expressive than single-head attention with the same parameter count?
> A single head computes one attention distribution per query — it can only attend strongly along one relational axis at a time (syntactic or positional or coreferential, etc.). Multiple heads in parallel let each one specialise in a different pattern; the final $W^O$ projection learns to combine them. Same parameter budget, more relational structure captured per layer.

> [!question]- After running $h$ attention heads in parallel and producing $h$ output sequences each of dim $d_v$, what two operations turn them back into a single sequence of dim $d_{\text{model}}$?
> First, **concatenate** the $h$ outputs along the feature axis to get a sequence of dim $h \cdot d_v$. Second, project through a learned matrix $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ to mix the heads and return to the original model dimension. The concatenation preserves all heads' information; $W^O$ learns how to combine them.

> [!question]- What kinds of patterns do trained transformer heads typically specialise into, and what enforces this specialisation?
> Empirically, different heads end up doing different things — one attends to the previous token (positional), one tracks coreference (e.g. pronouns to antecedents), one captures subject-verb agreement, one diffusely averages context, etc. Nothing in the architecture *enforces* this; it emerges from gradient descent because diversifying head behaviours reduces loss more than redundantly fitting the same pattern. The $W^O$ layer learns to combine the diverse heads, completing the division of labour.
