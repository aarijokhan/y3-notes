---
name: decoding-strategies
description: "Algorithms that turn the next-word distribution of an autoregressive language model into actual generated text. Greedy and beam search aim for high-probability sequences (deterministic-ish); ancestral sampling, temperature-adjusted sampling, top-K, and top-P sample from the distribution to encourage diversity while controlling for low-probability noise. Different strategies for different goals: translation favours beam search; creative generation favours top-P; evaluation often uses temperature 0 (greedy)."
type: concept
sources:
  - raw/week-09/w09-l19-transcript.txt
  - raw/week-09/w09-l20-transcript.txt
  - raw/week-09/w09-slides.pdf
  - raw/week-09/w09-problem-set.pdf
status: stable
updated: 2026-04-27
---

*An autoregressive [[language-model|LM]] outputs a probability distribution over the next word — but you wanted a string of text. The decoding strategy is what bridges that gap. Naively, you can pick the highest-probability word at each step (greedy), keep a beam of top hypotheses (beam search), or sample randomly (ancestral sampling). Each has failure modes — greedy is deterministic and bland, full sampling pulls in low-probability nonsense, beam search can find probability-maximising but boring outputs. The solutions — temperature, top-K, top-P — let you dial between "high probability" and "diverse" by trimming the distribution before sampling.*

## The setup

An autoregressive LM gives you, at each step, a probability distribution over the next token:

$$p_t = P(w_t \mid w_{1:t-1})$$

A **decoding strategy** is a rule for picking $w_t$ from $p_t$, then appending it to the context and asking for $p_{t+1}$, until either an `<EOS>` token is sampled or a maximum length is hit. Different rules optimise for different things — fluency vs. probability vs. diversity. There's no single right answer; the choice depends on the application.

## Sampling from a distribution: PDF → CDF

Before any of the strategies, the basic primitive: how do you draw a single word from a (discrete) probability distribution? Naive `argmax` always returns the same word — that's not sampling. The standard recipe converts the **PDF** (probability per word) into a **CDF** (cumulative distribution) and inverts a uniform random number through it:

1. List the words and their probabilities $(p_1, p_2, \ldots, p_V)$.
2. Compute cumulative sums $c_i = \sum_{j \le i} p_j$. The full CDF rises from 0 to 1 in steps of size $p_i$.
3. Draw $r \sim \text{Uniform}(0, 1)$.
4. Return the word $w_i$ such that $c_{i-1} < r \le c_i$ — i.e., the word whose CDF interval contains $r$.

A word with high probability "owns" a wide slice of $[0, 1]$ and is hit often; a low-probability word "owns" a narrow slice and is hit rarely. This is what people mean by "sampling according to the probability distribution."

> [!warning] CAUTION — Greedy ≠ random sampling ≠ uniform random
> Three things easy to confuse:
> - **Greedy / argmax**: pick the highest-probability word. Deterministic, no randomness.
> - **Random sampling according to $p$**: draw with probability $p_i$ via the CDF trick. Higher-probability words are more likely but not certain.
> - **Uniform random**: draw with probability $1/V$ regardless of $p$. *Doesn't use the model at all.*
>
> A common bug: writing `random.choice(words)` and thinking you've sampled from the distribution. You've sampled *uniformly*. The CDF mapping is what makes the sample respect the model.

## The strategies

### Greedy search

At each step, pick $w_t = \arg\max_w P(w \mid w_{1:t-1})$. Always the locally most probable continuation.

**Pros:** Deterministic, fast, single forward pass per step. Often used for evaluation or when reproducibility matters.

**Cons:** Deterministic — produces the same output every time given the same prompt, so no diversity. *Locally* greedy decisions can hurt globally: choosing the highest-probability first word may lead into a low-probability subtree, when a slightly worse first word would lead to a much better overall sentence.

### Beam search

Keep the top-$k$ partial sequences ("hypotheses") at each step, where $k$ is the **beam size**. At each step, expand each of the $k$ hypotheses by one word, score the resulting $k \cdot V$ partial sequences by their cumulative probability, and keep the top $k$.

**Toy example, beam size 2.** Start with `The`. After one step, $P(\text{nice} \mid \text{The}) = 0.5$ and $P(\text{dog} \mid \text{The}) = 0.4$ — we keep both (greedy would have kept only "nice"). After expanding *"The dog"* with $P(\text{has} \mid \text{The dog}) = 0.9$, the cumulative probability is $0.4 \times 0.9 = 0.36$, beating *"The nice woman"* at $0.5 \times 0.4 = 0.20$. Beam search finds the better sequence that greedy misses because the path went through a locally suboptimal first word.

**Pros:** Approximates the global argmax over sequences without enumerating $V^n$ candidates. Standard in machine translation and other tasks where probability-maximising fluency matters.

**Cons:** Computationally expensive (each step does $k$ forward passes). Often *too* probability-focused — beam search outputs are high-probability but generic and repetitive (the so-called "neural text degeneration" problem). For creative or diverse generation, beam search is the wrong tool.

> [!info] ASIDE — Why beam search is impractical as exact search
> Exhaustive search would give the truly maximum-probability sequence — but with vocabulary size $|V| \approx 10^5$ and sequence length 50, the search space is $10^{250}$ continuations. Beam search at $k = 5$ explores $5 \times 10^5 = 5 \times 10^5$ candidates per step instead of $10^5$ — still tractable, and empirically a good approximation. Larger beams reach better hypotheses up to a point, then plateau; very large beams sometimes *worsen* output quality (the "beam search curse" — the model's true argmax is unnaturally short or repetitive).

### Ancestral sampling

Draw $w_t \sim P(\cdot \mid w_{1:t-1})$ at each step using the CDF method. Pure stochastic generation according to the model.

**Pros:** Maximally diverse — every run produces different text. Asymptotically faithful to the model's distribution.

**Cons:** Vulnerable to the long tail. The next-word distribution often has a small number of very-likely words plus a long tail of low-probability ones — and even though each tail word has tiny individual probability, *collectively* the tail can have substantial mass. So sampling will occasionally pick a word from the tail, and these are typically low-quality (incoherent, off-topic, ungrammatical). A single bad word also derails subsequent context, since the LM will continue on the basis of what it actually generated.

### Temperature-adjusted sampling

Modify the next-word distribution before sampling by introducing a **temperature** $T$ in the softmax that produced it:

$$p_i \propto \exp(z_i / T)$$

where $z_i$ is the pre-softmax logit. Then sample from the modified distribution.

- $T = 1$: the original distribution.
- $T < 1$: distribution becomes *sharper* — high-probability words become more probable, low-probability ones less probable. In the limit $T \to 0$, this approaches greedy.
- $T > 1$: distribution becomes *flatter* — low-probability words gain probability mass. More diversity, more risk of incoherence.

**The intuition:** temperature dilutes (or concentrates) the model's preferences. Useful when the model's raw distribution is too peaky (sample the tail to see what it almost said) or too flat (focus on what it most wants to say).

### Top-$K$ sampling

Truncate the distribution to the $K$ highest-probability words; renormalise; sample from the truncated distribution. Used in GPT-2's generation by default.

**Pros:** Cheap, removes the long tail of garbage. Empirically much better than full ancestral sampling.

**Cons:** $K$ is fixed across steps. Some contexts have a sharp peak (one obvious continuation, $K = 1$ would suffice), others have many plausible options ($K = 50$ may not cover them). Using a fixed $K$ truncates too much in flat distributions and not enough in peaky ones.

### Top-$P$ (nucleus) sampling

Truncate the distribution to the smallest set of words whose cumulative probability is $\ge p$ (e.g., $p = 0.9$ keeps the words covering 90% of the probability mass); renormalise; sample. Adapts the truncation size to the shape of each step's distribution.

**Pros:** In sharp distributions, automatically picks a small set; in flat distributions, picks a large set. The right shape of "trim the tail" without a fixed cutoff.

**Cons:** Slightly more expensive (need to sort). One extra hyperparameter.

> [!example]+ Sharpness adapts: top-$K$ vs. top-$P$ in two contexts
> After *"The"*, the next-word distribution is broad — many nouns ("nice", "dog", "car", "woman", ...) have similar probability. Top-$K$ with $K = 6$ might keep nice/dog/car/woman/guy/man (cumulative ≈ 0.68); top-$P$ with $p = 0.94$ keeps a *larger* set (≈ 10 words) since the distribution is so flat. After *"The car"*, the next-word distribution sharpens to a few verbs ("drives" 0.5, "is" 0.3, "turns" 0.2). Top-$K$ with $K = 6$ would still keep 6 words even though the probability past the top 3 is essentially zero — including noise. Top-$P$ with $p = 0.97$ correctly keeps just 3.

## Practical guidance

| Application | Default strategy | Reason |
|---|---|---|
| Machine translation | Beam search (size 4–8) | Fluency matters more than diversity; argmax-ish is OK |
| Summarisation | Beam search or top-$P$ | Faithfulness to source |
| Creative writing / chat | Top-$P$ ($p = 0.9$) + temperature 0.7–1 | Diversity + coherence |
| Code generation | Greedy or beam | Often you want one canonical answer |
| Evaluation / benchmarking | Greedy ($T = 0$) | Reproducibility |

## Connections

- **Generation pipeline for** [[language-model|any autoregressive LM]] — n-gram, RNN, GPT all use the same decoding strategies on top of their next-word distribution.
- **Builds on** [[softmax]] — temperature is a parameter modifying the softmax that produced the distribution.
- **Doesn't apply to** [[language-model#Two flavours: autoregressive vs. masked|masked LMs]] — generating from BERT requires fundamentally different schemes (e.g., iterative mask-and-fill).
