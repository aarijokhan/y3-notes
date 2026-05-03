---
title: "Week 2 Flashcards — N-gram Language Models"
week: 2
topic: "N-gram Language Models — Counting, Perplexity, Smoothing"
deck: NLP::Week-02
---

TARGET DECK
NLP::Week-02

# Week 2 Flashcards

## Language Models

> [!question]- What is a language model?
> A **language model** assigns a probability $P(w_1, w_2, \ldots, w_n)$ to a sequence of words, or equivalently the probability $P(w_n \mid w_1 \ldots w_{n-1})$ of the next word given prior context. It is the core probabilistic tool for tasks where the output is text — speech recognition, machine translation, spelling correction, autocomplete.
<!--ID: 1777386639893-->





> [!question]- What is the chain rule of probability for a sentence, and why can't we compute it directly?
> $$P(w_1, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1})$$
> We can't compute it directly because estimating $P(w_n \mid w_1 \ldots w_{n-1})$ requires having seen that exact prefix in training data. With vocabulary $V$, there are $V^{n-1}$ possible prefixes — for English this explodes to billions even for 4-word contexts. **No corpus is large enough**; the data requirement grows exponentially with sentence length.
<!--ID: 1777386639894-->





> [!question]- What is the Markov assumption, and how does it make language modelling tractable?
> The **Markov assumption** says the probability of the next word depends only on the last $N-1$ words, not the full history:
> $$P(w_i \mid w_1 \ldots w_{i-1}) \approx P(w_i \mid w_{i-N+1} \ldots w_{i-1})$$
> This shrinks the conditioning context from arbitrary length to a fixed small window — making counts dense enough to estimate reliably from a finite corpus. It is an obviously wrong assumption (long-range dependencies exist) but is enough to make n-gram models work.
<!--ID: 1777386639895-->





## N-gram Models

> [!question]- What is an n-gram?
> An **n-gram** is a contiguous sequence of $n$ tokens from a text. A **unigram** is one word, a **bigram** is two adjacent words, a **trigram** is three. An n-gram language model estimates $P(w_i \mid \text{previous } n-1 \text{ words})$ by counting n-grams in a training corpus.
<!--ID: 1777386639896-->





> [!question]- What is the maximum likelihood estimate (MLE) for a bigram probability?
> $$P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$
> Count how often the bigram $(w_{i-1}, w_i)$ appears, divide by how often the unigram $w_{i-1}$ appears. This is the value of $P$ that maximises the probability of the training data — hence "maximum likelihood".
<!--ID: 1777386639897-->





> [!question]- Why do n-gram models use `<s>` and `</s>` tokens?
> - `<s>` lets the bigram $P(w_1 \mid \langle s \rangle)$ assign probability to the **first word** — without it, the first word has no context to condition on
> - `</s>` lets the model assign a probability to **ending the sentence** — making sentence probabilities sum to 1 across all possible lengths
> - Without sentence boundary tokens, the model can't compare sentences of different lengths fairly
<!--ID: 1777386639898-->





> [!question]- Why work in log space when computing sentence probabilities?
> Multiplying many small probabilities causes **floating-point underflow** — the product becomes smaller than the smallest representable float (≈$10^{-308}$). Working in log space converts products to sums: $\log(p_1 p_2 \ldots p_n) = \log p_1 + \log p_2 + \ldots + \log p_n$. Log is monotonic, so $\arg\max$ is preserved.
<!--ID: 1777386639899-->





## Sampling and Quality

> [!question]- How do you generate text by sampling from an n-gram model?
> 1. Start from `<s>` (and any other required prefix tokens)
> 2. Sample the next word from $P(w \mid \text{current context})$ — draw from the categorical distribution
> 3. Append the sampled word to the context, slide the window
> 4. Repeat until `</s>` is sampled
> Higher-order n-grams produce more coherent (but more memorised) text.
<!--ID: 1777386639901-->





> [!question]- Why does a 4-gram model on Shakespeare produce text that looks like memorisation?
> With Shakespeare's vocabulary of ~29,066 types, **99.96% of possible bigrams are never observed** — the model has seen too few unique 4-gram contexts to generalise. When it samples, the only continuations available are the ones from training. The model is reproducing training fragments rather than generating new combinations. This is the memorisation-vs-generalisation tradeoff that scaling n-gram order makes worse.
<!--ID: 1777386639902-->





## Evaluation

> [!question]- What is the difference between extrinsic and intrinsic evaluation?
> - **Extrinsic**: evaluate by embedding the LM in a downstream task (speech recognition, MT) and measuring task performance. Gold standard but expensive and task-specific.
> - **Intrinsic**: evaluate the model directly using a metric defined on the LM itself, like perplexity on a held-out test set. Faster and cheaper, but a better intrinsic score doesn't guarantee a better task score.
> - In practice: use intrinsic during development, validate with extrinsic before deployment.
<!--ID: 1777386639906-->





> [!question]- Why do you need a train/dev/test split? Why three sets, not two?
> - **Training set**: used to estimate model parameters (counts).
> - **Development (dev) set**: used to tune hyperparameters and compare model variants — touched repeatedly during development.
> - **Test set**: used **once**, at the end, to report final performance.
> - If you tuned on the test set, you'd implicitly train on it — your reported score would overestimate generalisation. The dev set absorbs the repeated-evaluation contamination, leaving the test set clean.
<!--ID: 1777386639908-->





> [!question]- What is perplexity?
> $$\text{PP}(W) = P(w_1 \ldots w_N)^{-1/N}$$
> The **inverse probability of the test set, normalised per word** ($N$th root). Lower is better. Range $[1, \infty)$. Minimising perplexity is equivalent to maximising test-set probability. Always computed on a held-out test set, never the training set.
<!--ID: 1777386639911-->





> [!question]- Give two equivalent intuitions for perplexity.
> 1. **Average branching factor**: perplexity $k$ means the model is as uncertain at each position as if choosing uniformly among $k$ equally likely words. PP = 1 means perfect prediction; PP = $|V|$ means uniform guessing over the vocabulary.
> 2. **Shannon game / surprise**: it is the average number of guesses a perfect Shannon-game player would need given the model's distribution. Equivalently, $\text{PP} = 2^H$ where $H$ is the cross-entropy of the model on the test data.
<!--ID: 1777386639912-->





> [!question]- Why can't you compare perplexity across different test sets?
> Perplexity depends on **both** the model and the data. A model with PP 95 on Twitter and one with PP 120 on WSJ tells you nothing about which is better — Twitter and WSJ have different vocabularies, structures, and intrinsic entropy. **Only same-test-set comparisons are valid.**
<!--ID: 1777386639915-->





## Smoothing

> [!question]- What is the zero-probability problem?
> Any n-gram unseen in training gets MLE probability **zero**. Since sentence probability is a product of n-gram probabilities, **one zero anywhere zeroes the entire sentence** — and perplexity (which depends on $1/P$) becomes undefined (division by zero). This is devastating because language is creative: novel n-grams appear constantly at test time.
<!--ID: 1777386639916-->





> [!question]- What is Laplace (add-one) smoothing for bigrams?
> Add 1 to every bigram count, and add $V$ (vocabulary size) to each unigram-count denominator to keep probabilities normalised:
> $$P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$$
> Pretends every possible bigram has been seen one extra time. Guarantees no zeros. Simple but **too aggressive for n-gram LMs** because $V$ is huge and the bigram space is mostly empty — too much mass is shifted to unseen events.
<!--ID: 1777386639917-->





> [!question]- Why can Laplace smoothing actually *lower* the probability of an observed sentence?
> The denominator gets $+V$ added (vocabulary size, often thousands). With $V$ large, the denominator grows substantially relative to the $+1$ in the numerator — pulling down the probability of every observed bigram. The mass removed flows to the millions of zero-count cells. If 99%+ of cells are zero (typical for n-grams), most mass leaves observed events. This is why add-one is usually inadequate for n-gram LMs (though fine for Naive Bayes, where the vocabulary is smaller and only relative ranking matters).
<!--ID: 1777386639918-->





> [!question]- What is add-k smoothing, and how is it different from add-one?
> Generalises Laplace: add a fractional $k$ (e.g., $k = 0.01$) instead of $1$:
> $$P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + kV}$$
> Smaller $k$ → less mass redistributed to unseen events → less distortion of observed counts. The optimal $k$ is tuned on a dev set. Still inferior to interpolation/backoff for serious LM work.
<!--ID: 1777386639921-->





## Interpolation and Backoff

> [!question]- What is linear interpolation for n-gram smoothing?
> Mix probabilities from multiple n-gram orders:
> $$\hat{P}(w_i \mid w_{i-2}, w_{i-1}) = \lambda_1 P(w_i \mid w_{i-2}, w_{i-1}) + \lambda_2 P(w_i \mid w_{i-1}) + \lambda_3 P(w_i)$$
> with $\sum \lambda_j = 1$. The $\lambda$ weights are learned on a held-out set (often via EM). The lower-order terms supply estimates when the higher-order count is zero or unreliable. **Generalises gracefully**: even if no trigram count exists, bigram and unigram still contribute.
<!--ID: 1777386639922-->





> [!question]- What is the difference between interpolation and backoff?
> - **Interpolation**: always uses **all** n-gram orders, weighted: $\hat{P} = \lambda_1 P_3 + \lambda_2 P_2 + \lambda_3 P_1$.
> - **Backoff**: uses the highest-order n-gram **if its count is non-zero**, otherwise **falls back** to a lower order.
> - Interpolation generally outperforms backoff because it uses all evidence simultaneously rather than discarding higher-order information when available.
<!--ID: 1777386639923-->





> [!question]- What is absolute discounting?
> Subtract a fixed constant $d$ (typically $\approx 0.75$) from every non-zero count, then redistribute the saved mass to unseen events via a lower-order model:
> $$P_{\text{AD}}(w_i \mid w_{i-1}) = \frac{\max(C(w_{i-1}, w_i) - d, 0)}{C(w_{i-1})} + \lambda(w_{i-1}) P(w_i)$$
> Empirically justified by Church & Gale (1991), who found bigram counts in held-out data are consistently $\approx c - 0.75$ relative to training. Smarter than add-k because the discount is fixed, not proportional.
<!--ID: 1777386639924-->





> [!question]- What is the key idea behind Kneser-Ney smoothing?
> Replace the lower-order unigram $P(w)$ with a **continuation probability** — how many *distinct contexts* $w$ has appeared in:
> $$P_{\text{cont}}(w) \propto |\{v : C(v, w) > 0\}|$$
> A word like *Kong* has very high raw frequency but appears only after *Hong*, so its continuation probability is low — correctly predicting it is unlikely after a novel context. *Glasses* appears after many words, so its continuation probability is high. Combined with absolute discounting → **interpolated Kneser-Ney**, the production default for n-gram LMs.
<!--ID: 1777386639925-->


> [!question]- What is Kneser-Ney smoothing?
> A smoothing method for n-gram models that handles unseen bigrams by backing off to a **continuation probability** instead of raw word frequency.
>
> Two key ideas:
> 1. **Absolute discounting** — subtract a fixed $d \approx 0.75$ from every seen bigram count, freeing up probability mass
> 2. **Continuation probability** — when backing off, prefer words that appear after *many* distinct words, not just frequent words
>
> $$P_{\text{KN}}(w_i \mid w_{i-1}) = \frac{\max(C(w_{i-1}, w_i) - d,\ 0)}{C(w_{i-1})} + \lambda(w_{i-1})\ P_{\text{cont}}(w_i)$$
>
> **Example:** "Francisco" is frequent but only ever follows "San" → low continuation probability → bad fallback guess. "Glasses" follows many words → high continuation probability → good fallback guess.





> [!question]- What is stupid backoff and when is it used?
> A heuristic backoff (Brants et al., 2007): if the higher-order count exists, use it; otherwise multiply the lower-order estimate by a fixed constant ($0.4$). **Does not produce a valid probability distribution** — masses don't sum to 1 — but it's extremely fast and works well at web scale (trillions of words). Used when training data is so large that distributional differences between smoothing methods vanish, leaving only computational cost as the differentiator.
<!--ID: 1777386639926-->


> [!question]- What are the core limitations of n-gram language models?
> 1. **Data sparsity / zero probabilities** — most n-grams never appear in training, so MLE assigns zero probability to unseen sequences, breaking sentence scoring entirely.
> 2. **Fixed, short context window** — the Markov assumption limits conditioning to $n-1$ words. Long-range dependencies (e.g. subject–verb agreement across a clause) are invisible to the model.
> 3. **Exponential memory growth** — storing counts for all possible n-grams requires $O(V^n)$ space; trigram models on large vocabularies already require gigabytes.
> 4. **No generalisation across similar words** — "cat" and "cats" are completely unrelated entries; the model cannot transfer what it knows about one to the other (no notion of similarity).
> 5. **Memorisation over generation** — higher $n$ produces more fluent-looking text but by copying training fragments, not by understanding language.
> 6. **Smoothing is a patch, not a fix** — techniques like Kneser-Ney redistribute probability mass but cannot invent knowledge about contexts the model has never seen.
<!--ID: 1777549264713-->







