---
type: concept
sources:
  - raw/week-03/w03-slides.pdf
  - raw/week-03/w03-l07-transcript.txt
  - raw/week-03/w03-l08-transcript.txt
  - raw/week-03/w03-lab-01.pdf
  - raw/week-03/w03-lab-02.pdf
  - raw/week-03/w03-lab-01-solutions.pdf
  - raw/week-03/w03-lab-02-solutions.pdf
status: draft
updated: 2026-04-24
---

*Naive Bayes classifies a document by scoring each class with a prior times a product of per-word likelihoods, assuming words are conditionally independent given the class.*

## The Classifier in One Line

Given a document $d$ and a class set $C$, pick the class with the highest posterior — the **maximum a posteriori** (MAP) class. Apply [[bayes-rule|Bayes' rule]], and drop the denominator $P(d)$ since it's constant across classes and can't change the $\arg\max$:

$$c_{MAP} = \underset{c \in C}{\arg\max}\ P(c \mid d) = \underset{c \in C}{\arg\max}\ \underbrace{P(d \mid c)}_{\text{likelihood}} \cdot \underbrace{P(c)}_{\text{prior}}$$

This is the core decision rule. Everything that follows is about estimating those two terms.

The prior $P(c)$ is not decorative — it's the **base rate** of the class in the training corpus, and ignoring it is the machine-learning version of [[bayes-rule#Base Rate Neglect: The Steve Problem|base-rate neglect]]: a model that relies only on $\prod_i P(x_i \mid c)$ will over-predict rare classes whose word patterns happen to match well.

## The "Naive" Assumption

A document $d$ is a sequence of features $x_1, \ldots, x_n$ — typically the words. In general, $P(x_1, \ldots, x_n \mid c)$ has $O(|V|^n \cdot |C|)$ parameters, which cannot be estimated from any realistic corpus.

Naive Bayes makes two simplifying assumptions:

1. **Bag of Words**: position doesn't matter — $P(\text{word at position } i \mid c)$ is the same for every $i$.
2. **Conditional independence**: given the class, the feature probabilities are independent.

Together:

$$P(x_1, \ldots, x_n \mid c) \approx \prod_{i=1}^{n} P(x_i \mid c)$$

This is "naive" because words in a real document are obviously not independent — seeing *"San"* makes *"Francisco"* much more likely. The assumption is wrong. It works anyway, because for classification you only need the *relative ordering* of class scores to be right, not the absolute probabilities.

## The Multinomial Naive Bayes Classifier

Combining the two assumptions gives the final decision rule:

$$c_{NB} = \underset{c \in C}{\arg\max}\ P(c) \prod_{i \in \text{positions}} P(x_i \mid c)$$

In log space (to avoid floating-point underflow from multiplying many small probabilities):

$$c_{NB} = \underset{c \in C}{\arg\max}\ \left[ \log P(c) + \sum_{i \in \text{positions}} \log P(x_i \mid c) \right]$$

Taking logs doesn't change the ranking of classes. And this log-space form reveals a property: the score is a **linear function** of the input features, so Naive Bayes is a **linear classifier**.

## Learning: Maximum Likelihood Estimation

Two types of parameters to estimate from a training corpus. Both follow the same [[maximum-likelihood-estimation-nlp|MLE]] pattern that n-gram LMs use: partition the data by the conditioning context, then count and normalize within each partition.

**Class priors** — how often does each class occur?

$$\hat{P}(c_j) = \frac{N_{c_j}}{N_{\text{total}}}$$

where $N_{c_j}$ is the number of documents labelled $c_j$ and $N_{\text{total}}$ is the total number of training documents.

**Word likelihoods** — how often does each word appear in documents of each class?

$$\hat{P}(w_i \mid c_j) = \frac{\text{count}(w_i, c_j)}{\sum_{w \in V} \text{count}(w, c_j)}$$

Operationally: **concatenate all documents of class $c_j$ into one mega-document** and compute the relative frequency of each word. The denominator is the total token count of that mega-document.

### The MLE zero-probability problem

If a word $w$ never appears with class $c$ in training, MLE assigns $\hat{P}(w \mid c) = 0$. Because the final score is a product, **one zero anywhere zeroes the whole class** — no matter what other evidence points to that class. If *fantastic* was never seen in positive-class training data, a glowing review containing it will always score zero for `positive`.

## Laplace (Add-One) Smoothing

The fix is the same as for n-gram models (see [[smoothing]]): pretend every word was seen one extra time in every class.

$$\hat{P}(w_i \mid c) = \frac{\text{count}(w_i, c) + 1}{\left(\sum_{w \in V} \text{count}(w, c)\right) + |V|}$$

Add 1 to the numerator; add $|V|$ to the denominator (one for each vocabulary word, so the distribution still sums to 1). More generally you can add $\alpha$ (add-$\alpha$ smoothing); $\alpha = 1$ is Laplace.

Unlike n-gram LMs — where add-one smoothing is too aggressive in practice — it works well for Naive Bayes text classification because the [[bag-of-words|BoW]] vocabulary is smaller than the bigram space and the classifier only needs relative ranking between classes.

## Handling Unknown Words and Stop Words

**Unknown words** (appear at test time but not in training vocabulary): **ignore them**. Remove them from the test document and compute the score over the remaining tokens. Building a separate unknown-word model doesn't help — knowing which class has more unknown words is not informative.

**Stop words** (very frequent words like *the*, *a*, *and*): you *can* remove them, but in practice it rarely helps. Most NB text classifiers use all words without a stopword list — irrelevant features contribute roughly equal likelihood to every class and cancel out.

## Training Algorithm (Multinomial NB)

**In words.** From a training corpus:

1. **Extract vocabulary** $V$ from all training documents.
2. **Compute priors.** For each class $c_j$, count how many documents carry that label and divide by the total number of documents: $P(c_j) = |\text{docs}_j| / |\text{total docs}|$.
3. **Build a per-class mega-document.** Concatenate every document of class $c_j$ into a single big document $\text{Text}_j$. This collapses per-document boundaries so you can count word occurrences in the whole class at once.
4. **Compute word likelihoods.** For each word $w_k \in V$, count its occurrences $n_k$ in $\text{Text}_j$, then apply Laplace smoothing: $\hat{P}(w_k \mid c_j) = (n_k + \alpha) / (n + \alpha \cdot |V|)$, where $n$ is the total token count in $\text{Text}_j$.
5. **Store** one prior per class and one likelihood table of size $|C| \times |V|$.

Two passes over the data: one over documents (for priors), one over the concatenated mega-documents (for likelihoods). $\alpha = 1$ gives Laplace smoothing; larger $\alpha$ smooths more aggressively.

**Pseudocode** (matching the slides):

```
Input: training documents D with labels, class set C
Output: priors P(c_j) and likelihoods P(w_k | c_j)

From the training corpus, extract Vocabulary V.

For each class c_j in C:
    docs_j  ← all documents with class = c_j
    P(c_j)  ← |docs_j| / |total # documents|

    Text_j  ← concatenation of all documents in docs_j
    n       ← total token count in Text_j

    For each word w_k in Vocabulary:
        n_k             ← # of occurrences of w_k in Text_j
        P(w_k | c_j)    ← (n_k + α) / (n + α · |V|)
```

## Worked Example (from the lab)

Training corpus (5 reviews, `+` = comedy, `−` = action):

| Doc | Tokens | Class |
|---|---|---|
| 1 | fun, couple, love, love | comedy |
| 2 | fast, furious, shoot | action |
| 3 | couple, fly, fast, fun, fun | comedy |
| 4 | furious, shoot, shoot, fun | action |
| 5 | fly, fast, shoot, love | action |

**Priors**: $\log P(\text{comedy}) = \log(2/5) = -0.398$, $\log P(\text{action}) = \log(3/5) = -0.222$.

**Vocabulary**: $V = \{\text{fun, couple, love, fast, furious, shoot, fly}\}$, so $|V| = 7$. Comedy mega-doc has 9 tokens; action mega-doc has 11.

**Laplace-smoothed likelihood for `fun | comedy`**: $(3 + 1)/(9 + 7) = 4/16$.

**Test document** $D$ = *"fast, couple, shoot, fly"*. Score:

- $\log\text{sum}(\text{comedy}) = -0.398 + \log(2/16) + \log(3/16) + \log(1/16) + \log(2/16) = -4.135$
- $\log\text{sum}(\text{action}) = -0.222 + \log(3/18) + \log(1/18) + \log(4/18) + \log(2/18) = -3.863$

$-3.863 > -4.135$, so $D$ is classified as **action**.

## Binary Multinomial Naive Bayes

For tasks like [[sentiment-analysis|sentiment]], word *occurrence* matters more than word *frequency* — seeing *fantastic* once tells you the review is positive; seeing it five times adds little. **Binary multinomial NB** clips counts at 1 *per document* before training.

### Learning Algorithm (Binary NB)

**In words.** Same as multinomial NB, with one preprocessing step added at the front:

1. **Binarize every training document.** For each document, remove duplicates so every word appears at most once. The document *"great scenes great film"* becomes *"great scenes film"*.
2. Then run the standard [[#Training Algorithm (Multinomial NB)|multinomial NB training algorithm]] on the binarized documents: extract vocabulary, compute priors, concatenate per-class mega-documents, count words, apply Laplace smoothing.
3. **At test time**, binarize the input document in the same way before scoring with the usual equation $c_{NB} = \arg\max_c P(c) \prod_{i} P(w_i \mid c)$.

The key invariant: **binarization is within-doc, not within-corpus**. A word's count across the full corpus can still exceed 1 — if *great* appears in three different training documents, the class mega-document has count 3. It's the counts *within each document* that are clipped to 0 or 1.

**Pseudocode**:

```
Input: training documents D with labels, class set C
Output: priors P(c_j) and likelihoods P(w_k | c_j)

For each document d in D:
    d ← set(d)    // keep only unique words (per-document binarization)

Run the multinomial NB training algorithm on the binarized documents.
```

This is **different from Bernoulli Naive Bayes**, which uses a binary feature per vocabulary word for the whole document and explicitly models word *absence* (one term per vocabulary word at test time, including `1 − P(w | c)` for words that didn't appear). Binary multinomial NB still uses the multinomial likelihood formula — it just feeds it binarized counts.

### Worked Example (from the slides)

Four training documents, two positive and two negative:

| Class | Document |
|---|---|
| − | it was pathetic the worst part was the boxing scenes |
| − | no plot twists or great scenes |
| + | and satire and great plot twists |
| + | great scenes great film |

**Step 1 — per-document binarization.** Remove within-document duplicates:

| Class | Binarized document |
|---|---|
| − | it was pathetic the worst part boxing scenes |
| − | no plot twists or great scenes |
| + | and satire great plot twists |
| + | great scenes film |

Words that were duplicated within a single document (*was* in doc 1, *and*/*great* in doc 3, *great* in doc 4) each drop to one occurrence.

**Step 2 — count per class.** Side-by-side comparison:

| Word | NB counts (+ / −) | Binary counts (+ / −) |
|---|---|---|
| and | 2 / 0 | 1 / 0 |
| boxing | 0 / 1 | 0 / 1 |
| film | 1 / 0 | 1 / 0 |
| great | 3 / 1 | **2 / 1** |
| it | 0 / 1 | 0 / 1 |
| no | 0 / 1 | 0 / 1 |
| or | 0 / 1 | 0 / 1 |
| part | 0 / 1 | 0 / 1 |
| pathetic | 0 / 1 | 0 / 1 |
| plot | 1 / 1 | 1 / 1 |
| satire | 1 / 0 | 1 / 0 |
| scenes | 1 / 2 | 1 / 2 |
| the | 0 / 2 | **0 / 1** |
| twists | 1 / 1 | 1 / 1 |
| was | 0 / 2 | **0 / 1** |
| worst | 0 / 1 | 0 / 1 |

The highlighted rows are where binarization changes something: `great` in the positive class drops from 3 to 2 (it appeared twice in "great scenes great film"); `the` and `was` in the negative class drop from 2 to 1 (both appeared twice in the first document).

**Takeaway from the counts table**: binarization mostly matters for words that a single document repeats. `scenes` had counts 1/2 in both tables — the `/2` on the negative side came from two *different* documents each mentioning it once, so binarization leaves it alone. `great` in the positive class had three occurrences from two documents — binarization drops the within-doc repeat but keeps the cross-doc signal. Raw multinomial NB lets a single repetitive document dominate the likelihood; binary NB normalizes out that effect.

### When Binary NB Wins

Binary NB and multinomial NB often give different predictions on the same document. See `wiki/concepts/sentiment-analysis.md` for a worked case where they disagree on *"A good, good plot and great characters, but poor acting"* — multinomial NB says positive, binary NB says negative. Binarization helps whenever the task cares about *whether* a word occurred, not *how many times*.

## Relationship to Language Modelling

### The equivalence (and its precondition)

Naive Bayes is a general classifier. It can use **any sort of feature** — URL patterns, email headers, dictionary lookups, network features, hand-crafted indicator functions — not just words.

But when two conditions hold:

1. We use **only word features** (nothing but tokens).
2. We use **all** of the words in the document (no feature selection, no stopword filtering).

…then each class's likelihood $\prod_i P(w_i \mid c)$ is exactly a **unigram [[n-gram-language-models|language model]]** conditioned on the class. Naive Bayes classification becomes: score the document under each class's per-class unigram LM and pick the highest-scoring class.

### Generative view

The slides present this as a generative graphical model. The class is a parent variable that "generates" each word independently:

```
                   ┌───────────┐
                   │ c = China │
                   └─────┬─────┘
         ┌─────────┬─────┼──────┬──────────┐
         ▼         ▼     ▼      ▼          ▼
   X₁=Shanghai  X₂=and  X₃=Shenzhen  X₄=issue  X₅=bonds
```

To generate a document of class `China`, sample each word independently from $P(\cdot \mid c=\text{China})$ — exactly the same process as sampling from a unigram LM whose distribution is $P(\cdot \mid c=\text{China})$. Running this backwards (scoring a given document) is what the classifier does.

This also makes clear *why* the independence assumption is "naive": in real text, after sampling "Shanghai" the next word is not independent of it ("and" becomes much more likely, for instance). The arrows in the diagram should really have *also* pointed from each $X_i$ to $X_{i+1}$. The model cuts those edges for tractability.

### Worked example (from the slides)

Two per-class unigram LMs, one for `pos` and one for `neg`, each assigning a probability to every word in the vocabulary:

| Word | $P(w \mid \text{pos})$ | $P(w \mid \text{neg})$ |
|---|---|---|
| I | 0.1 | 0.2 |
| love | 0.1 | 0.001 |
| this | 0.01 | 0.01 |
| fun | 0.05 | 0.005 |
| film | 0.1 | 0.1 |

Score the sentence **"I love this fun film"** under each class by multiplying word probabilities:

$$P(s \mid \text{pos}) = 0.1 \times 0.1 \times 0.01 \times 0.05 \times 0.1 = 5 \times 10^{-7}$$

$$P(s \mid \text{neg}) = 0.2 \times 0.001 \times 0.01 \times 0.005 \times 0.1 = 1 \times 10^{-9}$$

$P(s \mid \text{pos}) > P(s \mid \text{neg})$, so (with equal priors) the classifier labels the sentence as **positive**. The signal comes primarily from `love` (100× more likely under `pos`) and `fun` (10× more likely) — the two words where the class-conditional LMs most disagree.

### Why the equivalence matters

Seeing Naive Bayes as "one unigram LM per class" explains *why* the classifier works despite the naive assumption: it's solving the same kind of problem as an n-gram LM, only with a small number of target distributions (one per class) instead of one unified distribution. Every technique from n-gram modelling transfers — [[smoothing|Laplace smoothing]], log-space arithmetic, sparsity handling — because these are properties of the underlying categorical estimator, not of the LM application.

## Why It Works Despite "Naive" Assumptions

- **Very fast, low storage** — training is a single pass of counting; the model is a table of class priors and per-class word likelihoods.
- **Works on small training data** — parameters are well-estimated from modest corpora.
- **Robust to irrelevant features** — they contribute roughly equal likelihood to every class and cancel in the $\arg\max$.
- **Strong in domains with many equally-important features** — decision trees suffer from *fragmentation* when many features are equally informative (ties force arbitrary splits); NB just multiplies them together.
- **Optimal if the independence assumption holds** — then it is the Bayes Optimal Classifier.
- **Reliable baseline** — the first thing you should try on any new text classification task before reaching for anything fancier.

## Applications Beyond Sentiment

- **Spam filtering** (e.g. SpamAssassin): features like "mentions millions of dollars", "From starts with many numbers", "subject is all caps", "one hundred percent guaranteed".
- **Language identification**: features are character n-grams (not word features). Characters work better than words because different languages share little vocabulary. Important to train on *varieties* of each language (African-American English, Indian English) or [[harms-in-classification|performance disparities]] emerge.
- **Authorship identification**: the *Federalist Papers* problem — Mosteller and Wallace (1963) used Bayesian methods to attribute 12 disputed essays.

## Related

- [[bayes-rule]] — the probabilistic identity Naive Bayes is built on
- [[bag-of-words]] — the assumed document representation
- [[text-classification]] — the general task Naive Bayes solves
- [[sentiment-analysis]] — the canonical application; where binary NB shines
- [[smoothing]] — Laplace smoothing reused from n-gram LMs
- [[n-gram-language-models]] — each NB class is equivalent to a unigram LM
- [[classification-evaluation]] — how to measure the quality of an NB classifier

## Active Recall

> [!question]- Starting from $P(c \mid d) \propto P(d \mid c) P(c)$, how does Naive Bayes simplify $P(d \mid c)$ and why?
> It applies two assumptions: **bag of words** (position doesn't matter, so $P(x_i \mid c)$ is the same at every position) and **conditional independence** given the class. Together these let you factor $P(x_1, \ldots, x_n \mid c) = \prod_i P(x_i \mid c)$ — a product of single-word likelihoods instead of a joint distribution over all positions. Without this, $|V|^n$ parameters would be needed per class, which no corpus can estimate.

> [!question]- Why does Naive Bayes compute classification in log space, and what's the guarantee?
> Multiplying many small probabilities underflows 64-bit floating point. Taking logs turns products into sums, keeping the arithmetic numerically stable. The guarantee: the log is a monotonic function, so $\arg\max \log P = \arg\max P$ — the ranking of classes is identical. A side effect: the log-space form reveals NB as a linear classifier (a max of a linear function of the inputs).

> [!question]- What goes wrong without Laplace smoothing, and why does adding 1 fix it?
> Without smoothing, any word that never appears with class $c$ in training gets $\hat{P}(w \mid c) = 0$. Since the final score is a product, one zero zeroes the whole class — even if every other word in the document strongly suggests $c$. Laplace pretends every vocabulary word was seen one extra time in each class. This makes no zero possible while barely disturbing the estimates of words with large counts.

> [!question]- How is binary multinomial Naive Bayes different from regular multinomial NB, and when does it win?
> Binary NB **binarizes token counts within each document** before training and scoring: each word counts 0 or 1 per document, not its raw frequency. Corpus-wide counts can still exceed 1. It tends to win on tasks where *occurrence* carries the signal and *repetition* adds noise — sentiment analysis is the canonical case: seeing *fantastic* once already tells you; seeing it five times doesn't add five times the evidence.

> [!question]- Why is each class in a bag-of-words Naive Bayes classifier equivalent to a unigram language model?
> A class's likelihood under NB is $\prod_i P(w_i \mid c)$ — the probability of generating the document as independent draws from $P(\cdot \mid c)$. That is exactly a **unigram LM** conditioned on the class. Classification becomes: evaluate the document under each class's unigram LM and pick the highest-scoring one.

> [!question]- Walk through the Naive Bayes training algorithm in 5 bullet points.
> (1) Extract the vocabulary $V$ from the training corpus. (2) For each class $c_j$, compute $P(c_j) = |\text{docs}_j|/|\text{total docs}|$. (3) Concatenate all documents of class $c_j$ into a single mega-document $\text{Text}_j$. (4) For each word $w_k \in V$, count its occurrences $n_k$ in $\text{Text}_j$ and set $\hat{P}(w_k \mid c_j) = (n_k + 1)/(n + |V|)$ with Laplace smoothing. (5) Store the priors and per-class log-likelihoods; at test time, sum the appropriate logs and take $\arg\max$.

> [!question]- Why should unknown words simply be ignored at test time rather than assigned a special UNK probability?
> Knowing which class has more unknown words is not generally informative — unknown words appear in every class roughly in proportion to unseen vocabulary. Adding a UNK term would usually wash out across classes and not help the decision. Ignoring them (removing from the test document entirely) is cleaner and loses nothing useful. This is the opposite of the n-gram LM case, where unknown words break the probability formula entirely.

> [!question]- Given the corpus `[d1: (good=3, poor=0, great=3, +), d2: (good=0, poor=1, great=2, +), d3: (good=1, poor=3, great=0, −), d4: (good=1, poor=5, great=2, −), d5: (good=0, poor=2, great=0, −)]`, compute Laplace-smoothed $P(\text{good} \mid +)$ and $P(\text{good} \mid -)$.
> Positive class has 9 tokens total (3+0+3+0+1+2 from d1, d2); negative class has 14 tokens total (1+3+0+1+5+2+0+2+0 from d3, d4, d5). Vocabulary size is 3. **$\hat{P}(\text{good} \mid +) = (3+1)/(9+3) = 4/12 = 0.333$.** **$\hat{P}(\text{good} \mid -) = (2+1)/(14+3) = 3/17 \approx 0.176$.** Good is about twice as likely under positive — the model has learned that *good* is a positive-class signal.
