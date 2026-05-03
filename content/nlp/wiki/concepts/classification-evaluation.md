---
type: concept
sources:
  - raw/week-03/w03-slides.pdf
  - raw/week-03/w03-l09-transcript.txt
  - raw/week-03/w03-lab-02.pdf
  - raw/week-03/w03-lab-02-solutions.pdf
  - raw/week-03/3 NB and Sentiment Classification - Consolidation-1.pdf
status: draft
updated: 2026-04-24
---

*A classifier's score isn't a single number — it's a confusion matrix, from which accuracy, precision, recall, and F1 are derived. Accuracy lies on imbalanced classes; the others don't.*

## The Confusion Matrix

For a binary classifier, every test example falls into one of four cells based on its **gold label** (the human-annotated correct answer) and the **system output label**:

| | gold positive | gold negative |
|---|---|---|
| **system positive** | true positive (tp) | false positive (fp) |
| **system negative** | false negative (fn) | true negative (tn) |

These four counts are the basis for every evaluation metric below.

## Accuracy — and Why It Fails on Imbalanced Classes

$$\text{accuracy} = \frac{\text{tp} + \text{tn}}{\text{tp} + \text{fp} + \text{tn} + \text{fn}}$$

"Of all predictions, what fraction were correct?" Intuitive, but it **breaks under class imbalance**.

**Example (from the slides)**: 1,000,000 social media posts; 100 are about Delicious Pie Co; 999,900 are unrelated. A stupid classifier that says "not about pie" for every post achieves:

- 999,900 true negatives, 100 false negatives, 0 true positives, 0 false positives.
- **Accuracy = 999,900 / 1,000,000 = 99.99%**.

Completely useless at finding pie-related posts — its recall on the positive class is zero — yet it scores near-perfect on accuracy. For any task where you care about a minority class, accuracy is the wrong headline number.

## Precision and Recall

Both are defined **with respect to a specific class** (usually the positive class):

$$\text{precision} = \frac{\text{tp}}{\text{tp} + \text{fp}} \qquad \text{recall} = \frac{\text{tp}}{\text{tp} + \text{fn}}$$

- **Precision**: of all items the system labelled positive, what fraction actually are? → "how trustworthy are its positive predictions?"
- **Recall**: of all items that actually are positive, what fraction did the system find? → "how much of the positive class did it catch?"

The "say no to everything" classifier above has tp = 0, so both precision and recall collapse to 0. Neither is fooled by class imbalance the way accuracy is.

**Trade-off**: tightening a classifier (raising a decision threshold) typically raises precision but lowers recall; loosening it does the opposite. Which matters more depends on the cost of each error type — a cancer screener prefers recall (missing a case is deadly); a spam filter prefers precision (incorrectly filtering legitimate mail is disruptive).

## F1 Score

A single number combining precision and recall:

$$F_1 = \frac{2 P R}{P + R}$$

This is the **harmonic mean** of precision and recall. Harmonic mean punishes extremes — if either P or R is near zero, $F_1$ is near zero. The "say no" classifier with P = R = 0 has $F_1 = 0$; no amount of gaming the other metric saves it.

### Why "harmonic mean"

The **harmonic mean** of $n$ positive numbers is

$$\text{HM}(a_1, \ldots, a_n) = \frac{n}{\sum_{i=1}^{n} \frac{1}{a_i}}$$

It's the reciprocal of the arithmetic mean of reciprocals. For two numbers $P$ and $R$:

$$\text{HM}(P, R) = \frac{2}{\frac{1}{P} + \frac{1}{R}} = \frac{2PR}{P + R}$$

That second form is exactly $F_1$. The reciprocals are why HM blows up when any input is tiny: if $P = 0.01$, then $1/P = 100$, which dominates the denominator sum and pulls the HM toward 0 no matter what $R$ is. Arithmetic mean doesn't have this property — $\text{AM}(1.0, 0.01) = 0.505$, but $\text{HM}(1.0, 0.01) \approx 0.02$.

### The general F-measure

$F_1$ is a special case of the general **F-measure**, which allows weighting P and R differently. Two equivalent parameterizations:

$$F_\alpha = \frac{1}{\alpha \frac{1}{P} + (1 - \alpha) \frac{1}{R}} \qquad F_\beta = \frac{(\beta^2 + 1) P R}{\beta^2 P + R}$$

The two forms are related by $\beta^2 = (1 - \alpha) / \alpha$. In the $\alpha$ form, $\alpha$ is the relative weight on precision; in the $\beta$ form, $\beta$ is the relative emphasis on recall (think of $\beta$ as "recall is $\beta$ times as important as precision").

$F_1$ corresponds to $\alpha = \tfrac{1}{2}$ or equivalently $\beta = 1$ — precision and recall weighted equally. $\beta > 1$ (e.g. $F_2$) weights recall more heavily; $\beta < 1$ (e.g. $F_{0.5}$) weights precision more.

### Worked calculation (from the lab)

Confusion matrix:

| Actual / Predicted | pos | neg |
|---|---|---|
| **pos** | 80 | 20 |
| **neg** | 30 | 70 |

For the `pos` class: tp = 80, fp = 30, fn = 20, tn = 70.

- Accuracy = $(80 + 70) / 200 = 0.75$
- Precision = $80 / (80 + 30) = 0.73$
- Recall = $80 / (80 + 20) = 0.80$
- $F_1 = 2 \cdot 0.73 \cdot 0.80 / (0.73 + 0.80) = 0.76$

## Multi-class Precision and Recall

When $|C| > 2$, precision and recall are defined **per class**: for class $c$, treat "is $c$" vs "is not $c$" as a binary problem and count tp/fp/fn/tn accordingly.

### Worked example (from the slides)

Three-way email classification — `urgent`, `normal`, `spam` — with the confusion matrix (rows = system output, columns = gold):

| | gold urgent | gold normal | gold spam |
|---|---|---|---|
| **system urgent** | 8 | 10 | 1 |
| **system normal** | 5 | 60 | 50 |
| **system spam** | 3 | 30 | 200 |

Per-class precision (true positives over all predictions of that class — the row sum):

$$\text{precision}_u = \frac{8}{8 + 10 + 1} = 0.42 \qquad \text{precision}_n = \frac{60}{5 + 60 + 50} = 0.52 \qquad \text{precision}_s = \frac{200}{3 + 30 + 200} = 0.86$$

Per-class recall (true positives over all gold instances of that class — the column sum):

$$\text{recall}_u = \frac{8}{8 + 5 + 3} = 0.50 \qquad \text{recall}_n = \frac{60}{10 + 60 + 30} = 0.60 \qquad \text{recall}_s = \frac{200}{1 + 50 + 200} = 0.80$$

The classifier is good at spam, mediocre at normal, weak at urgent. A single-number summary needs a way to combine these three per-class scores — see below.

## Macro vs Micro Averaging

Combining per-class precision (or recall, or $F_1$) into one number takes two forms:

### Macroaveraging

Compute precision (or recall, or $F_1$) **for each class separately**, then **average** across classes.

$$\text{macro-precision} = \frac{1}{|C|} \sum_{c \in C} \text{precision}_c$$

**Treats every class equally** regardless of how many examples it has. Good when all classes matter equally (e.g. sentiment analysis with balanced positive/negative/neutral). Bad when a rare class with few examples produces a noisy estimate that drags the average down.

### Microaveraging

**Pool** all the true positives, false positives, and false negatives across classes first, then compute a single precision.

$$\text{micro-precision} = \frac{\sum_c \text{tp}_c}{\sum_c (\text{tp}_c + \text{fp}_c)}$$

**Treats every example equally.** Dominated by the majority class. Good when you care about average-case performance; bad when minority classes matter.

**The two can differ dramatically.** Continuing the urgent/normal/spam example, first convert the 3×3 matrix into three per-class binary confusion matrices (one per class: "is this class" vs "isn't"):

**Class 1 — Urgent:**

| | true urgent | true not |
|---|---|---|
| **system urgent** | 8 | 11 |
| **system not** | 8 | 340 |

$\text{precision}_u = 8 / (8 + 11) = 0.42$

**Class 2 — Normal:**

| | true normal | true not |
|---|---|---|
| **system normal** | 60 | 55 |
| **system not** | 40 | 212 |

$\text{precision}_n = 60 / (60 + 55) = 0.52$

**Class 3 — Spam:**

| | true spam | true not |
|---|---|---|
| **system spam** | 200 | 33 |
| **system not** | 51 | 83 |

$\text{precision}_s = 200 / (200 + 33) = 0.86$

**Pooled (for microaveraging):** sum the tp, fp, fn, tn across all three binary matrices:

| | true yes | true no |
|---|---|---|
| **system yes** | 268 | 99 |
| **system no** | 99 | 635 |

Now compute each average:

$$\text{macro-precision} = \frac{0.42 + 0.52 + 0.86}{3} = 0.60$$

$$\text{micro-precision} = \frac{268}{268 + 99} = 0.73$$

The **micro value (0.73) is higher than the macro value (0.60)** because the large, easy class (spam, 200 true positives) dominates the pooled sum, while the small hard class (urgent, precision 0.42) drags the macro average down. Macro-averaging exposes the per-class weakness; micro-averaging hides it.

### Worked multi-class (from the lab)

Confusion matrix:

| Actual / Predicted | pos | neut | neg |
|---|---|---|---|
| **pos** | 100 | 20 | 10 |
| **neut** | 330 | 120 | 20 |
| **neg** | 15 | 25 | 95 |

- **Accuracy** = $(100 + 120 + 95) / 735 = 0.43$
- **Positive**: precision = 100/445 = 0.22, recall = 100/130 = 0.77, $F_1$ = 0.34
- **Neutral**: precision = 120/165 = 0.73, recall = 120/470 = 0.26, $F_1$ = 0.38
- **Negative**: precision = 95/125 = 0.76, recall = 95/135 = 0.70, $F_1$ = 0.73
- **Macro-average**: precision = 0.57, recall = 0.57, $F_1$ = 0.57
- **Micro-average**: precision = 0.75, recall = 0.61, $F_1$ = 0.67

The classifier is strongly biased toward predicting `pos` (it over-fires), which hurts its precision on that class. Macro-averaging exposes the per-class imbalance; micro-averaging hides it.

## Devsets and Cross-Validation

Before evaluating on the test set, you need a separate "dress rehearsal" set for tuning — otherwise every hyperparameter choice leaks information from the test set into the model.

### Three-way split

```
┌──────────────┬──────────────────┬──────────┐
│ Training set │ Development set  │ Test set │
│              │    ("devset")    │          │
└──────────────┴──────────────────┴──────────┘
```

- **Training set**: fit parameters (counts, priors, smoothed likelihoods).
- **Development set (devset)**: tune hyperparameters — smoothing $\alpha$, feature choices, thresholds, whether to use binary or multinomial NB.
- **Test set**: report the final number. Used **once**, at the very end.

Train on training, **tune on devset, report on test**. The devset prevents **overfitting to the test set** — if you keep rerunning on the test set and tweaking until the number improves, you've leaked test-set information into your model and your reported score overestimates real-world performance.

**The paradox**: you want as much data as possible for training *and* as much as possible for the devset. Every example you put in one is an example not in the other. Cross-validation resolves this.

### $k$-fold cross-validation

Instead of a fixed devset, rotate which slice of the data acts as devset.

```
Iter 1: [ Dev │        Training                   ]
Iter 2: [     │ Dev │  Training                   ]
Iter 3: [           │ Dev │      Training         ]
 …                                                  → Test set (untouched)
Iter k: [                  Training          │ Dev ]
```

Split the non-test data into $k$ equal folds (typically $k = 10$). Run $k$ training iterations; in each, use one fold as the devset and the remaining $k-1$ as training. Pool the dev-set performance scores across folds to get a more stable estimate. The test set stays untouched across all folds — it's only used at the end.

**Why pool**: a single 90/10 split might get an easy (or hard) devset by chance. Averaging across 10 different devsets cancels out that variance. The trade-off is cost — $k$-fold is $k$ times more expensive to train.

## Statistical Significance: Hypothesis Testing

Given classifiers A and B with F1 of 0.81 and 0.83 on the same test set, is B *really* better — or did it get lucky on this test set? Asking the question formally:

### Setup: effect size and hypotheses

- Let $M(\text{A}, x)$ be classifier A's performance on test set $x$ under some metric $M$ (accuracy, F1, whatever).
- The **effect size** is the observed difference: $\delta(x) = M(\text{A}, x) - M(\text{B}, x)$.
- Null hypothesis $H_0$: A is **not** better than B, so $\delta(x) \le 0$. Any observed positive $\delta$ is coincidence.
- Alternative $H_1$: A is better than B, so $\delta(x) > 0$.

We want to **reject $H_0$**. The question is: assuming $H_0$ is true, how likely is it that we would see a $\delta$ as large as the one we observed?

### The p-value

$$p\text{-value} = P(\delta(X) \ge \delta(x) \mid H_0 \text{ is true})$$

$X$ is a random variable ranging over test sets drawn from the underlying population. The p-value is the probability that, across those hypothetical test sets, we would see an effect size at least as large as the one actually observed — *if* the null hypothesis were true.

- **Very small p-value** (say below 0.01 or 0.05) → the observed effect is unlikely under $H_0$; we **reject** $H_0$ and call the result **statistically significant**.
- Not-small p-value → we cannot distinguish the observation from noise; we fail to reject $H_0$.

The threshold (0.05, 0.01) is a social convention, not a mathematical constant.

### Parametric vs non-parametric tests

Parametric tests (e.g. the paired t-test) assume the differences $\delta(X)$ follow a specific distribution, usually Gaussian. In NLP, the metrics are typically non-linear functions of counts (F1, BLEU, etc.) and *not* Gaussian-distributed. So NLP overwhelmingly uses **non-parametric, sampling-based tests**:

1. **Approximate randomization** — randomly swap labels between A and B on each test item many times to see how often you get an effect as large as $\delta(x)$ by chance.
2. **Paired bootstrap** — resample the test set with replacement to simulate many alternative test sets (see below).

Both are **paired tests**: they assume each test item has a pair of observations (one from each system on the *same* test item). Pairing is much more powerful than unpaired testing because it removes per-example variance.

## The Paired Bootstrap Test

The bootstrap (Efron & Tibshirani, 1993) repeatedly draws large numbers of smaller samples **with replacement** from an original larger sample — each draw is a **bootstrap sample**. It works for any metric: accuracy, precision, recall, F1, BLEU, etc.

### Intuition: build a distribution from one test set

You have one real test set $x$ of size $n$. You can't collect more test sets, but you can *simulate* new ones by resampling rows from $x$ with replacement. Each simulated test set $x^{(i)}$ has the same size $n$ but a different composition — some rows duplicated, others missing. Run the bootstrap $b$ times (say 10,000) and you have a distribution of $\delta(x^{(i)})$ values, from which you can estimate how "accidental" the original $\delta(x)$ is.

### A concrete example

Consider a baby test set of $n = 10$ documents under accuracy, with the four possible outcomes per document encoded as (A correct / B correct, A correct / B wrong, …):

| doc | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A% | B% | $\delta$ |
|-----|---|---|---|---|---|---|---|---|---|----|----|----|----------|
| $x$ | AB | Aḃ | AB | Ȧ B | Aḃ | Ȧ B | Aḃ | AB | Ȧ Ḃ | Aḃ | .70 | .50 | .20 |

(Each column shows which of A, B was correct; A struck through = A wrong.) On the real test set, A scores 70%, B scores 50%, and $\delta(x) = 0.20$.

Now draw 10,000 bootstrap samples $x^{(i)}$. Each is 10 columns drawn uniformly at random with replacement from the 10 columns above. Compute A%, B%, and $\delta$ on each.

### The subtlety: shift to zero mean

Naïvely, you might count: *in what fraction of bootstrap samples is $\delta(x^{(i)}) \ge 0$?* That's wrong, because **you drew the bootstrap samples from $x$, which is itself biased by $\delta(x) = 0.20$ in favour of A**. Under $H_0$ (A no better than B), the expected $\delta$ is 0, not 0.20. So you have to shift the observed bootstrap $\delta$'s down by $\delta(x)$ to centre the distribution at zero, then ask how often the shifted value exceeds the original $\delta(x)$.

Equivalently: count how often $\delta(x^{(i)}) \ge 2\delta(x)$:

$$p\text{-value}(x) = \sum_{i=1}^{b} \mathbb{1}\!\left(\delta(x^{(i)}) - \delta(x) \ge \delta(x)\right) = \sum_{i=1}^{b} \mathbb{1}\!\left(\delta(x^{(i)}) \ge 2\delta(x)\right)$$

If only 47 out of 10,000 bootstrap samples show $\delta(x^{(i)}) \ge 2\delta(x) = 0.40$, then the p-value is $47/10000 = 0.0047 < 0.01$ — the observed 0.20 gap is unlikely to be an artefact of the particular test set, and we reject $H_0$: **A is significantly better than B**.

### Algorithm (after Berg-Kirkpatrick et al., 2012)

```
function Bootstrap(test set x, num samples b) returns p-value(x)
    Calculate δ(x)                      # observed effect size
    s ← 0
    for i = 1 to b do
        Draw bootstrap sample x⁽ⁱ⁾ of size n:
            for j = 1 to n do
                Select an item of x uniformly at random (with replacement)
                and add it to x⁽ⁱ⁾
        Calculate δ(x⁽ⁱ⁾)
        if δ(x⁽ⁱ⁾) > 2·δ(x):          # "accidentally" large under H₀
            s ← s + 1
    p-value(x) ≈ s / b
    return p-value(x)
```

### Why it's paired

Each bootstrap sample $x^{(i)}$ is a re-selection of *whole test items*, each of which carries **both** A's and B's output. This preserves the pairing — we never compare A on one set of items to B on a different set. Per-example variance (some examples are easy, some are hard) cancels out, making the test more powerful than comparing two separately-sampled distributions.

### Pros

- No distributional assumptions — works for F1, BLEU, ROUGE, any non-linear metric.
- Works on small test sets where parametric tests misestimate significance.
- Conceptually simple to implement: resample, recompute, count.

## Related

- [[naive-bayes]] — one of the classifiers this evaluates
- [[text-classification]] — the setting in which these metrics live
- [[sentiment-analysis]] — where class imbalance makes these choices real
- [[evaluation-methodology]] — train/dev/test split; intrinsic vs extrinsic evaluation (from [[week-02]])
- [[harms-in-classification]] — why aggregate metrics can hide disparate performance across groups

## Active Recall

> [!question]- Why does accuracy fail on imbalanced classes, and what metric choice repairs it?
> Accuracy counts all correct predictions equally. When one class is 99% of the data, a classifier that predicts that class for every example achieves 99% accuracy — while being completely useless at finding the minority class. **Precision and recall** (computed with respect to the minority class) repair this, because both go to zero when the minority class is never predicted correctly.

> [!question]- What is the difference between precision and recall, and which one matters more for spam filtering? What about for cancer screening?
> **Precision** = of items labelled positive, what fraction are correct. **Recall** = of items that are actually positive, what fraction were caught. Spam filtering cares about **precision** — wrongly marking legitimate mail as spam (a false positive) is very costly, so you want high confidence when you flag spam. Cancer screening cares about **recall** — missing a cancer case (a false negative) can be fatal, so you want to catch as many as possible, even at the cost of some false alarms to retest.

> [!question]- Why is F1 the harmonic mean of precision and recall rather than the arithmetic mean?
> The harmonic mean punishes extreme values — if either P or R is close to 0, harmonic mean is close to 0. Arithmetic mean lets a near-zero value be rescued by the other being near 1 (e.g. P=1.0, R=0.01 has arithmetic mean 0.505 but $F_1 \approx 0.02$). Because a useful classifier needs both precision *and* recall, $F_1$ correctly punishes classifiers that sacrifice one for the other.

> [!question]- When does macro-averaging give a very different answer from micro-averaging, and which should you report?
> They diverge whenever class sizes are imbalanced. **Micro** is dominated by the large class — a classifier that does well on the common class and poorly on rare ones gets a high micro score. **Macro** treats every class equally — it exposes poor performance on rare classes. Report **macro** when every class matters (sentiment analysis with equally important polarities), **micro** when you care about average-case performance (search where you mostly encounter common queries). Reporting both is often most informative.

> [!question]- What is the bootstrap method and why is it appropriate for comparing two classifiers?
> The bootstrap is resampling the test set with replacement to build up an empirical distribution of the performance metric. You repeat many times (1000+), compute the metric for each classifier on each resample, and look at the distribution of differences. It's appropriate because it makes **no distributional assumptions** about the data — NLP metrics like $F_1$ are nonlinear in the underlying counts and are typically not normally distributed, so parametric tests (like a paired t-test on accuracy) misestimate the significance.

> [!question]- Given a binary confusion matrix with tp=80, fp=30, fn=20, tn=70, compute accuracy, precision, recall, and F1 for the positive class.
> Total = 200. **Accuracy** = (80+70)/200 = **0.75**. **Precision** = 80/(80+30) = 80/110 = **0.73**. **Recall** = 80/(80+20) = 80/100 = **0.80**. **$F_1$** = 2(0.73)(0.80)/(0.73+0.80) = 1.168/1.53 ≈ **0.76**. The model has slightly higher recall than precision — it catches 80% of positive cases but 27% of its positive predictions are wrong.

> [!question]- Why do you need a separate devset, and when would you prefer cross-validation over a fixed devset?
> The **devset** is a dress-rehearsal for the test set — you use it to tune hyperparameters (smoothing $\alpha$, feature choices, decision thresholds) without ever touching the test set. Tuning on the test set leaks test-set information into the model and overstates real-world performance. **Cross-validation** is preferable when data is limited: a fixed 90/10 split might sample an unusually easy or hard devset by chance. $k$-fold cross-validation (typically $k=10$) rotates through $k$ different devset splits and pools the results, giving a more stable performance estimate at $k\times$ the training cost.

> [!question]- State the null and alternative hypotheses when testing whether classifier A is better than classifier B, and explain what a p-value of 0.03 means.
> Let $\delta(x) = M(A, x) - M(B, x)$. Null hypothesis $H_0: \delta(x) \le 0$ (A is not better than B); alternative $H_1: \delta(x) > 0$ (A is better than B). A p-value of 0.03 means: *if* $H_0$ were true, there is a 3% probability of seeing an effect size at least as large as the one we observed by chance alone. Since 0.03 < 0.05, we reject $H_0$ at the 0.05 significance level — but would fail to reject it at the 0.01 level. The p-value is a probability of observing the data given the null, not a probability that the null is true.

> [!question]- In the paired bootstrap, why do we count $\delta(x^{(i)}) \ge 2\delta(x)$ rather than $\delta(x^{(i)}) \ge 0$?
> The bootstrap samples are drawn from the original test set $x$, which is already biased by the observed $\delta(x)$ in favour of A. Under the null hypothesis, the *expected* $\delta$ is 0 — so the resampled distribution is centred on $\delta(x)$, not on 0. To ask "how surprising is the observed $\delta(x)$ under $H_0$?" we shift by $\delta(x)$: count how often $\delta(x^{(i)}) - \delta(x) \ge \delta(x)$, i.e. $\delta(x^{(i)}) \ge 2\delta(x)$. This correctly tests whether the effect size exceeds what sampling variation alone would produce under the null.

> [!question]- Why are paired tests more powerful than unpaired tests for comparing classifiers?
> **Paired** means each test item is evaluated by *both* classifiers, and we compare them item-by-item. This removes per-item variance — some test items are intrinsically easier, others harder — from the comparison. Unpaired tests compare aggregate scores on two independent samples and have to account for that per-item variance as noise, reducing statistical power. Since in classifier comparison we always run both systems on the same test set, pairing is free and strictly more informative.
