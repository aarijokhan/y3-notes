---
type: concept
name: learning principles
description: Three meta-principles that constrain how learning experiments should be designed. Occam's Razor — prefer simpler models. Sampling Bias — biased data produces biased models, with no recovery. Data Snooping — using the test set in any decision contaminates it.
sources:
  - raw/week-11/Wk_11_Lec_2.pdf
  - raw/week-11/Tuesday_ December 9_ 2025 at 9_56_03 AM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*Three meta-principles that govern how learning experiments should be designed and interpreted, beyond the specific algorithms. **Occam's Razor**: among hypotheses that fit, prefer the simplest. **Sampling Bias**: if training data is sampled from a distribution that differs from the deployment distribution, no learning theorem rescues you. **Data Snooping**: any data that has affected any step of the learning process — including preprocessing, feature design, hyperparameter choice, or model selection — has lost its ability to fairly assess the result.*

## 1. Occam's Razor

> *Entities must not be multiplied beyond necessity.* — William of Occam (1287–1347)

> *An explanation of the data should be made as simple as possible, but no simpler.* — Albert Einstein (probably; widely attributed)

The **Occam's Razor** principle in machine learning: **the simplest model that fits the data is also the most plausible.** Two questions to make this precise:

### What does "simple" mean?

| Object | Simple means |
|---|---|
| Hypothesis $h$ | Small $\Omega(h)$ — few parameters, short description |
| Hypothesis set $\mathcal{H}$ | Small $\Omega(\mathcal{H})$ — small VC dimension, few hypotheses |

These two are related: $\Omega(h)$ small $\impliedby \Omega(\mathcal{H})$ small. A hypothesis drawn from a low-complexity set is automatically a low-complexity hypothesis. The two senses of simplicity are aligned, even though they're conceptually distinct.

### Why is simpler better?

"Better" doesn't mean "more elegant" — it means **better out-of-sample performance**. A simpler model has tighter generalisation bounds (smaller $\Omega(\mathcal{H})$ in the VC bound), so when training error is comparable, simpler wins on test error.

The practical rule: **start linear, then ask whether the data is being over-modelled**. Don't add flexibility unless the simpler model is structurally inadequate. This pairs with [[regularization-ml|regularisation]] — even within a complex hypothesis set, the regulariser pushes the optimiser toward "effectively simpler" hypotheses.

> [!warning] COMMON MISCONCEPTION — "Simple" means "fewer features"
> Not always. A 2-feature degree-10 polynomial has more capacity than a 100-feature linear model. Complexity is about the **hypothesis set's expressivity**, captured by VC dimension, not the input dimension. A linear classifier in $\mathbb{R}^{1000}$ can be simpler than a degree-3 polynomial in $\mathbb{R}^2$.

## 2. Sampling Bias

> *If the data is sampled in a biased way, learning will produce a similarly biased outcome.*

The technical statement: if training data is drawn from $P_1(\mathbf{x}, y)$ but deployment is under $P_2 \neq P_1$, the [[generalization-bound|VC bound]] **fails** — its proof assumes training and test draws come from the same distribution.

The philosophical statement: studying maths hard but being tested on English gives no strong test-performance guarantee. The mismatch between what you trained on and what you're evaluated on is fatal — no clever algorithm rescues a model trained on the wrong distribution.

### Examples in Practice

- **Survey bias**: a poll of land-line phone owners predicts presidential elections poorly when many voters use only mobiles.
- **Survivorship bias**: a model of "successful startups" trained only on companies that survived ignores the companies that failed.
- **Class imbalance**: training on 99% benign, 1% malignant means the model can achieve 99% accuracy by always predicting "benign" — but generalises terribly when the true rate matters.
- **Temporal drift**: a model trained on 2020 data deployed in 2025 sees a shifted input distribution.

### What Can You Do About It?

If $P_{\text{train}} \neq P_{\text{deploy}}$ and you can't re-collect data:

- **Re-weight training examples** to match the deployment distribution (importance sampling).
- **Distributionally robust** training: optimise for worst-case performance across a family of distributions.
- **Domain adaptation**: explicit transfer-learning techniques.

But these are partial fixes. The cleanest approach is **prevention**: collect training data from the actual deployment distribution. If the distribution will shift, plan to retrain.

## 3. Data Snooping

> *If a data set has affected any step in the learning process, its ability to assess the outcome has been compromised.*

The strongest of the three principles. **Any** influence — preprocessing choice, feature engineering, model selection, hyperparameter tuning — counts as "the data affecting a step". Once snooped, the data is no longer independent of the trained model, and no test-set guarantee survives.

### Examples of Data Snooping

- **Normalising the entire dataset** (training + test) before splitting: the test set's mean/variance has leaked into the training preprocessing.
- **Looking at the test set's labels** to design features: those features are tuned to whatever pattern existed in the test set.
- **Trying many models, reporting the best test error**: the test set was used as a selection criterion. The reported number is biased downward.
- **Reading a paper, then "discovering" a feature engineering trick that turns out to help on the same benchmark**: research-community-level snooping.
- **Cross-validating to pick a model, then reporting the best CV error**: the CV error is the *selection* score, not an unbiased estimate of $E_{\text{out}}$ for the chosen model. (You'd need a fresh test set for that.)

### The Discipline

The cleanest workflow:

1. **Split** $\mathcal{D}$ into $\mathcal{D}_{\text{train}}$ and $\mathcal{D}_{\text{test}}$ before doing *anything*.
2. **Lock $\mathcal{D}_{\text{test}}$ in the safe** — never look at it, never normalise with it, never preprocess with it.
3. Use $\mathcal{D}_{\text{train}}$ for everything: feature design, model fitting, hyperparameter tuning (via further internal validation/cross-validation splits).
4. **One** evaluation on $\mathcal{D}_{\text{test}}$ at the end. If you peek and decide to "try one more thing," you've burned it — no longer a clean estimate.

The lecture's warning, made concrete: a financial trading example shows that "snooping" (using test-period statistics to normalise data before training) produces an apparently profitable strategy with cumulative profit $> 30\%$, while the same strategy with proper normalisation (training-only) loses money. **The model didn't learn signal — it learned the test set's distribution.**

> [!info]+ ASIDE — Why data snooping is the hardest principle to follow
> Sampling bias is hard to detect but at least conceptually clear ("get representative data"). Occam's Razor is a default ("start simple"). Data snooping is **insidious**: every paper, every prior experience, every glance at exploratory data analysis is a potential snoop. In the strictest sense, the moment you saw that the dataset has 1{,}000 features rather than 10, your hypothesis class is "informed" by the data. Practitioners aim for *minimal* snooping, not zero.

## How the Three Interact

| If you violate... | Effect on $E_{\text{out}}$ |
|---|---|
| Occam's Razor (model too complex) | $E_{\text{out}} \gg E_{\text{in}}$ — overfitting; bound is loose |
| Sampling Bias (wrong distribution) | $E_{\text{out}}$ on deployment $\gg E_{\text{out}}$ on training distribution; bound doesn't apply |
| Data Snooping (test set used) | Reported $E_{\text{test}}$ is optimistic; true $E_{\text{out}}$ is worse |

All three corrupt the chain of inference from training to deployment. They're separately deadly and jointly catastrophic.

## Related

- [[generalization-bound]] — the formal theorem whose assumptions all three principles correspond to: bounded complexity (Occam), i.i.d. sampling (no sampling bias), independent test set (no snooping).
- [[validation]] — disciplined data hygiene relies on it.
- [[overfitting-ml|overfitting]] — Occam's Razor's failure mode.
- [[regularization-ml|regularization]] — Occam's Razor mechanically enforced.

## Active Recall

> [!question]- Occam's Razor says simpler is better. What does "simpler" mean precisely, and why is simpler "better" in machine-learning terms?
> "Simpler" has two senses, related: a hypothesis $h$ is simple if $\Omega(h)$ is small (few parameters, short description); a hypothesis set $\mathcal{H}$ is simple if $\Omega(\mathcal{H})$ is small (small VC dimension). The two are linked because a hypothesis drawn from a low-complexity set is automatically low-complexity. "Better" means **better out-of-sample performance**, not aesthetic elegance. The connection: the [[generalization-bound|VC bound]] $E_{\text{out}} \leq E_{\text{in}} + \Omega(\mathcal{H})$ has a tighter penalty term for simpler models, so when training fits are comparable, simpler wins on test error.

> [!question]- A model trained on data from $P_1$ is deployed on data from $P_2 \neq P_1$. Which assumption of the VC bound fails, and what's the practical implication?
> The i.i.d. assumption — training and test data are independent draws from the *same* joint distribution. When $P_1 \neq P_2$, every theorem that bounded $E_{\text{out}}$ in terms of $E_{\text{in}}$ is silent: the bound's proof relied on training error being a sample average from the same distribution as test error. Practical implication: deployment performance can be arbitrarily worse than training performance, and there's no algorithmic fix that doesn't either re-collect data from $P_2$ or apply explicit distribution-correction techniques (importance weighting, domain adaptation).

> [!question]- Give three concrete examples of data snooping, and explain in each case what was contaminated.
> 1. **Normalising the entire dataset before train/test split**: the test set's mean and variance leaked into the preprocessing step. The "test set" is no longer independent of the training pipeline; reported test error is downward-biased.
> 2. **Trying many models on the test set and reporting the best**: the test set was used as a selection criterion across $M$ models. The reported number is the *minimum* over $M$ correlated estimates, not an unbiased estimate of $E_{\text{out}}$ for one model.
> 3. **Reading prior papers' results on a benchmark, then "noticing" a feature trick**: even without direct test-set access, the researcher's hypothesis class is informed by what worked on the test set in earlier work. Community-level snooping is real but invisible at the level of one experiment.

> [!question]- What's the cleanest workflow to avoid data snooping when you need to do feature engineering, model selection, *and* report a final test error?
> 1. Split $\mathcal{D}$ into $\mathcal{D}_{\text{train}}$ and $\mathcal{D}_{\text{test}}$ at the very start, *before* any analysis.
> 2. Lock $\mathcal{D}_{\text{test}}$ away — don't look at it, don't normalise with it, don't preprocess with it.
> 3. Do everything (feature engineering, hyperparameter tuning) using $\mathcal{D}_{\text{train}}$, with internal validation/cross-validation as needed.
> 4. Evaluate on $\mathcal{D}_{\text{test}}$ **once**, at the end. Report that number.
>
> If after step 4 you decide you want to try something else, $\mathcal{D}_{\text{test}}$ is now contaminated — you'd need a *fresh* held-out set for the next round of evaluation. Most published "test errors" violate this discipline at least somewhat; the question is by how much.
