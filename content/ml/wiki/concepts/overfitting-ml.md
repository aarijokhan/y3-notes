---
type: concept
name: overfitting
description: When training error goes down while test error goes up — "fitting the training data more than is warranted". Caused by some combination of model capacity, limited data, and noise (both stochastic and deterministic). The condition that motivates regularisation, validation, and most of practical ML.
sources:
  - raw/week-10/Wk_10_Lec_1-1.pdf
  - raw/week-10/Thursday_ December 4_ 2025 at 8_10_06 AM_Captions_English (United States).txt
  - raw/week-10/Thursday_ December 4_ 2025 at 8_54_53 AM_Captions_English (United States).txt
status: draft
updated: 2026-05-04
---

*A model **overfits** when, in the process of reducing training error $E_{\text{in}}$, it increases test error $E_{\text{out}}$. The resulting fit captures noise and idiosyncrasies of the specific training set rather than the underlying structure of the target. Symbolically: low $E_{\text{in}}$, high $E_{\text{out}}$ — bad generalisation. The dual failure is **underfitting**, where both errors are high because the model class is too restricted to capture $f$.*

## Diagnosis

Overfitting and underfitting live at opposite ends of the [[bias-variance-decomposition|bias–variance]] spectrum:

| Symptom | Diagnosis |
|---|---|
| $E_{\text{in}}$ low, $E_{\text{out}}$ high | **Overfitting** (high variance, low bias) |
| Both errors high and similar | **Underfitting** (high bias, low variance) |
| Both errors low and similar | Sweet spot — correctly specified model with enough data |
| Both errors low, different | Possible data leak; investigate |

The defining feature of overfitting is the *gap* between training and test performance. A model with $E_{\text{in}} = 0$ that gets every training point exactly right but predicts wildly on new data is the classic example: a degree-10 polynomial through 10 noisy points has $E_{\text{in}} \approx 0$ and $E_{\text{out}}$ in the thousands.

## Causes

The lecture lists six common causes. Each compounds: real overfitting is usually multiple at once.

- **Model too complex.** A deep network with millions of parameters trained on 1{,}000 images memorises the images rather than learning generalisable patterns.
- **Too little training data.** A sentiment classifier trained on 50 customer reviews fits unique phrases instead of sentiment patterns; with 50{,}000 it generalises.
- **Too many training epochs.** Training error keeps falling while validation error bottoms out and then climbs. Early-stopping, the standard fix, is also a form of regularisation.
- **Lack of regularisation.** Without constraints, weights are free to grow large to fit specific outliers — reducing $E_{\text{in}}$ at the cost of generalisation.
- **High-variance features or noisy labels.** Random "ID number" features can become decision-tree splits; mislabelled examples can drag the boundary.
- **Poor data processing.** Unstandardised inputs cause distance-based methods (SVM, k-NN) to weight high-magnitude features arbitrarily; data leakage between training and validation invalidates the whole exercise.

## The Two Learners

A vivid framing from the lecture. Target $f$ is degree-10 with very small noise; $N = 15$ training points. Two learners:

- **Learner $O$ (Overfit).** Picks $g_{10} \in \mathcal{H}_{10}$ — uses a degree-10 polynomial to "match" the target's complexity.
- **Learner $R$ (Restrict).** Picks $g_2 \in \mathcal{H}_2$ — uses a degree-2 polynomial despite knowing the target is degree-10.

$R$ "gives up" the ability to fit the target exactly. Yet:

- **Noisy low-order target** ($y = $ degree-2 + noise): $\mathcal{H}_2$ gets $E_{\text{out}} = 0.127$; $\mathcal{H}_{10}$ gets $9.00$.
- **Noiseless high-order target** ($y = $ degree-10, no noise): $\mathcal{H}_2$ gets $E_{\text{out}} = 0.120$; $\mathcal{H}_{10}$ gets $7680$.

**$R$ wins by a lot, even when its hypothesis class can't represent the truth.** With only 15 points, the high-capacity learner $O$ has so much variance that the bias saving evaporates.

The lesson is uncomfortable: **using the right hypothesis class isn't enough**; you need enough data to constrain it. With insufficient $N$, deliberate underfitting beats accurate complexity-matching. This is the bias–variance trade-off shouting in your ear.

## Stochastic vs Deterministic Noise

Two kinds of "noise" drive overfitting:

**Stochastic noise** is randomness in the labels: $y = f(\mathbf{x}) + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$. Truly random, irreducible, the same kind of noise that gave rise to the $\sigma^2$ floor in the [[bias-variance-decomposition|bias–variance decomposition]].

**Deterministic noise** is the part of the target $f$ that the hypothesis set $\mathcal{H}$ *cannot* represent. If $f$ is degree-50 and $\mathcal{H} = \mathcal{H}_2$ (degree-2 polynomials), the residual $f - $ (best degree-2 approximation) is the *deterministic* noise: a fixed function of $\mathbf{x}$, perfectly predictable in principle, but invisible to the chosen model class.

Both look the same to the learner. The trained model can't tell whether the residuals come from random label noise or from the target's structural complexity exceeding $\mathcal{H}$. From the learner's perspective, both are "stuff I can't fit", and both pull the fit towards spurious patterns when capacity is high relative to data.

The four impact factors of overfitting (from the lecture's heat-map experiment):

| Factor | Direction |
|---|---|
| Data size $N$ ↓ | Overfitting ↑ |
| Stochastic noise $\sigma^2$ ↑ | Overfitting ↑ |
| Deterministic noise $Q_f$ ↑ (target complexity) | Overfitting ↑ |
| Excessive model power | Overfitting ↑ |

## Why MLE Cannot Prevent It

[[maximum-likelihood-estimation-ml|MLE]] minimises in-sample negative log-likelihood. It has no mechanism to prefer "simpler" hypotheses over more complex ones — the likelihood always favours whatever maximises in-sample fit. So an MLE-fit polynomial of high enough degree will interpolate every training point and overfit terribly.

The Bayesian remedy is to combine the likelihood with a prior on $\mathbf{w}$ that penalises large weights. The MAP estimate then balances likelihood (fit) against prior (simplicity) — exactly [[ridge-regression|ridge regression]] for a Gaussian prior. This is the probabilistic-mechanics view of why [[regularization-ml|regularisation]] works.

## What to Do

The lecture's two cures are the standard practical toolkit:

1. **[[regularization-ml|Regularisation]]** ("putting the brakes"). Add a penalty term that biases the optimiser towards simpler hypotheses. L2 (ridge), L1 (lasso), early stopping, dropout — all are regularisation in disguise.
2. **Validation / cross-validation** ("checking the bottom line"). Hold out a portion of the data, fit on the rest, and select hyperparameters (model class, $\lambda$, epochs) by held-out performance. Measures rather than assumes that you're not overfitting.

The Bayesian connection ties these together: regularisation is MAP with a prior, validation is empirical model selection. Both attack the same enemy from different angles.

## Related

- [[bias-variance-decomposition]] — the formal lens; overfitting is high variance, underfitting is high bias.
- [[generalization-bound]] — the worst-case bound that overfitting violates spectacularly.
- [[regularization-ml|regularization]] — the primary structural cure.
- [[ridge-regression]] — L2 regularisation, MAP with Gaussian prior.
- [[bayesian-linear-regression]] — the principled framework for trading fit against simplicity.

## Active Recall

> [!question]- A degree-2 learner ($\mathcal{H}_2$) and a degree-10 learner ($\mathcal{H}_{10}$) are both told that the true target is degree-10 with very small noise. With $N = 15$ training points, the degree-2 model achieves much lower test error. Why does the "wrong" model win?
> Because variance dominates bias when $N$ is small relative to model capacity. $\mathcal{H}_{10}$ has enough flexibility to fit the noise as well as the signal — its hypothesis swings wildly between the 15 training points, producing huge test error. $\mathcal{H}_2$ pays a bias cost (it cannot represent the degree-10 target exactly) but its low variance means the trained model is stable across training sets. The lesson: knowing the right hypothesis class is not enough — you need data sufficient to constrain it.

> [!question]- Distinguish stochastic noise from deterministic noise. Why does the trained model treat them similarly?
> *Stochastic noise* is randomness in the labels themselves: $y = f(\mathbf{x}) + \epsilon$ with $\epsilon$ random. *Deterministic noise* is the part of the target $f$ that the hypothesis class $\mathcal{H}$ cannot represent — a fixed function of $\mathbf{x}$, but invisible from the learner's perspective. To the trained model, both look like residuals it can't reduce, and both encourage overfitting when capacity is high relative to data: the model attempts to fit spurious structure that won't generalise. The cure for both is the same — regularise capacity or get more data.

> [!question]- A model trained on 1{,}000 images has training accuracy 99% and test accuracy 60%. List three remedies in order of likely effectiveness, and why.
> 1. **More data.** The 39-point gap suggests the model is fitting idiosyncrasies of the 1{,}000 examples; more data is the most direct fix for variance, and dramatically more effective per unit of effort than algorithmic tweaks.
> 2. **Regularisation** (L2 weight decay, dropout, data augmentation). Each penalty effectively reduces the hypothesis set the model can converge to. Choose the regularisation strength via validation.
> 3. **Reduce model capacity.** Fewer parameters, smaller network. Most heavy-handed; lowers the bias floor too.
> Order matters because data is the highest-leverage fix, regularisation is reliable and cheap, and capacity reduction trades variance for bias and may underfit.

> [!question]- Why does training for too many epochs cause overfitting, and what's the standard fix?
> Each gradient step lets the model fit the training data more closely, and after the model has captured the signal, additional epochs let it fit *noise*. Validation error typically falls early then rises as overfitting sets in. **Early stopping** terminates training when validation error starts to rise — equivalent to a regulariser that constrains how far the optimiser can move from initialisation. It's free (no extra hyperparameters beyond the validation schedule) and effective.
