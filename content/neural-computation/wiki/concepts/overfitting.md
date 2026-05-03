---
name: overfitting
description: A failure mode where a model fits training data so closely that it memorises noise, losing the ability to generalise to unseen data; paired with its opposite (underfitting) it defines the capacity-tradeoff core to ML.
type: concept
sources:
  - raw/week-03/w03-l08-transcript.txt
  - raw/week-03/w03-slides.pdf
status: stable
updated: 2026-04-24
---

*The central tension of machine learning: a model that's too weak can't capture the pattern, but a model that's too strong memorises the training examples rather than learning the underlying rule. Either way, performance on new data is bad.*

## The spectrum: underfitting, good fit, overfitting

| Regime | Model capacity | Training performance | Test performance | Symptom |
|---|---|---|---|---|
| **Underfit** | Too low | Poor | Poor | Can't capture the trend — training loss stays high |
| **Good fit** | Matched to problem | Reasonable | Reasonable | Captures the pattern, tolerates noise |
| **Overfit** | Too high | Near-perfect | Poor | Memorises training data including noise |

The classic illustration:

- **Underfit classification:** a single straight line drawn through a dataset with a clearly curved decision boundary. Lots of misclassified training points, and new points are mispredicted in exactly the same way.
- **Overfit classification:** a wildly wiggly boundary that zig-zags to classify *every* training point correctly, including points that are probably noise. New points fall into the wrong pockets of the wiggly region.
- **Good fit:** a smooth curve that follows the general shape of the data without chasing individual noisy points.

The same pattern in regression: underfit = straight line through curved data, overfit = high-degree polynomial that threads through every training point, good fit = smooth curve capturing the underlying signal.

## Generalisation is the real goal

Training loss measures memorisation; test loss measures *generalisation* — how well the model performs on data it's never seen. Since we deploy the model to new inputs, only generalisation matters.

> [!tip] TIP — The student analogy
> A student who memorises every exam question from past years but doesn't understand the concepts will fail on new questions. That's **overfitting** — perfect recall of training data, poor performance on unseen material. A student who doesn't study at all fails both. That's **underfitting**. You want the student who understands the concepts well enough to solve problems they've never seen — that's generalisation.

## Causes

**Underfitting:**
- Model too simple (not enough parameters, wrong architecture).
- Trained for too few iterations.
- Features too impoverished to contain the signal.

**Overfitting:**
- Model too powerful (too many parameters relative to training data).
- Too few training examples for the model's capacity.
- Trained for too long, fitting noise once the signal is captured.
- No regularisation.

More complex models (deeper / wider networks, higher-degree polynomials) can *only* overfit. They can't underfit in the capacity sense, because they can always shrink to mimic a simpler model. The failure mode shifts from "too rigid" to "too flexible".

## The three-way data split

To detect and control overfitting, split your dataset into **three** disjoint partitions:

| Split | Typical fraction | Purpose |
|---|---|---|
| **Training** | ~60% | Gradient descent updates the weights using this data |
| **Validation** | ~20% | Monitor generalisation *during* training — tune hyperparameters, pick stopping point, compare models |
| **Test** | ~20% | Final, one-shot evaluation *after* all training decisions are locked in |

**Key rule: each split has a specific purpose and must not leak into another.**

- Training on the **test set** → the test evaluation no longer reflects generalisation. The test set would become just another piece of training data.
- Training on the **validation set** → you'd overfit to the validation set, and its measurements would stop being a faithful estimate of generalisation.
- Evaluating on the **training set** → tells you nothing about generalisation; high training accuracy is compatible with severe overfitting.

> [!tip] TIP — The university-exam analogy
> Training data is like practice exercises you study from. Validation data is like mid-term assessments — internal checks during the term. Test data is the final exam — you only see it once, and you don't study from it beforehand, or the evaluation is compromised. The final-exam rule ("no peeking") is exactly the rule for test sets.

## Fixing underfitting

Use a more powerful model: more layers, more neurons per layer, richer features. The underlying issue is "the model cannot represent the pattern", so give it more representational capacity.

## Fixing overfitting

Several levers, usually used in combination:

1. **More training data.** The single most reliable fix. A model with $N$ parameters overfits when training set size $\ll N$; adding examples increases the ratio of signal to noise.
2. **Less powerful model.** Fewer parameters (shallower / narrower network, lower-degree polynomial) gives the model less room to memorise.
3. **Early stopping.** Stop training before the model has had time to overfit (see below).
4. **Regularisation.** Add an explicit penalty on weight magnitude to the loss. See [[regularization]].
5. **[[dropout]]** — randomly disable neurons during training to prevent over-reliance on specific units.
6. **[[data-augmentation]]** — synthesise extra training examples by applying label-preserving transformations.
7. **[[transfer-learning]]** — start from weights pre-trained on a large dataset, dramatically reducing the effective data requirement.

## Early stopping

Monitor training loss and validation loss at each epoch:

- **Training loss** typically decreases monotonically — the model keeps memorising more of the training set.
- **Validation loss** decreases initially (the model is learning generalisable patterns), then bottoms out, then *increases* as the model starts overfitting to training-set noise.

The minimum of the validation loss is the sweet spot: further training improves training loss but hurts generalisation. **Early stopping** picks the epoch where validation loss is lowest and uses those weights as the final model.

Visually:
- Both losses decrease together in the "learning useful patterns" phase.
- They diverge at the "overfitting starts" point — training loss keeps falling, validation loss starts rising.
- Stop at the divergence point.

> [!info] ASIDE — Early stopping is a form of regularisation
> Even without an explicit penalty term, limiting the number of training iterations constrains how thoroughly the model can fit the training data. Stopping early is equivalent to restricting the effective capacity of the model, which is exactly what other regularisers do.

## Related

- [[regularization]] — an explicit loss term that discourages overfitting
- [[dropout]] — randomly disable neurons during training; an alternative regulariser
- [[data-augmentation]] — synthesise more training examples to reduce overfitting from a small dataset
- [[transfer-learning]] — bypass the data-hunger problem by starting from pre-trained weights
- [[gradient-descent-nc|gradient descent]] — the process whose stopping point early stopping controls
- [[loss-function]] — the training-loss metric; validation/test losses use the same formula on different data
- [[multi-layer-perceptron]] — the architecture whose capacity is the main dial for overfitting

## Active Recall

> [!question]- What exactly is the difference between underfitting and overfitting — both can produce poor test performance, so how do you tell them apart?
> Look at the *training* loss. Underfitting: training loss is high (and test loss is high too) — the model can't even fit what it was trained on. Overfitting: training loss is low (often near-zero) but test loss is high — the model memorised the training set including noise. The gap between training and test loss is the key diagnostic: small gap + both high = underfit; large gap with low training / high test = overfit.

> [!question]- You have a dataset of 10,000 samples. Your friend suggests training on all 10,000 and evaluating on the same 10,000 to measure the model's quality. What's wrong with this?
> It measures memorisation, not generalisation. The model has already seen every one of those 10,000 samples during training — scoring well on them tells you nothing about how it'll behave on new data. A sufficiently overparameterised model can achieve 100% training accuracy while having completely random test accuracy. You need a held-out test set, disjoint from the training data, to measure generalisation honestly.

> [!question]- Why should you not train on the validation set?
> The validation set's job is to measure generalisation *during* training — you use it to pick hyperparameters, decide when to stop, compare models. If the training process can update weights using validation data, the model will overfit to the validation set just as it can overfit to the training set. Its loss on the validation set will no longer be a faithful estimate of generalisation, and you'll lose the early-stopping signal. The validation set works precisely because the training process never touches it.

> [!question]- During training, validation loss starts increasing while training loss keeps decreasing. What's happening, and what should you do?
> The model has moved from "learning generalisable patterns" to "memorising training-set noise". Training loss keeps falling because the model keeps fitting training data better; validation loss rises because that fit no longer transfers to unseen data — it's memorisation, not generalisation. The standard response is **early stopping**: use the weights from the epoch where validation loss was lowest, and discard any training past that point.

> [!question]- You're overfitting. Which is usually the most effective fix: adding more data, using a smaller model, or adding regularisation?
> In practice, **more data** is by far the most effective when feasible — it addresses the root cause (the model had insufficient signal-to-noise ratio during training). When more data isn't available (most real-world cases), some combination of a smaller model, early stopping, and regularisation is used. Regularisation and smaller-model approaches are fundamentally equivalent in spirit: both reduce the effective capacity the model has for memorising noise.
