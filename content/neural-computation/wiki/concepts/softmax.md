---
name: softmax
description: The multi-class generalisation of sigmoid; converts a vector of raw scores into a probability distribution over classes that sums to one.
type: concept
sources:
  - raw/week-03/w03-l08-transcript.txt
  - raw/week-03/w03-slides.pdf
status: stable
updated: 2026-04-24
---

*Sigmoid turns one number into a probability in (0, 1). Softmax turns a vector of numbers into a probability distribution — one that's positive and sums to 1. It's what you use in the output layer when the task is multi-class classification.*

## Definition

For a vector of raw pre-activation scores $\mathbf{z} = (z_1, z_2, \dots, z_m)$ from the output layer of an MLP, the **softmax** of the $j$-th component is:

$$\text{softmax}(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^{m} e^{z_k}}$$

Two properties fall out immediately:

1. **Positive:** $e^{z_j} > 0$, so every output is strictly positive.
2. **Sums to 1:** the denominator is the sum of all numerators, so $\sum_j \text{softmax}(\mathbf{z})_j = 1$.

Together, these properties mean the output can be interpreted as a **probability distribution** over the $m$ classes: $\text{softmax}(\mathbf{z})_j = p(y = j \mid \mathbf{x})$.

## Why we need it

The [[sigmoid-function-nc|sigmoid function]] handles binary classification: one output neuron producing $p(y=1 \mid \mathbf{x}) \in (0, 1)$, with the other class's probability implicitly being $1 - \hat{y}$. That doesn't scale to three or more classes — you can't cleanly encode "class 1 vs class 2 vs class 3" with a single number between 0 and 1.

For $m$-class classification, the output layer has $m$ neurons. Each neuron produces a raw score $z_j$, but those raw scores aren't directly usable as class probabilities (they can be negative, they don't sum to anything meaningful). Softmax takes the whole vector and rescales it into a proper probability distribution.

## Example

A network has an output layer of 10 neurons (one per digit, 0–9) trained on MNIST. On some input image, the raw outputs are:

$$\mathbf{z} = (0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 3.5, 0.2, 0.3)$$

These aren't probabilities — they're unbounded raw scores. Softmax turns them into something like:

$$\text{softmax}(\mathbf{z}) \approx (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.80, 0.02, 0.02)$$

which reads as: "80% probability it's a 7, 2% each for the others". Now the network's output is directly a probability distribution over digits. The predicted class is $\arg\max_j \text{softmax}(\mathbf{z})_j$ — the index with the highest probability.

## One-hot encoding: the matching label format

If softmax produces a probability distribution over classes, the training labels should match that shape. The standard format is **one-hot encoding**.

For a sample whose true class is $j$, the label is a vector of length $m$ with:
- A $1$ in position $j$ ("hot").
- A $0$ in every other position ("cool").

Example for MNIST, label = 3:

$$\mathbf{y} = (0, 0, 0, 1, 0, 0, 0, 0, 0, 0)^T$$

Formally:

$$y_{i, j} = \begin{cases} 1 & \text{if class of sample } i \text{ is } j \\ 0 & \text{otherwise} \end{cases}$$

This one-hot vector pairs naturally with the softmax output vector of length $m$, and together they plug into the multi-class cross-entropy loss.

> [!info] ASIDE — Why one-hot and not just an integer label?
> You could label sample 3 as just the integer "3", but then the network (which has 10 output neurons) wouldn't know how to match its prediction against a scalar target. One-hot encoding maps the label into the same space as the prediction, so the loss function can compare component-by-component. It also generalises nicely to tasks where the label is genuinely multi-valued (e.g. "the image could be a 3 or a 5 with 50/50 probability").

## The softmax–sigmoid connection

Softmax is a strict generalisation of the sigmoid. In the binary case with $m = 2$ and scores $z_1, z_2$:

$$\text{softmax}(\mathbf{z})_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{z_2 - z_1}} = \sigma(z_1 - z_2)$$

So two-class softmax collapses to a sigmoid of the score *difference*. Conceptually: sigmoid is "softmax with only two classes, and the two scores reduce to one degree of freedom".

## Training with softmax

Softmax is used as the output-layer activation; the loss that pairs naturally with it is **categorical cross-entropy** (the multi-class extension of [[binary-cross-entropy]]):

$$L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{m} y_{i,j} \ln \text{softmax}(\mathbf{z}_i)_j$$

For each training example $i$, only one $y_{i,j}$ is 1 (and the rest are 0), so the inner sum collapses to a single term — $-\ln(\text{predicted probability of the true class})$. The loss is low when the network assigns high probability to the correct class and grows unboundedly as the network becomes confidently wrong.

(As with binary cross-entropy under sigmoid, the pairing is not arbitrary: the gradient of softmax + cross-entropy simplifies to $\text{softmax}(\mathbf{z}) - \mathbf{y}$, a clean prediction-minus-label form. This avoids the saturation problem that squared error would introduce.)

## Alternatives and gotchas

- **Mean squared error can still be used** with softmax for classification — it's differentiable and will "work" — but cross-entropy is preferred because its gradient is better-behaved, particularly when the network is confidently wrong.
- Softmax saturates in the same way sigmoid does: if one $z_j$ is much larger than the others, its softmax approaches 1 and the gradient approaches 0 for that neuron. This is fine at the output layer (where you want confident predictions) but is *not* a reason to use softmax in hidden layers (you don't want saturation partway through the network).

## Related

- [[sigmoid-function-nc|sigmoid function]] — the binary-class special case
- [[binary-cross-entropy]] — the two-class loss; softmax + categorical cross-entropy is the multi-class analogue
- [[multi-layer-perceptron]] — softmax is used at the output layer of an MLP for multi-class tasks
- [[maximum-likelihood-estimation-nc|maximum likelihood estimation]] — the principle from which softmax + cross-entropy falls out (categorical distribution over classes)

## Active Recall

> [!question]- Given raw output scores $\mathbf{z} = (1, 2, 3)$ from a three-class network, compute the softmax probabilities.
> $e^1 \approx 2.72$, $e^2 \approx 7.39$, $e^3 \approx 20.09$. Sum $\approx 30.2$. Softmax $\approx (0.09, 0.24, 0.67)$. The model assigns roughly 67% probability to class 3, 24% to class 2, 9% to class 1. Sanity check: the three probabilities sum to 1. ✓

> [!question]- Why do we use one-hot encoding for class labels instead of just writing the class number as an integer?
> The network's output layer for an $m$-class problem produces an $m$-dimensional vector (of softmax probabilities). For the loss function to compare prediction and label component-by-component, the label must live in the same space — a vector of length $m$. One-hot is the minimal encoding that puts the label in this space and also has a clean probabilistic interpretation: the one-hot vector is the "perfect probability distribution" that assigns 100% to the correct class.

> [!question]- What is the relationship between softmax and sigmoid?
> Softmax is the multi-class generalisation of sigmoid. In the binary case, softmax applied to two scores $z_1, z_2$ reduces to $\sigma(z_1 - z_2)$ — a sigmoid of the score difference. Both functions are derived from maximum likelihood under categorical distributions (Bernoulli for sigmoid, multinoulli / categorical for softmax). Both produce outputs interpretable as probabilities. Sigmoid is just the $m = 2$ special case.

> [!question]- Can softmax outputs ever be exactly 0 or exactly 1?
> No. The numerator $e^{z_j}$ is always strictly positive, and the denominator is a sum of strictly positive terms, so every softmax output is in the open interval $(0, 1)$ — it can get arbitrarily close to 0 or 1 but never reach either exactly. This has a practical consequence: the cross-entropy loss $-\ln(\text{predicted probability})$ is always finite for a softmax output, so the loss is well-defined. If you ever see softmax output "exactly 1.0" in practice, it's floating-point saturation.

> [!question]- Why not use softmax as the activation for hidden layers as well as the output layer?
> Softmax couples the output values across a whole layer (they must sum to 1), which makes its derivative more complex and forces a competition between units. In a hidden layer, you typically want each unit to learn an independent feature — softmax's cross-unit coupling is the wrong inductive bias, and its saturation at confident values would also cause vanishing gradients within the network. Hidden layers use per-unit activations like sigmoid, tanh, or ReLU; softmax is reserved for the output layer of multi-class classifiers.
