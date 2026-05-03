---
name: binary-cross-entropy
description: The loss function for binary classification with a sigmoid output; derived from maximum likelihood under a Bernoulli assumption on the labels.
type: concept
sources:
  - raw/week-02/w02-l05-transcript.txt
  - raw/week-02/w02-l06-transcript.txt
  - raw/week-02/w02-slides.pdf
status: stable
updated: 2026-04-24
---

*The classification counterpart to squared error. Where squared error is what max likelihood prescribes under Gaussian noise, cross-entropy is what max likelihood prescribes for classes drawn from a Bernoulli with $p = \sigma(z)$.*

## Definition

For a binary classification dataset with labels $y^j \in \{0, 1\}$ and sigmoid-perceptron predictions $\hat{y}^j \in (0, 1)$:

$$L(\boldsymbol{\theta}) = -\sum_{j=1}^{n} \Big[ y^j \ln \hat{y}^j + (1 - y^j) \ln(1 - \hat{y}^j) \Big]$$

The minus sign in front turns a log-likelihood maximisation into a loss minimisation.

## What each term does

For each training sample, exactly one of the two terms is active (because $y^j \in \{0, 1\}$):

| Label $y^j$ | Active term | Penalises |
|---|---|---|
| $1$ | $-\ln \hat{y}^j$ | $\hat{y}^j$ close to 0 (big loss if the model is confidently wrong) |
| $0$ | $-\ln(1 - \hat{y}^j)$ | $\hat{y}^j$ close to 1 (big loss if the model is confidently wrong) |

- Correct, confident predictions ($\hat{y}^j \to 1$ when $y^j = 1$, or $\hat{y}^j \to 0$ when $y^j = 0$) give $\ln(1) = 0$ loss.
- Wrong, confident predictions drive $\hat{y}^j \to 0$ when the true label is 1 (or vice versa), and $-\ln(\hat{y}^j) \to \infty$. The loss is unbounded above — wrongness is punished severely.

## Derivation from max likelihood

Assume each label is drawn from a Bernoulli distribution parameterised by the sigmoid output:

$$p_{\mathbf{w}, b}(y^j \mid \mathbf{x}^j) = \begin{cases} \hat{y}^j & \text{if } y^j = 1 \\ 1 - \hat{y}^j & \text{if } y^j = 0 \end{cases}$$

Using the $\{0, 1\}$ encoding, this collapses into one expression valid for both labels:

$$p_{\mathbf{w}, b}(y^j \mid \mathbf{x}^j) = (\hat{y}^j)^{y^j} (1 - \hat{y}^j)^{1 - y^j}$$

(Plug in $y^j = 1$: get $\hat{y}^j$. Plug in $y^j = 0$: get $1 - \hat{y}^j$. ✓)

Now apply the standard [[maximum-likelihood-estimation-nc|maximum likelihood estimation]] recipe:

1. Assume conditional independence: $p(y^1, \dots, y^n \mid \mathbf{x}^1, \dots, \mathbf{x}^n) = \prod_j p(y^j \mid \mathbf{x}^j)$.
2. Take the log to turn the product into a sum.
3. Flip the sign to turn a maximisation into a minimisation.

The result is exactly binary cross-entropy:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \; -\sum_{j=1}^{n} \Big[ y^j \ln \hat{y}^j + (1 - y^j) \ln(1 - \hat{y}^j) \Big]$$

This mirrors the derivation of [[loss-function|squared error]] under Gaussian noise — same pattern, different probability model.

## Why cross-entropy and not squared error for classification

Squared error *would* run — sigmoid is differentiable, so $\nabla L$ is non-zero. But it's a worse fit for two reasons:

1. **Probabilistic meaning.** Cross-entropy is what max likelihood recommends when your outputs are class probabilities. Squared error assumes Gaussian noise on a continuous target, which doesn't match the Bernoulli structure of binary labels.
2. **Gradient magnitude.** Squared error combined with sigmoid leads to gradients of the form $(y - \hat{y}) \sigma'(z)$, which vanishes whenever $\sigma'(z)$ saturates — even when $(y - \hat{y})$ is large. Cross-entropy cancels this sigmoid derivative cleanly, so the parameter gradient becomes simply $(\hat{y} - y) \mathbf{x}$ — proportional to the error, no matter where you are on the sigmoid curve. Learning is faster and more stable.

## Connection to [[gradient-descent-nc|gradient descent]]

Training a sigmoid classifier is just [[gradient-descent-nc|gradient descent]] on binary cross-entropy:

1. Forward: $\hat{y}^j = \sigma(b + \mathbf{w} \cdot \mathbf{x}^j)$.
2. Loss: $L = -\sum_j [y^j \ln \hat{y}^j + (1 - y^j) \ln(1 - \hat{y}^j)]$.
3. Gradient (after chain rule): $\frac{\partial L}{\partial w_i} = \sum_j (\hat{y}^j - y^j) x_i^j$, $\frac{\partial L}{\partial b} = \sum_j (\hat{y}^j - y^j)$.
4. Update: $\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \eta \nabla L_t$.

You'll implement exactly this in the week 2 Python exercise.

## Multiclass extension

For more than two classes, the natural output is a probability distribution over $m$ classes produced by [[softmax]], and the labels are represented as one-hot vectors of length $m$. The loss becomes **categorical cross-entropy**:

$$L = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{m} y_{i,j} \ln \hat{y}_{i,j}$$

where $y_{i,j}$ is 1 if sample $i$ belongs to class $j$ (else 0), and $\hat{y}_{i,j}$ is the softmax output for that class.

Because only one $y_{i,j}$ is non-zero per sample, the inner sum collapses to a single term — $-\ln(\hat{y}_{i, \text{true class}})$ — so the per-sample loss is just $-\log$ of the probability the model assigned to the correct class. Binary cross-entropy is the special case for $m = 2$ (with the two classes implicit in $\hat{y}$ and $1 - \hat{y}$). Same recipe, same MLE derivation, different number of classes.

## Related

- [[sigmoid-function-nc|sigmoid function]] — produces the binary probabilities that cross-entropy consumes
- [[softmax]] — produces the multi-class probabilities for the categorical extension
- [[loss-function]] — the general concept; this is the classification specialisation
- [[maximum-likelihood-estimation-nc|maximum likelihood estimation]] — the derivation of cross-entropy from first principles
- [[gradient-descent-nc|gradient descent]] — what minimises it

## Active Recall

> [!question]- Write the binary cross-entropy loss and explain which term is active for each label.
> $L = -\sum_j [y^j \ln \hat{y}^j + (1 - y^j) \ln(1 - \hat{y}^j)]$. When $y^j = 1$ the $(1 - y^j)$ factor zeroes the second term, leaving $-\ln \hat{y}^j$. When $y^j = 0$ the $y^j$ factor zeroes the first term, leaving $-\ln(1 - \hat{y}^j)$. Each training example contributes one of the two terms.

> [!question]- A sigmoid classifier outputs $\hat{y} = 0.01$ for a sample whose true label is $y = 1$. Compute the per-sample cross-entropy loss and explain why it is large.
> $-[1 \cdot \ln(0.01) + 0 \cdot \ln(0.99)] = -\ln(0.01) \approx 4.6$. The loss is large because the model is confidently wrong — it said the probability of class 1 was only 1%, but the truth is class 1. Cross-entropy grows without bound as confidently-wrong predictions approach $\hat{y} = 0$ (or $\hat{y} = 1$ for a class-0 example).

> [!question]- Why is cross-entropy preferred over squared error when training a sigmoid classifier?
> (1) It is what maximum likelihood recommends under a Bernoulli model for the labels, so it has a principled probabilistic interpretation. (2) Its gradient with respect to the pre-activation $z$ is simply $(\hat{y} - y)$, which cancels the sigmoid's saturating derivative; squared-error gradients include an extra $\sigma'(z)$ factor that vanishes in the saturated tails, making learning slow when the model is confidently wrong.

> [!question]- In the max likelihood derivation of cross-entropy, which step converts a product of probabilities into the summation we see in the loss, and why doesn't this change the optimum?
> Taking the logarithm. $\log(\prod_j p^j) = \sum_j \log p^j$, and because log is monotonically increasing, $\arg\max$ of the log equals $\arg\max$ of the original — so the optimiser moves to the same parameter values. Log is then typical because it tames numerical underflow (products of many small probabilities go to 0) and gives gradients a clean additive form.
