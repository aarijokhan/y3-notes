---
type: week
week: 3
title: "Multi-Layer Networks, Backpropagation, and Generalisation"
dates: 2025-10-13 to 2025-10-19
sources:
  - raw/week-03/w03-l07-transcript.txt
  - raw/week-03/w03-l08-transcript.txt
  - raw/week-03/w03-slides.pdf
concepts:
  - "[[multi-layer-perceptron]]"
  - "[[computation-graph]]"
  - "[[backpropagation]]"
  - "[[softmax]]"
  - "[[overfitting-nc|overfitting]]"
  - "[[regularization-nc|regularization]]"
status: stable
updated: 2026-04-24
---

> [!question]+ THE CRUX: How do we train a network that has more than one layer — and once we can, how do we stop it from just memorising the training data?

*Stacking perceptrons into layers unlocks non-linear decision boundaries but makes gradient computation vastly harder. The trick is to represent the network as a computation graph and compute all gradients in a single backward pass — that's backpropagation, and it rests entirely on the chain rule. Once training works, a new problem appears: powerful models tend to memorise rather than generalise, so we need explicit tools (data splits, early stopping, regularisation) to stop them.*

## Where we left off

After week 2 we had a trained single [[perceptron]]: a [[loss-function]] via max likelihood, [[gradient-descent-nc|gradient descent]] to minimise it, and a [[sigmoid-function-nc|sigmoid function]] activation that makes the classification case differentiable. Good for linearly separable data, good for regression with a linear trend.

But take one look at non-linearly-separable data — XOR, concentric circles, spirals — and the single perceptron fails. No straight line exists that separates the classes. The model's representational capacity is simply too low.

## Stacking perceptrons: the MLP

The fix is to combine multiple perceptrons into layers, and stack multiple layers. Each layer's outputs feed the next layer's inputs, and the composition can represent wildly non-linear decision surfaces. This is the [[multi-layer-perceptron]] (MLP) — also called a **feed-forward neural network**, because information flows in one direction, input to output.

The structure:

- **Input layer** — just a placeholder for the features, no computation.
- **Hidden layers** — one or more layers of sigmoid (or other smooth) perceptrons.
- **Output layer** — one neuron for regression / binary classification, $m$ neurons for $m$-class classification.

The notation is worth committing to memory, because it recurs in every subsequent week:

$$z^\ell_j = \sum_k w^\ell_{jk}\, a^{\ell-1}_k + b^\ell_j, \qquad a^\ell_j = \sigma(z^\ell_j)$$

Superscripts are layer indices; subscripts are unit indices. $a^1_k = x_k$ at the input layer.

> [!question]- A fully connected MLP has layer widths $(3, 3, 3, 3)$. How many learnable parameters does it have?
> 36. Each of the three non-input layers has $3 \times 3 = 9$ weights plus 3 biases = 12 parameters. Three such layers = 36 total. The input layer has no parameters because no computation happens there.

## The training problem: gradient descent still works, but computing the gradient is hard

The optimisation framework hasn't changed. We still minimise a loss function with [[gradient-descent-nc|gradient descent]]:

$$\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \eta \, \nabla L_t$$

The catch is $\nabla L$. For a single perceptron it's a short chain-rule calculation. For an MLP with millions of parameters, direct computation is both:

- **Infeasible by hand** — you can't write down a closed-form derivative for every parameter.
- **Wasteful if done naively** — many intermediate derivatives get reused across different parameters' gradients, so recomputing them is exponentially slow.

The loss depends on *every* parameter, because the forward pass chains them all together. A weight in layer 2 influences activations in layer 3, which influence layer 4, which influence the final output. The output depends on *all* the parameters, not just the last layer.

> [!question]- In a 4-layer network, gradient descent on its own can only update the parameters of the final layer. Why? What's the missing piece that lets it update earlier layers too?
> If you apply gradient descent directly to the loss, you compute $\partial L / \partial w^L$ for the last layer's weights — these appear explicitly in the final output. But $\partial L / \partial w^\ell$ for earlier $\ell$ requires the chain rule: trace how a change in $w^\ell$ propagates through every subsequent layer to change the loss. You need a *systematic way* to do this for every parameter — which is exactly what backpropagation provides.

## The computation graph: making derivatives mechanical

To systematically compute gradients, represent the whole network as a [[computation-graph]] — a directed acyclic graph of elementary operations. Every intermediate quantity gets its own node; every arrow is one elementary operation (addition, multiplication, activation, squaring, etc.).

The squared-error loss $L = ((wx + b) - y)^2$ becomes a chain of five nodes — $w \to r \to q \to p \to L$ — each connected by a single operation. A whole MLP becomes a bigger graph with branching, but built from the same elementary pieces.

Once the graph is drawn, the chain rule is mechanical. Walk any path from parameter to loss, multiply the local derivatives along the way, and you have the gradient.

## Chain rule: univariate and multivariate

Recall the univariate [[chain rule]]: if $f = f(g)$ and $g = g(t)$, then

$$\frac{\partial f}{\partial t} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial t}$$

But in a computation graph, a single variable often influences the output through **multiple paths**. For $z = (a+b)(b+c)$, the variable $b$ reaches $z$ through both $x = a + b$ and $y = b + c$. The **multivariate chain rule** says: sum contributions from every path.

$$\frac{\partial z}{\partial b} = \frac{\partial z}{\partial x}\frac{\partial x}{\partial b} + \frac{\partial z}{\partial y}\frac{\partial y}{\partial b}$$

In a neural network, early-layer weights routinely influence the output through many paths (every downstream neuron they touch). The multivariate chain rule is what correctly accumulates these contributions.

## Backpropagation: the two-pass algorithm

[[Backpropagation]] is the algorithm that walks the computation graph twice:

**Forward pass** — input to output. At each node, compute its value from its predecessors and cache it. At the end, you have the loss and all intermediate activations.

**Backward pass** — output back to input. At each node $t$, compute $\partial L / \partial t$ using the multivariate chain rule:

$$\frac{\partial L}{\partial t} = \sum_{i : g_i \text{ successor of } t} \frac{\partial L}{\partial g_i} \cdot \frac{\partial g_i}{\partial t}$$

Crucially, because we go *backward*, $\partial L / \partial g_i$ is already computed by the time we reach $t$. We just multiply by local derivatives (which use cached forward-pass values) and sum.

### Worked example

For $z = (a+b)(b+c)$ with $a = -1$, $b = 3$, $c = 4$:

**Forward:** $x = 2$, $y = 7$, $z = 14$.

**Backward:** $\partial z / \partial z = 1$; $\partial z / \partial x = y = 7$; $\partial z / \partial y = x = 2$; $\partial z / \partial a = 7 \cdot 1 = 7$; $\partial z / \partial c = 2 \cdot 1 = 2$; $\partial z / \partial b = 7 \cdot 1 + 2 \cdot 1 = 9$ (the two paths combine via multivariate chain rule).

Every intermediate gradient is computed **once** and reused for every predecessor. That's the efficiency win that makes training a billion-parameter network possible.

> [!tip] TIP — Backpropagation is not a new algorithm, it's a reorganisation
> Backprop doesn't invent any new math — it's the chain rule applied systematically on a graph. What makes it clever is *the order of computation*: go backward so each gradient is available when its predecessors need it, cache every intermediate result, and the whole thing runs in time linear in the number of graph nodes. The same principle underlies every modern automatic-differentiation framework (PyTorch, TensorFlow, JAX).

## Training loop

Putting it all together, one training step is:

1. **Forward pass:** compute $\hat{\mathbf{y}}$ from inputs and current weights. Cache all intermediate activations.
2. **Compute loss:** $L(\boldsymbol{\theta}) = \text{loss}(\hat{\mathbf{y}}, \mathbf{y})$.
3. **Backward pass:** walk the graph backwards, computing $\nabla L$ over all parameters.
4. **Update:** $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla L$.
5. Repeat until convergence or the iteration budget.

At **inference time** (using a trained network to predict), you only need the forward pass. The backward pass and the weight updates are training-only machinery; once training finishes, you throw them away and keep the learned weights.

## Multi-class classification: softmax

Binary classification pairs sigmoid with [[binary-cross-entropy]]. Multi-class classification (digit recognition, letter recognition, etc.) generalises both. The output layer has $m$ neurons — one per class — and produces raw scores $\mathbf{z} = (z_1, \dots, z_m)$. [[softmax]] converts these scores into a probability distribution:

$$\text{softmax}(\mathbf{z})_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

Outputs are positive and sum to 1, so they're readable as $p(y = j \mid \mathbf{x})$. Labels are represented as **one-hot vectors** of length $m$: a 1 in the true-class position, 0s elsewhere. And the loss is the multi-class version of cross-entropy — exactly the same MLE derivation as binary cross-entropy, just over a categorical distribution instead of Bernoulli.

## Once training works, generalisation becomes the real problem

With backpropagation and softmax, you can now train arbitrarily deep networks for arbitrary numbers of classes. The next problem: the model might memorise the training set rather than learn generalisable patterns. This is [[overfitting-nc|overfitting]].

**Underfitting** — model too simple, can't even fit the training data.
**Overfitting** — model too powerful, fits training data near-perfectly including the noise, fails on unseen data.
**Good fit** — captures the underlying pattern, tolerates noise.

> [!info] ASIDE — The student analogy
> A student who memorises past exam answers but doesn't understand the material fails on new questions. That's overfitting. A student who doesn't study at all fails both familiar and new questions. That's underfitting. You want the student who understands the concepts well enough to solve problems they've never seen.

The key observation: the model's **training loss** measures memorisation; only the **test loss** (on unseen data) measures generalisation. If the two diverge — training loss low, test loss high — you've overfit.

## Three-way data split

To detect overfitting and control it, partition your data into three disjoint sets:

| Split | Fraction | Purpose |
|---|---|---|
| **Training** | ~60% | Fit the weights with gradient descent |
| **Validation** | ~20% | Tune hyperparameters, pick the stopping point |
| **Test** | ~20% | Final one-shot evaluation |

Each split has its own job and must not leak into another. **Training on the test set** compromises your final evaluation (it's no longer "unseen"). **Training on the validation set** compromises the signal you use to tune hyperparameters (the validation loss stops being a faithful estimate of generalisation).

> [!tip] TIP — It's just like a university term
> Training data = practice exercises. Validation = mid-term assessments. Test = final exam. The final-exam rule — you don't study from the exam beforehand — is the test-set rule.

## Fixing overfitting: early stopping and regularisation

**Early stopping:** monitor validation loss during training. It drops initially (the model is learning useful patterns), reaches a minimum, then rises (the model starts memorising noise). Stop training at the minimum — those weights generalise best.

**Regularisation** (specifically, [[regularization-nc|weight decay]]): add an explicit penalty on weight magnitudes to the loss:

$$L(\boldsymbol{\theta}) = L_{\text{orig}}(\boldsymbol{\theta}) + \lambda \sum_j \theta_j^2$$

The hyperparameter $\lambda$ controls the strength of the penalty. The optimiser now has to balance fitting the data against keeping weights small. Smaller weights → smoother decision boundary → better generalisation.

The two techniques are complementary: early stopping limits *how long* the model is allowed to overfit; regularisation limits *how extreme* the weights can become. Both are used together in modern practice.

## Summary: the week 3 pipeline

We now have the full training pipeline for deep networks:

1. **Architecture:** stack perceptrons into a multi-layer network.
2. **Representation:** model the network as a computation graph.
3. **Training:** forward pass to compute outputs, backward pass (backprop) to compute gradients, gradient descent to update weights.
4. **Output heads:** sigmoid + binary cross-entropy for binary classification, softmax + categorical cross-entropy for multi-class, linear + MSE for regression.
5. **Generalisation:** split data into train/val/test, monitor validation loss, use early stopping and regularisation.

## Concepts introduced this week

- [[multi-layer-perceptron]] — architecture of stacked perceptron layers; handles non-linearly-separable data
- [[computation-graph]] — DAG representation of a function as nodes (variables) and edges (operations)
- [[backpropagation]] — efficient gradient computation via forward and backward passes over the graph
- [[softmax]] — multi-class output activation producing a probability distribution
- [[overfitting-nc|overfitting]] — the memorisation failure mode; paired with underfitting and the generalisation ideal
- [[regularization-nc|regularization]] — weight decay / L2 penalty to reduce effective model capacity

## Connections

- **Builds on** [[week-02]]: gradient descent still does the optimisation; sigmoid still makes classification differentiable. This week's novelty is (a) stacking perceptrons and (b) the algorithm that computes gradients efficiently for the stack.
- **Builds on** [[week-01]]: the MLE framework still derives losses. Binary cross-entropy extends to categorical cross-entropy via the same recipe, now with softmax producing class probabilities.
- **Sets up** week 4: convolutional neural networks (CNNs) — a specialised architecture for image inputs that isn't fully connected. Same backprop, different connection pattern.

## Open questions

- The exact gradient of softmax + categorical cross-entropy was sketched but not derived. Worth working through; the result simplifies to $(\hat{y} - y)$ just like sigmoid + binary cross-entropy.
- Modern deep networks use activations beyond sigmoid — ReLU, tanh, GELU — that avoid the vanishing-gradient problem at depth. Those appear in later weeks but sit on top of the same backprop machinery.
- Architecture design (how many layers? how wide?) was left as "human trial and error". Neural Architecture Search exists but is beyond this module's scope.
