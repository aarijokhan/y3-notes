---
week: 3
topic: "Multi-Layer Networks, Backpropagation, and Generalisation"
deck: NeuralComputation::Week-03
---

TARGET DECK
NeuralComputation::Week-03

## MLP architecture

> [!question]- What is a **multi-layer perceptron (MLP)**, and why do we need one?
> An MLP is a stack of perceptron layers — each layer's outputs feed the next layer's inputs, with smooth (e.g. sigmoid, ReLU) activations between them. We need it because a *single* perceptron can only handle linearly separable data; stacking layers lets the network represent non-linear decision boundaries (XOR, spirals, etc.).

> [!question]- Write the standard MLP notation for layer $\ell$.
> $$z^\ell_j = \sum_k w^\ell_{jk}\, a^{\ell-1}_k + b^\ell_j, \qquad a^\ell_j = \sigma(z^\ell_j)$$
> Superscripts are layer indices, subscripts are unit indices. $z^\ell_j$ is the **pre-activation** of unit $j$ in layer $\ell$, $a^\ell_j$ is its **activation**, and $a^1_k = x_k$ is the input.

> [!question]- A fully connected MLP has layer widths $(3, 3, 3, 3)$. How many learnable parameters does it have?
> **36.** Each of the three non-input layers has $3 \times 3 = 9$ weights plus 3 biases = 12 parameters. Three such layers gives $3 \times 12 = 36$. The input layer has no parameters because no computation happens there.

## Why naive gradient descent doesn't scale

> [!question]- Why is computing $\nabla L$ for an MLP "infeasible by hand" and "wasteful if done naively"?
>
> - **Infeasible:** an MLP can have millions of parameters; you cannot write a closed-form derivative for each.
> - **Wasteful:** the chain rule produces many shared intermediate factors; recomputing them per parameter is exponential. Backprop reuses them, computing each gradient exactly once.

> [!question]- In a 4-layer network, why can vanilla GD only update the **final** layer's weights without backprop?
> Final-layer weights appear directly in the output, so $\partial L / \partial w^L$ is straightforward. For earlier $\ell$, $\partial L / \partial w^\ell$ requires tracing how a change in $w^\ell$ propagates through every subsequent layer. We need a *systematic* way to do this — backpropagation is exactly that.

## Computation graphs and the chain rule

> [!question]- What is a **computation graph**?
> A directed acyclic graph (DAG) representation of a function as nodes (intermediate variables) connected by edges (elementary operations: add, multiply, activation, square, etc.). Every neural network can be drawn as one. Once drawn, the chain rule becomes mechanical — walk paths from a parameter to the loss, multiply local derivatives.

> [!question]- State the **multivariate chain rule** for a variable $b$ that influences $z$ through several intermediate paths $g_1, g_2, \dots$.
> $$\frac{\partial z}{\partial b} = \sum_i \frac{\partial z}{\partial g_i} \cdot \frac{\partial g_i}{\partial b}$$
> When a variable reaches the output through multiple paths in the graph, the contributions from every path are **summed**. This matters because in an MLP, an early-layer weight typically affects the output through many downstream neurons.

## Backpropagation

> [!question]- What two passes does **backpropagation** run, and what does each do?
>
> 1. **Forward pass** — input to output. Compute every node's value from its predecessors and **cache** the values.
> 2. **Backward pass** — output to input. At each node, compute $\partial L / \partial \text{node}$ using the multivariate chain rule. Because we proceed backwards, each successor's gradient is already available when we need it.

> [!question]- Why is backprop linear in the number of graph nodes rather than exponential?
> Each intermediate gradient $\partial L / \partial t$ is computed **once** and reused for every predecessor of $t$. Going backward guarantees the gradient at each successor is ready before we visit the current node. Cache + reuse turns what looks like exponential branching of the chain rule into a single pass over the graph.

> [!question]- Worked backprop: for $z = (a + b)(b + c)$ with $a = -1, b = 3, c = 4$, compute $\partial z / \partial b$.
> Forward: $x = a + b = 2$, $y = b + c = 7$, $z = xy = 14$. Backward: $\partial z / \partial x = y = 7$, $\partial z / \partial y = x = 2$. Both $x$ and $y$ depend on $b$ with $\partial x / \partial b = 1, \partial y / \partial b = 1$, so by the multivariate chain rule $\partial z / \partial b = 7 \cdot 1 + 2 \cdot 1 = 9$.

> [!question]- During inference, do we need the **backward pass**?
> No. The backward pass and weight updates are **training-only** machinery. Once training finishes, you keep only the learned weights and run forward passes to make predictions. Discarding the gradient graph at inference is what makes deployed models cheap.

## Softmax and multi-class

> [!question]- What is **softmax**, and what does it produce?
> $$\text{softmax}(\mathbf{z})_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$$
> It converts a vector of raw scores $\mathbf{z}$ into a **probability distribution** — outputs are positive and sum to 1, so they can be read as $p(y = j \mid \mathbf{x})$. It is the multi-class analogue of sigmoid.

> [!question]- What is a **one-hot vector** for the label $y = 2$ in a 4-class problem?
> $\mathbf{y} = (0, 1, 0, 0)$ — a length-$m$ vector with a 1 in the position of the true class and 0s elsewhere. (Class indexing here starts at 0, so position 2 is the third entry; if 1-indexed, it is the second entry.) Softmax outputs are compared against this one-hot via categorical cross-entropy.

> [!question]- Which probability distribution underlies softmax + categorical cross-entropy?
> A **categorical distribution** over $m$ classes — the multi-class generalisation of Bernoulli. The same MLE recipe used for BCE under Bernoulli applies, just with the categorical PMF, yielding $L = -\sum_j y_j \ln \hat{y}_j$.

## Overfitting and generalisation

> [!question]- What is the difference between **underfitting**, **overfitting**, and a **good fit**?
>
> - **Underfitting:** model too simple, can't even fit the training data. High training loss.
> - **Overfitting:** model too powerful, fits training data including its noise, fails on unseen data. Low training loss, high test loss.
> - **Good fit:** captures the underlying pattern, tolerates noise. Both losses low.

> [!question]- Why is **training loss alone** not enough to detect overfitting?
> Training loss measures **memorisation** — how well the model fits the data it has already seen. Overfitting is failure on *new* data. Only the **test loss** (or validation loss) measures generalisation. If training loss is low but test loss is high, the model has overfit.

> [!question]- What are the three data splits, and what is each one used for?
>
> | Split | Fraction | Purpose |
> |---|---|---|
> | **Training** (~60%) | Fit the weights with gradient descent |
> | **Validation** (~20%) | Tune hyperparameters, pick the stopping point |
> | **Test** (~20%) | Final one-shot evaluation |
>
> Each must stay disjoint. Training on test or validation data corrupts the corresponding signal.

## Regularisation

> [!question]- What is **early stopping**, and what does it monitor?
> Train the network and monitor the **validation loss** at every epoch. Validation loss typically drops, reaches a minimum, then rises as the model starts to memorise noise. Stop training at that minimum and keep those weights — they generalise best.

> [!question]- What is **L2 regularisation** (weight decay), and what is its loss?
> $$L(\boldsymbol{\theta}) = L_{\text{orig}}(\boldsymbol{\theta}) + \lambda \sum_j \theta_j^2$$
> Add a penalty on weight magnitudes to the loss. $\lambda$ is the regularisation strength. The optimiser must now balance fitting the data against keeping weights small — smaller weights yield smoother decision boundaries that generalise better.

> [!question]- Early stopping vs L2 regularisation — what does each technique limit?
>
> - **Early stopping** limits **how long** the model is allowed to overfit (truncates training).
> - **L2 / weight decay** limits **how extreme** the weights can become (caps representational capacity).
>
> They are complementary and almost always used together in modern training.
