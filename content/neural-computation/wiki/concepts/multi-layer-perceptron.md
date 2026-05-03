---
name: multi-layer-perceptron
description: A feed-forward neural network made of multiple layers of perceptrons; the simplest architecture that can handle non-linearly-separable data.
type: concept
sources:
  - raw/week-03/w03-l07-transcript.txt
  - raw/week-03/w03-l08-transcript.txt
  - raw/week-03/w03-slides.pdf
status: stable
updated: 2026-04-24
---

*Stack multiple perceptrons in a layer, stack multiple layers one after another, feed the output of each layer into the next. That's a multi-layer perceptron — and it's what you need when a single straight line can't separate your data.*

## Why we need more than one perceptron

A single [[perceptron]] can only draw one hyperplane — one straight line (or linear boundary) through input space. That's fine for linearly separable classification and for data with a linear trend, but it fails on anything curvier. The XOR pattern, or any dataset with positive examples on opposite corners, simply cannot be split by a single line.

The fix: use *multiple* perceptrons. Two perceptrons in a first layer each draw their own boundary, carving space into regions. A perceptron in the next layer combines their outputs to produce a more complex decision surface. Stack enough of these and you can approximate any reasonable function — that's the power of an **MLP**.

MLPs are also called **feed-forward neural networks** — "feed-forward" because information flows only in one direction, from input to output, never backward during inference. (Training is a different story — see [[backpropagation]].)

> [!tip] TIP — Layers as a hierarchy of abstraction
> The deep idea behind layered networks isn't really "more capacity"; it's **compositional learning**. Each layer solves a simpler sub-problem, and stacking layers builds up to solve a complex one.
>
> Think of building a house. You can't go directly from atoms to a finished house. But you *can* go atoms → materials (wood, metal, glass) → components (walls, beams, windows) → rooms → house. Each step is a manageable transformation; the complexity emerges from the composition.
>
> Or think of how you learned to read: letters → words → grammar → meaning. You don't memorise every possible sentence — you build up layers of understanding.
>
> A neural network does the same thing automatically. Early layers learn simple features (edges, textures); middle layers compose those into parts (shapes, motifs); later layers compose parts into concepts (objects, classes). Each layer breaks the problem into something the next layer can build on. This is what makes deep networks dramatically more powerful than shallow ones, even when they have the same total parameter count.

## Anatomy of an MLP

An MLP consists of:

- **Input layer** — just the features; no computation happens here, it's a placeholder for $\mathbf{x}$.
- **Hidden layers** — one or more layers of perceptrons between input and output. "Hidden" because they're not directly observable from outside; they process intermediate representations.
- **Output layer** — the final layer, whose size matches the task (1 unit for binary classification / regression, $k$ units for $k$-class classification).

In a **fully connected** MLP, every neuron in layer $\ell$ is connected to every neuron in layer $\ell - 1$. This is the default architecture; other connection patterns (e.g. convolutional, recurrent) come later in the module.

Each individual hidden/output neuron is a **soft perceptron** — a perceptron with a smooth activation like the [[sigmoid-function-nc|sigmoid function]] (not sign, because [[gradient-descent-nc|gradient descent]] needs differentiability).

## Notation (this is important)

The notation the module uses is worth memorising — it appears in exam questions and in every subsequent week.

| Symbol | Meaning |
|---|---|
| $L$ | Total number of layers (superscript $1$ = input layer, $L$ = output layer) |
| $m$ | Width of a layer (number of neurons); can vary between layers |
| $a^\ell_j$ | Activation of neuron $j$ in layer $\ell$ — the *output* after the activation function |
| $z^\ell_j$ | Weighted input to neuron $j$ in layer $\ell$ — the raw pre-activation value |
| $w^\ell_{jk}$ | Weight from neuron $k$ in layer $\ell - 1$ to neuron $j$ in layer $\ell$. **Superscript = destination layer; subscript = (destination unit, source unit).** |
| $b^\ell_j$ | Bias of neuron $j$ in layer $\ell$ |

![[mlp-notation-diagram.png]]

The core equations for a single neuron in layer $\ell$:

$$z^\ell_j = \sum_{k} w^\ell_{jk}\, a^{\ell-1}_k + b^\ell_j$$
$$a^\ell_j = \sigma(z^\ell_j)$$

where $\sigma$ is an activation function (sigmoid, ReLU, tanh, …). The input layer uses $a^1_k = x_k$ directly — there is no weighted sum or activation at layer 1.

> [!tip] TIP — Why use $a$ instead of $x$ for activations
> At the input layer, the "activations" are just the raw features, so it looks odd to not call them $x$. But layers 2 onwards receive *activations from the previous layer* — not raw inputs — and using $a$ uniformly across all layers makes the recurrence clean: $a^\ell_j$ depends on $a^{\ell-1}_k$, regardless of which layer you're in. A single piece of notation covers every layer.

## Why activations *must* be non-linear

Stacking layers without non-linearity buys you nothing. Suppose every neuron just computed $z^\ell = W^\ell a^{\ell-1} + b^\ell$ — no $\sigma$. Compose two layers:

$$a^3 = W^3 a^2 + b^3 = W^3 (W^2 a^1 + b^2) + b^3 = (W^3 W^2)\, a^1 + (W^3 b^2 + b^3)$$

That's just *one* linear transformation with combined weights $W^3 W^2$ and combined bias $W^3 b^2 + b^3$. No matter how many layers you stack, a network of pure linear layers is mathematically equivalent to a single linear layer — same expressive power as a single perceptron, completely defeating the point of having multiple layers.

The non-linear activation function (sigmoid, tanh, ReLU) is what breaks this collapse. With a non-linearity between layers, $a^3 = \sigma(W^3 \sigma(W^2 a^1 + b^2) + b^3)$ does *not* simplify to a single linear transformation. Each layer can now genuinely transform the representation in ways the previous layer couldn't, and the composition is genuinely richer than any single layer.

So the activation function isn't just a differentiability fix (see [[sigmoid-function-nc|sigmoid function]]) — it's also what makes multi-layer networks more powerful than single-layer ones in the first place. Without it, depth is wasted.

## Counting parameters

Exam questions routinely ask you to count parameters. The recipe:

**For layer $\ell$ with $m_\ell$ neurons, receiving input from a layer with $m_{\ell-1}$ neurons:**
- Weights: $m_\ell \times m_{\ell-1}$ (one per connection, fully connected)
- Biases: $m_\ell$ (one per neuron in layer $\ell$)
- Total parameters for layer $\ell$: $m_\ell \times (m_{\ell-1} + 1)$

**For the whole network:** sum over all layers $\ell = 2, \dots, L$. The input layer contributes no parameters.

### Worked example

For the network shown in the lecture (input: 3 features, three hidden/output layers of 3 neurons each, so $L = 4$, each layer width 3):

- Layer 1 → Layer 2: $3 \times 3 + 3 = 12$ parameters
- Layer 2 → Layer 3: $3 \times 3 + 3 = 12$ parameters
- Layer 3 → Layer 4: $3 \times 3 + 3 = 12$ parameters
- **Total: 36 parameters.**

(Note: the number of activations — the $a^\ell_j$ values — are *not* learnable parameters. Only weights and biases are.)

![[mlp-parameter-counting.png]]

## Depth and width

- **Depth** ($L$) — the number of layers. Deeper networks can represent more complex functions, but are harder to train (vanishing gradients, longer backprop chains).
- **Width** ($m$) — the number of neurons per layer. Wider layers give each layer more representational capacity.

The distinction: a **deep** network has many layers; a **wide** network has many neurons per layer. "Depth" and "width" are independent knobs.

> [!info] ASIDE — How do we choose depth and width?
> There's no scientific answer. It's currently a human-expertise / trial-and-error process. Pick based on problem complexity, past experience, and experimentation. The subfield of **Neural Architecture Search (NAS)** and tools like **AutoML** try to automate this — by searching over possible architectures — but for the purposes of this module, treat layer count and width as hyperparameters chosen by the practitioner.

## Worked example: a 3-perceptron MLP for non-linearly-separable data

Consider data in 2D with positives ($+$) at $(-1, 1)$ and $(1, -1)$ and negatives ($-$) at $(-1, -1)$ and $(1, 1)$ — opposite-corner pattern, no straight line separates the two classes. We can solve it with three perceptrons: two in a hidden layer, one in the output.

**Hidden layer.** Pick two perceptrons whose decision boundaries each cut off one positive corner from the rest of the plane:

- Perceptron 1: $\mathbf{w}_1 = \tfrac{1}{\sqrt{2}}(-1, 1)$, bias $b_1 \in (-\sqrt{2}, 0)$. Its boundary is the line $-x_1 + x_2 = $ const; outputs $+1$ for $(-1, 1)$ (one positive corner) and $-1$ for the other three points.
- Perceptron 2: $\mathbf{w}_2 = \tfrac{1}{\sqrt{2}}(1, -1)$, bias $b_2 \in (-\sqrt{2}, 0)$. Symmetric — outputs $+1$ for $(1, -1)$ (the other positive corner) and $-1$ elsewhere.

After the hidden layer, the only way both outputs $z_1, z_2$ are simultaneously $-1$ is for one of the negative-class corners. The positive-class corners produce $(+1, -1)$ or $(-1, +1)$.

**Output perceptron.** Now the question is reduced: classify $(z_1, z_2)$ such that $(-1, -1)$ → negative, anything else → positive. That's a *linearly separable* problem in $(z_1, z_2)$-space, so a single perceptron suffices: $\mathbf{w}_3 = (1, 1) / \sqrt{2}$, $b_3 \in (0, \sqrt{2})$. The boundary "is at least one input $+1$?" splits the negative case from the two positive cases.

The lesson: each hidden-layer perceptron carves space into a half-plane, and the output perceptron combines those half-planes into a non-linear region. Two perceptrons + one output = enough to solve a problem no single perceptron can.

## Training an MLP is harder than training a perceptron

[[gradient-descent-nc|gradient descent]] still applies — the core update $\boldsymbol{\theta}^{t+1} = \boldsymbol{\theta}^t - \eta \nabla L_t$ is unchanged. The hard part is computing $\nabla L$. A single perceptron has just $D + 1$ partial derivatives; a deep MLP can have millions or billions. The output loss depends on *every* parameter in *every* layer, because the forward pass chains them together.

Naive application of the chain rule works but would be redundant — the same intermediate derivatives get recomputed many times. The efficient algorithm that computes all these gradients in one backward pass is [[backpropagation]], built on top of the [[computation-graph]] representation of the network.

## Related

- [[perceptron]] — the building block
- [[sigmoid-function-nc|sigmoid function]] — the default hidden-unit activation
- [[computation-graph]] — the data structure backprop operates on
- [[backpropagation]] — how we actually train MLPs
- [[softmax]] — the output-layer activation for multiclass tasks

## Active Recall

> [!question]- For a fully connected MLP with layer widths $(4, 5, 3, 2)$, how many learnable parameters are there in total?
> Layer 1→2: $5 \times 4 + 5 = 25$. Layer 2→3: $3 \times 5 + 3 = 18$. Layer 3→4: $2 \times 3 + 2 = 8$. Total: $25 + 18 + 8 = 51$. The input layer (width 4) contributes no parameters because no computation happens there.

> [!question]- In the notation $w^3_{2,5}$, what does each index mean?
> Superscript $3$: this weight belongs to layer 3 (i.e. it feeds *into* a neuron in layer 3). Subscript $(2, 5)$: the weight connects neuron 5 in layer 2 (source) to neuron 2 in layer 3 (destination). Convention: superscript = destination layer; subscript = (destination unit, source unit).

> [!question]- Why does a single perceptron fail on XOR-like data, and how does an MLP fix it?
> A single perceptron can only draw one linear boundary; XOR's positive examples sit on opposite corners of the plane, so no single line separates them. An MLP uses multiple perceptrons in a first layer to carve space into regions, and then a further perceptron combines those regions — effectively composing linear boundaries into a non-linear one. Two hidden-layer perceptrons plus an output perceptron is enough to solve XOR.

> [!question]- What is the difference between $z^\ell_j$ and $a^\ell_j$?
> $z^\ell_j = \sum_k w^\ell_{jk} a^{\ell-1}_k + b^\ell_j$ is the raw pre-activation — the weighted sum of inputs plus the bias, *before* any non-linearity. $a^\ell_j = \sigma(z^\ell_j)$ is the activation — the output after applying the activation function. The neuron computes $z$ and then emits $a$; downstream layers only see $a$.

> [!question]- Why are the hidden layers called "hidden"?
> They sit between the input and the output, so their activations aren't directly observed from outside the network. You see what goes in (input) and what comes out (output), but not the intermediate representations formed inside — they're hidden from the external observer's view.

> [!question]- A friend says "since each layer is just $z = Wa + b$, stacking ten of them gives a much more expressive model than one layer." Why are they wrong?
> Composing linear maps yields another linear map. $W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$ — same shape as a single linear layer, just with combined coefficients. Without a non-linear activation between layers, depth adds nothing. The non-linearity (sigmoid, tanh, ReLU) is what actually makes stacked layers express functions a single layer cannot. Depth is only useful when paired with non-linearity.
