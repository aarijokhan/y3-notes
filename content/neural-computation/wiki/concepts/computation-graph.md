---
name: computation-graph
description: A directed acyclic graph that represents a composite function as nodes (variables / intermediate results) and edges (elementary operations), enabling systematic gradient computation via backpropagation.
type: concept
sources:
  - raw/week-03/w03-l07-transcript.txt
  - raw/week-03/w03-l08-transcript.txt
  - raw/week-03/w03-slides.pdf
status: stable
updated: 2026-04-24
---

*Break a messy nested expression into an explicit graph of small steps. Each node holds an intermediate result; each edge is one elementary operation. Once the function is drawn as a DAG, computing gradients with the chain rule becomes mechanical.*

## Definition

A **computation graph** is a **directed acyclic graph (DAG)** where:

- **Nodes** represent variables — inputs, outputs, or intermediate results.
- **Edges** represent elementary operations (addition, multiplication, squaring, applying an activation, etc.). The edge goes from operand(s) to result.

Because it's acyclic, there's no loop — values flow in one direction, from inputs through intermediate results to the final output. Because it's directed, each node has clearly defined **predecessors** (nodes it depends on) and **successors** (nodes that depend on it).

## Why bother: chain rule gets systematic

Given a composite expression like $L = \big((w_i x_i + b) - y\big)^2$, you could compute $\partial L / \partial w$ by a flurry of chain-rule applications in one go. But for a bigger network the expression is a hundred levels deep with shared sub-expressions, and hand-derivation becomes error-prone and compute-heavy.

A computation graph fixes this by:

1. **Factoring** the composite expression into a series of simple, named intermediate results — one per node.
2. **Localising** the derivative work: each edge contributes a single elementary derivative that depends only on its two endpoints.
3. **Chaining** those local derivatives through the graph to recover any $\partial(\text{output})/\partial(\text{parameter})$.

Once the graph is drawn, the chain rule mechanically tells you how to combine the local derivatives along any path from input to output.

## Worked example: rewriting a regression loss as a graph

Take the squared-error loss for a single training example:

$$L = \big((w x + b) - y\big)^2$$

Introduce intermediate variables to isolate each elementary operation:

$$r(w) = w x \qquad q(r) = r + b \qquad p(q) = q - y \qquad L(p) = p^2$$

Draw this as a chain:

$$w \;\xrightarrow{\cdot x}\; r \;\xrightarrow{+b}\; q \;\xrightarrow{-y}\; p \;\xrightarrow{(\cdot)^2}\; L$$

Each arrow is one elementary operation. Each node is a function of its predecessor.

Now to compute $\partial L / \partial w$: walk the chain and multiply the local derivatives.

| Edge | Local derivative |
|---|---|
| $L \leftarrow p$ | $\partial L / \partial p = 2p$ |
| $p \leftarrow q$ | $\partial p / \partial q = 1$ |
| $q \leftarrow r$ | $\partial q / \partial r = 1$ |
| $r \leftarrow w$ | $\partial r / \partial w = x$ |

Chain rule gives:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial q} \cdot \frac{\partial q}{\partial r} \cdot \frac{\partial r}{\partial w} = 2p \cdot 1 \cdot 1 \cdot x = 2px$$

The graph transforms a single tangled derivative into a systematic walk.

### What the chain rule does in a graph

![[3b1b-tree-arrows.svg]]

The chain rule's job in a computation graph is to *trace a path*. Pick any input node $w^{(L)}$ and any output node $C_0$; multiply the local derivatives along every edge on the path connecting them; the product is $\partial C_0 / \partial w^{(L)}$. The yellow arrows above show the path for the simplest case — one input, one output, no branching. When the graph branches, you sum over paths (next section).

## Branching graphs and shared subexpressions

Straight-line chains are easy. Real computations branch and merge — and this is where the graph representation earns its keep.

Consider $z = (a + b)(b + c)$. The variable $b$ appears *twice*. Draw the graph:

- $a, b, c$ are leaf (input) nodes.
- $x = a + b$ (intermediate).
- $y = b + c$ (intermediate).
- $z = x \cdot y$ (output).

Here $b$ feeds into both $x$ and $y$, so $z$ depends on $b$ through **two distinct paths**. Computing $\partial z / \partial b$ requires summing contributions from each path — this is the **multivariate chain rule**, covered in [[backpropagation]]:

$$\frac{\partial z}{\partial b} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial b} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial b}$$

> [!info] ASIDE — Branching is why computation graphs matter for networks
> A neural network is *full* of branching: a single weight early in the network influences many downstream neurons (and therefore many paths to the loss). Without an explicit graph, bookkeeping these paths by hand becomes unmanageable. With the graph, each node just accumulates contributions from all its successors — a local computation.

## Node types

When drawing a computation graph for a neural network (e.g. the week 3 linear-regression graph), the nodes fall into three visual categories:

- **Input nodes** (blue circles in the slides) — features $\mathbf{x}$ and ground-truth labels $y$. Given, not learned.
- **Parameter nodes** (orange circles) — learnable weights $\mathbf{w}$ and biases $b$. These are what gradient descent updates.
- **Computation nodes** (yellow squares) — intermediate results: weighted sums, activations, differences, the loss itself. Produced by combining predecessor nodes via elementary operations.

Only parameter nodes receive gradient updates; input nodes stay fixed, and computation nodes are recomputed on each forward pass.

## Reusing intermediate derivatives (efficiency)

In the $z = (a+b)(b+c)$ graph, both $\partial z / \partial a$ and $\partial z / \partial b$ require $\partial z / \partial x$. A naive pen-and-paper approach recomputes it. A real implementation:

- Traverses the graph backwards from the output.
- Computes each intermediate partial *once*.
- Caches it for use by all downstream dependents.

This is the core insight that makes [[backpropagation]] efficient enough for networks with billions of parameters: *the same intermediate partials appear in many final derivatives, so compute them once and reuse*.

## Related

- [[backpropagation]] — the algorithm that walks the computation graph in both directions (forward for values, backward for gradients)
- [[multi-layer-perceptron]] — a neural network is a computation graph with a specific structure

## Active Recall

> [!question]- What does it mean for a computation graph to be "directed" and "acyclic"? Why do both properties matter?
> **Directed:** each edge has a direction (operand → result), so every node has well-defined predecessors (what it depends on) and successors (what depends on it). This lets us order computations — forward and backward passes both need that ordering. **Acyclic:** no loops. Cycles would mean a value depends on itself, making both evaluation and gradient computation ill-defined. Together, the two properties guarantee a well-defined topological order in which you can evaluate (forward) or differentiate (backward).

> [!question]- For $L = \big((wx+b) - y\big)^2$, draw the computation graph and compute $\partial L / \partial b$.
> Graph: $w \xrightarrow{\cdot x} r \xrightarrow{+b} q \xrightarrow{-y} p \xrightarrow{(\cdot)^2} L$. $b$ enters at the $q = r + b$ node. Local derivatives on the path $b \to q \to p \to L$: $\partial q / \partial b = 1$, $\partial p / \partial q = 1$, $\partial L / \partial p = 2p$. Chain: $\partial L / \partial b = 2p \cdot 1 \cdot 1 = 2p = 2((wx+b) - y)$.

> [!question]- In the graph for $z = (a + b)(b + c)$, why does $\partial z / \partial b$ have two terms whereas $\partial z / \partial a$ has only one?
> $a$ reaches $z$ through exactly one path ($a \to x \to z$), so the chain rule gives one product. $b$ reaches $z$ through two paths ($b \to x \to z$ and $b \to y \to z$), so we must *sum* the contributions from both paths — this is the multivariate chain rule. In network terms: parameters that feed into multiple downstream neurons must accumulate gradient contributions from every path to the loss.

> [!question]- You're given a computation graph for $g = \big((a \cdot b + c \cdot d + e) - f\big)^2$. Identify which leaf variables are **inputs**, which are **learnable parameters**, and which is the **ground-truth label** of a linear regression problem.
> Working backward from the regression structure $L = (\hat{y} - y)^2$ where $\hat{y} = w_1 x_1 + w_2 x_2 + b$: $g$ is the loss, $f$ is the ground-truth label $y$ (it gets subtracted from the prediction), and the predicted-value subtree is $a b + c d + e$. Comparing to $w_1 x_1 + w_2 x_2 + b$, each multiplication is a (parameter × input) pair and the standalone $e$ is the bias. So one valid assignment is **parameters: $a, c, e$ (the two weights and the bias)**; **inputs: $b, d$ (the two features)**; **ground-truth label: $f$**. The two products are interchangeable — $a, b$ and $c, d$ can swap roles within each multiplication.

> [!question]- Why do modern deep learning frameworks (PyTorch, TensorFlow) build a computation graph rather than just computing derivatives symbolically?
> Building the graph makes gradient computation *local and reusable*: each node only needs to know how its output depends on its immediate inputs. The framework can then walk the graph backwards, accumulating gradients, and crucially can share intermediate results between the many derivatives it needs to compute. Symbolic differentiation of a full expression produces enormous formulae with massive redundancy; the graph-based approach (automatic differentiation) is exponentially more efficient for the kind of deeply nested expressions neural networks define.
