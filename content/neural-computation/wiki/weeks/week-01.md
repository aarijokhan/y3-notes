---
type: week
week: 1
title: "The Perceptron and the Optimisation Problem"
dates: 2025-09-29 to 2025-10-05
sources:
  - raw/week-01/w01-l01-transcript.txt
  - raw/week-01/w01-l02-transcript.txt
  - raw/week-01/w01-l03-transcript.txt
  - raw/week-01/w01-slides.pdf
concepts:
  - "[[perceptron]]"
  - "[[dot-product]]"
  - "[[decision-boundary-nc|decision boundary]]"
  - "[[loss-function]]"
  - "[[maximum-likelihood-estimation-nc|maximum likelihood estimation]]"
status: stable
updated: 2026-04-20
---

> [!question]+ THE CRUX: How does a single artificial neuron turn data into decisions — and what breaks when the problem gets harder?

*A perceptron splits the world with a hyperplane; learning means finding the hyperplane that splits best; and "best" requires a loss function, which maximum likelihood tells us should be squared error.*

## Not everything smart is machine learning

Let's start by clearing up a common confusion. AI is a big tent. Deep Blue beat Kasparov at chess in 1997, but it wasn't doing machine learning — it was searching millions of moves per second using the minimax algorithm, following hand-coded evaluation rules. Navigation apps find shortest paths with Dijkstra or A*, not neural networks. These are engineered algorithms: clever, powerful, and entirely hand-designed.

Machine learning is different. Instead of writing rules, you provide data and let the system learn the rules by adjusting parameters. The nested hierarchy looks like this: **Artificial Intelligence** (anything that performs tasks typically requiring human intelligence) contains **Machine Learning** (algorithms that learn from data without explicit instructions), which in turn contains **Neural Computation / Deep Learning** (ML that uses multi-layered neural networks).

Within ML, we distinguish **supervised learning** — where every training example comes with a correct answer — from **unsupervised learning**, where the algorithm discovers structure on its own (clustering, dimensionality reduction). This module focuses overwhelmingly on supervised learning with neural networks.

> [!info] ASIDE — A brief history of AI winters
> AI isn't new — it has been around since the 1940s. The field has gone through dramatic cycles of hype and disappointment. The early perceptron era (1950s–60s) ended when Minsky and Papert showed its limitations. The backpropagation revival (1980s) faded when training deep networks proved impractical. The current boom — driven by massive data, GPU compute, and architectural innovations like transformers — began around 2012. Whether it sustains depends on whether the field delivers on its promises.

## The simplest learner: a single neuron

So what does the simplest possible learning machine look like? The [[perceptron]] is modelled on a biological neuron — dendrites receive input signals, the cell body aggregates them, and the axon fires if the total exceeds a threshold. The artificial version is mathematically clean: take inputs $x_1, \dots, x_D$, multiply each by a weight $w_i$, add a bias $b$, and pass the result through a sign function.

Think of it like cooking: the inputs are your ingredients, the weights tell you how much of each to use, and the output is the dish. Change the weights and you change the recipe.

> [!info] ASIDE — Frank Rosenblatt's Perceptron (1958)
> The original perceptron was a physical machine built at Cornell — a room-sized contraption of wires and motors. The New York Times reported it as "the embryo of an electronic computer that the Navy expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence." The hype was premature, but the core idea was sound.

## Drawing lines through data

What does a perceptron actually do geometrically? Imagine plotting data points in 2D — say, letters "A" and "C" represented by two pixel values. The perceptron draws a straight line (a [[decision-boundary-nc|decision boundary]]) that separates the two classes. Points on one side get labelled $+1$, points on the other get $-1$.

The [[dot-product]] is the mechanism behind this. Computing $\mathbf{w} \cdot \mathbf{x}$ tells you the signed distance from a point to the boundary — positive means same side as $\mathbf{w}$, negative means opposite side. The sign function converts that distance into a hard classification.

The weight vector $\mathbf{w}$ controls the *tilt* of the line (the boundary is always perpendicular to $\mathbf{w}$), while the bias $b$ controls the *position* (how far the line sits from the origin). All of this generalises seamlessly to higher dimensions — in 3D the boundary is a plane, in 784 dimensions (a $28 \times 28$ image) it's a hyperplane.

> [!tip] TIP — Build intuition in low dimensions
> You can't visualise a hyperplane in 784-dimensional space, and that's fine. Build your geometric intuition in 2D and 3D, where you can draw pictures and check your reasoning. Then trust that the algebra generalises — it does, and the math is identical.

> [!question]- If a perceptron's bias $b$ is negative, which direction does the decision boundary shift relative to $\mathbf{w}$?
> A negative $b$ shifts the boundary in the direction of $\mathbf{w}$ (toward the $+1$ side). You can verify this: the boundary sits at distance $-b/\|\mathbf{w}\|$ from the origin along $\mathbf{w}$. If $b < 0$, that distance is positive, so the boundary moves in the direction $\mathbf{w}$ points.

## When a straight line isn't enough

Here's where the perceptron hits a wall. Some datasets can't be separated by a single straight line. Consider four points at $(\pm 1, \pm 1)$ where opposite corners share a label — the XOR pattern. No matter how you tilt or shift a single line, you'll always misclassify at least one point.

The fix is to use *multiple* perceptrons. Two perceptrons in a first layer each draw their own line, carving the space into regions. A third perceptron in a second layer combines their outputs to make the final classification. This is a tiny multi-layer perceptron — the simplest neural *network*. We'll see how to train these in week 3 with backpropagation.

## From classification to regression

Now, take the same perceptron and *remove* the sign function. Instead of outputting $+1$ or $-1$, it outputs the raw number $\hat{y} = b + \mathbf{w} \cdot \mathbf{x}$. This is linear regression — predicting a continuous value like commute time from features like distance and day of the week. The weight $w_i$ becomes the slope along each input dimension, and $b$ is the y-intercept.

Same architecture, different activation, different job. Classification splits space; regression fits a surface through it.

## The optimisation problem: how do we learn?

We now have a model with tuneable knobs ($\mathbf{w}$ and $b$), but how do we set them? This is the central question: **learning is optimisation**.

The idea is simple in principle. Define a [[loss-function]] that measures how bad the current parameters are — the total error between predictions and truth. Then find the parameters that make the loss as small as possible:

$$\mathbf{w}^*, b^* = \arg\min_{\mathbf{w}, b} \; L(\mathbf{w}, b)$$

Consider a concrete example: three thermometers read 19°C, 17°C, and 24°C. What's the true temperature? If we guess 21°C, the total absolute error is $|19 - 21| + |17 - 21| + |24 - 21| = 9$. If we guess 22°C, the error is 10 — worse. We want the guess that minimises the total error.

But which error to use? Absolute error? Squared error? This isn't just a design choice — there's a principled answer.

## Why squared error: the MLE connection

[[Maximum-likelihood-estimation]] provides the answer. If we assume each measurement is independently drawn from a normal distribution centred on the true value — a reasonable assumption for noisy measurements — then the parameter value most likely to have produced the data is the one that minimises the **sum of squared errors**.

The derivation is elegant: write the product of Gaussian probabilities, take the log (which preserves the optimum because $\log$ is monotonically increasing), drop constant terms, flip the sign from max to min, and you arrive at $\arg\min \sum (x_i - \hat{x})^2$. Squared error isn't arbitrary — it's what probability theory recommends under Gaussian noise.

> [!question]- Why does maximising the likelihood under a Gaussian noise assumption lead to minimising the sum of squared errors?
> The Gaussian PDF contains the term $\exp(-(x_i - \hat{x})^2 / 2\sigma^2)$. Taking the log of the product of these across all data points gives a sum of $-(x_i - \hat{x})^2$ terms (plus constants). Maximising this negative sum is equivalent to minimising $\sum(x_i - \hat{x})^2$. The key move is that log converts the product to a sum and is monotonically increasing, so the optimum doesn't shift.

## What comes next

We now know *what* to minimise — a loss function, specifically squared error — but we don't yet know *how*. With billions of parameters each taking any real value, brute-force search is out of the question. Week 2 introduces **gradient descent**: the algorithm that efficiently navigates the loss landscape by following the slope downhill.

> [!question]- In week 1 we established that learning is optimisation. What are the two missing pieces that week 2 needs to address?
> First, we need an *algorithm* to efficiently find the parameter values that minimise the loss (gradient descent). Second, we need to compute *how the loss changes* as each parameter changes (derivatives/gradients), which tells us which direction to adjust each parameter.

## Concepts introduced this week

- [[perceptron]] — a single artificial neuron that classifies (with sign function) or regresses (without)
- [[dot-product]] — the algebraic operation that measures signed distance to a hyperplane
- [[decision-boundary-nc|decision boundary]] — the hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$ that separates classes
- [[loss-function]] — measures prediction error; the thing we minimise during training (MAE, SSE, MSE)
- [[maximum-likelihood-estimation-nc|maximum likelihood estimation]] — derives squared error loss from Gaussian noise assumptions

## Connections

- **Sets up** [[week-02]]: gradient descent and its variants — the algorithm that actually solves the optimisation problem posed here.
- **Foreshadows** week 3: multi-layer perceptrons and backpropagation, which overcome the linear separability limitation.

## Open questions

- The perceptron learning algorithm (how to iteratively update weights for classification) was mentioned historically but not covered in detail — it may appear in week 2.
- The relationship between different activation functions (sign, sigmoid, ReLU) and how they affect learning is deferred to later weeks.
