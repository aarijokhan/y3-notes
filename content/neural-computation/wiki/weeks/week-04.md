---
type: week
week: 4
title: "Images and Convolutional Neural Networks"
dates: 2025-10-20 to 2025-10-26
sources:
  - raw/week-04/w04-l10-transcript.txt
  - raw/week-04/w04-l11-transcript.txt
  - raw/week-04/w04-slides.pdf
  - raw/week-04/w4-notion.zip
concepts:
  - "[[image-representation]]"
  - "[[activation-functions]]"
  - "[[convolution]]"
  - "[[pooling]]"
  - "[[convolutional-neural-network]]"
  - "[[shift-invariance-equivariance]]"
status: stable
updated: 2026-04-26
---

> [!question]+ THE CRUX: A fully connected MLP has too many parameters to handle real images. What's the architectural trick that lets a network process images efficiently — and what new properties does it gain in the process?

*Images break MLPs because every pixel becomes its own input, and a fully connected first layer alone needs millions of parameters. The fix is **weight sharing** — instead of a unique weight per (input, output) pair, the same small kernel slides across the image, reusing its weights at every spatial location. That sliding dot product is **convolution**, and a network built from it is a CNN. As a side effect, CNNs naturally handle objects appearing in different positions: convolution is shift-equivariant, pooling adds shift-invariance.*

## Where we left off

Week 3 built the full training pipeline for arbitrary multi-layer networks: stack [[multi-layer-perceptron|perceptrons]] into layers, represent the network as a [[computation-graph]], compute gradients with [[backpropagation]], add [[regularization]] and [[overfitting|early stopping]] to keep the model honest. The architecture was always **fully connected** — every neuron in layer $\ell$ connected to every neuron in layer $\ell - 1$.

That works fine for small inputs. It does *not* work for images.

## Why MLPs fail on images

Images are big. A modest $1000 \times 1000$ greyscale image is a million pixels — so a fully connected first layer with even just a million hidden units would need $10^{12}$ weights. That's a trillion parameters in *one* layer. No memory, no data, no training budget can support that. The same issue scales with colour images (3× more inputs) and modern resolutions.

There's a deeper reason MLPs are wrong for images, beyond just parameter count:

- **No translation structure.** The MLP treats pixel $(5, 5)$ as completely unrelated to pixel $(5, 6)$. There's no built-in notion that nearby pixels are correlated, or that an object is the same object whether it appears in the top-left or bottom-right.
- **No shift handling.** Train an MLP to recognise a "7" in the centre of an image, then move the 7 to a corner. The network sees a totally different input and is likely to fail. The MLP would have to learn *every* spatial location independently.

Both problems point at the same fix: **share weights across spatial positions**.

> [!question]- A $1000 \times 1000$ image is fed to a fully connected MLP whose first hidden layer has $10^6$ neurons. How many weights does just the first layer have, and why is this fatal?
> Each input pixel connects to each hidden neuron, so the count is $10^6 \times 10^6 = 10^{12}$ weights — one trillion parameters in a single layer. This is fatal for three reasons: memory (storing the weights alone is far beyond modern GPUs), compute (each forward/backward pass touches all of them), and statistics (you'd need orders of magnitude more training data to fit them without massive overfitting).

## Weight sharing: the bridge to convolution

Imagine a 1D toy version: a column of $m$ pixels, fully connected to the next layer, $m^2$ weights. Now impose a constraint: *all weights along the diagonal share the same value.* And all weights one above the diagonal share another value. And one below, another, and so on. After fixing the bandwidth (only weights up to two diagonals away from the centre are non-zero), the layer has just **5 unique weights** total — regardless of $m$. Each output neuron uses the same five weights, applied to its local neighbourhood of inputs.

That's the picture in 2D too: each output neuron in the next layer connects only to a small **local** neighbourhood of the input, and *every output neuron uses the same weights*. The collection of shared weights is called a **kernel** (or filter, or window). The operation is called [[wiki/concepts/convolution|convolution]].

Convolution wasn't invented for neural networks — it has been a workhorse of signal and image processing since the mid-20th century. The neural-network innovation was making the kernel **learnable**.

## Image representation

Before going further, it helps to be explicit about what an image actually is. See [[image-representation]] for the details. The summary:

- A **greyscale 8-bit image** is a 2D matrix of integers in $[0, 255]$ — one byte per pixel, $2^8 = 256$ possible intensities.
- A **scientific 16-bit image** uses two bytes per pixel — $2^{16} = 65{,}536$ possible intensities.
- A **32-bit floating point image** stores real-valued intensities, useful when computations push values outside the integer range.
- A **colour (RGB) image** is *three stacked matrices* — one per channel — so a $W \times H$ RGB image is a tensor of shape $W \times H \times 3$. Each channel-pixel is 8 bits, so a 24-bit colour image has $2^{24} \approx 16.8$ million possible colours.

The depth dimension (channels) becomes important once we get to convolution: kernels always extend through the full depth of the input.

## Activation functions revisit

Week 2 introduced [[sigmoid-function-nc|sigmoid function]]; week 3 hinted that sigmoid saturates and that's a problem for deep networks. Week 4 makes this explicit and introduces alternatives. See [[activation-functions]] for the full story. The shortlist:

| Activation | Range | Problem |
|---|---|---|
| **Sigmoid** $\sigma(z) = 1/(1+e^{-z})$ | $(0, 1)$ | Saturates — gradient $\to 0$ at extremes |
| **Tanh** $\tanh(z)$ | $(-1, 1)$ | Centred at 0, but still saturates |
| **ReLU** $\max(0, z)$ | $[0, \infty)$ | Doesn't saturate for positive inputs; can "die" if always negative |
| **Leaky ReLU** $\max(0.01 z, z)$ | $\mathbb{R}$ | Doesn't saturate, doesn't die |

Modern CNNs almost always use **ReLU** in hidden layers — fast, non-saturating, and converges much quicker than sigmoid (≈ 6× faster in some benchmarks).

## The convolution operation

Place a small kernel matrix $\mathbf{w}$ on top of one part of an image $X$. Compute the elementwise product and sum:

$$G[i, j] = \sum_{u=-k}^{k} \sum_{v=-k}^{k} \mathbf{w}[u, v] \cdot X[i + u, j + v]$$

That single number is one output pixel. Slide the kernel across the whole image, repeating the operation, and you get a full output matrix $G = \mathbf{w} * X$. Each output is a **dot product** between the kernel and the local image patch — same machinery as the perceptron's $\mathbf{w} \cdot \mathbf{x} + b$, just over a small window.

> [!info] ASIDE — Mathematician's vs computer scientist's convolution
> Strict mathematical convolution flips the kernel (replace $X[i + u, j + v]$ with $X[i - u, j - v]$). In CNN literature and PyTorch/TensorFlow code, the un-flipped version is used and still called "convolution". The flip doesn't matter for learnability — the network just learns a flipped kernel — so we use the simpler computer-scientist form throughout.

The dot-product reading gives convolution its meaning: **dot product is a similarity measure**. A high value at $(i, j)$ means the local patch of the image looks like the kernel. So convolution scans the image asking "where does this pattern appear?", and the output map records the answers spatially. That's what **feature extraction** means in this context.

In the old days of image processing, kernels were *hand-designed* by humans — Sobel for edges, Gaussian for blur, sharpen, identity. Researchers would spend months tuning kernel values for one specific feature. The CNN innovation is that the network **learns the kernels** end-to-end from data, automatically discovering whatever filters work best for the task.

## Output size, padding, and stride

Convolving without any padding **shrinks** the output. A 3×3 kernel on a 7×7 image gives 5×5 output (lose one row and column at each border, since the kernel can't centre on the boundary pixels).

Two hyperparameters control this:

- **Padding $P$** — pad the input with zeros around the border. With padding $P = 1$ for a 3×3 kernel, the output has the same spatial size as the input ("same convolution"). For a 5×5 kernel, $P = 2$ keeps the size the same. The general rule: $P = (F - 1)/2$ for an odd-sized kernel $F$.
- **Stride $S$** — step size when sliding the kernel. Stride 1 visits every position; stride 2 skips every other position, halving the output spatial size.

The size of the output is captured by the **magic formula**: with input volume $W_1 \times H_1 \times D_1$, applying $K$ kernels of size $F$ with stride $S$ and padding $P$ gives output volume $W_2 \times H_2 \times D_2$ where

$$W_2 = \frac{W_1 - F + 2P}{S} + 1, \qquad H_2 = \frac{H_1 - F + 2P}{S} + 1, \qquad D_2 = K$$

If the formula gives a non-integer, the chosen stride/padding/filter combination doesn't tile the image cleanly — you'd lose information at the borders. Choose $F$, $S$, $P$ so the formula yields integers.

> [!question]- A $32 \times 32$ image is convolved with a $5 \times 5$ filter, stride 1, no padding. What size is the output?
> $(32 - 5 + 0)/1 + 1 = 28$. The output is $28 \times 28$ — two rows lost at the top, two at the bottom, two columns each side.

> [!question]- A $7 \times 7$ image is convolved with a $3 \times 3$ filter and stride 3. Why does this fail strictly speaking, and what would happen?
> $(7 - 3)/3 + 1 = 4/3 + 1 \approx 2.33$, which isn't an integer. With stride 3, the kernel can't tile the image cleanly: positions $(1, 1)$, $(1, 4)$, $(1, 7)$ would be visited along the first row, but the last position only has one column of input to work with on the right and would either need handling (truncate, lose information) or fail.

## Convolution layer: the learnable building block

A **convolution layer** wraps the convolution operation in three additional ideas:

1. **Multiple filters.** One filter detects one type of feature (a vertical edge, say). Real layers use $K$ filters in parallel, each looking for a different pattern. The output is then a stack of $K$ activation maps, depth $D_2 = K$.
2. **Filters span the full input depth.** If the input has $D_1$ channels, every filter has shape $F \times F \times D_1$. The dot product is computed across the spatial window *and* through the depth, so the output of one filter is a 2D map.
3. **Activation function.** After the dot product (and a per-filter bias), a non-linearity like ReLU is applied. The result is called an **activation map** (or feature map).

A CNN is built by stacking such layers, often interspersed with [[pooling]] layers (no learnable parameters; just downsampling). At the end, fully connected layers map the final feature representation to class scores. See [[convolutional-neural-network]] for the full architecture story.

> [!tip] TIP — Layers as a hierarchy of features
> Early conv layers learn **low-level** features: edges, corners, dark spots, simple textures. Middle layers compose those into **mid-level** features: eyes, nose, ears, wheels. Final layers compose into **high-level** features: faces, cars, objects. This hierarchy emerges naturally from training on classification — and matches what neuroscience says happens in the visual cortex. Going from the input to the output, each layer zooms out: smaller spatial resolution but more abstract features.

## Pooling: cheap, non-learnable downsampling

After a few convolution layers, you typically want to reduce spatial resolution. [[pooling]] does this without adding any learnable parameters. The two flavours:

- **Max pooling** — take the maximum value in each $F \times F$ window. The intuition: "if a feature was strong anywhere in this region, keep that signal."
- **Average pooling** — take the mean. Less aggressive; smooths rather than picks.

Default settings are $F = 2, S = 2$, which halves spatial size in both dimensions. Pooling operates **independently on each channel** — the depth dimension is untouched. So pooling a $224 \times 224 \times 64$ tensor gives $112 \times 112 \times 64$.

Pooling has two roles in the CNN architecture:

1. **Compute and memory savings.** Smaller activation maps in later layers means fewer ops, fewer parameters in the eventual fully connected layers.
2. **Shift invariance.** Small translations of the input map to (approximately) the same pooled output, since the max value within a window is unchanged when an active feature shifts a pixel or two.

## A worked architecture: VGG16

VGG16 (Simonyan & Zisserman, 2014) was a state-of-the-art ImageNet model whose design choices became canonical:

- **Small kernels everywhere.** Every convolution is $3 \times 3$ with padding 1 (same convolution).
- **Doubling depth, halving spatial size.** Filter counts: 64 → 128 → 256 → 512 → 512 across the five conv blocks. After each block, a $2 \times 2$ pooling halves the spatial dimensions: $224 \to 112 \to 56 \to 28 \to 14 \to 7$.
- **Three fully connected layers at the end.** $7 \times 7 \times 512$ flattens to a 25{,}088-dim vector, passed through FC layers of sizes 4096, 4096, 1000 (one per ImageNet class).

Total parameters: **138 million**, mostly in the FC layers. The convolutional layers, despite doing most of the work, account for fewer parameters per layer than even one of the FC layers — that's the weight-sharing payoff.

> [!question]- VGG16's first conv layer takes a $224 \times 224 \times 3$ input and applies 64 filters of size $3 \times 3$. How many learnable parameters does this layer have?
> Each filter has $3 \times 3 \times 3 = 27$ weights (the depth must match the input depth) plus 1 bias = 28 parameters. With 64 filters: $64 \times 28 = 1792$ parameters. Compare to a fully connected first layer of the same output size, which would need $224 \cdot 224 \cdot 3 \cdot (224 \cdot 224 \cdot 64) \approx 4.8 \times 10^{11}$ — five orders of magnitude more. That gap is what makes CNNs feasible.

> [!question]- Why does the depth of a convolution filter have to match the depth of the input feature map?
> The filter performs a dot product across the entire local volume — the spatial window *and* the depth. If the input has $D$ channels, each spatial position contains a $D$-dimensional vector, and the filter must contribute one weight per (spatial offset, channel) pair to multiply through. Mismatched depths would leave channels unaccounted for. The filter's spatial size ($F \times F$) is a design choice; its depth is determined by the input.

## What CNNs gain by design: invariance and equivariance

Two distinct symmetry properties fall out of the architecture, both with the same root cause — weight sharing — but acting on different parts of the network. See [[shift-invariance-equivariance]] for the full discussion.

- **Shift equivariance** comes from convolution: shift the input, and the output of a convolutional layer shifts by the same amount. The features detected don't change; only their positions do.
- **Shift invariance** comes from pooling (and, ultimately, from flattening and FC layers): even if a feature appears at a different location, the final classification output stays the same.

Together: convolution layers track *where* features are; pooling and downstream layers eventually disregard *where* and just ask *what*. That's exactly what classification needs (a cat is a cat regardless of where in the frame it sits) and what segmentation does *not* want (a cat on the left and a cat on the right need different pixel-level predictions).

## Beyond image classification

The same CNN backbone can be adapted to other vision tasks:

- **Classification + localisation.** Predict both a class and a bounding box. The class head uses cross-entropy loss; the box head treats the four box coordinates $(x, y, w, h)$ as a regression problem with L2 loss. The total loss is a weighted sum.
- **Object detection** (Faster R-CNN, YOLO). Find every object in the image and label each with a class and box. CNNs serve as the "backbone" for feature extraction; specialised heads do the detection.
- **Semantic segmentation.** Label *every pixel* with a class. Output is a $H \times W$ map of labels. The natural architecture is a **fully convolutional network (FCN)**: replace fully connected layers with convolutions, so the output preserves spatial structure. Often combined with upsampling layers to undo the downsampling that pooling introduces. (More on this in week 5.)

The unifying observation: a CNN extracts a hierarchy of features that's useful for many visual tasks. Swap the head, retrain (or fine-tune), and you have a different model.

## Summary: the week 4 pipeline

1. **Inputs are images.** Multi-channel tensors $H \times W \times D$. Activations are floats.
2. **Architectures share weights spatially.** Each layer applies the same kernel at every position — a sliding dot product = convolution.
3. **Layers stack.** Conv → activation → conv → … → pool → conv → … → flatten → FC → softmax. Depth grows; spatial size shrinks.
4. **Pooling discards spatial precision** to gain shift invariance and computational efficiency.
5. **The output of the conv stack is a learned feature representation** — used either for classification (FC head) or for spatially structured tasks (more conv layers, no FC).

## Concepts introduced this week

- [[image-representation]] — bit depth, channels, RGB; why MLPs hit a parameter wall on images
- [[activation-functions]] — ReLU, leaky ReLU, tanh; non-saturating activations as a fix for the vanishing gradient
- [[convolution]] — sliding dot product over a window; the operation that replaces full connectivity
- [[pooling]] — non-learnable downsampling (max or average) that adds shift invariance
- [[convolutional-neural-network]] — the architecture: stacked conv + pool + FC layers with multiple filters per layer
- [[shift-invariance-equivariance]] — symmetry properties that emerge from weight sharing and pooling

## Connections

- **Builds on** [[week-03]]: the same backprop machinery still trains a CNN — convolution is just another differentiable layer in the [[computation-graph]]. Weight sharing means one weight's gradient is the sum of contributions from every spatial position where it was used (multivariate chain rule, again).
- **Builds on** [[week-02]]: the dot product at the core of convolution is the same dot product that drove the perceptron. ReLU/leaky ReLU replace [[sigmoid-function-nc|sigmoid function]] in hidden layers but the gradient-descent training loop is unchanged.
- **Builds on** [[week-01]]: a single convolution operation = a perceptron applied to a local image patch, with the perceptron's weights shared across all patches.
- **Sets up** week 5: ResNet (deeper than VGG but with fewer parameters via residual connections), upsampling, and segmentation networks. Also: training tricks (data augmentation, batch normalisation) that make deep CNNs trainable.

## Open questions

- The relationship between receptive field size, network depth, and kernel size was hinted at (each layer increases the receptive field) but not formalised. Useful to revisit when comparing architectures.
- How exactly does backpropagation work through a convolution layer? It does — every operation in the network is differentiable — but the bookkeeping (gradients through shared weights, gradients through pooling's max operation) is worth working out by hand once.
- Why $3 \times 3$ everywhere in VGG? Two stacked $3 \times 3$ convs have the same receptive field as one $5 \times 5$, but with fewer parameters and an extra non-linearity in between. The argument generalises and is one reason small kernels dominate modern designs.
