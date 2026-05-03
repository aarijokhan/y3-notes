---
name: convolutional-neural-network
description: A neural network whose hidden layers are stacked convolution layers (with pooling), ending in fully connected layers for classification. Designed for image inputs; massively reduces parameter count vs an MLP through weight sharing.
type: concept
sources:
  - raw/week-04/w04-l10-transcript.txt
  - raw/week-04/w04-l11-transcript.txt
  - raw/week-04/w04-slides.pdf
  - raw/week-04/w4-notion.zip
status: stable
updated: 2026-04-26
---

*A CNN replaces the MLP's fully connected hidden layers with stacked convolution layers, plus pooling for downsampling. The result is a network that scales to images by sharing weights across spatial positions, while still learning a hierarchy of features end-to-end.*

## What a CNN looks like

A canonical CNN has three kinds of layer in roughly this order:

1. **Convolution layers** — many filters per layer, often paired with ReLU activation. These do the feature extraction. See [[convolution]].
2. **Pooling layers** — periodic downsampling (typically $2 \times 2$ max pooling, stride 2). See [[pooling]].
3. **Fully connected (FC) layers** — at the end, after flattening the spatial structure. These produce the final class scores, with [[softmax]] on the output for multi-class classification.

A typical pattern: `[CONV → ReLU → CONV → ReLU → POOL] × N → FLATTEN → [FC → ReLU] × M → FC → SOFTMAX`. The conv-pool block repeats $N$ times to extract increasingly abstract features at decreasing spatial resolution; the FC head turns the final feature representation into class scores.

![[cnn-stacked-conv-layers.png]]

At the level of feature volumes, the picture is a chain of decreasing-spatial / increasing-depth blocks: the input ($32 \times 32 \times 3$) is convolved with several $5 \times 5 \times 3$ filters and a ReLU to give a $28 \times 28 \times 6$ activation map; that's convolved with $5 \times 5 \times 6$ filters to give $24 \times 24 \times 10$; and so on. Spatial dimensions shrink (by valid convolution or pooling); the channel dimension grows because each layer applies many filters in parallel.

![[cnn-architecture-101x101.png]]

A complete CNN pipeline (here from a $101 \times 101$ input) makes the alternation visible: convolution layers (Conv1 … Conv4) interleaved with subsampling/pooling layers (Sub1 … Sub3), then a fully connected head (FC5, FC6) producing the final class scores. Spatial resolution drops from $101$ → $91$ → $46$ → … → $6$; channel depth rises from 1 → 80 → 96 → 128 → 160. The annotations show how the kernel size shrinks deeper into the network — large kernels early to cover lots of input pixels at first, smaller kernels later because pooling has already enlarged each neuron's receptive field.

## The convolution layer

A **convolution layer** is the operational unit of a CNN. It wraps the [[convolution]] operation in three additional ideas:

1. **Multiple filters in parallel.** A single filter detects one type of feature. Real layers use $K$ filters, each looking for a different pattern. Their outputs stack along the depth dimension to give a 3D output.
2. **Filter depth matches input depth.** If the input has $D_1$ channels, every filter has shape $F \times F \times D_1$. The dot product is computed across both spatial *and* depth dimensions, so each filter outputs a 2D map.
3. **Activation function applied per output.** After the dot product (and a per-filter bias), an activation like ReLU. The result of the layer — `conv → activation` — is called an **activation map** (or feature map).

For input $W_1 \times H_1 \times D_1$ and $K$ filters of size $F$ with stride $S$ and padding $P$, the output volume is $W_2 \times H_2 \times K$ where the magic formula gives $W_2 = (W_1 - F + 2P)/S + 1$ (and similarly for $H_2$). The depth of the output equals the number of filters, set by the layer designer.

> [!tip] TIP — Two depth rules to keep straight
> The depth dimension causes confusion. Two rules together fix it:
>
> 1. **The filter's depth is *not* a choice — it equals the input's depth.** If the previous layer's output has depth $D_1$, every filter in this layer has shape $F \times F \times D_1$. The depth dimension is consumed by the per-filter dot product.
> 2. **The output's depth *is* a choice — it equals the number of filters $K$.** You decide how many filters to apply; each one produces a 2D activation map; stacking them gives output depth $K$.
>
> So the depth pattern is: $D_1$ (fixed by previous layer) → $K$ (your choice this layer) → becomes the next layer's input depth, fixing *its* filter depth. The number of filters is a hyperparameter at every layer; the filter depth is implied.

### Counting parameters in a conv layer

Each filter has $F \times F \times D_1$ weights (one per element of the local volume) plus 1 bias. With $K$ filters:

$$\text{Parameters per layer} = K \cdot (F \cdot F \cdot D_1 + 1)$$

The crucial observation: **this number does not depend on $W_1$ or $H_1$**. Doubling the image size doesn't double the parameter count of a conv layer — it just produces a larger output volume. That's the weight-sharing payoff.

> [!question]- A convolution layer takes a $56 \times 56 \times 128$ input and applies 256 filters of size $3 \times 3$ with stride 1 and padding 1. How many parameters does it have, and what's the output shape?
> Each filter: $3 \cdot 3 \cdot 128 + 1 = 1{,}153$ parameters. Total: $256 \cdot 1{,}153 = 295{,}168$. Output spatial size: $(56 - 3 + 2)/1 + 1 = 56$ (same convolution preserves size). Output: $56 \times 56 \times 256$.

## Why multiple filters

A single filter can detect a single type of pattern — say, vertical edges. Real images contain many feature types: vertical edges, horizontal edges, diagonal edges, dots, corners, textures. A layer needs **multiple filters in parallel** to extract all of them.

This is the only way the network can build a rich feature representation. With one filter, the next layer sees only "where vertical edges are". With 64 filters, it sees "where vertical edges are, where horizontal edges are, where dots are, ..." — a much richer 64-channel description that subsequent layers can compose into more complex features.

> [!info] ASIDE — The extreme case: $1 \times 1$ convolutions
> A $1 \times 1$ kernel sounds useless — it has no spatial extent, just a single weight per input channel. But its *depth* is still the full input depth $D_1$, so it performs a $D_1$-dimensional dot product per spatial position. In other words: a $1 \times 1$ convolution mixes information **across channels** at each pixel, without combining neighbouring pixels.
>
> ![[1x1-convolution.png]]
>
> Concrete shape arithmetic: a $56 \times 56 \times 64$ input convolved with 32 filters of size $1 \times 1 \times 64$ gives a $56 \times 56 \times 32$ output. Each filter performs a 64-dimensional dot product per pixel. Spatial dimensions are unchanged; the channel count goes from 64 → 32.
>
> This is surprisingly useful. It lets a layer change the channel count cheaply (e.g., reduce 256 channels to 64 before an expensive $3 \times 3$ convolution — the "bottleneck" trick used in ResNet and Inception), and it lets the network learn channel-wise feature combinations without spatial pooling. They look trivial but are a workhorse in modern architectures.

## Stacking layers: the feature hierarchy

Putting many conv layers in a row produces a **hierarchy of features** with two key properties:

- **Increasing receptive field.** Each successive layer's neurons look at a wider patch of the original image. After many layers, a single neuron near the output can see the whole input.
- **Increasing abstraction.** Layer 1 detects low-level features (edges, dots) directly from pixels. Layer 2 composes those into mid-level features (corners, eyes, wheel arcs). Layer 3+ composes those into objects (faces, cars).

> [!tip] TIP — What "receptive field" means
> The **receptive field** of a neuron is the patch of the *original input image* that influences its value. It's an answer to "if I changed a pixel of the input, would this neuron's value change?" — the receptive field is the set of input pixels for which the answer is yes.
>
> - **Layer 1** with $3 \times 3$ kernels: each output neuron sees a $3 \times 3$ patch of the input. Receptive field = 3×3.
> - **Layer 2** with $3 \times 3$ kernels (no pooling): each layer-2 neuron sees a $3 \times 3$ patch of layer-1 activations, which themselves came from $3 \times 3$ patches of the input. Together, a layer-2 neuron sees a $5 \times 5$ patch of the original input. Receptive field = 5×5.
> - **Layer 3:** another step of $3 \times 3$ adds 2 more on each side. Receptive field = 7×7.
>
> Each pure conv layer with kernel size $F$ adds $F - 1$ to the receptive field. **Pooling enlarges it much faster:** a $2 \times 2$ pool with stride 2 doubles the receptive field of every subsequent layer (because each layer-3 neuron now sees a $2 \times 2$ patch of post-pool activations, which corresponds to a $4 \times 4$ patch of pre-pool activations). After several pool layers, deep neurons have receptive fields covering most or all of the input.
>
> **Why it matters:** the receptive field is what the neuron "can see". A neuron with a $3 \times 3$ receptive field can detect edges; a neuron with a $50 \times 50$ receptive field can detect parts of objects; a neuron whose receptive field covers the entire image can detect whole objects. The hierarchy of features is built by progressively enlarging the receptive field through depth and pooling.

This composition isn't programmed — it emerges naturally from training on classification. Visualisations of trained CNNs consistently show:

- **Early layers:** edges, oriented gradients, dark spots, simple textures.
- **Middle layers:** eyes, ears, wheels, parts of objects.
- **Late layers:** faces, recognisable objects, semantic concepts.

This matches the structure of the visual cortex in animals — V1 detects edges, V4 detects parts, IT detects objects. The CNN reproduces this hierarchy spontaneously because both systems are doing the same thing: solving a hard recognition problem by composing simple detectors.

## Why pooling sits between conv blocks

Three reasons to interleave pooling:

1. **Compute and memory.** Halving spatial dimensions cuts activation count by $4 \times$, making deeper networks feasible.
2. **Shift invariance.** Within each pool window, small spatial shifts are absorbed by the max operation. After several pool layers and the FC head, the final classification is largely insensitive to where in the image the object sat. See [[shift-invariance-equivariance]].
3. **Larger receptive fields.** Each pool layer doubles the effective receptive field of subsequent conv layers — a $3 \times 3$ kernel after a pool covers a $6 \times 6$ patch of the pre-pool input, and a $12 \times 12$ patch of the original input after another pool. This lets later layers see the bigger picture.

There's no rigid rule about where pools go. Some networks pool after every conv (early CNNs); some pool after every two or three convs (VGG); some skip pooling and use strided convolutions instead (ResNet).

## The fully connected head

After the conv stack, the spatial feature representation has to become a class score. Two steps:

**Flattening.** The final $W \times H \times D$ feature volume is reshaped into a single vector of length $W \cdot H \cdot D$. This loses spatial structure entirely (treating the whole feature volume as one long input vector to the FC layers).

**Fully connected layers.** One or more standard MLP-style layers, often with ReLU between them, ending in a layer with one neuron per class and softmax to produce probabilities.

The FC layers tend to dominate the parameter count in classical CNNs. In VGG16, the FC layers contain ~80% of the network's 138M parameters even though they're only 3 of 16 weight-bearing layers. This is the trade-off of the FC head: learnable but very parameter-heavy.

> [!info] ASIDE — Replacing FC with global average pooling
> Modern architectures (ResNet, EfficientNet) often use **global average pooling** before the final FC layer instead of flattening: average each channel of the final feature map across all spatial positions, producing a $D$-dim vector. Then a single FC layer projects to the class scores. This drastically cuts parameters and tends to generalise better. The basic CNN we study still uses flatten + FC for clarity; just know the alternative exists.

## Tensor size vs parameter count

Two different quantities people often confuse — and you'll be asked about both in exams. They measure different things.

| Quantity | What it counts | Where it lives |
|---|---|---|
| **Tensor size** | How many *activation values* exist at a layer's output | In memory, recomputed every forward pass |
| **Parameter count** | How many *learnable weights* the layer has | In the model file, fixed across all forward passes |

**Intuitive picture:** tensor size is "how many numbers come out of this layer per image"; parameter count is "how many numbers does the network *remember* about this layer". Tensor size scales with input size (a bigger image produces bigger feature maps); parameter count does *not* (the kernel is the same size whatever image you feed it).

### Counting tensor size

Tensor size = $W \times H \times D$ — width × height × depth of the output volume. Just multiply.

For a conv layer with input $W_1 \times H_1 \times D_1$, applying $K$ filters of size $F$ with stride $S$ and padding $P$:

$$\text{Tensor size} = W_2 \times H_2 \times K, \quad W_2 = \frac{W_1 - F + 2P}{S} + 1$$

(Same formula for $H_2$.) For pool layers, no filter count: depth is preserved, only spatial dimensions shrink.

### Counting parameters

The intuition is "count the numbers inside the kernel-block, multiply by how many kernels":

| Layer type | Parameters | Intuition |
|---|---|---|
| **Conv** | $K \cdot (F \cdot F \cdot D_1 + 1)$ | One filter has $F \times F \times D_1$ weights (shape of the local volume) plus 1 bias; $K$ such filters. |
| **Pool** | $0$ | Pooling has no learnable weights — it's a fixed function. |
| **FC** | $n_{\text{in}} \cdot n_{\text{out}} + n_{\text{out}}$ | Every input connects to every output (weight per pair) plus one bias per output neuron. |
| **Activation (ReLU etc.)** | $0$ | Element-wise function, no weights. |

Notice what's *not* in any of the conv-layer formulas: $W_1$ and $H_1$. **A conv layer's parameter count doesn't depend on the input image's spatial dimensions** — that's the weight-sharing payoff. Only the kernel size $F$, input depth $D_1$, and number of filters $K$ matter.

For FC layers, in contrast, the parameter count *does* depend on input size — every input value connects to every output, so doubling the input doubles the parameters. This is why the FC head dominates parameter counts: by the time the conv stack flattens, the input vector is huge.

### Worked example: a single conv layer

Take a conv layer with input $32 \times 32 \times 3$, applying 16 filters of size $5 \times 5$ with stride 1 and padding 2.

**Tensor size:** $W_2 = (32 - 5 + 4)/1 + 1 = 32$. Output: $32 \times 32 \times 16$ = **16{,}384** activations per image.

**Parameter count:** Each filter has $5 \cdot 5 \cdot 3 + 1 = 76$ parameters. With 16 filters: $16 \cdot 76 =$ **1{,}216** parameters total.

Two very different numbers despite both being "about" this one layer. The activations grow with image size; the parameters don't.

### Worked example: a single FC layer

Take an FC layer mapping a 25{,}088-dim input vector to 4{,}096 output units (this is VGG16's first FC layer, after flattening $7 \times 7 \times 512$).

**Tensor size:** Just the output dimension. Output: $4{,}096$ activations per image.

**Parameter count:** $25{,}088 \cdot 4{,}096 + 4{,}096 = 102{,}764{,}544 + 4{,}096 \approx$ **103M** parameters in this single layer.

Compare: VGG16's *thirteen* conv layers together total ~15M parameters; this one FC layer alone is ~103M.

> [!tip] TIP — Why FC layers explode parameter counts
> Every FC neuron has its own private weight to every input value. There's no sharing. So when the input vector is huge (as it is right after flattening a deep feature volume), the parameter count is essentially "input size × output size" — which scales linearly in both. Conv layers escape this by reusing the same kernel everywhere; FC layers cannot, because they have no spatial structure to share over. This is why modern architectures (ResNet etc.) use **global average pooling** before the FC head — it collapses the spatial dimensions so the FC layer's input is tiny.

> [!question]- A conv layer takes a $112 \times 112 \times 64$ input and applies 128 filters of $3 \times 3$ with stride 1 and padding 1. Compute (a) the output tensor size and (b) the number of parameters.
> (a) $W_2 = (112 - 3 + 2)/1 + 1 = 112$. Output: $112 \times 112 \times 128 = 1{,}605{,}632$ activations. (b) Each filter: $3 \cdot 3 \cdot 64 + 1 = 577$ parameters. Total: $128 \cdot 577 = 73{,}856$ parameters. Note that the tensor is ~22× larger than the parameter count — typical for early conv layers.

> [!question]- An FC layer with $4{,}096$ inputs and $4{,}096$ outputs. Compute the tensor size and the parameter count.
> Tensor size: $4{,}096$ (just the output dim). Parameters: $4{,}096 \cdot 4{,}096 + 4{,}096 = 16{,}781{,}312$ — about 16.8M. The parameter count is roughly $4{,}096 \times$ the tensor size, because every output has its own weight to every input.

> [!question]- Why does a conv layer's parameter count not depend on the input's spatial dimensions $W_1, H_1$?
> Because the kernel is reused at every spatial position (weight sharing). The kernel itself has shape $F \times F \times D_1$ — fixed by the layer's hyperparameters, not by how big the image is. Doubling the image size doubles the *output volume* (more spatial positions to apply the kernel at) but doesn't change *how many distinct weights* exist in the kernel.

## Two case studies

### AlexNet (2012)

The breakthrough that put CNNs on the map. Won ImageNet 2012 by a huge margin (16.4% error vs the previous year's 25.8%), kicking off the modern deep-learning era.

- Input: $227 \times 227 \times 3$.
- 5 convolution layers with 96, 256, 384, 384, 256 filters (kernel sizes 11, 5, 3, 3, 3).
- 3 max-pooling layers (after conv1, conv2, conv5).
- 3 fully connected layers: 4096, 4096, 1000.
- Total: ~60M parameters.

Two things AlexNet did that we've already absorbed: ReLU activations everywhere (instead of sigmoid/tanh) and training on GPUs.

### VGG16 (2014)

Two years later, a much deeper and more uniform design. Won ImageNet 2014 with 7.3% error.

- Input: $224 \times 224 \times 3$.
- 13 convolution layers, all $3 \times 3$ with padding 1.
- 5 max-pooling layers (after every block of 2-3 convs), each halving spatial size.
- Filter counts double per block: 64 → 128 → 256 → 512 → 512.
- Spatial size halves per pool: $224 \to 112 \to 56 \to 28 \to 14 \to 7$.
- 3 fully connected layers: 4096, 4096, 1000.
- Total: 138M parameters (roughly 2× AlexNet despite being deeper).

VGG's lesson: **small kernels stacked deep** is more parameter-efficient than large kernels at any single layer. Two stacked $3 \times 3$ convolutions cover a $5 \times 5$ receptive field with fewer parameters and an extra non-linearity in between.

### Walking through VGG16's layer-by-layer parameter count

| Layer | Output shape | Parameters |
|---|---|---|
| Input | $224 \times 224 \times 3$ | 0 |
| conv1: 64 × 3×3 | $224 \times 224 \times 64$ | $(3 \cdot 3 \cdot 3) \cdot 64 + 64 = 1{,}792$ |
| conv2: 64 × 3×3 | $224 \times 224 \times 64$ | $(3 \cdot 3 \cdot 64) \cdot 64 + 64 = 36{,}928$ |
| pool: 2×2/2 | $112 \times 112 \times 64$ | 0 |
| conv3: 128 × 3×3 | $112 \times 112 \times 128$ | $(3 \cdot 3 \cdot 64) \cdot 128 + 128 = 73{,}856$ |
| conv4: 128 × 3×3 | $112 \times 112 \times 128$ | $(3 \cdot 3 \cdot 128) \cdot 128 + 128 = 147{,}584$ |
| pool: 2×2/2 | $56 \times 56 \times 128$ | 0 |
| ... | ... | ... |
| pool (final) | $7 \times 7 \times 512$ | 0 |
| FC1 | $4096$ | $(7 \cdot 7 \cdot 512) \cdot 4096 + 4096 = 102{,}764{,}544$ |
| FC2 | $4096$ | $4096 \cdot 4096 + 4096 = 16{,}781{,}312$ |
| FC3 | $1000$ | $4096 \cdot 1000 + 1000 = 4{,}097{,}000$ |

The FC1 layer alone has ~103M of the 138M total parameters — by far the biggest chunk. The 13 conv layers together account for only ~15M.

> [!tip] TIP — Where parameters live
> Convolution layers do most of the work but have few parameters per layer (weight sharing). FC layers have many parameters per layer (no sharing). The "where do the parameters live" answer for almost any classical CNN: in the FC head. This is precisely why modern designs reduce or eliminate the FC head — it's the highest-parameter, lowest-leverage part of the architecture.

### ResNet (2016) — when stacking layers stops working

VGG showed that deeper helped, up to a point. Going much beyond ~20 layers, plain CNNs run into the **degradation problem**: training error gets *worse* as depth increases, despite the deeper network being strictly more expressive in theory. The fix is a small structural change — [[residual-connection|residual connections]] that add the input of each layer block back to its output. ResNet-34 (34 layers, ~21M parameters) matches or beats VGG19 (~143M parameters) on ImageNet, and the same architectural primitive scales to 50, 100, even 1000 layers. See [[residual-connection]] for the why.

## Training a CNN

Same as any other neural network: forward pass to compute predictions and loss, backward pass to compute gradients, gradient descent to update weights. See [[backpropagation]] and [[gradient-descent-nc|gradient descent]].

Two CNN-specific considerations:

- **Backprop through convolution.** A weight is used at every spatial position (weight sharing), so its gradient is the *sum* of contributions from every position — exactly the multivariate chain rule from week 3, applied automatically. Implementation-wise, the gradient of a convolution turns out to be another convolution (with a flipped kernel and rearranged inputs), which is why GPU-optimised convolution kernels are central to CNN training.
- **Backprop through pooling.** No parameters, but gradient routing matters. Max pooling routes the gradient only to the position that contained the max; average pooling spreads the gradient equally. See [[pooling]].

## CNNs aren't only for classification

The conv layers extract a feature representation; *what you do with it* depends on the task.

- **Classification:** flatten + FC + softmax (the canonical pipeline).
- **Classification + localisation:** two FC heads on the feature representation — one for class scores (softmax loss), one for box coordinates $(x, y, w, h)$ (L2 loss). Add the losses.

![[classification-localization.png]]

The same conv backbone produces a feature vector; one FC head predicts the class label (cat) using softmax + cross-entropy, another predicts the bounding box $(x, y, w, h)$ as a *regression* problem with L2 loss. Localisation is just regression dressed up — the loss is the squared distance to the ground-truth box.

- **Object detection:** more complex (Faster R-CNN, YOLO), but the CNN backbone is the same. See [[shift-invariance-equivariance]] for why a backbone trained for classification transfers to detection.
- **Semantic segmentation:** replace the FC head with more conv layers, often with upsampling, to produce a per-pixel output. **Fully convolutional networks (FCNs)** are the natural design; the [[u-net|U-Net]] is the canonical encoder-decoder elaboration with skip connections from encoder to decoder.

![[semantic-segmentation-fcn.png]]

In an FCN, there are no FC layers at all — the input is convolved end-to-end into a $C \times H \times W$ score volume (one channel per class), and an argmax across channels gives a per-pixel class prediction. The cross-entropy loss is computed pixel-wise against a ground-truth segmentation mask. The same conv operation that classified whole images now classifies every pixel.

The CNN architecture is a feature extractor — once trained, it's task-agnostic to a meaningful degree. This is why "CNN backbone" is a standard term: the same conv layers, with different heads, solve many vision tasks.

## Related

- [[convolution]] — the operation each layer performs
- [[pooling]] — the downsampling primitive between conv blocks
- [[activation-functions]] — ReLU is the default in CNN hidden layers
- [[image-representation]] — explains why MLPs can't replace this and why CNNs work
- [[shift-invariance-equivariance]] — symmetries that make CNNs effective for image tasks
- [[multi-layer-perceptron]] — the predecessor whose limitation on images motivated CNNs
- [[backpropagation]] — still the training mechanism; the chain rule handles weight sharing automatically
- [[residual-connection]] — the structural fix for training very deep CNNs (ResNet)
- [[u-net]] — encoder-decoder CNN architecture for segmentation
- [[normalization]] — batch normalisation is standard between CNN conv blocks
- [[transfer-learning]] — pre-trained CNN backbones as the standard starting point for new vision tasks

## Active Recall

> [!question]- Sketch the typical layer sequence in a classification CNN, and explain the role of each layer type.
> `[CONV → ReLU → CONV → ReLU → POOL] × N → FLATTEN → [FC → ReLU] × M → FC → SOFTMAX`. **Conv layers** extract features by sliding learned kernels across the input. **ReLU** introduces non-linearity. **Pool layers** downsample and add shift invariance. **Flatten** reshapes the spatial feature volume into a vector. **FC layers** mix all features into class-relevant scores. **Softmax** normalises into a probability distribution.

> [!question]- A CNN's first conv layer takes a $32 \times 32 \times 3$ input. The layer has 16 filters of size $5 \times 5$ with padding 2 and stride 1. What's the output shape, and how many parameters does the layer have?
> Output spatial size: $(32 - 5 + 4)/1 + 1 = 32$ (padding 2 makes it "same"). Output shape: $32 \times 32 \times 16$. Each filter: $5 \times 5 \times 3 + 1 = 76$ parameters. Total: $16 \times 76 = 1{,}216$ parameters.

> [!question]- VGG16 uses only $3 \times 3$ kernels throughout. Why is this design choice better than using larger kernels?
> Two stacked $3 \times 3$ convolutions have the same $5 \times 5$ receptive field as one $5 \times 5$ convolution, but with fewer parameters per filter ($2 \times 9 = 18$ vs $25$ — and after accounting for input depth, the saving compounds). Plus, two stacks introduce *two* non-linearities instead of one, increasing expressiveness. Three stacked $3 \times 3$ convs cover $7 \times 7$ with even bigger savings. Small, deeply stacked kernels are more parameter-efficient and more expressive than large shallow ones.

> [!question]- Why does the convolution layer's parameter count not depend on the input image's spatial size?
> Each filter has shape $F \times F \times D_1$, regardless of $W_1$ or $H_1$. With $K$ filters, the layer has $K \cdot (F \cdot F \cdot D_1 + 1)$ parameters — purely a function of the layer's hyperparameters and the input depth. The spatial dimensions $W_1, H_1$ only affect the *output volume size*, not the parameter count. This is the weight-sharing payoff: the same kernel is applied at every position, so doubling image size doesn't double parameters.

> [!question]- In VGG16, almost all parameters live in just three layers (the FC layers). Why?
> Conv layers share weights across all spatial positions, so they have very few parameters relative to their input size. FC layers have a unique weight per (input, output) pair: the first FC layer alone connects a flattened $7 \times 7 \times 512 = 25{,}088$-dim vector to 4096 units, that's $25{,}088 \cdot 4096 \approx 103$M parameters. Conv layers throughout the network total ~15M; FC layers alone ~123M. Modern architectures often replace flatten + FC with global average pooling to slash this — VGG's design is a snapshot of the older approach.

> [!question]- A friend says "convolutional networks are just MLPs with some weights forced to be equal and others forced to be zero." Is this fair?
> Mostly yes — and it's a useful conceptual framing. Forcing weights to be zero outside a small spatial neighbourhood gives the **local connectivity** of conv layers; forcing weights to be equal across spatial positions gives **weight sharing**. Both constraints reduce parameter count and bake in priors (locality and translation invariance) that suit images. So conv layers are a *constrained* MLP. The constraints aren't arbitrary; they encode the inductive bias that makes CNNs efficient and effective on images.

> [!question]- What does "feature map" or "activation map" mean, and how is it produced?
> A 2D output of one convolution filter (after applying its activation function) — one channel of the layer's output volume. It's produced by sliding the filter over the input volume, computing the dot product (across spatial and depth dimensions) at every position, applying the bias, then applying the activation. A layer with $K$ filters produces $K$ stacked activation maps as its output (depth $K$).
