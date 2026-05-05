---
week: 5
topic: "Tricks of the Trade and Dense Prediction"
deck: NeuralComputation::Week-05
---

TARGET DECK
NeuralComputation::Week-05

## Two goals frame everything

> [!question]- The week-5 "tricks" all serve one or both of two goals. What are they?
>
> 1. **Faster convergence** — the network reaches a good solution in fewer epochs.
> 2. **Better generalisation** — the network performs well on unseen data.
>
> Convergence-focused: weight init, LR schedule, batch norm, residual connections. Generalisation-focused: data augmentation, dropout. Both: transfer learning.

## Weight initialisation

> [!question]- Why does initialising **all weights to zero** (or all the same value) fail?
> Every neuron in a hidden layer would compute the same pre-activation, the same activation, and the same gradient — so they would update identically and stay interchangeable forever. The hidden layer effectively collapses to a single neuron regardless of width. We need **symmetry breaking**, achieved by sampling weights from a probability distribution.

> [!question]- What is the difference between **Xavier (Glorot)** and **He (Kaiming)** initialisation?
> Both pick the variance of the random distribution to keep activation magnitudes stable layer-to-layer, but tuned for different activations:
>
> - **Xavier:** designed for tanh / sigmoid — symmetric activations.
> - **He:** designed for ReLU — accounts for ReLU killing roughly half the activations, so it scales weights up to compensate.
>
> Rule of thumb: choose ReLU $\to$ use He init.

## Learning rate schedules

> [!question]- Why does training benefit from **decaying** the learning rate over time?
> Far from the minimum, large $\eta$ moves quickly across the loss landscape — useful early. Near the minimum, large $\eta$ overshoots and oscillates without settling — useful to shrink. A schedule "strides first, tiptoes later".

> [!question]- Name three **learning-rate schedules** and how each decides when to drop $\eta$.
>
> - **Step decay** — drop $\eta$ by a fixed factor (e.g. 10×) every $n$ epochs.
> - **Exponential decay** — multiply $\eta$ by a decay factor every step.
> - **Reduce-on-plateau** — drop $\eta$ when validation loss flatlines for several epochs.
> - (Bonus) **Cosine annealing** — smoothly anneal $\eta$ along a cosine curve.

## Normalisation

> [!question]- What is **input normalisation** for image data, and why is it done?
> Rescale raw inputs to a small, well-behaved range — divide 8-bit pixel intensities by 255 to get $[0, 1]$, or apply Z-score $(x - \mu)/\sigma$ for zero mean / unit variance. Large input magnitudes cascade through the network and produce huge activations; normalisation prevents this from the start. Compute $\mu, \sigma$ once on training data and reuse on val/test.

> [!question]- What does **batch normalisation** do, mathematically?
> For each mini-batch and each channel:
>
> 1. Compute the per-channel mean $\mu_B$ and variance $\sigma_B^2$.
> 2. Normalise: $\hat{x} = (x - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$ — mean 0, variance 1.
> 3. Rescale and shift with learnable parameters: $y = \gamma \hat{x} + \beta$.
>
> The network learns its own preferred scale via $\gamma, \beta$, but the cascade of unstable magnitudes is broken at every layer.

> [!question]- Why does batch norm **accelerate training** and reduce sensitivity to initialisation?
> By forcing each layer's pre-activation distribution back to mean 0 / unit variance after every step, batch norm prevents the distribution from drifting (the original "internal covariate shift" story) and stops activations from blowing up or vanishing as they pass through depth. The network can use larger learning rates without diverging, and small differences in init don't propagate into instability.

## Residual connections

> [!question]- What is the **degradation problem** in plain deep CNNs?
> A plain CNN with more layers performs *worse* on the **training** set than a shallower one — not overfitting, since training error itself rises. The cause is optimisation difficulty: vanishing gradients, saturated activations, and pathological loss landscapes. Theoretically a deeper net should be at least as expressive, but in practice you can't train it.

> [!question]- What is a **residual connection**, and how does it fix the degradation problem?
> Add a shortcut around a block of layers: $y = F(x) + x$, where $F$ is the block's transformation and $x$ is the input copied directly. Two effects:
>
> 1. **Identity is the easy default:** $F(x) = 0$ makes the block a passthrough, so unneeded layers turn themselves off.
> 2. **Gradient highway:** $\partial L / \partial x = \partial L / \partial y \cdot (1 + \partial F / \partial x)$. The "+1" survives even when $\partial F / \partial x$ is tiny, so gradients reach early layers undamaged.

## Data augmentation

> [!question]- What is **data augmentation**, and through what mechanism does it improve generalisation?
> Apply random **label-preserving transformations** (flips, rotations, crops, brightness, noise) to training images, generating new examples for free. The mechanism is *not* "more data" — those variants come from the same scenes — but **invariance**: the network is forced to learn features stable under those transformations, which is more abstract than memorising pixel values.

## Dropout

> [!question]- What is **dropout** during training?
> At each training step, set each neuron's output to zero independently with probability $p$. Different neurons are dropped each step, so each iteration trains a slightly different "thinned" sub-network. At inference, dropout is **switched off** and outputs are scaled to match training-time expected magnitudes (handled by `model.eval()`).

> [!question]- Give the **two interpretations** of why dropout improves generalisation.
>
> 1. **Implicit ensembling.** Each step trains one of $2^N$ thinned sub-networks sharing weights. At inference the full network averages over them — and ensembles generalise.
> 2. **Anti-co-adaptation.** Without dropout, neurons can become dependent on specific others always firing. Dropout makes that pair-trick fail randomly, forcing each neuron to be useful on its own.

## Transfer learning

> [!question]- What is **transfer learning**, and what makes it work?
> Initialise your network from weights already trained on a large general dataset (typically ImageNet) instead of from scratch. The early conv layers learn generic visual features (edges, textures, simple shapes) that are useful for almost any visual task — only the final task-specific classifier needs to be retrained for your problem.

> [!question]- What is the difference between **feature extraction** and **fine-tuning** in transfer learning?
>
> - **Feature extraction:** freeze the pre-trained convolutional layers, replace the final FC layer with a new one sized for your task, train only the new head. Cheap, low risk of overfitting.
> - **Fine-tuning:** also unfreeze some/all of the pre-trained layers and update them with a small learning rate. More flexible, more powerful, but needs more data and care.

## U-Net

> [!question]- What problem does **U-Net** solve that a naive FCN does not?
> Semantic segmentation needs both **context** (large receptive field, from pooling) and **spatial precision** (per-pixel boundaries, which pooling destroys). A naive encoder + upsampler can't recover precise edges from a heavily pooled bottleneck because the information was discarded. U-Net's skip connections pipe high-resolution encoder features directly to the decoder at the matching level, giving precision *and* context simultaneously.

> [!question]- What does the **U-Net architecture** look like?
>
> - **Encoder (contracting path):** standard CNN with repeated pooling — spatial size shrinks, channel count grows. Captures context.
> - **Bottleneck:** the deepest, most abstract feature map.
> - **Decoder (expanding path):** mirror of the encoder using upsampling — spatial size grows back, channels drop. Recovers spatial structure.
> - **Skip connections:** at each resolution level, the encoder's feature map is **concatenated** to the corresponding decoder feature map, so subsequent convolutions can combine high-res detail with deep semantics.

> [!question]- How do U-Net's skip connections differ from ResNet's?
> ResNet **adds**: $y = F(x) + x$ — same channel count required, used to ease gradient flow through depth. U-Net **concatenates**: encoder features are stacked alongside decoder features along the channel axis, doubling channels at the join, and subsequent convs learn how to combine them. Add fixes optimisation; concat fuses different information sources.

## Upsampling

> [!question]- Compare the three **upsampling** methods used in CNN decoders.
>
> | Method | Learnable? | Output quality | Notes |
> |---|---|---|---|
> | **Nearest neighbour** | No | Blocky | Copy each pixel into a $2 \times 2$ block |
> | **Bilinear interpolation** | No | Smooth | Weighted average of 4 nearest neighbours; modern default |
> | **Transposed convolution** | Yes | Flexible but can produce checkerboard artefacts | Learnable upsampling kernel |
