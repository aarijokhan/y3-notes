---
type: week
week: 5
title: "Tricks of the Trade and Dense Prediction"
dates: 2025-10-27 to 2025-11-02
sources:
  - raw/week-05/w05-l12-transcript.txt
  - raw/week-05/w05-l13-transcript.txt
  - raw/week-05/w05-slides.pdf
  - raw/week-05/w05-problem-set.pdf
  - raw/week-05/w5-notion.zip
concepts:
  - "[[weight-initialization]]"
  - "[[learning-rate]]"
  - "[[normalization]]"
  - "[[data-augmentation]]"
  - "[[dropout]]"
  - "[[residual-connection]]"
  - "[[transfer-learning]]"
  - "[[upsampling]]"
  - "[[u-net]]"
status: stable
updated: 2026-04-26
---

> [!question]+ THE CRUX: We have CNNs. So why doesn't simply stacking more layers automatically give better performance — and what's the bag of tricks that lets us actually train deep networks well, plus adapt them to tasks beyond image classification?

*Two desiderata frame the entire week: faster convergence and better generalisation. Most of the "tricks" are aimed at one or both. Some are about getting gradient flow right (weight init, batch norm, residual connections). Some are about not memorising the training data (data augmentation, dropout). One — transfer learning — sidesteps the data shortage entirely. The week ends with **U-Net**, an encoder-decoder architecture for per-pixel prediction that uses several of the same tricks (skip connections, upsampling) at once.*

## Where we left off

Week 4 introduced [[convolutional-neural-network|CNNs]] and showed how weight sharing lets a network process images without an exploding parameter count. We built the canonical pipeline `[CONV → ReLU → POOL] × N → FLATTEN → FC → SOFTMAX`, walked through AlexNet and VGG16 as case studies, and saw how this same backbone supports localisation and (briefly) segmentation through fully convolutional networks.

The architecture works in principle. But getting it to *actually train* — converging quickly, generalising well, and going deeper than ~20 layers without falling apart — needs more than just the architecture itself. That's what this week is about.

## Two goals everything maps to

> [!tip] TIP — The frame to hold in your head
> Every trick this week serves at least one of two ends:
> 1. **Faster convergence** — the network reaches a good solution in fewer epochs.
> 2. **Better generalisation** — the network performs well on data it wasn't trained on.
>
> Some tricks help both. None hurt either. The list of techniques is long but the goals are short — and most of the items below have a clear primary-purpose home on this two-axis map.

| Trick | Primary purpose |
|---|---|
| Weight initialisation | Convergence (gradient flow at $t = 0$) |
| Learning rate schedule | Convergence (stride first, tiptoe later) |
| Batch normalisation | Convergence (gradient flow throughout training) |
| Residual / skip connections | Convergence (enables real depth) |
| Data augmentation | Generalisation (variety against overfitting) |
| Dropout | Generalisation (no neuron is irreplaceable) |
| Transfer learning | Both (less data needed, faster training) |

## Faster convergence — getting the math to behave

### Weight initialisation: don't all start equal

The very first decision when training a network is what to set the weights to before gradient descent begins. The naive choice — all zeros, or all ones — fails catastrophically: every neuron in a hidden layer would compute the same pre-activation, the same activation, and the same gradient, then update identically. They stay interchangeable forever, learning the same feature in parallel. The hidden layer effectively collapses to a single neuron regardless of how wide it is.

The fix is to draw weights from a probability distribution so that no two neurons start equal — break the symmetry from step zero. Modern initialisation schemes (Xavier for tanh/sigmoid, He/Kaiming for ReLU) go one step further: they pick the random distribution's *variance* to keep activation magnitudes stable from layer to layer. Get this wrong and gradients vanish or explode from the first forward pass. See [[weight-initialization]].

### Learning rate schedule: stride first, tiptoe later

Week 2 introduced the [[learning-rate]] $\eta$ as a hyperparameter; week 5 makes it a *time-varying* one. Far from the minimum, large $\eta$ moves you across the loss landscape quickly. Near the minimum, large $\eta$ overshoots and oscillates without settling. A schedule decays $\eta$ over training — step decay (drop by 10× every $n$ epochs), exponential decay, reduce-on-plateau (drop when validation loss flatlines), cosine annealing.

The clearest diagnostic: if your validation loss has flatlined for several epochs, drop the learning rate. You typically see a fresh sharp drop in loss, then another plateau. Repeat. The lecturer described it as *"if you're saturating, drop $\eta$"* — that intuition is the basis of reduce-on-plateau scheduling.

### Normalisation: keep numbers in a sane range

[[activation-functions|ReLU]] solved the worst of the vanishing-gradient problem at the activation level. But the network can still produce activations on wildly varying scales as the data flows through layers — and once activations grow into the thousands, even a non-saturating activation can't save you from numerical instability or saturated downstream sigmoids elsewhere.

The fix is normalisation, applied at two levels:

- **Input normalisation.** Preprocess raw inputs so they enter the network on a sensible scale. For 8-bit images, divide by 255 to get $[0, 1]$, or apply Z-score normalisation $(x - \mu)/\sigma$ to centre at zero with unit variance. Compute $\mu, \sigma$ once on training data and reuse them.
- **Batch normalisation.** Normalise *inside* the network, after every layer's pre-activations. For each mini-batch, compute the per-channel mean and variance; rescale to mean 0, variance 1; then multiply by learnable $\gamma$ and add learnable $\beta$, letting the network choose its own scale. Batch norm dramatically accelerates training and makes the network less sensitive to initialisation.

Both stories rest on the same observation: deep networks are unstable when their internal numbers drift far from zero, and normalisation is the cheap fix. See [[normalization]].

> [!info] ASIDE — Internal covariate shift
> The original motivation for batch norm was a phenomenon called *internal covariate shift*: as earlier layers update during training, their output distribution drifts, and later layers spend their effort chasing this moving target instead of learning. Batch norm restabilises the distribution at every layer after every step, removing the drift. (More recent papers question whether covariate shift is the *actual* mechanism by which batch norm helps, but the practical benefit is undisputed.)

> [!question]- A friend trains a CNN and gets enormous activation magnitudes at the third layer ($\sim 10^4$). The network then either produces NaN losses or gets stuck. Without changing the architecture, what two normalisation interventions would you suggest?
> First, **input normalisation** — make sure the raw inputs are scaled to a small range (divide pixel intensities by 255 or apply Z-score). Large input magnitudes cascade through the network and produce exactly this kind of explosion. Second, add **batch normalisation** layers after each conv (or after the conv + ReLU, depending on the convention). Batch norm forces each layer's pre-activation distribution to mean 0 / variance 1 within each batch, breaking the cascade and keeping numbers stable through depth. Together these address both sources of the problem: large inputs and unchecked drift through layers.

### Residual connections: making depth pay off

ResNet (2016) is the architectural innovation that makes "very deep" actually work. Plain CNNs of 20+ layers run into the **degradation problem** — training error gets *worse* with more layers, not better, despite the deeper network being strictly more expressive in theory. The cause is optimisation difficulty: gradients vanish through long chains of multiplications, activations saturate, the loss landscape gets pathological.

The fix is structural. Take a small block of layers computing $F(x)$. Add a shortcut that copies the input $x$ around the block. Sum the two paths: $y = F(x) + x$. The block's job is now to learn the *residual* — the change to apply to $x$ — rather than the entire transformation. Two consequences:

1. **Identity is the easy default.** Setting $F(x) = 0$ makes the block a passthrough, so unneeded layers can effectively turn themselves off. Adding more blocks can only help.
2. **Gradient highway.** Backprop through $y = F(x) + x$ gives $\partial L / \partial x = \partial L / \partial y \cdot (1 + \partial F / \partial x)$. The "+1" survives even when the inner gradient is tiny — gradients flow uninterrupted through the identity path back to early layers.

ResNet-34 with skip connections beats plain VGG19 with ~7× fewer parameters. The trick has since become a building block of essentially every deep architecture, including transformers and U-Net. See [[residual-connection]].

> [!question]- A 56-layer plain CNN performs *worse* on the **training** set than a 20-layer plain CNN. Is this overfitting, and how does adding residual connections fix it?
> Not overfitting — overfitting shows up as a gap between training and test, but here the deeper network is failing to fit even the training data. This is the **degradation problem**: optimisation difficulty in deep networks (vanishing gradients, ill-conditioned loss landscapes). Residual connections fix it two ways. First, identity is now easy to learn ($F(x) = 0$ makes the block a passthrough), so unneeded layers stop hurting. Second, the identity path provides a gradient highway with constant scaling factor 1, so gradients reach early layers undamaged regardless of depth. With these, deep networks can finally fulfil their theoretical promise.

## Better generalisation — fighting overfitting

### Data augmentation: cheap variety

Real datasets are smaller than deep networks need. Acquiring more data is slow and expensive, especially in domains like medical imaging where each image needs an expert to annotate. Data augmentation generates new training examples for free by applying *label-preserving* transformations: flips, rotations, brightness changes, noise, crops. One image becomes ten, fifty.

The mechanism isn't that you actually have more data — those flipped variants come from the same scenes. It's that you've forced the network to learn *features that are invariant to those transformations*. A model that recognises a penguin from any orientation, in any lighting, with any noise level, has necessarily learned something more abstract than the literal pixel values of the original photo. That abstraction is what generalises. See [[data-augmentation]].

### Dropout: the rough-love regulariser

The most counter-intuitive trick of the week. During training, randomly disable each neuron (set its output to zero) with probability $p$, independently per training step. Different neurons get dropped each step, so every iteration trains a slightly different "thinned" sub-network.

Two reads of why this helps:

- **Ensemble interpretation.** Each step trains one of $2^N$ thinned sub-networks, all sharing weights. Training is effectively training a giant ensemble simultaneously. At inference, the full network is used — implicitly averaging over all the sub-networks. Ensembles generalise.
- **Anti-co-adaptation.** Without dropout, neurons can become co-dependent (neuron A only works because neuron B is always firing). Dropout makes that pair-trick fail randomly, so each neuron is forced to be useful on its own.

At inference, dropout is switched off — all neurons are used, but their outputs are scaled to match training-time expected magnitudes (this is "inverted dropout"; PyTorch handles it via `model.train()` vs `model.eval()`). See [[dropout]].

> [!question]- "Dropout deliberately breaks the network during training". Why does that improve test-time performance, given that we never use dropout at test time?
> Dropout enforces redundancy and prevents over-reliance on specific neurons, both of which produce a more robust internal representation. The network can't memorise specific pixel-to-prediction pathways because any one of those pathways might be missing on the next training step. To minimise the loss across all possible thinned sub-networks, the weights have to encode features that generalise — a kind of implicit ensembling. At test time, with all neurons active, you get the benefit of every learned feature without the noise. The training-time chaos pays off as a more principled set of learned weights, even though you never use the chaotic sub-networks in production.

### Transfer learning: stand on the shoulders of giants

Modern CNNs need millions of labelled examples to train from scratch — a luxury most real problems lack. Transfer learning sidesteps this by initialising the network from weights *already trained* on a large general dataset (typically ImageNet's 1.2M images). The early conv layers — which detect edges, textures, simple shapes — are useful for almost any visual task. Only the final task-specific classifier needs to be retrained for your problem.

In practice: take a pre-trained CNN, strip off its final FC layer, attach a new one sized for your task (e.g. benign-vs-malignant skin lesion, 2 outputs instead of 1000), then either freeze the convolution layers and train just the new head (feature extraction), or unfreeze some/all of them with a small learning rate and let them adapt (fine-tuning). Either way you start from an already-good feature extractor and only have to learn the task-specific composition. Convergence is fast and effective even with a few hundred target examples. See [[transfer-learning]].

## Dense prediction — beyond classification

The last topic of the week shifts away from "tricks" to a new architecture, **U-Net**, that uses several of those tricks at once and makes per-pixel prediction practical.

### The problem with FCNs

Last week introduced fully convolutional networks (FCNs) for semantic segmentation: replace the FC head with conv layers, and the output preserves spatial structure. But naive FCNs without pooling are *expensive* — keeping full resolution through every layer is computationally prohibitive. And FCNs that *do* pool (most of them) lose fine spatial detail by the time they reach the segmentation head.

You can't get both context (from pooling) and precision (from full resolution) with a single forward pipeline. You need them at the same time.

### U-Net: encoder + decoder + skip connections

U-Net (Ronneberger et al., 2015) resolves the tension with a U-shaped architecture:

- **Encoder (contracting path).** A standard CNN that pools repeatedly. Spatial size shrinks: $572 \to 280 \to \dots \to 32$. Channel count grows: $64 \to 128 \to \dots \to 1024$. This captures *context* — what's in the image.
- **Bottleneck.** The deepest, most abstract feature representation.
- **Decoder (expanding path).** Mirror image of the encoder, using [[upsampling]] (or transposed convolution) to grow back to full resolution. Spatial size doubles per level; channel count drops back down. This recovers *spatial structure* — where things are in the image.
- **Skip connections.** At each resolution level, the encoder's feature map is **concatenated** to the corresponding decoder feature map (not added — this is unlike ResNet's skip connections). The decoder then has both the deep abstract features from below *and* the high-resolution detail from across. Subsequent convolutions learn how to combine them.

The skip connections are the genius of U-Net. Without them, the decoder has only the bottleneck's blurry abstract output to work from — it knows roughly where the object is, but can't draw precise boundaries. With them, every spatial precision lost in the encoder is restored to the decoder at the same resolution level, ready to be combined with the semantic depth from the bottleneck. The final segmentation is both semantically correct and spatially precise.

See [[u-net]] for the full architecture.

> [!question]- A classmate proposes building a segmentation network as just an encoder followed by upsampling — no skip connections. They argue the upsampling layers will recover the spatial detail. What's wrong with this reasoning?
> The upsampling layers don't *recover* detail; they fill in pixels by interpolation or learnable kernels, but the information about exact edges was already discarded by the encoder's pooling. Once the encoder has reduced a $128 \times 128$ region to a $4 \times 4$ feature map, the precise pixel-level boundaries of objects in that region are *gone* — no upsampling can reconstruct them from $4 \times 4$ summary statistics. Skip connections solve this by piping the high-resolution features directly from the encoder to the decoder, bypassing the bottleneck. The decoder's upsampling provides spatial size; the skip connections provide spatial precision.

### Upsampling: three flavours

The decoder's upsampling step can use:

- **Nearest neighbour.** Copy each pixel into a $2 \times 2$ block. Cheap, no parameters, but blocky output.
- **Bilinear interpolation.** Smooth weighted average of the four nearest neighbours, weights set by distance. Cheap, no parameters, smooth output. The default in modern segmentation networks.
- **Transposed convolution.** Learnable upsampling kernel — the network can learn what pattern to use. More flexible but susceptible to checkerboard artifacts.

Bilinear interpolation is the most common modern choice; it has no learnable parameters but produces smooth output. See [[upsampling]] for the formula and a worked example.

## What the tricks add up to

This week is best understood not as a list of techniques but as an answer to a single question: *what does it take to make a deep network actually work in practice?* The answer is layered:

1. **Get the math right at $t = 0$** — sensible weight initialisation.
2. **Keep the math right throughout training** — normalisation, learning rate scheduling.
3. **Make depth pay off** — residual connections.
4. **Don't memorise** — data augmentation, dropout, weight decay.
5. **Don't reinvent the wheel** — transfer learning when your dataset is small.
6. **Adapt the architecture to the task** — encoder-decoder + skip connections for dense prediction.

Most of these compound. A modern training pipeline uses several at once: a pre-trained ResNet backbone (transfer learning + residual connections), batch norm everywhere, augmentation, dropout in the FC head, a learning rate schedule, all together. None of this is exotic; all of it is now standard.

## Concepts introduced this week

- [[weight-initialization]] — why not zero, why scale matters, Xavier vs He
- [[normalization]] — input normalisation, batch norm, layer/instance/group norm variants
- [[data-augmentation]] — synthesising variety, label-preserving transformations
- [[dropout]] — random masking as ensemble + anti-co-adaptation
- [[residual-connection]] — $F(x) + x$, the degradation problem, gradient highway
- [[transfer-learning]] — pre-trained backbones, freezing vs fine-tuning
- [[upsampling]] — nearest neighbour, bilinear, transposed conv
- [[u-net]] — encoder-decoder with skip connections for segmentation
- [[learning-rate]] (extended) — schedules: step decay, exponential, reduce-on-plateau, cosine

## Connections

- **Builds on** [[week-04]]: every CNN-specific trick (batch norm at conv layers, residual connections in residual blocks, U-Net's encoder being a CNN) operates on the architecture we built last week. The "tricks" are *additions* to the CNN pipeline, not replacements.
- **Builds on** [[week-03]]: backprop and gradient descent still drive everything. Each trick is just a new operation in the [[computation-graph]], differentiable like everything else.
- **Builds on** [[week-02]]: the activation functions ([[activation-functions|ReLU and friends]]) we picked then are what makes scale-aware initialisation matter. Choose ReLU → use He init.
- **Sets up week 6:** autoencoders use the same encoder-decoder pattern as U-Net but for representation learning rather than segmentation. The encoder-decoder framing recurs throughout the rest of the module.

## Open questions

- The exact relationship between batch normalisation, weight decay, and residual connections — they all stabilise training but for nominally different reasons. Modern theory papers still debate whether one is "really" doing the work the others get credit for.
- Why does dropout help less in CNNs than in FC layers? The standard answer is "weight sharing already regularises", but the precise mechanism for why batch norm + light dropout > heavy dropout in CNNs is still empirical.
- We covered transfer learning briefly; the practical question of *which* pre-trained model to choose, and *how many* layers to fine-tune, is project-specific and worth revisiting if you do the medical-imaging Python practical.
