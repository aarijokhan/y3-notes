---
name: latent-representation
description: "A vector $z$ (or $h$) inside a neural network that encodes the input in a lower-dimensional, structured form — \"latent\" meaning hidden, in the sense of not being directly observable like the input pixels. The exact role of \"the\" latent representation depends on the architecture: in an autoencoder it's the bottleneck code we keep; in SimCLR it's a discardable projection used only for the loss."
type: concept
sources:
  - raw/week-06/w06-slides.pdf
  - raw/week-06/w6-notion.zip
status: stable
updated: 2026-04-26
---

*"Latent" literally means hidden. A latent representation is a vector inside the network that you don't see directly — it's an internal encoding the network builds en route from input to output. The key word is "representation": each latent vector is supposed to **stand for** the input in a more useful form. Pixels become embeddings; embeddings are what algorithms can actually work with. The slippery part is that "the latent representation" picks out different vectors depending on which architecture you're looking at — and confusing which one is "the keeper" is one of the most common errors in week 6.*

## Definitions

- **Latent vector** $z$ (or $h$): the activations of some hidden layer in the network, viewed as a coordinate vector. The dimensionality is usually much smaller than the input dimension.
- **Latent space** $\mathcal{Z}$: the vector space these vectors live in. Often $\mathbb{R}^v$ with $v \ll d$.
- **Latent representation of $x$**: the specific latent vector the network produces when fed $x$. Sometimes called the **embedding** or **code**.

"Latent" because the vector isn't part of the input or the output — it's an intermediate, hidden state of the computation. We can extract it (it's just an activation), but we don't supervise it directly; it emerges from training.

## Properties we want

For the latent representation to be useful, we want $\mathcal{Z}$ to be:

1. **Lower-dimensional** than $\mathcal{X}$ — so it's a compression, forcing the network to throw away the unimportant.
2. **Structured** — semantically similar inputs map to nearby points; semantically different inputs map to distant points (this gives clustering for free, [[autoencoder]]).
3. **Smooth / continuous** — small changes in $z$ correspond to small, meaningful changes in the decoded image (enables [[#latent-walks-probing-the-space|latent walks]]).
4. **Useful downstream** — a small classifier on $z$ should solve tasks the raw input couldn't.

These properties are not guaranteed by the architecture; they emerge from training under the right objective. A well-trained autoencoder produces all four; a poorly-tuned one (e.g. no bottleneck) produces none of them.

## The Tale of Two Representations

> [!warning] CAUTION — "latent representation" picks out different vectors in AE vs SimCLR
> The same words refer to different network locations. Get this wrong and downstream interpretation breaks.
>
> | | Autoencoder | [[contrastive-learning|SimCLR]] |
> |---|---|---|
> | Encoder $f$ produces | $z$ (the bottleneck) | $h$ |
> | Projection head $g$ produces | (no head) | $z$ |
> | Loss is computed on | $\hat{x} = g_\theta(z)$ vs $x$ | $\mathrm{sim}(z_i, z_j)$ across the batch |
> | The "latent representation we keep" | $z$ | $h$ |
> | What you discard after training | (nothing) | $z$ and the projection head $g$ |

### Why the same letter, different meaning?

In an autoencoder, $z$ is the *only* hidden vector that matters — it's the bottleneck, the choke point through which information must pass. We named it "the latent representation" because it's the only candidate.

In SimCLR, there are *two* candidate latent vectors: $h = f(\tilde{x})$ from the encoder, and $z = g(h)$ from the projection head. Both are "hidden" in the literal sense. But $z$ is optimised purely for the contrastive task — it has been compressed in a way that *discards information specific to that task* (e.g. forgetting rotation, colour). $h$ retains richer general-purpose features. We keep $h$, throw away $z$.

So when a SimCLR paper says "we use the representation $h$ for downstream tasks," $h$ is what they mean by "the latent representation." When an autoencoder paper says "the latent code $z$ is used for clustering," $z$ is what they mean. Same word, different vector.

The general rule: **the latent representation is whichever vector you keep after training**, regardless of which Greek letter it carries.

## Latent walks: probing the space

Once trained, you can investigate what each dimension of the latent space corresponds to by sweeping it.

For an autoencoder with a 2-d bottleneck on MNIST 7s:

- Encode a real "7" → get $z_1 = (z_1^{(1)}, z_1^{(2)})$.
- Hold $z_1^{(2)}$ fixed; sweep $z_1^{(1)}$ across a range; decode at each step; look at the sequence of reconstructions.

What you see: the digit physically morphs along one axis of variation (curvature / rotation / thickness / etc.). The network has implicitly assigned an *interpretable physical attribute* to that latent dimension. We did not tell it to; it discovered it.

When an axis of $\mathcal{Z}$ corresponds cleanly to a single human-interpretable factor (and other axes don't also vary that factor), the representation is **disentangled**. Disentanglement is desirable but rare — most trained AEs produce mixed factors, where a single physical attribute is encoded across multiple latent dimensions.

## Why "latent" rather than "feature"?

Subtle distinction worth knowing:

- **Feature** is older terminology, often referring to *named* attributes (edge orientation, colour histogram, SIFT descriptor) — things a human or a hand-designed algorithm decided to extract.
- **Latent variable / representation** is statistical / probabilistic terminology — the variable is *hidden*, not directly observed, and we infer it from data. The name comes from probabilistic modelling (latent factor models, latent Dirichlet allocation, etc.) where latent variables explain observed data via some generative process.

In modern deep learning the terms blur. "Latent" carries a connotation of "the network learned this for itself, we don't know what it means until we probe." "Feature" carries a connotation of "this represents something." For week 6's purposes, treat them as synonyms but lean on "latent" when emphasising that the encoding is unsupervised and discovered.

## What the latent representation is *not*

- **Not the output.** The output of an AE is $\hat{x}$ (a reconstruction); the latent is $z$ (the compressed code). The output of a SimCLR encoder is $h$; the projection $z$ is computed from it but isn't kept.
- **Not the parameters.** The parameters $\phi, \theta$ are the network's weights — they don't change per input. The latent representation is per-input.
- **Not the same across runs.** Different random seeds produce different latent representations of the same input. The dimensions are not "named" — only the *space* has structure, and even that is only partially shared between training runs.

> [!question]- A friend says: "After I train an autoencoder, the latent code $z$ is just a list of numbers — how is it any more meaningful than the raw pixels?"
> Because of *what* those numbers are. The pixels are intensities at fixed coordinates — they say nothing on their own; they only mean something when you look at the whole grid. The latent code is the *encoder's distillation* of the input down to its most informative axes. Each entry of $z$ corresponds to some emergent feature the encoder discovered (often something like "this much skin tone," "this much rotation," "this much stroke thickness"), chosen because it explains the most pixels at reconstruction time. The pixel values don't cluster MNIST digits; the latent code does. The pixel values can't be linearly classified; the latent code often can. The numbers look the same on the page, but one set is unstructured ambient coordinates and the other is a structured summary the network paid for in training capacity.

> [!question]- Why should you reach for the encoder output $h$ rather than the projection $z$ when extracting a representation from a trained SimCLR model?
> Because $z$ is *over-specialised*. The contrastive loss demanded that $z$ be invariant to the augmentations applied (cropping, flipping, colour jitter, blur). To win the contrastive game, $g$ learned to delete that information from $h$ before producing $z$. That deletion is exactly what the contrastive objective wanted — but it throws away features that *some* downstream task might need (e.g. colour for a retrieval task, orientation for a rotation-aware classifier). $h$, sitting one layer earlier, is shaped by the same loss (gradients flowed through it) but hasn't yet been compressed to the strict invariant form. So $h$ retains more general-purpose features. Empirically, linear classifiers on $h$ outperform linear classifiers on $z$ by ~10 percentage points on ImageNet — the projection-head trick is one of SimCLR's key practical contributions.

## Connections

- [[autoencoder]] — the bottleneck $z$ *is* the latent representation; reconstruction loss shapes its meaning.
- [[contrastive-learning]] — the encoder output $h$ is the kept latent representation; the projection $z$ is discarded.
- [[representation-learning]] — the broader goal of learning useful $z$/$h$, of which "latent representation" is the per-input output.
- [[self-supervised-learning]] — provides the training signal that gives the latent space its structure.
- [[u-net]] — the encoder/decoder architecture is similar, but U-Net's bottleneck is *not* used as a "kept" representation — the goal is the segmentation output, not a compressed code.
