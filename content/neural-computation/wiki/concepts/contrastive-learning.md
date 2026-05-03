---
name: contrastive-learning
description: Self-supervised representation learning by "same vs different" — augmentations of the same image should land close in feature space; different images should land far apart. SimCLR (Chen et al. 2020) is the canonical instantiation; matches fully-supervised ResNet-50 accuracy on ImageNet linear evaluation.
type: concept
sources:
  - raw/week-06/w06-slides.pdf
  - raw/week-06/w6-notion.zip
status: stable
updated: 2026-04-26
---

> [!tip] TIP — Detective vs artist
> An autoencoder is an *artist*: shown a dog, it must redraw the dog pixel-by-pixel. To do well it must memorise everything — including the grass texture and the lighting on the dog's fur, neither of which has anything to do with "dog-ness." Contrastive learning is a *detective*: shown a photo of a suspect and a photo of the same suspect in a hat and sunglasses, all that's required is to confirm "same person." No drawing needed. The detective can ignore hat / sunglasses / lighting / pose because those don't change *identity* — and that's exactly the kind of representation we want.

*Contrastive learning replaces "rebuild this image" with "tell same-content apart from different-content." Two augmented views of one dog should map to nearby vectors; that dog and a penguin should map to distant vectors. The network never has to draw a single pixel — it just has to know which images are "the same thing." That turns out to be enough to learn representations as good as fully-supervised ones, on no labels at all.*

## The core idea

Pick a batch of images. For each image $x$, generate two random augmentations $\tilde{x}_i, \tilde{x}_j$ — crop, flip, colour jitter, blur, etc. They look pixel-different but show the same object. Encode all augmentations through a network; in the resulting feature space:

- **Pull** the two views of the *same* image together (positive pair).
- **Push** them away from views of all *other* images in the batch (negatives).

That's it. No reconstruction, no labels — just a "find-your-twin" game played in feature space.

The network has to learn what's *invariant* under the augmentations: a dog is still a dog whether cropped, recoloured, or blurred. Learning this invariance forces the encoder to extract content (object identity, structure) and ignore appearance (colour, exact crop).

## SimCLR: the canonical recipe

[Chen, Kornblith, Norouzi, Hinton — *A Simple Framework for Contrastive Learning of Visual Representations*, 2020.]

### The pipeline

For a minibatch of $N$ images $\{x_k\}_{k=1}^N$:

1. **Augment.** Sample two augmentation operators $t, t' \sim \mathcal{T}$ per image. Get $\tilde{x}_{2k-1} = t(x_k)$, $\tilde{x}_{2k} = t'(x_k)$. Now you have $2N$ augmented views.
2. **Encode.** Pass each view through a base encoder $f$ (typically ResNet-50): $h_i = f(\tilde{x}_i)$. The $h$ vectors are the [[latent-representation|latent representations]] — what we'll keep at the end.
3. **Project.** Pass each $h$ through a small MLP projection head $g$: $z_i = g(h_i)$. The $z$ vectors are where the loss is computed. Note: although $z$ is also a "latent" vector in the literal sense (hidden, internal), in SimCLR the *kept* representation is $h$, not $z$ — see [[latent-representation]] for why.
4. **Contrast.** Compute pairwise similarities $s_{i,j} = z_i^\top z_j / (\|z_i\| \|z_j\|)$ (cosine similarity). Apply contrastive loss.
5. **Update.** Backprop through $g$ and $f$ jointly.

After training, **throw away $g$** and use $f$ (and the representation $h$) for downstream tasks.

### The loss (NT-Xent)

For a positive pair $(i, j)$ — two augmentations of the same image — within a batch of $2N$ views:

$$\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\mathrm{sim}(z_i, z_k) / \tau)}$$

where $\mathrm{sim}(u,v) = u^\top v / (\|u\| \|v\|)$ and $\tau$ is the **temperature** hyperparameter.

Reading the formula:

- **Numerator (pull).** Similarity between the positive pair. The loss decreases as this grows. The network maximises $\mathrm{sim}(z_i, z_j)$.
- **Denominator (push).** Sum of similarities with *all other* views in the batch ($2N - 2$ negatives). The loss decreases as these *shrink*. The network minimises $\mathrm{sim}(z_i, z_k)$ for $k \neq j$.
- **Temperature $\tau$.** Scales the logits. Low $\tau$ → sharp decisions, hard distinctions; high $\tau$ → softer, more permissive. Typical $\tau = 0.1$–$0.5$.

The total batch loss averages this over all positive pairs.

It's just a softmax cross-entropy classification problem in disguise: "given $z_i$, which of the $2N - 1$ other vectors is its augmentation twin?"

## Why a projection head?

This is the surprising design choice — and why "the representation" can mean two different things.

> [!warning] CAUTION — In SimCLR, $h$ is the keeper, not $z$
> | | Comes from | Used for | After training |
> |---|---|---|---|
> | $h$ | Encoder $f$ | Downstream tasks | **Keep** |
> | $z$ | Projection head $g$ | Computing the contrastive loss | **Discard** |
>
> Empirically, training a linear classifier on $h$ outperforms training one on $z$ — by a lot. The standard explanation: the contrastive loss forces $z$ to be invariant to *exactly* the augmentations applied (rotation, colour, etc.). That's good for the contrastive task but throws away information that's useful for some downstream tasks. The projection head $g$ absorbs that information loss, leaving $h$ to retain richer general-purpose features.

This is a clean separation worth remembering: the proxy task gets its specialised vector; the *kept* representation comes from one layer earlier.

## Augmentation choice matters enormously

The augmentations $\mathcal{T}$ define what the encoder is told to be invariant to. Bad choice → bad representation. SimCLR's recipe (cropping + colour distortion is the killer combo) was found empirically.

Why both? Cropping alone leaves a positional / colour-statistics shortcut: two crops of the same image have similar colour histograms, so the network just matches colour statistics and ignores content. Adding heavy colour distortion breaks that shortcut, forcing actual content recognition. This is a Clever-Hans-style failure mode caught and patched ([[clever-hans-effect]]).

Standard SimCLR augmentations: random crop+resize, random horizontal flip, random colour jitter, random colour drop (greyscale), Gaussian blur. Rotation works for some datasets, hurts others (e.g. natural-orientation matters for digits).

## Negatives matter — bigger batch is better

The loss's denominator is summed over *all* other views in the batch. More views = more negatives = harder discrimination task = better representations. This is why SimCLR uses huge batch sizes (4096+ on TPUs). Methods like MoCo address this with a memory bank of cached negatives; methods like BYOL claim to remove the need for negatives entirely (the *why* is debated).

## Results

The headline: SimCLR with a linear classifier on top of $h$ matches the top-1 accuracy of fully-supervised ResNet-50 on ImageNet.

| Method | Architecture | Top-1 |
|---|---|---|
| Supervised ResNet-50 | ResNet-50 | ~76% |
| SimCLR | ResNet-50 | 69.3% |
| SimCLR (2×) | ResNet-50 (2×) | 74.2% |
| SimCLR (4×) | ResNet-50 (4×) | 76.5% |

With only 1% of ImageNet labels, SimCLR (4×) reaches 85.8% top-5 — beating heavily-engineered semi-supervised baselines that *do* use the label distribution structure. In other words: with a strong self-supervised pretrained encoder, you barely need labels at all.

## Application: unsupervised dataset visualization (t-SimCNE)

A nice byproduct of strong contrastive representations: if you constrain the projection space $\mathcal{Z}$ to be 2-dimensional and replace cosine similarity with a Cauchy kernel $1 / (1 + \|z_i - z_j\|^2)$, the contrastive objective becomes a *visualization* method. **t-SimCNE** (Bohm, Berens, Kobak 2023) does exactly this and produces 2-d embeddings of CIFAR-10 where:

- Each class forms a coherent macro-cluster — all without ever showing a class label.
- *Within* each macro-cluster, sub-structure emerges that captures finer semantics: "bright horses" / "dark horses" / "mounted horses" / "horse heads" all separate; "red cars" / "metallic cars" / "colourful cars" / "duplicate cars" form their own neighbourhoods.

The takeaway is that contrastive learning isn't just for downstream classifiers. The induced latent geometry is rich enough to be a competitor to t-SNE / UMAP for unsupervised data exploration.

## What contrastive learning gets you over autoencoders

| | [[autoencoder]] | Contrastive |
|---|---|---|
| Loss target | Reconstruct pixels | Tell same-vs-different |
| What it preserves | Whatever explains the most pixels (pose, colour, large structure) | Whatever survives augmentation (content, semantic identity) |
| Throws away | Fine details; everything below the bottleneck capacity | Anything an augmentation can change (colour, crop, exact pixel layout) |
| Downstream representation | Bottleneck $z$ | Encoder output $h$ |
| Performance vs supervised | Substantially worse | Comparable |

The shift from "rebuild the picture" to "recognise the same thing" is the whole story. Reconstruction wastes capacity on pixel detail; contrast forces the encoder to focus on what's actually *invariant* about an object.

> [!question]- Why does throwing away the projection head $g$ make the representation *better* than keeping it?
> The projection $z = g(h)$ is optimised purely to win the contrastive game — to maximise similarity to positive pairs and minimise it to negatives. To do that well, $z$ should be *invariant* to anything the augmentations changed (rotation, colour, crop). That invariance is the goal of the loss but it deletes information — features like "what colour is this object" or "what orientation" can't be recovered from $z$. Some downstream tasks need those features. By keeping $h$ (one layer back) we get a representation that is *implicitly* shaped by the contrastive loss but hasn't yet been compressed to the strict augmentation-invariant form. $h$ has more general-purpose information than $z$. Empirically, this gap is large — sometimes 10+ percentage points on linear evaluation.

> [!question]- A friend says: "Contrastive learning needs negatives because otherwise the network would map everything to the same point." Defend this — what's the failure mode they're describing?
> Without negatives, the loss only has a "pull" term: bring $z_i$ and $z_j$ close. The trivial solution is $f(\cdot) = \text{constant}$ — every image maps to the same vector. Now positive pairs are at distance zero (perfect), but the representation is degenerate (every image is "the same"), useless for any downstream task. This is **representation collapse**. The negatives in the contrastive loss are what prevent it: by forcing $z_i$ to be *far* from other images' embeddings, the network can't shrink everything to one point. (Methods like BYOL avoid collapse without explicit negatives via architectural tricks like a momentum encoder + stop-gradient — the mechanism is subtle and still being analysed.)

## Connections

- [[latent-representation]] — clarifies why $h$ (encoder output) is the kept latent representation in SimCLR while $z$ (projection) is discarded — the opposite of the autoencoder convention.
- [[self-supervised-learning]] — contrastive is the strongest current example.
- [[representation-learning]] — the goal contrastive serves.
- [[autoencoder]] — the alternative SSL family; contrastive outperforms reconstruction-based methods on most vision benchmarks.
- [[data-augmentation]] — turns from a regulariser into the *training signal itself*; the augmentation set $\mathcal{T}$ defines what the encoder learns to ignore.
- [[transfer-learning]] — SimCLR-pretrained encoders are now a default starting point for vision transfer, in place of (or alongside) ImageNet-supervised backbones.
- [[clever-hans-effect]] — augmentation choice is the main defence against the network solving the contrastive task via colour-statistics shortcuts instead of content.
