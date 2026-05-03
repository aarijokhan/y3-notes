---
name: representation-learning
description: The broad agenda of learning a useful vector encoding $z$ of raw data $x$, such that downstream tasks become easier. The encoder is the prize; the proxy task used to train it (reconstruction, contrast, pretext) is just scaffolding. Modern deep learning is largely representation learning.
type: concept
sources:
  - raw/week-06/w06-slides.pdf
  - raw/week-06/w6-notion.zip
status: stable
updated: 2026-04-26
---

*Raw data is high-dimensional and dumb. A million pixels do not say "this is a cat"; they say "intensities at a million coordinates." Representation learning is the deliberate effort to translate that mess into a short vector $z$ where each dimension means something — features that downstream algorithms can actually work with. The "right" representation makes hard problems easy: a linear classifier on top of a good representation can match a deep CNN trained from raw pixels with full supervision.*

## What "representation" means here

Given raw input $x \in \mathcal{X}$ (pixels, words, audio), find a function $f_\phi : \mathcal{X} \to \mathcal{Z}$ that produces a vector $z \in \mathcal{Z}$ — the **representation** — with two properties:

1. **Lower-dimensional / structured.** $\mathcal{Z}$ has many fewer dimensions than $\mathcal{X}$, or at least more semantically meaningful axes.
2. **Useful for downstream tasks.** Classification, retrieval, clustering, generation — whatever you want to do with $x$ becomes easier given $z$.

The latent space $\mathcal{Z}$ is sometimes called the **embedding** space, especially in NLP (word embeddings) or retrieval contexts. The per-input vector $z = f_\phi(x)$ is the [[latent-representation]] of $x$ — "latent" because it's an internal vector the network builds, not directly observed in the input or output.

> [!info] ASIDE — Beyond images: word embeddings as representation learning
> The same idea recurs across domains. **Word embeddings** (word2vec, GloVe) take raw word strings and learn a vector representation where semantically similar words end up close (`king` near `queen`, `Paris` near `France`). The training task is different — typically predicting a word from its context — but the recipe is identical: invent a self-supervised proxy task, train an encoder, throw the proxy head away, keep the embedding. The whole modern transformer pipeline (BERT, GPT, sentence embeddings, CLIP) is representation learning at scale.

## Why we need it

Raw inputs are bad inputs:

- **Too many dimensions.** A $1280 \times 720$ image has ~1M numbers. Most ML algorithms choke or overfit on that scale.
- **The wrong axes.** "Pixel intensity at row 73, column 412" is not a feature anyone cares about. We care about "is there a face here," "what's the pose," "what's the colour scheme" — none of which are individual pixels.
- **Sparse meaning.** Real images sit on a tiny manifold inside the vast pixel space. Only $\sim 10^{19}$ of the $\sim 10^{1888}$ possible $28\times 28$ greyscale images correspond to anything meaningful. The representation should describe positions on that manifold, not in the ambient pixel space.

The old way: humans hand-designed features (PCA, SIFT, HOG). The new way: let a neural network *learn* the encoder from data.

## How representations get learned (this module)

Three flavours covered in week 6, each defined by what proxy task forces the encoder to extract structure:

| Method | Training signal | What's kept |
|---|---|---|
| [[autoencoder]] | Reconstruct $x$ through a bottleneck | Bottleneck code $z$ |
| [[pretext-task]] (Doersch context prediction, rotation prediction) | Predict a label fabricated from the data | Mid-layer features (e.g. AlexNet `fc6`) |
| [[contrastive-learning]] (SimCLR) | Augmentations of same image close, others far | Encoder output $h$ (not the projection $z$) |

Plus the supervised special case:

- [[transfer-learning]] — train a CNN on a large *labelled* source dataset (ImageNet); the early conv layers' features transfer to other tasks. Same idea, different proxy task (classification on a related domain instead of self-supervision).

What unites all four: the proxy task is *throwaway*. After training, you discard the part of the network specific to it and keep the encoder. The encoder is the deliverable.

## How to know a representation is good

You can't tell by looking at $z$ — it's just a vector of numbers. The verdict is empirical:

- **Linear evaluation.** Freeze the encoder, train a linear classifier on top with limited labels. If accuracy is high, the representation has separated classes for you. SimCLR's $h$ matches *fully supervised* ResNet-50 accuracy on ImageNet under linear evaluation — a representation good enough that "draw a line through it" suffices.
- **Few-shot / semi-supervised.** Same as above but with very few labels. A good representation works with 1% of the labels.
- **Clustering.** Plot $z$ for unlabelled data; if natural classes form distinct clusters (without ever showing class labels), the representation is capturing the relevant structure. Autoencoders cluster MNIST digits this way ([[autoencoder]]).
- **Nearest-neighbour search.** Pick a query, find images closest in $z$-space. If they're semantically similar (same object, similar pose), the representation is good.

## Disentanglement: an aspirational property

A representation is **disentangled** if individual dimensions of $z$ correspond to independent factors of variation. For an MNIST autoencoder with $z \in \mathbb{R}^2$: $z_1$ = stroke thickness, $z_2$ = digit angle. Sweeping one while holding the other fixed varies *only* that physical attribute.

Disentanglement is desirable because it makes the representation interpretable and controllable (e.g. "make this digit thicker" → adjust $z_1$). It's also rare and unreliable — autoencoders sometimes achieve it, sometimes not, depending on architecture, init, data. There's no architectural trick that guarantees it.

## The Tale of Two Representations

> [!warning] CAUTION — "$z$" means different things in AEs vs. SimCLR
> Don't conflate the bottleneck $z$ of an autoencoder with the projection-head output $z$ of SimCLR. They occupy the same letter but different roles.
>
> | | Autoencoder | SimCLR |
> |---|---|---|
> | Encoder produces | $z$ (bottleneck) | $h$ |
> | Projection head produces | (no head) | $z$ |
> | After training, keep | $z$ | $h$ |
> | Why | $z$ = the compressed representation | $h$ retains general info; $z$ specialised to the contrastive task |

In both cases, the *kept* vector is what we mean by "the representation."

> [!question]- Why is representation learning often described as "the heart of modern deep learning"?
> Because the deep network's final layer (the supervised classifier) is small and easy — what's hard, and what makes deep learning work, is the *learned features* that come out of the layers below. A pre-trained ResNet's fc7 features can be transferred to a totally different task with a tiny new head; that's the whole power of [[transfer-learning]]. Self-supervised methods generalise this — the *representation* is a deliverable, the original task is incidental. SimCLR shows that with no labels at all you can produce features as good as fully-supervised ones. The representation, not the classifier, is the thing the network is really *for*.

## Connections

- [[latent-representation]] — the per-input vector that *is* the representation; clarifies the AE-vs-SimCLR naming clash.
- [[autoencoder]] — simplest representation-learning model; reconstruction as proxy task.
- [[contrastive-learning]] — modern, stronger proxy task; matches supervised performance.
- [[pretext-task]] — older self-supervised approach; useful baseline.
- [[transfer-learning]] — supervised representation learning (uses labels for the proxy task).
- [[self-supervised-learning]] — the umbrella covering AE, pretext, and contrastive.
- [[clever-hans-effect]] — what happens when the representation is good for the *wrong reason*: it solves the proxy via a shortcut, doesn't capture the intended structure.
