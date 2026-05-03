---
name: self-supervised-learning
description: Supervised learning where the labels are fabricated automatically from the data instead of supplied by a human. Encompasses autoencoders (predict input), pretext tasks (predict patch position, rotation), and contrastive learning (predict same-vs-different). The umbrella that unifies week 6.
type: concept
sources:
  - raw/week-06/w06-slides.pdf
  - raw/week-06/w6-notion.zip
status: stable
updated: 2026-04-26
---

*Self-supervised learning (SSL) is supervised learning's clever cousin. Same machinery — input, target, loss, backprop — but the target is computed from the data, not provided by an annotator. Mask part of an image and predict it back. Rotate an image and predict the angle. Augment an image two ways and predict that they're the same. The proxy task is throwaway; the encoder you get out of training it is the real product.*

## The framing

- **Supervised:** dataset $\{(x_i, y_i)\}_{i=1}^N$, with $y$ from a human. Loss $\mathcal{L}(\hat{y}, y)$ trains $f_\theta : \mathcal{X} \to \mathcal{Y}$.
- **Unsupervised (in spirit):** dataset $\{x_i\}_{i=1}^N$, no labels. Goal is to find structure.
- **Self-supervised:** dataset $\{x_i\}$, no human labels — but the *algorithm* fabricates a target $y_{\text{pretext}}(x)$ from $x$ alone, then runs supervised training against it.

Mechanically, SSL is just supervised learning with a label-generating function $y_{\text{pretext}}: \mathcal{X} \to \mathcal{Y}_{\text{pretext}}$ in front of the dataset. Practically, the difference is that you can do it on *unlimited* data — anything you can scrape, you can train on, because you don't need annotators.

## The recipe

1. **Design a pretext task** whose label is computable from the raw data and whose solution requires understanding the content.
2. **Train a network** $\text{enc} \to \text{head}$ to solve the pretext task.
3. **Throw away the head.** Keep the encoder.
4. **Use the encoder for downstream tasks** — fine-tune, train a small head with limited labels, do retrieval, etc.

Step 1 is where the design effort goes. The pretext task must be:

- **Solvable from the data alone** — no human labels required.
- **Hard enough to require understanding content** — not solvable by pixel statistics or low-level shortcuts (or it falls into the [[clever-hans-effect]] trap).
- **Aligned with what you want downstream** — the features that solve the pretext should be useful for what you actually care about.

## The three families covered this week

### Reconstruction

The pretext task is "rebuild the input." The label-generating function is the identity: $y(x) = x$. Realised by the [[autoencoder]] family. Strength: simple, well-understood. Weakness: spends capacity on pixel-level detail that may be irrelevant downstream.

### Pretext tasks (handcrafted proxies)

The pretext task is some fabricated classification or regression problem. Examples:

- **Context prediction** (Doersch et al. 2015) — sample two image patches, classify which of 8 spatial positions the second occupies relative to the first. To answer correctly, the network must recognise the object. See [[pretext-task]].
- **Rotation prediction** (Gidaris et al. 2018) — rotate an image by 0°, 90°, 180°, or 270°; predict which. Forces the network to know the canonical orientation of objects (cars upright, faces upright).
- **Jigsaw / colourisation / inpainting** — variations on "scramble part of the image, predict the missing piece."

Strength: explicitly forces semantic understanding. Weakness: easy to design pretexts that admit shortcuts ([[clever-hans-effect]]).

### Contrastive learning

The pretext task is "tell same-image-pair from different-image-pair." Realised by [[contrastive-learning|SimCLR]] and its descendants. Strength: matches fully supervised performance, scales beautifully. Weakness: needs many negatives per batch, hyperparameter-sensitive (temperature, augmentation choices).

## Why SSL is a big deal

- **Data unlocks.** Labelled data is bottlenecked by annotators; unlabelled data is essentially infinite. SSL lets the model train on the infinite pile.
- **Better representations.** SimCLR's representations match supervised ones on ImageNet linear evaluation. The supervised pre-training advantage is gone.
- **Domain transfer.** A model SSL-pretrained on web-scraped images works well as a starting point for medical imaging, satellite imagery, etc. — domains where labelled data is scarcest.
- **Modern foundation models** (vision and language) are essentially all self-supervised: BERT, GPT, CLIP, MAE, DINO, all variants of "predict-something-about-the-data" trained on massive unlabelled corpora.

## Pre-train + fine-tune: the dominant pattern

Once you have an SSL-pretrained encoder, the downstream workflow mirrors [[transfer-learning]]:

1. **Freeze the encoder, train a small head** with limited labels. Safer; cannot overfit through the encoder.
2. **Or fine-tune both** for a small number of SGD iterations, with a small learning rate. More flexible; risks overfitting.

The encoder is treated exactly like an ImageNet-pretrained backbone — the only difference is that it was pre-trained without labels.

> [!question]- Self-supervised learning is sometimes called "supervised learning in disguise." Why?
> Because the training loop is identical to supervised learning — input, target, loss, gradient — once the pretext target has been generated. The only difference is the source of the label: a human annotator vs. an algorithm that derives the label from the data. SGD doesn't know or care where the target came from. The cleverness of SSL is entirely in *designing the pretext task*; the optimisation machinery is just standard supervised training.

> [!question]- Why doesn't unsupervised learning (in the strict, no-target sense) really exist in modern deep learning?
> Because deep networks are gradient-trained, and gradients require a loss, and a loss requires a target. With no target whatsoever, there is no signal to optimise — the network has nothing to learn. What people *call* "unsupervised learning" — autoencoders, contrastive learning, clustering algorithms with internal objectives — all secretly rely on some target that the algorithm computes for itself. Self-supervised is the honest name for what's going on. The exception is purely structural / non-gradient methods like classical k-means or PCA, which use no neural network — and even those have an implicit objective (reconstruction error, within-cluster variance) that plays the role of a loss.

## Connections

- [[autoencoder]] — reconstruction-based SSL; the entry point.
- [[contrastive-learning]] — discriminative SSL; current state of the art.
- [[pretext-task]] — handcrafted-target SSL; the bridge between AEs and contrastive methods.
- [[representation-learning]] — the goal SSL serves.
- [[transfer-learning]] — closely related: pre-train on a source task, transfer to a target. SSL replaces the source task's labels with self-generated ones.
- [[clever-hans-effect]] — the constant risk in SSL: the network solves the pretext task by a shortcut and learns nothing about the intended content.
