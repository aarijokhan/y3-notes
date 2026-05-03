---
name: latent-diffusion-model
description: A diffusion model (Rombach et al. 2022 — the architecture behind Stable Diffusion) that runs the forward and reverse diffusion processes in the small latent space of a pretrained autoencoder, rather than directly in high-dimensional pixel space. Drastically faster sampling and training, with a clean place to inject text or image conditioning via cross-attention.
type: concept
sources:
  - raw/week-08/w08-l18-transcript.txt
  - raw/week-08/w08-slides.pdf
status: stable
updated: 2026-04-27
---

*Pixel-space [[diffusion-model|DDPM]] is slow because it has to denoise millions of pixels for $T \approx 1000$ steps. Latent diffusion fixes this with one architectural insight: most pixels are redundant — train an autoencoder to compress images into a small latent grid (say $64 \times 64$ instead of $512 \times 512$), then run the entire diffusion process in that latent space, and decode once at the end. Same training objective, same U-Net machinery, but everything happens on a representation that's $\sim 50\times$ smaller. The result is Stable Diffusion: high-resolution text-to-image generation runnable on consumer GPUs.*

## The architectural shift

A standard [[diffusion-model|DDPM]] operates on pixel-space images $x_t \in \mathbb{R}^{H \times W \times C}$. For high-resolution generation ($512 \times 512 \times 3 = 786{,}432$ values), 1000 U-Net forward passes per sample are computationally extreme.

A **latent diffusion model (LDM)** inserts a frozen autoencoder around the diffusion process:

1. **Encoder** $\mathcal{E}$ — compresses an image $x_0$ into a small latent representation $z_0 = \mathcal{E}(x_0) \in \mathbb{R}^{h \times w \times c}$, where $h \times w \ll H \times W$ (typical compression factor: 4×, 8×, or 16× spatially).
2. **Diffusion in latent space** — the entire DDPM forward and reverse processes operate on $z_t$, not $x_t$. The U-Net predicts noise on the latent grid.
3. **Decoder** $\mathcal{D}$ — at the end of sampling, decode the final latent $z_0$ back to pixel space: $x_0 = \mathcal{D}(z_0)$.

The autoencoder is **trained separately** (not jointly with the diffusion model), then frozen. Once $\mathcal{E}, \mathcal{D}$ are trained, the diffusion model only ever sees latents.

## The loss function

Vanilla DDPM loss (operating in pixel space):

$$L_{\text{DM}} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,I), t} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|_2^2 \right]$$

Latent diffusion loss (operating in latent space):

$$L_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,I), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t)\|_2^2 \right]$$

The only change: $x_t$ is replaced by $z_t$ in the U-Net's input. Everything else — noise schedule, sampling algorithm, training procedure — is identical to vanilla DDPM. The expectation now ranges over training latents $\mathcal{E}(x)$ rather than training images $x$.

## Why it works

Two facts about images make latent diffusion practical:

1. **Pixel-space images are redundant.** Most natural images can be compressed by a learned autoencoder by 4–16× spatially with negligible perceptual loss — neighbouring pixels are highly correlated, fine textures can be regenerated from coarse summaries, etc. So you can lose a lot of pixel-level information without losing what makes the image *look* like an image.
2. **Diffusion's hard work is in the structure, not the pixels.** The semantic content of an image (faces, objects, scene layout) lives at coarse spatial scales. Diffusion at this scale is enough to capture the structure; the autoencoder's decoder fills in the fine pixel detail (texture) that doesn't need to be diffused over.

In practice: a 512×512 image autoencodes to a 64×64 latent (8× spatial compression, 64× fewer "pixels"). The U-Net acts on the 64×64 grid — far cheaper per step. Training is cheaper too because you can fit larger batches.

## Conditional latent diffusion: text-to-image

The architecture is built for conditioning. The U-Net is augmented with **cross-attention** layers that consume an external embedding $\tau_\theta(y)$:

$$L_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0,I), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|_2^2 \right]$$

Concretely:

- $y$ is the conditioning input — a text prompt, a semantic map, a class label, a low-res image, a layout, etc.
- $\tau_\theta$ is a domain-specific encoder for $y$ — for text, a transformer-based text encoder (often a frozen CLIP encoder); for images, a CNN; for class labels, an embedding lookup.
- The U-Net's cross-attention layers consume $\tau_\theta(y)$ as keys and values, with the latent feature map as queries — letting every spatial location in $z_t$ attend to the conditioning information.

This is the architectural template behind **Stable Diffusion** — Stability AI's open-source release that made high-quality text-to-image generation broadly accessible. Closed-source systems built on the same principle: DALL-E 2, Imagen, Midjourney (varied details, same core idea).

## The full inference pipeline

To generate an image from a text prompt:

```
Encode condition:    e = τ_θ("Paris in milky way")        # CLIP text encoder
Sample noise:        z_T ~ N(0, I)                          # in latent space (small!)
for t = T, ..., 1:                                          # reverse diffusion in latent space
    z_{t-1} = denoise(z_t, t, e)                            # U-Net + cross-attention with e
Decode latent:       x_0 = D(z_0)                           # back to pixel space
return x_0
```

The slow loop is the diffusion sampling, but it's now in the *small* latent space, so each step is cheap. The decoder runs once, at the very end. Compare with pixel-space diffusion where every one of the $T$ U-Net calls would be on the full-resolution image.

## What you give up

The compression isn't free — the encoder lossily compresses, so:

- The autoencoder needs to be **good** for the LDM to be good. Stable Diffusion uses a [[generative-adversarial-network|GAN]]-trained perceptual autoencoder so reconstructions look right perceptually even if not pixel-perfect.
- Some fine pixel-level effects can't be perfectly captured (very fine text, intricate patterns, etc.) — they live below the autoencoder's resolution.
- The latent space is *learned*, not interpretable. You don't get clean disentanglement.

In exchange you get: 5–50× faster training, 5–50× faster sampling, and the ability to train on consumer hardware. Worth it.

## Why this is *the* architecture for modern text-to-image

Three reasons LDM dominates production text-to-image:

1. **Compute scales.** You can train at high resolution because most compute happens in latent space. Pixel-space diffusion at 512×512+ is prohibitively expensive.
2. **Conditioning is clean.** Cross-attention with frozen text encoders gives strong text-following without retraining the text encoder. Swap CLIP for a better encoder, swap the U-Net's cross-attention scheme — modular.
3. **Released open.** Stability AI's open release of Stable Diffusion (weights + code, Apache-licensed) seeded a massive ecosystem (Diffusers library, ControlNet, LoRAs, fine-tunes). Closed alternatives (DALL-E, Midjourney, Imagen) used the same principle without sharing weights.

> [!info] ASIDE — "Latent" here means three different things
> Don't confuse the three uses of "latent" floating around this module:
> - **Autoencoder latent** ([[latent-representation]]): encoder output $z = f_\phi(x)$, kept as a learned compressed representation of a specific input. Used for clustering / classification.
> - **GAN/VAE latent** ([[generative-adversarial-network]] $z$): a random sample from a prior $\mathcal{N}(0, I)$, fed into a generator. No encoder, no input image — just a fresh noise vector.
> - **Latent diffusion latent** (this page, $z_0, z_t$): a *spatial grid* (e.g. 64×64×4) that is the autoencoder's encoded form of an input image, *and* the space in which the diffusion process operates. So an LDM mixes the autoencoder sense (output of $\mathcal{E}$) with a diffusion-sense (the $z_t$ chain). At inference, $z_T$ is sampled from $\mathcal{N}(0, I)$ — playing the GAN-prior role — and the reverse process produces $z_0$ which the decoder turns into a pixel image.
>
> Same word, three different vectors with three different roles. Treat each in context.

> [!question]- A friend says: "Why not just train a normal pixel-space DDPM with a smaller U-Net for speed?" Why is latent diffusion better?
> A smaller U-Net loses *capacity* without changing the problem — you'd still be predicting noise on $H \times W$ pixels each step, just with fewer parameters to do it with. The result: faster but lower-quality (the network can't model the data well). Latent diffusion changes the *problem* — it shrinks the spatial dimensions the U-Net operates on, so even a large, capable U-Net runs cheaply per step. You get speed *and* capacity. The autoencoder takes the perceptually-irrelevant pixel detail off the diffusion model's hands, freeing it to focus on the structural decisions that actually matter. The right way to think about it: pixel-space diffusion does double duty (modelling structure *and* low-level pixel details); LDM factors those concerns — the autoencoder handles pixels, diffusion handles structure.

> [!question]- The autoencoder in an LDM is trained separately and then frozen during diffusion training. Why not jointly train them?
> Several reasons. **Stability:** the autoencoder converges to a clean reconstruction objective in isolation; jointly training would entangle its loss with the diffusion loss and risk degenerate solutions (e.g. an encoder that maps everything to noise, which the diffusion model then learns to reverse trivially). **Reusability:** a single trained autoencoder can serve many diffusion models (different conditioning, different domains, different fine-tunes) — you only train the expensive autoencoder once. **Modularity:** decoupling means you can swap the autoencoder for a better one (higher fidelity, different latent dimensionality) without retraining the diffusion U-Net. The Stable Diffusion ecosystem heavily exploits this — countless fine-tunes share the same autoencoder backbone.

## Connections

- **Built on** [[diffusion-model]] — LDM is "DDPM in latent space." Same forward process, same reverse process, same noise-prediction objective, same U-Net training. The shift is: where do you do the diffusion?
- **Built on** [[autoencoder]] — the encoder/decoder pair is a standard autoencoder, trained separately and then frozen. Decoupling is essential.
- **Built on** [[u-net]] — the noise predictor is a U-Net, but acting on the small latent grid rather than full-resolution pixels. Augmented with self-attention and cross-attention.
- **Built on** [[conditional-generative-model]] — text-to-image is conditional generation $p_\theta(x \mid \text{text})$; LDM provides the architecture and the cross-attention mechanism for injecting the text condition.
- **Latent disambiguation:** see [[latent-representation]]. In LDM, "latent" refers to a *spatial grid* that's the autoencoder's compressed representation; not the per-input vector of an autoencoder, and not the random noise prior of a GAN. Same word, three different meanings — keep them straight.
- **Practical instance** — Stable Diffusion (Stability AI 2022) is the canonical LDM; the architecture is also the basis of Imagen, DALL-E 3, and most modern text-to-image diffusion systems.
