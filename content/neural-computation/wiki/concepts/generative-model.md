---
name: generative-model
description: A model that learns the underlying probability distribution $p_{data}(x)$ of a dataset, well enough to draw new samples from it. Density estimation in disguise — the goal is "make a network whose output, when sampled, looks like real data." Includes GANs, VAEs, diffusion models, and autoregressive models.
type: concept
sources:
  - raw/week-07/w07-l14-transcript.txt
  - raw/week-07/w07-slides.pdf
status: stable
updated: 2026-04-26
---

*A generative model is a probability distribution wearing a neural network. The data — images, text, audio — lives on some unknown distribution $p_{data}(x)$. We never see $p_{data}$ directly; we only see samples drawn from it (the training set). The goal is to fit a parameterised distribution $p_\theta(x)$ that's close enough to $p_{data}$ that new samples $\tilde{x} \sim p_\theta$ look indistinguishable from real data. "Looks like a face that doesn't exist" is just $p_\theta$ doing its job.*

## The setup: density estimation

We have a dataset $\{x_i\}_{i=1}^N$ — say, MNIST digits or photos of faces. These are samples from some underlying distribution $p_{data}(x)$. We don't know $p_{data}$:

- We can't write it down mathematically (faces aren't a Gaussian).
- We can't compute $p_{data}(x)$ for an arbitrary $x$ (no closed form).
- All we have are samples.

A generative model fits a parameterised distribution $p_\theta(x)$ — typically a neural network — to the data, with the explicit goal:

$$p_\theta(x) \approx p_{data}(x)$$

This is **density estimation** with a neural network in place of the closed-form parametric family that classical statistics would use.

## What we want from $p_\theta$

There are two distinct things you might want from a fitted distribution:

1. **Evaluation** — given an $x$, compute (or at least score) $p_\theta(x)$. Useful for anomaly detection, likelihood-based comparison, or training via maximum likelihood.
2. **Sampling** — draw $\tilde{x} \sim p_\theta$. The new $\tilde{x}$ is a *generated* example: not in the training set, but plausibly from the same distribution.

Different model families prioritise these differently. GANs only do (2); VAEs do both approximately; autoregressive models do both exactly. For most modern applications — generating images, text, audio — sampling is what matters.

## Why this is hard for images

The pixel space $\mathcal{X}$ is enormous: a $28 \times 28$ greyscale image has $256^{784} \approx 10^{1888}$ possible configurations — more than atoms in the observable universe ($10^{80}$). Real images occupy a vanishingly small subset of that space; almost all configurations are noise. The structure of the distribution lives on a thin manifold inside the ambient space.

A second hardness: pixels are *strongly correlated*. You can't decompose

$$p(x) \neq \prod_i p(x^{(i)})$$

because the colour of pixel $(15, 23)$ depends on the colour of $(15, 22)$ in deeply context-dependent ways. The naive factorisation works for pure noise (where pixels *are* independent) but fails for any meaningful image distribution. Capturing the joint distribution faithfully is the central technical challenge.

## Families of generative models

Four broad families dominate modern generative modelling:

| Family | Mechanism | Examples |
|---|---|---|
| **Latent variable** | Sample $z \sim p(z)$ from a simple prior (typically $\mathcal{N}(0, I)$); decode through a network: $x = G_\theta(z)$. | [[generative-adversarial-network|GANs]], VAEs |
| **Diffusion** | Start from pure noise; iteratively denoise toward a sample. | DDPM, Stable Diffusion (week 8) |
| **Autoregressive** | Factorise $p(x) = \prod_i p(x^{(i)} \mid x^{(<i)})$; predict one token / pixel at a time conditioned on the rest. | GPT, PixelCNN |
| **Normalising flows** | Invertible neural networks; transform a simple base distribution into the target distribution exactly. | RealNVP, Glow (not covered in this module) |

This module covers GANs (week 7), diffusion (week 8), and autoregressive models (week 9 onwards). Each has different trade-offs in sample quality, training stability, sampling speed, and likelihood evaluation.

## The latent-variable framing (used by GANs and VAEs)

The simplest recipe for a sampling-only generative model:

1. Pick a **prior distribution** $p(z)$ over a low-dimensional latent space — typically $\mathcal{N}(0, I)$ in $\mathbb{R}^d$. This is fixed by us, not learned.
2. Define a **generator** $G_\theta : \mathcal{Z} \to \mathcal{X}$ — a neural network with parameters $\theta$.
3. To sample: draw $z \sim p(z)$, then output $x = G_\theta(z)$.

The model distribution $p_\theta(x)$ is the *push-forward* of $p(z)$ through $G_\theta$ — the distribution you get by pushing each Gaussian sample through the network. We never write $p_\theta(x)$ in closed form; we just sample from it by sampling $z$ and decoding.

> [!warning] CAUTION — This $z$ is *not* the latent representation of an input
> In an [[autoencoder]] or [[contrastive-learning|SimCLR]], $z = f_\phi(x)$ — the encoder *takes* an input and produces a latent. In a GAN/VAE generator, $z$ is sampled from a *prior* — there is no input. The two uses share the letter "latent" but mean different things. See [[latent-representation]] for the AE-vs-SimCLR clash; the GAN $z$ adds a third meaning: random noise that the generator *expands into* a sample, with no encoding step.

## How do we train $p_\theta$?

You can't compute the loss "$p_\theta - p_{data}$" directly because you don't know either distribution explicitly. Each generative-model family solves this differently:

- **GANs** train an auxiliary network (the discriminator) to *distinguish* samples from the two distributions; the generator learns by trying to fool it. Implicit density estimation. See [[generative-adversarial-network]].
- **VAEs** maximise an evidence lower bound (ELBO) that approximates likelihood of training data under the model.
- **Diffusion models** train a network to predict the noise added at each step of a diffusion process; sampling is reversed denoising.
- **Autoregressive models** maximise exact log-likelihood by training each conditional $p(x^{(i)} \mid x^{(<i)})$ as a classification problem.

The clever part of each family is the *training objective* that lets you optimise $p_\theta \to p_{data}$ without ever evaluating either explicitly.

## Generation vs reconstruction

Worth nailing the distinction with [[autoencoder|autoencoders]]:

| | Autoencoder | Generative model |
|---|---|---|
| Goal | Reconstruct *the same* input | Generate a *new, similar* sample |
| Input at inference | A specific $x$ | Random $z \sim p(z)$ (or a condition) |
| Output relationship to input | $\hat{x} \approx x$ | $\tilde{x}$ is from the same *distribution*, not the same instance |
| Why we care | The latent $z$ is the deliverable | The generated $\tilde{x}$ is the deliverable |

An AE that generates a perfect copy of a training image has succeeded; a generative model that does the same has *memorised* and failed. Generation requires producing samples not in the training set.

> [!question]- A friend says: "If I want to generate new images, I'll just train an autoencoder, then sample random points in latent space and decode them." Why does this approach mostly fail?
> Because the autoencoder's latent space is shaped only by the reconstruction loss — it has no constraint to fill its latent space densely or smoothly. Most points in $\mathcal{Z}$ that you sample randomly will be in *unvisited regions* of the latent space (the encoder mapped real inputs to a thin manifold inside $\mathcal{Z}$, not all of it). Decoding from those holes produces garbage. This is exactly the problem **VAEs** were designed to solve — they explicitly regularise the latent distribution toward $\mathcal{N}(0, I)$, so random samples fall in covered regions. GANs sidestep the problem entirely by *only* sampling from $p(z) = \mathcal{N}(0, I)$ and training the generator to map all of it to plausible outputs; there's no encoder, no reconstruction, just $z \to x$.

> [!question]- Why is naively factorising $p(x) = \prod_i p(x^{(i)})$ fine for pure noise but disastrous for natural images?
> Pure noise *is* per-pixel independent — by definition, each pixel is drawn independently from a uniform or Gaussian distribution. The factorisation is exact. Natural images have strong long-range correlations: knowing one pixel of a face tells you a lot about its neighbours (skin colour continuity), about distant pixels (the eyes are roughly symmetric), and about global structure (faces have a typical layout). Treating pixels as independent throws all that away — you'd generate an image where each pixel is plausible *in isolation* (right brightness, right colour) but the joint configuration is total noise. Capturing the joint distribution is the whole point of generative modelling; the per-pixel marginal is trivially easy and useless on its own.

## Connections

- [[generative-adversarial-network]] — the implicit-density approach; this week's main subject.
- [[conditional-generative-model]] — extends generative models to learn $p_\theta(y \mid x)$ rather than $p_\theta(x)$, allowing controlled generation from a condition.
- [[autoencoder]] — *not* a generative model in the strict sense, but the architectural ancestor of VAEs and the encoder-decoder pattern that recurs throughout. See the table above for the goal-level distinction.
- [[latent-representation]] — clarifies the various uses of $z$ in different architectures; in a GAN, $z$ is sampled from a prior, not produced by an encoder.
- [[bayes-theorem]] — the probabilistic backbone for understanding *conditional* generative models (what does it mean to learn a posterior?).
- [[self-supervised-learning]] — generative modelling is a form of self-supervised learning (no labels needed; the data supervises itself).
