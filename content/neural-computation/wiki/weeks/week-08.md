---
type: week
week: 8
title: "Diffusion Models: From Noise to Image, One Small Step at a Time"
sources:
  - raw/week-08/w08-l16-transcript.txt
  - raw/week-08/w08-l17-transcript.txt
  - raw/week-08/w08-l18-transcript.txt
  - raw/week-08/w08-slides.pdf
  - raw/week-08/w08-problem-set.pdf
  - raw/week-08/w08-problem-set-solution.pdf
concepts:
  - "[[diffusion-model]]"
  - "[[latent-diffusion-model]]"
status: stable
updated: 2026-04-27
---

> [!question]+ THE CRUX: GANs go from noise to image in one shot, and they're hard to train. What if instead we broke the problem into 1000 tiny denoising steps and trained a single network — supervised, no adversary — to walk the chain backwards? Could that produce better images more reliably?

*This week's answer is yes, and it's now the dominant paradigm for image generation. **Diffusion models** turn generation into iterated denoising: a fixed forward process gradually destroys data into Gaussian noise; a U-Net is trained to predict, given a noisy image and a timestep, the noise that was added; sampling reverses the process step by step. Training is just supervised regression — boringly stable. Sampling is slow (1000 sequential network calls) but produces near-photorealistic images at high resolution. Stable Diffusion — the architecture behind every text-to-image system in production — is a diffusion model running in the latent space of an autoencoder, with text injected via cross-attention.*

## Where we left off

Week 7 was about [[generative-adversarial-network|GANs]] — the prior dominant approach to image generation. We saw the genius (adversarial training implicitly minimises Jensen-Shannon divergence between real and generated distributions) and the problems (training instability, mode collapse, hyperparameter sensitivity). We also covered the [[bayes-theorem|Bayesian]] backbone of conditional generation: a [[conditional-generative-model|conditional generative model]] learns a posterior $p(y \mid x)$ and samples from it, beating regression's blurry conditional-mean output.

This week introduces **diffusion models**, which since 2020 have largely displaced GANs for state-of-the-art image generation. The same goals (sample from $p_{data}$), the same conditional extensions (text-to-image, super-resolution, etc.), but a fundamentally different approach to the underlying optimisation problem.

## The framing — same problem, new strategy

Like all [[generative-model|generative models]], a diffusion model fits $p_\theta(x) \approx p_{data}(x)$. The shift is in *how*:

| | GAN | Diffusion |
|---|---|---|
| Generation steps | One shot ($z \to x$) | Many steps ($x_T \to x_{T-1} \to \cdots \to x_0$) |
| Training signal | Adversarial (discriminator) | Supervised (predict the noise) |
| Sample quality | High but mode-prone | Very high, diverse |
| Sampling speed | Fast | Slow ($T \approx 1000$ network calls) |

The diffusion model breaks the hard problem ("turn this Gaussian noise into a face") into 1000 easy ones ("remove a tiny bit of noise"). Each step is small enough to be solvable by a single supervised regression; their composition produces a sample. The trade is speed for stability and quality.

> [!info] ASIDE — The "ink in water" metaphor
> Drop a bead of ink into still water. It spreads gradually until uniformly distributed — entropy increases, structure is lost. Watching that backwards (a uniform purple solution spontaneously forming a single ink drop) is thermodynamically forbidden. But a diffusion model, in pixel space, *learns* to do exactly that: take pure noise (uniform high-entropy mush) and run it backwards into a structured image.

## DDPM: forward and reverse

A **denoising diffusion probabilistic model** ([[diffusion-model]], Ho et al. 2020) has two processes:

1. **Forward diffusion** — fixed, no learned parameters. Take clean data $x_0$, add a small amount of Gaussian noise to get $x_1$, then $x_2$, ..., up to $x_T$ where the noise has destroyed all structure and $x_T \approx \mathcal{N}(0, I)$. Each step:
$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t I\right)$$
where $\beta_t \ll 1$ is the *noise schedule* (chosen by us, e.g. linear or cosine).

2. **Reverse diffusion** — learned. Sample $x_T \sim \mathcal{N}(0, I)$ and denoise step by step:
$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I\right)$$
The mean $\mu_\theta$ is predicted by a neural network. After $T$ such steps, $x_0$ is the generated image.

> [!tip] TIP — Why $\sqrt{1 - \beta_t}$ and not just $1$?
> The scaling keeps each step a *gentle mix* of "previous image" and "fresh noise" rather than just piling on. Without it, the variance of $x_t$ would explode. With it, the variance stays bounded and the chain terminates at a known distribution: $\mathcal{N}(0, I)$. That's the same simple distribution we'll sample from at test time.

### The forward shortcut

The composition of all forward steps from $x_0$ to $x_t$ collapses (because Gaussians compose nicely) into a *single* closed-form Gaussian. Define $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$. Then:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t) I\right)$$

So instead of running the forward process step-by-step (1000 sequential operations) to make a noisy training example, you can jump from $x_0$ to $x_t$ in *one* operation. This is what makes training feasible.

### The reverse process is intractable — so we approximate it

In general, the true reverse $q(x_{t-1} \mid x_t)$ has no closed form — it's intractable. But: if $\beta_t$ is small at each step (which it is, by design), the true reverse is *well-approximated* by a Gaussian. So we let the network output the mean of that Gaussian:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I)$$

The network only has to learn the **mean**. The variance is a function of the schedule, fixed.

## The clever trick — predict the noise, not the mean

A theoretical derivation (see [[diffusion-model]]) shows that predicting $\mu_\theta$ directly is *equivalent* to predicting the noise $\epsilon$ that was added in the forward step from $x_0$ to $x_t$. The two are linked by:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t) \right)$$

where $\alpha_t = 1 - \beta_t$ and $\epsilon_\theta(x_t, t)$ is the U-Net output (an image-shaped tensor representing predicted noise).

Why this matters: predicting noise is a much cleaner regression target than predicting a clean image. The noise has consistent statistics across timesteps (zero-mean unit-variance Gaussian); a clean image's distribution is image-specific. Loss becomes simple MSE between true and predicted noise:

$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

Standard supervised learning. Backprop. Stable training. The "noise prediction" parameterisation is the unsung hero of why DDPMs work in practice.

## The training algorithm (supervised regression in disguise)

```
repeat:
    x_0 ~ q(x_0)                          # sample clean training image
    t   ~ Uniform({1, ..., T})            # pick a random timestep
    ε   ~ N(0, I)                         # sample fresh Gaussian noise
    x_t = √(α̅_t) · x_0 + √(1-α̅_t) · ε    # forward shortcut: jump to time t in one op
    L = || ε - ε_θ(x_t, t) ||²            # MSE between true and predicted noise
    take gradient step on ∇_θ L
until converged
```

Three things to note:

- **It's just supervised learning.** We made the noise ourselves (so we know the target); the network learns to predict it from the noisy image and timestep. Standard regression.
- **No iterative forward roll-out at training time.** The shortcut means each gradient step is one network forward+backward pass.
- **One network, all timesteps.** The same U-Net learns to denoise at every $t$ — random sampling of $t$ each iteration interleaves the regimes. The timestep embedding tells it which noise level to expect.

## The sampling algorithm (the slow part)

```
x_T ~ N(0, I)                          # start from pure noise
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else 0
    x_{t-1} = (1/√α_t) · (x_t - ((1-α_t)/√(1-α̅_t)) · ε_θ(x_t, t)) + σ_t · z
return x_0
```

The bracketed expression is $\mu_\theta(x_t, t)$; adding $\sigma_t z$ samples from the reverse Gaussian. Repeat $T$ times. Each iteration requires a U-Net forward pass — and since each step's output feeds the next, the iterations cannot be parallelised. With $T = 1000$, generating a single image takes 1000 sequential network calls. **This is the central practical drawback** of diffusion models, and the motivation for everything in the "extensions" section below.

## The U-Net that does the work

The architecture predicting $\epsilon_\theta(x_t, t)$ is a [[u-net|U-Net]] — encoder-decoder with skip connections, often augmented with [[residual-connection|ResNet blocks]] and self-attention layers. Critical details:

- **Input:** noisy image $x_t$ + timestep embedding (the scalar $t$ embedded into a vector via a small MLP).
- **Output:** predicted noise $\epsilon_\theta$, *same shape as the input image*. Image-to-image, where the output is interpreted as noise rather than a clean image.
- **Conditioning** (when used): scalar $c$ added to the timestep embedding; image $c$ concatenated channel-wise with $x_t$; text $c$ injected via cross-attention layers (week 9 material).

A common trap (from the week-08 problem set): if the U-Net is too shallow (insufficient pooling/upsampling layers), its receptive field is too small to enforce *globally consistent* attributes. Faces come out with mismatched eye colours, mismatched earrings, mismatched shoes — each pixel is predicted from local context only, and consistency-requiring features drift apart. The fix is more downsampling or self-attention layers that aggregate global information.

## What the model is *really* learning

A diffusion model is fundamentally a **denoiser**. It is *not*, strictly speaking, an image generator. Every training example, every loss evaluation is the same problem: "given this noisy image and what timestep it is, what was the noise?" Generation is a side effect — start from pure noise (which the model is trained to handle, since $x_T$ in the forward process is essentially pure noise), apply denoising 1000 times, and you end up at a clean image consistent with $p_{data}$.

> [!warning] CAUTION — $x_T$ is NOT a latent code
> A common confusion: "isn't the noise at $x_T$ playing the same role as the latent $z$ in a GAN/VAE?" *Sort of* — both are Gaussian samples that the model expands into structured output. But $x_T$ is *not* an encoded representation of any specific $x_0$. The forward process is a Markov chain that destroys all information about $x_0$ over $T$ steps; $x_T$ is pure noise containing no information about the original. A diffusion model has no encoder. There is no compressed latent for an input image. (See [[latent-representation]] for the AE-vs-SimCLR-vs-LDM disambiguation; the noise at $x_T$ adds yet another use of "latent-like vector" — pure prior sample, no encoded content.)

## The trilemma: quality, diversity, speed

| Family | Quality | Diversity | Speed |
|---|---|---|---|
| [[generative-adversarial-network\|GAN]] | High | Low (mode collapse) | Fast (1 forward pass) |
| Diffusion | Very high | High | Slow ($T$ passes) |
| VAE | Moderate | High | Fast |

Diffusion wins quality and diversity, loses speed. Most extensions of DDPM are either (a) accelerating sampling, (b) adding conditioning, or (c) both.

## Extensions

### Conditional diffusion

Same recipe with $c$ added as input to the U-Net:

$$p_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t, c),\, \Sigma_\theta(x_t, t, c))$$

How $c$ enters depends on its type:

- **Scalar** (class label) — embed as a vector, add to the timestep embedding.
- **Image** (low-res, masked, etc.) — concatenate channel-wise with $x_t$.
- **Text** — encode with a transformer text encoder (CLIP, T5), inject via cross-attention layers throughout the U-Net.

The probabilistic framing is the same as for cGANs — see [[conditional-generative-model]] for the posterior-estimation perspective; same logic applies here.

### Cascaded diffusion

Generate at low resolution (e.g. 32×32), then chain of conditional super-resolution diffusion models scales up: 32→64→128→256. Each stage is conditioned on the previous stage's output. Faster than one giant model trained at full resolution, and produces sharper outputs at high resolution. Used as a stepping stone before latent diffusion supplanted it for most uses.

### Super-resolution diffusion (SR3)

Train $p(x \mid y)$ where $x$ is high-res and $y$ is low-res. Concatenate upsampled $y$ to $x_t$ in the U-Net. Beats both bicubic interpolation (no learned content) and MSE-regression super-resolution (blurry — converges to the conditional mean) by a wide margin.

### Image-to-image (Palette)

Same architecture, $y$ is any image-domain condition: greyscale → colour, masked → inpainted, JPEG → clean, panorama uncropping. One model architecture, many tasks. Fundamentally the cGAN pix2pix recipe with diffusion replacing the adversarial training.

### Diffusion for segmentation

Counterintuitively, the U-Net features from a diffusion model (trained for noise prediction, not segmentation) turn out to be excellent representations. Run forward diffusion partway, extract intermediate U-Net feature maps, train a small MLP classifier on top for per-pixel class prediction. A nice cross-over between diffusion and [[representation-learning]].

### Latent diffusion (Stable Diffusion)

The decisive practical breakthrough — see [[latent-diffusion-model]]. Encode each image into a small latent grid via a frozen autoencoder, run the entire diffusion process in that latent space, and decode once at the end. Massive speed-up (everything happens on a 64×64 grid instead of 512×512), plus a clean place to inject text conditioning via cross-attention. The architecture behind virtually every modern text-to-image system (Stable Diffusion, Imagen, DALL-E 3, Midjourney).

> [!question]- A friend's diffusion model produces faces where the eyes don't match. They want to generate at higher resolution. Diagnose and prescribe.
> The receptive-field bug ([[diffusion-model#Common pitfalls]]). A shallow U-Net's bottleneck doesn't see the whole face — each pixel's prediction depends only on local context, so left and right eyes are predicted independently and drift apart. Fix: deeper U-Net (more pooling levels) so the bottleneck features cover the whole face, or add self-attention layers so any pixel can directly attend to any other. Modern diffusion U-Nets always have self-attention for exactly this reason. For higher resolution, switch to [[latent-diffusion-model|latent diffusion]] — the autoencoder takes the resolution off the diffusion model's hands, letting you scale to 1024+ without proportional compute.

> [!question]- Both forward and reverse processes are stochastic, and the U-Net is deterministic. How does that work?
> The U-Net is a deterministic function: same input, same output. Stochasticity in the forward process comes from the explicit noise $\epsilon$ added at each step (or via the closed-form shortcut, where $\epsilon$ is sampled and added to $x_0$). Stochasticity in the reverse process comes from the explicit $\sigma_t z$ term added at each sampling step on top of the U-Net's deterministic mean prediction. So even with the same starting noise $x_T$, two runs of the reverse process produce different images because each step injects fresh noise — the U-Net's contribution is always deterministic, but the *overall* sampling chain is not. This separation of "deterministic mean prediction" from "explicit injected noise" is the same as in any Gaussian sampling — the network learns the mean, the algorithm samples from a Gaussian centred on it.

## Concepts introduced this week

- [[diffusion-model]] — DDPM architecture: forward and reverse diffusion, noise prediction objective, U-Net + timestep embedding, training algorithm (Algorithm 1) and sampling algorithm (Algorithm 2), the trilemma.
- [[latent-diffusion-model]] — Stable Diffusion: run diffusion in the latent space of a frozen autoencoder; massive speed-up; clean cross-attention conditioning for text.

## Connections

- **Builds on** [[u-net]] — the noise predictor is a U-Net (often with [[residual-connection|ResNet blocks]] and self-attention). The encoder-decoder architecture pattern recurs throughout this module; here it solves a regression problem (predict noise) rather than segmentation or reconstruction.
- **Builds on** [[generative-model]] — diffusion is one of the four families (latent variable, diffusion, autoregressive, normalising flows); within this module's narrative, it's the successor to GANs that prioritises stable training and high quality at the cost of sampling speed.
- **Builds on** [[conditional-generative-model]] — the conditional extensions (text-to-image, super-resolution, etc.) follow the same posterior-estimation framing as conditional GANs; see week 7 for the broader picture.
- **Builds on** [[bayes-theorem]] — the reverse process is a chain of Bayesian inference steps; the DDPM derivation involves repeatedly applying Bayes between forward and reverse processes.
- **Builds on** [[autoencoder]] — latent diffusion uses a frozen autoencoder around the diffusion process. The autoencoder's role here is *only* compression (and decoding), not the latent-representation use of week 6.
- **Sets up week 9 (transformers, attention)** — modern diffusion U-Nets all use self-attention and cross-attention, which are formalised in the transformer architecture covered next week. The cross-attention mechanism that consumes text embeddings in conditional latent diffusion is a transformer concept; we'll see it properly developed there.

## Open questions

- **Sampling speed.** $T = 1000$ steps is the bottleneck. Active research areas: DDIM (deterministic, skip-step sampling), DPM-Solver (higher-order ODE solvers in closed form), distilled diffusion (train a fast student to mimic the slow teacher), consistency models (one-step generation). [[latent-diffusion-model|Latent diffusion]] is partly a speed fix.
- **Why noise prediction works so much better than mean prediction.** The DDPM paper showed they're mathematically equivalent under reparameterisation, but empirically noise prediction trains far more reliably. The deeper reason has theoretical conjectures but no consensus.
- **Classifier-free guidance.** A trick (not covered) for boosting conditional fidelity by training the model on both conditional and unconditional samples and combining them at inference. Universal in production text-to-image, the ad-hoc-but-it-works fix for "stronger prompt adherence."

## Problem-set lessons

- **Q1 (true/false on DDPMs):**
  - "Forward process is deterministic" → **False.** Gaussian noise is added at each step; same starting image, different runs → different $x_T$.
  - "Reverse process is deterministic" → **False.** Sampling injects fresh $\sigma_t z$ at each step; same $x_T$, different runs → different $x_0$.
  - "U-Net role is to map noise to clean image, like a GAN generator" → **False.** The U-Net predicts the *noise component* at each reverse step; mapping noise to a clean image is the *whole sampling loop*'s job, not the U-Net's.
  - "U-Net is deterministic" → **True.** Same input, same output. Stochasticity is in the sampling algorithm around it, not in the U-Net itself.
  - "DDPMs trained by backpropagation" → **True.** Standard MSE loss + backprop. No adversary.
  - "Noise at $x_T$ corresponds to a latent code" → **False.** It's pure Gaussian noise — no information about $x_0$. Diffusion has no encoder; $x_T$ is not a representation of anything.
- **Q2 (mismatched eye colours):** Receptive-field too small. A two-level U-Net's bottleneck doesn't see the whole face, so left and right eyes are predicted from independent local features. Fix: deeper U-Net (more downsampling), or add self-attention layers that aggregate global context.
