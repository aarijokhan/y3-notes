---
name: diffusion-model
description: A generative model (Ho et al. 2020, "DDPM") that learns to reverse a fixed noising process — the forward process gradually corrupts data into Gaussian noise over $T$ steps; a U-Net is trained to predict, given a noisy sample $x_t$ and timestep $t$, the noise that was added; sampling iteratively denoises a fresh Gaussian sample back to a clean image. Current state-of-the-art for image generation.
type: concept
sources:
  - raw/week-08/w08-l16-transcript.txt
  - raw/week-08/w08-l17-transcript.txt
  - raw/week-08/w08-l18-transcript.txt
  - raw/week-08/w08-slides.pdf
  - raw/week-08/w08-problem-set.pdf
status: stable
updated: 2026-04-26
---

*A diffusion model breaks the hard problem of "go from noise to a face in one step" into the easy problem of "remove a little noise" applied 1000 times. The trick is fully symmetric: define a fixed forward process that destroys data into Gaussian noise step by step, then train a neural network to walk that process backwards. The forward process is just math (noise + scaling); the reverse process is the learning. At test time, sample pure Gaussian noise and apply the network 1000 times — out comes a face that doesn't exist.*

## The two processes

A DDPM (denoising diffusion probabilistic model) consists of two processes operating on a sequence of variables $x_0, x_1, \ldots, x_T$ with $T$ typically around 1000:

- **Forward diffusion** $x_0 \to x_1 \to \cdots \to x_T$ — fixed (no learned parameters). Take a clean data sample $x_0 \sim p_{data}$ and gradually add Gaussian noise until $x_T \approx \mathcal{N}(0, I)$ — pure noise that has lost all information about $x_0$.
- **Reverse diffusion** $x_T \to x_{T-1} \to \cdots \to x_0$ — learned. Sample $x_T \sim \mathcal{N}(0, I)$ and progressively denoise to produce a fresh sample $x_0 \sim p_\theta(x)$. The reverse process is the *generative path* and where all training happens.

> [!info] ASIDE — Why "diffusion"?
> The name borrows from physics. A drop of ink in still water spreads gradually until the pigment is uniformly distributed and the original shape is gone — a system relaxing from low entropy to high entropy via small random kicks. The forward process here is the same idea applied to pixels: each step nudges the image with a tiny amount of Gaussian noise, and after enough steps all structure is destroyed and you're left with pure noise. The reverse process is the (much harder) thermodynamic miracle: watching the diffusion run backwards.

## Forward diffusion: the noising chain

Each forward step is a small Gaussian perturbation:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t I\right)$$

In sample form:

$$x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\beta_t \ll 1$ is the **noise schedule** — a sequence of small positive numbers (typically $\beta_1 \approx 10^{-4}$ ramping linearly to $\beta_T \approx 0.02$). The schedule is a hyperparameter we choose; common choices are linear or cosine.

> [!tip] TIP — Why the $\sqrt{1 - \beta_t}$ scaling?
> The scaling keeps each step a *gentle mix* of "old image" and "new noise" rather than just piling noise on top. Without it, the variance of $x_t$ would grow without bound as $t$ increases — values would explode out of the meaningful range. With it, the variance stays approximately fixed. After many steps, $x_T$ ends up distributed as $\mathcal{N}(0, I)$ — the same simple distribution we can sample from at test time. It's how the chain *terminates at noise* in a controlled way, rather than diverging.

### The closed-form shortcut: ancestral sampling

The composition of all forward steps from $x_0$ to $x_t$ collapses into a single Gaussian. Define the **cumulative product**

$$\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$$

Then

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t) I\right)$$

In sample form:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This is **enormously useful** for training: instead of running the forward process step by step (1000 sequential operations) to produce a noisy training example at timestep $t$, you can sample $x_t$ in *one operation* directly from $x_0$. The schedule is designed so $\bar{\alpha}_T \to 0$ — meaning $x_T$ has zero contribution from $x_0$ and is pure noise as required.

## Reverse diffusion: the generative path

To generate a sample, start from pure noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoise:

$$x_T \to x_{T-1} \to \cdots \to x_1 \to x_0$$

The ideal reverse step is the conditional distribution $q(x_{t-1} \mid x_t)$ — the true posterior over the slightly cleaner image given the slightly noisier one. **In general this is intractable** — there's no closed form and we can't sample from it directly.

The DDPM workaround: *approximate it with a Gaussian whose mean is predicted by a neural network*. As long as $\beta_t$ is small at each step, the true posterior is well-approximated by a Gaussian — so the parametric form

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I\right)$$

is an honest approximation (not a heuristic). Here:

- $\mu_\theta(x_t, t)$ is the network's prediction — a learnable mean of the same shape as the image.
- $\sigma_t^2$ is fixed in vanilla DDPM (a function of $\beta_t$, not learned).

In sample form:

$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$

The network only has to learn the **mean** of the reverse step. The variance is given by the schedule.

## The big simplification: predict the noise, not the mean

A theoretical derivation (in the DDPM paper) shows that predicting $\mu_\theta$ directly is equivalent to predicting the *noise* $\epsilon$ that was added in the forward step from $x_0$ to $x_t$. The two are linked by:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t) \right)$$

where $\alpha_t = 1 - \beta_t$, and $\epsilon_\theta(x_t, t)$ is a neural network whose output has the same shape as $x_t$ (it predicts the noise image).

Why predict noise instead of the mean directly? Two reasons:

1. **Easier learning.** Predicting noise is a cleaner regression target than predicting a clean image. The noise is approximately the same scale at every timestep; the clean image's range varies.
2. **Same loss, different framing.** The loss becomes simple MSE between predicted noise and actual noise — a *supervised* problem with abundant training data (we made the noise ourselves).

## The network: a U-Net that takes a timestep

The architecture predicting $\epsilon_\theta(x_t, t)$ is typically a [[u-net|U-Net]] (encoder-decoder with skip connections), often augmented with [[residual-connection|ResNet blocks]] and self-attention layers. Two crucial details:

- **Input is the noisy image $x_t$, output is the predicted noise $\epsilon_\theta$** — both have the same shape (same number of pixels, same channels).
- **The timestep $t$ is also fed in** as a learned vector embedding (via fully-connected layers from the scalar $t$). The network needs to know "which step of denoising are we at?" — different timesteps require different denoising strategies (lots of noise removal vs fine-detail polishing).

The U-Net is **deterministic**: same input, same output. Stochasticity in the overall reverse process comes from the explicit noise term $\sigma_t z$ added at each sampling step, not from the network.

## Training algorithm (DDPM Algorithm 1)

```
repeat:
    x_0 ~ q(x_0)                      # sample a clean training image
    t   ~ Uniform({1, ..., T})        # pick a random timestep
    ε   ~ N(0, I)                     # sample fresh Gaussian noise
    # Forward shortcut: jump straight from x_0 to x_t in one operation
    x_t = √(α̅_t) · x_0 + √(1 - α̅_t) · ε
    # Loss: MSE between true noise and predicted noise
    L = || ε - ε_θ(x_t, t) ||²
    take gradient step on ∇_θ L
until converged
```

Three things worth noting:

- **It's just supervised learning.** The forward shortcut gives us an $(x_t, \epsilon)$ pair for free; the network learns to predict $\epsilon$ from $x_t$ and $t$. Loss = MSE. Gradient descent. Backprop. All standard.
- **No iterative forward roll-out.** The shortcut means each training step costs one network forward+backward pass, not $T$ passes. Critical for training feasibility.
- **The network sees all timesteps interleaved.** Random sampling of $t$ each iteration means the same network learns to denoise at $t = 1$ (lots of noise) and $t = T$ (just noise) — the timestep embedding tells it which regime it's in.

## Sampling algorithm (DDPM Algorithm 2)

```
x_T ~ N(0, I)                          # start from pure Gaussian noise
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else 0
    # Apply one reverse step using predicted noise
    x_{t-1} = (1/√α_t) · (x_t - ((1-α_t)/√(1-α̅_t)) · ε_θ(x_t, t)) + σ_t · z
return x_0
```

The big bracketed expression is exactly $\mu_\theta(x_t, t)$ — the predicted mean of the reverse Gaussian. Adding $\sigma_t z$ samples from that Gaussian. After $T$ such steps, $x_0$ is your generated image.

> [!warning] CAUTION — Sampling is slow
> The reverse loop runs *sequentially* — each denoising step requires the result of the previous one, so the $T$ U-Net calls cannot be parallelised. With $T = 1000$ and a heavy U-Net, generating one image can take seconds even on a top GPU. This is the **major drawback** of diffusion models compared to GANs (which generate in one forward pass). The race to fix this — DDIM, DPM-Solver, distilled diffusion, latent diffusion — is one of the most active research areas in the field.

## What the model is *really* learning

The diffusion model is fundamentally a **denoiser**. It is *not* — strictly speaking — an image-generation model. It happens to generate images as a side effect of repeated denoising of pure noise. Every training example, every loss evaluation, every U-Net forward pass is solving the same problem: "given this noisy image and what timestep it is, what noise was added?"

Generation works because if you start from pure noise (which the model has been trained to handle, because $x_T$ in the forward process is essentially pure noise) and apply denoising 1000 times, you end up at a clean image consistent with $p_{data}$. The chain of small denoising steps does the work that a GAN's generator does in one step.

## Quality vs diversity vs speed: the trilemma

| Family | Quality | Diversity | Speed |
|---|---|---|---|
| [[generative-adversarial-network\|GAN]] | High | Low (mode collapse) | Fast (1 forward pass) |
| Diffusion | Very high | High (covers modes well) | Slow ($T$ forward passes) |
| VAE / normalising flows | Moderate | High | Fast |

Diffusion models excel at the first two and are weak on the third. The "race against the clock" is what motivates [[latent-diffusion-model|latent diffusion]] (do diffusion in a small latent space, decode once at the end), DDIM (skip steps deterministically), and other accelerated samplers.

## Conditional diffusion: feeding in $c$

The same architecture extends trivially to conditional generation $p_\theta(x \mid c)$ where $c$ is a class label, a low-resolution image, a text prompt, etc.

$$p_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t, c),\, \Sigma_\theta(x_t, t, c)\right)$$

The U-Net consumes $c$ in addition to $x_t$ and $t$:

- **Scalar $c$** (class label) — embed as a vector via lookup or small MLP, add to the timestep embedding.
- **Image $c$** (low-res, masked, etc.) — concatenate channel-wise with $x_t$ before the U-Net.
- **Text $c$** — encode with a text encoder (CLIP, T5), inject via cross-attention layers throughout the U-Net.

Loss is unchanged — predict the noise that was added, conditioned on $c$. This is the workhorse pattern behind every text-to-image model in production. See [[conditional-generative-model]] for the broader posterior-estimation framing that applies equally to cGANs and conditional diffusion.

## Key extensions (high-level)

- **Cascaded diffusion.** Generate at low resolution (e.g. 32×32), then use a chain of conditional super-resolution diffusion models to scale up (32→64→256). Each stage is conditioned on the previous stage's output. Faster than one giant model trained at full resolution, and produces sharper outputs at high res.
- **Super-resolution diffusion (SR3).** Train $p(x \mid y)$ where $x$ is high-res and $y$ is low-res. Concatenate upsampled $y$ to $x_t$ in the U-Net. Beats bicubic interpolation and MSE-regression super-resolution by a wide margin (regression converges to the conditional mean — blurry; diffusion samples from the posterior — sharp).
- **Image-to-image diffusion (Palette).** Same recipe with $y$ being any image-domain condition: greyscale→colour, masked→inpainted, low-quality JPEG→clean, panorama uncropping. Uses one model architecture across all tasks.
- **Diffusion for segmentation.** Run forward diffusion on an image, extract intermediate U-Net feature maps, train a small MLP classifier on top to predict per-pixel class labels. The diffusion U-Net's features turn out to be excellent representations even though it was trained for noise prediction (a [[representation-learning|representation-learning]] side effect).
- **[[latent-diffusion-model|Latent diffusion (Stable Diffusion)]].** The decisive practical breakthrough — perform diffusion in the small latent space of a frozen autoencoder rather than in pixel space. Massive speedup, plus a clean place to inject conditioning via cross-attention. Behind essentially every modern text-to-image system.

## Common pitfalls

- **Receptive field too small.** A shallow U-Net (e.g. only two pooling/upsampling levels) has a limited receptive field — pixel predictions are influenced only by local context. For tasks requiring *global* consistency (eyes the same colour, both shoes the same pair), this causes asymmetric / inconsistent outputs. Fix: deeper U-Net with more pooling levels, or add self-attention layers that aggregate global context. (Week-08 problem set Q2.)
- **Confusing $x_T$ with a latent code.** The noise at $x_T$ is *not* an encoding of $x_0$. It's pure Gaussian noise that has lost all information about the original. Diffusion models are not [[autoencoder|autoencoders]] — there is no compressed latent representation of the input. The noise plays the same role as the random $z$ at the start of a [[generative-adversarial-network|GAN]]: a fresh sample from a simple distribution that the model expands into structured output.
- **Both processes are stochastic.** The forward process is random (different noise sample → different $x_T$); the reverse process is random (different $z$'s in sampling → different $x_0$). Even with the same starting noise $x_T$, the reverse process produces different images on different runs because each step injects fresh $\sigma_t z$.
- **Predicting the mean directly is harder than predicting noise.** Empirically and theoretically. Always parameterise the network to output noise, and recover $\mu_\theta$ analytically.

> [!question]- A friend's diffusion model generates faces where the left and right eye colours don't match. What's likely wrong, and what should they change about the U-Net?
> The U-Net's **receptive field** is too small to enforce global consistency. Each pixel's prediction is influenced mainly by its local neighbourhood, so spatially distant features that should match (left eye, right eye) are predicted independently — and any random difference accumulates over the 1000 denoising steps. The fix is to make the U-Net deeper, with more pooling/upsampling levels, so the receptive field at the bottleneck spans the whole face — making both eyes "visible" to the same set of features. Alternatively (or additionally), insert self-attention layers in the U-Net so any pixel can directly query any other pixel regardless of distance — this is exactly why modern diffusion U-Nets all use self-attention. The same bug pattern appears for "both shoes match," "both earrings match," etc. — globally-consistent attributes always need either large receptive fields or attention.

> [!question]- Why do we ask the network to predict the noise $\epsilon$ rather than directly predict the slightly-denoised image $x_{t-1}$?
> Several reasons. **Theoretical:** the DDPM derivation shows the loss based on noise prediction is *equivalent* to the loss based on mean prediction (related by a closed-form reparameterisation), so we lose nothing by choosing the easier target. **Practical:** the noise has approximately the same statistics across all timesteps (zero-mean unit-variance Gaussian), making it a clean regression target. The clean image $x_0$ has a complicated, image-specific distribution; predicting a slightly-cleaner version $x_{t-1}$ from $x_t$ is also harder because most of $x_t$ is correct and only a small fraction needs change. Predicting noise focuses the network on *what's wrong*, which is more learnable. Once you have $\epsilon_\theta(x_t, t)$, you can recover $\mu_\theta$ analytically — no information lost.

> [!question]- A friend says: "Diffusion models are just GANs with extra steps. Both go from noise to images." How are they fundamentally different?
> Two key differences. **Training signal.** A GAN trains its generator implicitly via an adversarial discriminator (no direct loss on the generator's output) — fragile, prone to mode collapse, hard to balance. A diffusion model trains via direct supervised regression: predict a known noise that we made ourselves. Stable, well-behaved, no adversary needed. **Sampling structure.** A GAN's generator does the entire $z \to x$ mapping in a single forward pass. A diffusion model breaks it into $T$ tiny steps, each one a small denoising. The "extra steps" aren't a quirk — they're the *mechanism* that makes the problem easier per step and the training signal cleaner. The trade-off: GAN sampling is fast but training is hard; diffusion training is easy but sampling is slow. The two represent fundamentally different paradigms for generative modelling, not minor variants of the same idea.

## Connections

- **Built on** [[u-net]] — the noise-predictor architecture is a U-Net, often with [[residual-connection|ResNet blocks]] and self-attention. The encoder-decoder shape is well-suited to image-shape input/output.
- **Built on** [[bayes-theorem]] — the reverse process is Bayesian inference: given the observation $x_t$, infer the posterior over $x_{t-1}$. The DDPM derivation involves repeated application of Bayes between forward and reverse processes.
- **Built on** [[backpropagation]] / [[gradient-descent-nc|gradient descent]] — training is standard supervised learning with MSE loss. The forward shortcut makes each gradient step cheap.
- **Family member of** [[generative-model]] — competes with [[generative-adversarial-network|GANs]] and VAEs for image generation. Currently the dominant paradigm for image synthesis quality.
- **Extended by** [[latent-diffusion-model]] — performing diffusion in the latent space of a frozen autoencoder rather than pixel space; the basis of Stable Diffusion.
- **Extends to** [[conditional-generative-model|conditional generation]] — same recipe with a condition $c$ added as input. Text-to-image, image-to-image translation, super-resolution, inpainting all follow this pattern.
- **Compared to** [[autoencoder]] — both have an encoder-decoder shape (the U-Net), but the AE compresses input to a meaningful latent and reconstructs; diffusion adds noise (no learned encoding) and learns to remove it. The "noise at $x_T$" is *not* a latent code in the AE sense — it's pure noise containing no information about $x_0$.
- **Compared to** [[generative-adversarial-network|GAN]] — both go from noise to image, but: GAN uses one fast forward pass with adversarial training (unstable, mode-collapse-prone); diffusion uses many slow denoising passes with stable supervised training (high quality, high diversity).
