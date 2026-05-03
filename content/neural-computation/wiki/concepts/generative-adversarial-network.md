---
name: generative-adversarial-network
description: A generative model architecture (Goodfellow et al. 2014) consisting of a generator $G$ and a discriminator $D$ trained against each other. $G$ learns to produce samples; $D$ learns to tell real from generated; the two are locked in a min-max game whose equilibrium is $p_\theta = p_{data}$. Implicit density estimation — no explicit likelihood is ever computed.
type: concept
sources:
  - raw/week-07/w07-l14-transcript.txt
  - raw/week-07/w07-l15-transcript.txt
  - raw/week-07/w07-slides.pdf
  - raw/week-07/w07-problem-set.pdf
status: stable
updated: 2026-04-26
---

*A GAN is a forger and a detective locked in a room. The forger ($G$) makes counterfeit notes from random inspiration. The detective ($D$) inspects each note and decides "real or fake?" The forger improves by studying which fakes get caught; the detective improves by seeing more fakes. At equilibrium, the forger's counterfeits are indistinguishable from real currency — the discriminator can do no better than coin-flip — and the detective is no longer useful. Throw the detective away. The forger is the deliverable.*

## Architecture

Two networks, trained jointly:

- **Generator** $G_\theta : \mathcal{Z} \to \mathcal{X}$ with parameters $\theta$. Takes a sample $z \sim p(z)$ from a fixed prior (typically $\mathcal{N}(0, I)$) and produces a sample $G_\theta(z) \in \mathcal{X}$ in image space.
- **Discriminator** $D_\phi : \mathcal{X} \to [0, 1]$ with parameters $\phi$. Takes an input $x$ and outputs a scalar probability that $x$ is *real* (drawn from $p_{data}$) rather than fake (drawn from $p_\theta$). Final layer is a [[sigmoid-function-nc|sigmoid]]; it is just a [[binary-cross-entropy|binary classifier]].

The model distribution $p_\theta(x)$ is the push-forward of $p(z)$ through $G_\theta$ — implicit, never written down. To sample: draw $z \sim p(z)$, return $G_\theta(z)$.

> [!info] ASIDE — Forger / detective intuition
> Goodfellow's original framing: imagine a counterfeiter trying to print fake currency. They start with random inspiration ($z$) and produce fakes ($G(z)$). A detective inspects every note (real or fake) and announces "$D(x) = $ probability this is real." The counterfeiter studies which fakes the detective catches and refines their craft; the detective sees more fakes and gets better at spotting them. The objective is *equilibrium*: when the forger's fakes are indistinguishable from real notes, the detective can do no better than 50/50 guessing. At that point, the forger's distribution matches the real one.

## The discriminator's loss

The discriminator solves a standard binary classification problem: label $1$ for real samples ($x \sim p_{data}$), label $0$ for fakes ($x \sim p_\theta$, equivalently $x = G_\theta(z)$ for $z \sim p(z)$). Standard binary cross-entropy, written with the sign flipped to a *minimisation*:

$$\mathcal{L}_D(\phi, \theta) = -\underbrace{\mathbb{E}_{x \sim p_{data}}[\log D_\phi(x)]}_{\text{push real} \to 1} - \underbrace{\mathbb{E}_{z \sim p(z)}[\log(1 - D_\phi(G_\theta(z)))]}_{\text{push fake} \to 0}$$

The two terms are independent losses on the two batches: real samples (label 1) and fake samples (label 0). In practice, with mini-batches of size $N_{D,r}$ real samples and $N_{D,f}$ fake samples:

$$\mathcal{L}_D = -\frac{1}{N_{D,r}} \sum_i \log D_\phi(x_i) - \frac{1}{N_{D,f}} \sum_j \log(1 - D_\phi(G_\theta(z_j)))$$

For *fixed* generator parameters $\theta$, the discriminator is updated by SGD on $\phi$ to minimise this.

## The generator's loss — theoretical and practical

The generator's job is to *fool* the discriminator. The natural objective is the negative of $D$'s loss on the fake batch (since $G$ doesn't see real data, the real-data term is irrelevant — its gradient w.r.t. $\theta$ is zero):

**Theoretical loss** (matches the min-max derivation):

$$\mathcal{L}_G^{\text{theo}}(\theta, \phi) = \mathbb{E}_{z \sim p(z)}[\log(1 - D_\phi(G_\theta(z)))]$$

The generator wants to *minimise* this (push $D(G(z)) \to 1$, fooling the detective). The discriminator wants to maximise the same quantity. This is the **two-player min-max game**:

$$\min_\theta \max_\phi \; J_{\text{GAN}}(\theta, \phi) = \mathbb{E}_{x \sim p_{data}}[\log D_\phi(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D_\phi(G_\theta(z)))]$$

### The saturating-gradient problem

The theoretical loss has a fatal practical flaw early in training. When the generator is bad (its samples are obvious fakes), $D(G(z)) \approx 0$, so $\log(1 - D(G(z))) \approx \log(1) = 0$ — flat region of the curve. The gradient is *small* exactly when the generator is *worst* and most needs a strong learning signal. Conversely, when the generator is excellent and $D(G(z)) \approx 1$, the gradient is huge — even though training has nearly converged. Backwards from what we want.

**Practical loss** (the fix, used in real implementations):

$$\mathcal{L}_G^{\text{prac}}(\theta, \phi) = -\mathbb{E}_{z \sim p(z)}[\log D_\phi(G_\theta(z))]$$

Same goal — push $D(G(z)) \to 1$ — but the gradient is *large* when $D(G(z))$ is small (training start) and saturates when $D(G(z)) \to 1$ (training end). Exactly the right shape.

The two losses optimise the same equilibrium ($D(G(z)) = 1$) but with different gradient profiles. The practical loss is what every GAN implementation actually uses.

> [!tip] TIP — How to remember the loss flip
> Both forms of the generator loss are trying to *fool* $D$ into outputting "1" on fakes. The theoretical form $\log(1 - D)$ is "minimise the log-probability of being caught" — saturates badly when the model is bad. The practical form $-\log D$ is "maximise the log-probability of fooling the detective" — saturates badly when the model is good (which is fine — at that point you're done). Same equilibrium, opposite saturation behaviour.

## The training algorithm

Outer loop iterates many epochs. Inner step alternates D and G updates:

```
Initialize θ, φ randomly
for t = 1 ... T:
    # Train D for K steps (often K=1)
    for k = 1 ... K:
        Sample real batch {x_i} ~ p_data, size N_{D,r}
        Sample latent batch {z_i} ~ N(0,I), size N_{D,f}
        L_D = -(1/N_{D,r}) Σ log D_φ(x_i) - (1/N_{D,f}) Σ log(1 - D_φ(G_θ(z_i)))
        φ ← φ - α ∇_φ L_D
    # Train G for 1 step
    Sample latent batch {z_i} ~ N(0,I), size N_G
    L_G = -(1/N_G) Σ log D_φ(G_θ(z_i))     # practical generator loss
    θ ← θ - β ∇_θ L_G
```

Two stochastic gradient descents on opposite objectives, alternated. Backprop through $D$ provides the gradient signal that flows back into $G$ — the generator's output $G_\theta(z)$ is just an activation map fed into $D$, so the chain rule pushes gradients all the way back to $\theta$ via $\partial L_G / \partial \theta = \partial L_G / \partial G(z) \cdot \partial G(z) / \partial \theta$.

## Why this works — the Jensen-Shannon connection

Goodfellow's 2014 paper proved a theoretical guarantee:

**Optimal discriminator.** For *fixed* generator parameters $\theta$ and assuming $D$ has infinite capacity, the optimal discriminator at every point is

$$D_\phi^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_\theta(x)}$$

Substituting this back into the min-max objective $J_{\text{GAN}}$ and simplifying yields:

$$J_{\text{GAN}}(\theta, \phi^*) = 2 D_{JS}[p_{data}(x) \;\|\; p_\theta(x)] - 2\log 2$$

where $D_{JS}$ is the **Jensen-Shannon divergence** — a symmetric measure of distance between two distributions, with $D_{JS}[p \| q] = 0 \iff p = q$.

**Conclusion.** *If* $D$ is perfectly optimised at every step before $G$ is updated, *then* minimising $J_{\text{GAN}}$ over $\theta$ minimises $D_{JS}[p_{data} \| p_\theta]$ — driving the model distribution to exactly match the data distribution.

Caveats: in practice, $D$ is not perfectly optimised at every step (we run only $K$ SGD updates), $D$ doesn't have infinite capacity, and we use the practical generator loss instead of the theoretical one. So the theoretical guarantee is loose. But it provides intuition for *why* the adversarial setup works at all — there's a real divergence being minimised behind the scenes.

## Inference: just keep the generator

After training, **the discriminator is thrown away**. The trained generator alone defines the model distribution. To generate a new sample:

1. Draw $z \sim \mathcal{N}(0, I)$.
2. Compute $\tilde{x} = G_\theta(z)$.

That's it. The same generator can produce unlimited new samples by drawing fresh $z$ vectors. No labels, no encoder, no input image.

## Variants worth knowing

- **DCGAN** (Radford et al. 2016) — the first practical convolutional GAN. Generator uses transposed convolutions / [[upsampling]] (analogous to a U-Net decoder, but starting from $z \in \mathbb{R}^{100}$ and growing to a full image). Discriminator is a standard [[convolutional-neural-network|CNN]] classifier. DCGAN-style architectures are still the default for many image-generation tasks.
- **Latent interpolation in DCGAN** — sampling $z = \alpha z_1 + (1-\alpha) z_2$ for $\alpha \in [0, 1]$ produces a *smooth morph* between $G(z_1)$ and $G(z_2)$. Empirically observed; no theoretical guarantee that the latent space is smooth, but it usually is.
- **BigGAN** (Brock et al. 2019) — scaled-up DCGAN-style architecture trained on large datasets at high resolution; produced the first photorealistic class-conditional ImageNet generation.
- **StyleGAN** (Karras et al.) — the architecture behind `thispersondoesnotexist.com`; introduces style mixing and disentangled latent control.
- **Conditional GANs (cGAN / pix2pix)** — generate $y$ conditioned on an input $x$, instead of unconditionally. See [[conditional-generative-model]].

## Common pitfalls

- **Mode collapse.** $G$ discovers one or a few outputs that consistently fool $D$ and starts producing only those — generating a perfect "7" every time, ignoring all other digits. Symptom: low diversity in samples. Defences: minibatch discrimination, Wasserstein loss, careful architecture choice.
- **Discriminator dominance.** If $D$ becomes near-perfect early, $D(G(z)) \to 0$ for all $z$, the practical generator loss saturates, and $G$ can't learn. Defences: keep $D$ slightly weaker (lower learning rate, fewer updates per step), use the practical loss not the theoretical one, balance capacity.
- **Training instability.** The min-max equilibrium is fragile; small changes to architecture, learning rate, or batch size can derail training entirely. GAN training is notoriously empirical — DCGAN's main contribution was a recipe of architectural choices (no fully-connected layers, batch norm everywhere, LeakyReLU in $D$) that consistently train.

> [!question]- The discriminator is trained to maximise $\mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$. Why is this just binary cross-entropy in disguise?
> Because the two terms together are the negative log-likelihood of the labels under the discriminator's predictions — exactly [[binary-cross-entropy]]. Real samples have label $y = 1$; for them the BCE term is $-\log D(x)$. Fake samples have label $y = 0$; for them the BCE term is $-\log(1 - D(G(z)))$. Sum and negate: minimising BCE = maximising $\log D + \log(1-D)$. The "min-max" framing of GANs is just standard supervised binary classification on the discriminator's side, with the twist that the data distribution it's classifying against ($p_\theta$) is itself being learned by the generator.

> [!question]- A friend trains a GAN and notices that when the generator is at its worst (early training, $D(G(z)) \approx 0$), the gradients on $G$ are tiny. What loss function are they probably using, and how should they fix it?
> They are probably using the **theoretical generator loss** $\log(1 - D(G(z)))$, which is flat near $D(G(z)) = 0$ — exactly where training starts. The fix is to switch to the **practical generator loss** $-\log D(G(z))$, which has large gradients when $D(G(z))$ is small. Both losses share the same minimum (push $D(G(z)) \to 1$); they differ only in the *gradient profile* during training. The practical form gives strong learning signal early when the generator is bad, and saturates softly when the generator is good — exactly the desired schedule.

> [!question]- After GAN training is complete, why don't we use the discriminator for anything?
> Because its job was to provide a *training signal* for the generator, not to be useful in itself. At the equilibrium of training, $D(x) \approx 0.5$ for both real and fake inputs — the discriminator is no longer informative; it's been trained into a corner where it can't tell the two apart. Even slightly before equilibrium, $D$'s purpose is supervision, not classification of new inputs (we already have plenty of *real* data; we don't need a network to confirm it's real). All the value is in $G$, which knows how to map $\mathcal{N}(0, I)$ noise to plausible samples. So at inference: keep $G$, discard $D$.

> [!question]- A friend asks: "If GANs minimise Jensen-Shannon divergence between $p_\theta$ and $p_{data}$, why don't we just compute JSD and minimise it directly with gradient descent?"
> Because computing JSD directly requires evaluating *both* $p_{data}(x)$ and $p_\theta(x)$ at arbitrary points $x$ — and we have *neither* of those in closed form. $p_{data}$ is the unknown true distribution we only see samples from; $p_\theta$ is implicitly defined by the generator (the push-forward of a Gaussian through $G$, which we can sample from but not evaluate). The genius of the GAN setup is that adversarial training optimises JSD *without ever computing it* — the discriminator implicitly estimates the ratio $p_{data}/(p_{data} + p_\theta)$, and the generator's loss reduces to JSD when $D$ is optimal. Implicit density estimation is the only tractable approach when explicit densities aren't available.

## Connections

- **Built on** [[binary-cross-entropy]] — the discriminator loss is BCE; the practical generator loss is half of it.
- **Built on** [[sigmoid-function-nc|sigmoid function]] — the discriminator's final activation produces $D(x) \in [0,1]$.
- **Built on** [[gradient-descent-nc|gradient descent]] and [[backpropagation]] — gradients flow from $\mathcal{L}_G$ through $D$ back into $G$ (the generator's output is just another activation map that $D$ consumes).
- **Built on** [[convolutional-neural-network]] and [[upsampling]] — DCGAN's generator uses transposed convolutions to grow $z \in \mathbb{R}^{100}$ into a full image; its discriminator is a standard CNN.
- **Family member of** [[generative-model]] — the implicit-density-estimation branch.
- **Extended by** [[conditional-generative-model]] — adds a condition $x$ to control what gets generated; the basis of pix2pix, image-to-image translation, super-resolution, and class-conditional generation.
- **Latent vector** $z$ in a GAN is *not* an encoded representation — it's a sample from a fixed prior. Contrast with [[latent-representation]] in autoencoders / SimCLR where $z$ comes from an encoder.
- **Compared to** [[autoencoder]] — both have a "decoder-shaped" network, but the AE's decoder reconstructs a specific input via the bottleneck while the GAN's generator transforms random noise into novel samples. Different goals, different training signals.
