---
type: week
week: 7
title: "Generative Models, GANs, and Conditional Generation"
sources:
  - raw/week-07/w07-l14-transcript.txt
  - raw/week-07/w07-l15-transcript.txt
  - raw/week-07/w07-slides.pdf
  - raw/week-07/w07-problem-set.pdf
  - raw/week-07/w07-problem-set-solution.pdf
concepts:
  - "[[generative-model]]"
  - "[[generative-adversarial-network]]"
  - "[[bayes-theorem]]"
  - "[[conditional-generative-model]]"
status: stable
updated: 2026-04-26
---

> [!question]+ THE CRUX: Last week we learned to *encode* data into useful representations. This week the goal flips — given only training samples, can we build a network that *generates new samples* from the same distribution? What does it even mean to "learn a distribution" when we can't write it down, can't evaluate it, and only ever see samples from it?

*The week's answer is **implicit density estimation**: don't try to write down $p_{data}(x)$ — train a generator network whose samples are statistically indistinguishable from real data. The genius mechanism is **adversarial training**: pair the generator with a discriminator that tries to spot fakes, and let them duel. At equilibrium, the generator's samples match the real distribution. The week then extends this to **conditional generation** — given a sketch, produce a photo; given a low-res image, produce high-res — by feeding a condition into both networks and reframing the problem as posterior estimation $p(y \mid x)$ via [[bayes-theorem]].*

## Where we left off

Week 6 was about *learning representations* of data via autoencoders, contrastive learning, and pretext tasks. The deliverable was always the encoder — the latent vector $z$ was the prize, and reconstruction / contrast / context-prediction was scaffolding to train it.

This week the deliverable flips. We don't care about an encoder; we want a network whose *outputs*, when sampled, look like real data we've never seen. New faces. New street scenes. New microscopy images. The encoder pattern still appears (cGAN's discriminator and pix2pix's U-Net generator are both encoder-shaped), but now the goal is to generate, not to compress.

## The framing: density estimation we can't actually do

Real data $\{x_i\}$ comes from some unknown distribution $p_{data}(x)$. We never get to see $p_{data}$ directly:

- We can't write it down (faces aren't a Gaussian).
- We can't evaluate it at arbitrary points.
- We only have samples — the training set.

A **generative model** fits a parameterised distribution $p_\theta(x)$ — typically a neural network — with the goal $p_\theta \approx p_{data}$. Then to sample a new $x$, we sample from $p_\theta$.

This is **density estimation** with neural networks instead of classical parametric families. Two things make it hard for images:

1. **Pixel space is enormous** — $256^{784} \approx 10^{1888}$ configurations for a $28\times 28$ image, and meaningful images are a vanishingly thin manifold inside that vastness.
2. **Pixels are heavily correlated** — $p(x) \neq \prod_i p(x^{(i)})$. The distribution doesn't factor; the joint structure is what makes images images.

> [!tip] TIP — Why this is the same problem as week 6 (but flipped)
> Week 6 said: real images are a tiny island in pixel space, so we should learn the structure of that island via an encoder. Week 7 says: real images are a tiny island in pixel space, so we should learn to *sample* from it via a generator. The hardness is the same hardness; the deliverable is different.

See [[generative-model]] for the full landscape (latent variable models, diffusion, autoregressive, normalising flows).

## The latent-variable trick

The simplest sampling-only generative model:

1. Pick a **prior** $p(z) = \mathcal{N}(0, I)$ over a small latent space $\mathcal{Z}$.
2. Define a generator network $G_\theta : \mathcal{Z} \to \mathcal{X}$.
3. To sample: $z \sim \mathcal{N}(0, I)$; output $G_\theta(z)$.

The model distribution $p_\theta$ is the *push-forward* of the Gaussian through $G_\theta$ — implicitly defined; you can sample from it but never write it in closed form.

> [!warning] CAUTION — This $z$ is *not* the latent of an autoencoder
> In a [[autoencoder|AE]] or [[contrastive-learning|SimCLR]], $z = f_\phi(x)$ — the encoder consumes an input and outputs a latent. In a GAN/VAE, $z \sim p(z)$ is a sample from a *prior* — there is no input to encode. The two uses share the letter "latent" but differ fundamentally. See [[latent-representation]] for the AE-vs-SimCLR clash; the GAN $z$ adds a third meaning: random noise that the generator *expands into* a sample.

The remaining problem: how do we train $G_\theta$? We can't compute $p_\theta(x)$ to score it against $p_{data}(x)$. We don't have either explicitly. Each generative-model family solves this differently. GANs use adversarial training.

## GANs: training by adversarial duel

A **generative adversarial network** ([[generative-adversarial-network]]) pairs the generator $G_\theta$ with a second network — a **discriminator** $D_\phi$ — and trains them against each other.

- $G_\theta$ generates fake samples $G_\theta(z)$ from random $z$.
- $D_\phi$ takes inputs (real or fake) and outputs a probability that the input is real. It's a [[binary-cross-entropy|binary classifier]] with a [[sigmoid-function-nc|sigmoid]] head.
- $D$ is trained to *detect* fakes: push $D(x) \to 1$ for real $x$, $D(G(z)) \to 0$ for fakes.
- $G$ is trained to *fool* $D$: push $D(G(z)) \to 1$.

It's a forger-and-detective game. The forger produces counterfeits from random inspiration; the detective inspects each note and announces real-or-fake; both improve over time. At equilibrium, the forger's fakes are indistinguishable from real — $D$ can do no better than 50/50, and $G$'s output distribution matches $p_{data}$.

### The losses

**Discriminator loss** — standard binary cross-entropy on a mixed batch (label 1 for real, 0 for fake):

$$\mathcal{L}_D = -\mathbb{E}_{x \sim p_{data}}[\log D_\phi(x)] - \mathbb{E}_{z \sim p(z)}[\log(1 - D_\phi(G_\theta(z)))]$$

**Generator loss (theoretical)** — minimise the negative of $D$'s fake-batch loss (the real-batch term has zero gradient w.r.t. $\theta$ and is dropped):

$$\mathcal{L}_G^{\text{theo}} = \mathbb{E}_{z \sim p(z)}[\log(1 - D_\phi(G_\theta(z)))]$$

**Generator loss (practical)** — same equilibrium, better gradients:

$$\mathcal{L}_G^{\text{prac}} = -\mathbb{E}_{z \sim p(z)}[\log D_\phi(G_\theta(z))]$$

Together $G$ and $D$ play a **min-max game**:

$$\min_\theta \max_\phi \; J_{\text{GAN}}(\theta, \phi) = \mathbb{E}_{x \sim p_{data}}[\log D_\phi(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D_\phi(G_\theta(z)))]$$

### Why "theoretical" and "practical" generator losses differ

The theoretical form $\log(1 - D(G(z)))$ saturates exactly where you don't want it to — early in training when $D(G(z)) \approx 0$ (the generator's outputs are obvious garbage), the gradient is *flat*. The practical form $-\log D(G(z))$ has the opposite gradient profile: large when $D(G(z))$ is small (lots of learning signal early), saturating only as $G$ approaches $D$'s upper limit (where you're already done).

Both losses share the same minimum (push $D(G(z)) \to 1$). Every real GAN implementation uses the practical form. The theoretical form survives in textbooks because it's the one the JSD-minimisation proof uses.

> [!info] ASIDE — The Jensen-Shannon connection
> Goodfellow's 2014 paper proved: *if* $D$ has infinite capacity and is perfectly optimised at every step, *then* the optimal $D$ at every point is $D^*(x) = p_{data}(x) / (p_{data}(x) + p_\theta(x))$, and substituting back gives $J_{\text{GAN}}(\theta, \phi^*) = 2 D_{JS}[p_{data} \| p_\theta] - 2\log 2$. Minimising the GAN objective minimises **Jensen-Shannon divergence** between data and model — and JSD is zero iff the two distributions are equal. So adversarial training implicitly minimises a real divergence, even though we never compute it directly. Caveats: real $D$ has finite capacity and isn't perfectly optimised at every step; the practical loss differs from the theoretical one. The result is a guarantee in spirit, not a proof of practical convergence.

### The training algorithm

```
Initialize θ, φ randomly
for t = 1 ... T:
    for k = 1 ... K:                    # train D for K steps (often K=1)
        sample real batch {x_i} ~ p_data
        sample latent batch {z_i} ~ N(0,I)
        L_D = -avg(log D_φ(x_i)) - avg(log(1 - D_φ(G_θ(z_i))))
        φ ← φ - α ∇_φ L_D
    sample latent batch {z_i} ~ N(0,I)  # train G for 1 step
    L_G = -avg(log D_φ(G_θ(z_i)))       # practical generator loss
    θ ← θ - β ∇_θ L_G
```

Two stochastic gradient descents on opposite objectives, alternated. Backprop flows from $\mathcal{L}_G$ through $D$ back into $G$ — the generator's output is just an activation map fed into $D$, so the chain rule pushes gradients all the way back to $\theta$.

### After training: throw the discriminator away

Once trained, only $G$ is needed. To generate: sample $z \sim \mathcal{N}(0,I)$, compute $G_\theta(z)$. The discriminator was scaffolding for training; at inference it has no role.

> [!question]- A friend asks: "If GANs minimise Jensen-Shannon divergence between $p_\theta$ and $p_{data}$, why don't we just compute JSD and minimise it directly?" What's the answer in two sentences?
> Computing JSD requires evaluating both $p_{data}(x)$ and $p_\theta(x)$ at arbitrary points — and we have neither in closed form ($p_{data}$ is the unknown true distribution; $p_\theta$ is the implicit push-forward through $G$). The discriminator's job is to *implicitly* estimate the ratio $p_{data}/(p_{data}+p_\theta)$; adversarial training optimises JSD without ever computing it.

> [!question]- The discriminator achieves 99% training accuracy after a few steps; the generator's gradients vanish. What probably happened, and what's the standard response?
> $D$ became too strong too fast: $D(G(z)) \approx 0$ for all generated $z$, the practical generator loss $-\log D(G(z))$ saturates badly (huge magnitude but no gradient direction since $D$ outputs the same answer for everything), and $G$ can't improve. Standard responses: lower $D$'s learning rate, run fewer $D$ updates per $G$ update (already $K=1$ is often too many), reduce $D$'s capacity, or switch to a different objective (Wasserstein loss). The two networks need to stay roughly balanced; if $D$ wins decisively, training collapses.

## DCGAN and beyond

The original 2014 GAN used MLPs. Real images need convolutions:

- **DCGAN** (Radford et al. 2016) — generator made of transposed convolutions / [[upsampling]] (analogous to the decoder half of a [[u-net|U-Net]], starting from $z \in \mathbb{R}^{100}$ and growing to a full image); discriminator is a standard [[convolutional-neural-network|CNN]] classifier. DCGAN's main contribution was a *recipe* of architectural choices (no FC layers, batch norm everywhere, LeakyReLU in $D$, no pooling) that consistently train. Most modern GAN architectures descend from DCGAN.
- **Latent interpolation** — sampling $z = \alpha z_1 + (1-\alpha) z_2$ for $\alpha \in [0,1]$ produces a smooth morph between $G(z_1)$ and $G(z_2)$. Empirically observed; no theoretical guarantee, but reliable in practice.
- **BigGAN** (2019), **StyleGAN** (2019, behind `thispersondoesnotexist.com`), **StyleGAN-T** (text-to-image) — progressively scaled-up architectures that improved sample quality from "ugly digits" (2014) to "photorealistic faces" (2018) to "controllable text-to-image" (2023).

GANs were the state of the art for image generation roughly 2014–2020; they were displaced by **diffusion models** (which week 8 covers) but remain useful in many image-to-image tasks where their implicit-density approach is well-suited.

## Bayesian basics — the bridge to conditional models

The transition to conditional generation goes through Bayes. We pause for a refresher.

The setup: two random variables $x$ (observation) and $y$ (truth). The joint factors two ways:

$$p(x, y) = p(x \mid y)\,p(y) = p(y \mid x)\,p(x)$$

Equating gives **[[bayes-theorem|Bayes' theorem]]**:

$$p(y \mid x) = \frac{p(x \mid y)\,p(y)}{p(x)}$$

The four named pieces:

| Term | Name | Reading |
|---|---|---|
| $p(y)$ | Prior | What we believe about $y$ before seeing $x$ |
| $p(x \mid y)$ | Likelihood | How $x$ is produced given $y$ |
| $p(y \mid x)$ | Posterior | What we believe about $y$ after seeing $x$ |

The directionality matters: **likelihood goes forward (truth → observation), posterior goes backward (observation → truth).** The forward direction is usually easy to model (sensor physics); the backward direction is what we want at inference. Bayes converts one to the other.

Worked example from the slides: penguin flipper length $y$ measured by hand, computer-vision estimate $x$. The prior $p(y)$ is the dataset histogram of true flipper lengths (centred at $\sim 190$ mm). The likelihood $p(x \mid y = 192)$ is the histogram of CV measurements when truth is 192 (centred at 192 with noise spread). The posterior $p(y \mid x = 201)$ — the actual question we care about, "given the CV said 201, what's the true value?" — is *not* centred at 201; it's pulled toward the prior mean of 190 by Bayes shrinkage.

### MMSE — why MSE regression converges to the conditional mean

A classical exercise: given noisy measurements $\{x_i\}$, the estimator that minimises sum-of-squared-error is the **sample mean**. Generalising: a regression network trained with MSE loss converges to $f^*(x) = \mathbb{E}_{p(y \mid x)}[y]$ — the **conditional mean** over all plausible explanations of the observation.

This sets up the next section's payoff.

## Conditional generative models

Switch from learning the marginal $p_\theta(x) \approx p_{data}(x)$ to learning the **posterior** $\hat{p}_\theta(y \mid x) \approx p(y \mid x)$. The neural network is now a parameterised approximation to Bayes.

Tasks that fit this framing:

| Task | Condition $x$ | Target $y$ |
|---|---|---|
| Colourisation | Greyscale image | Colour image |
| Super-resolution | Low-res image | High-res image |
| Inpainting | Image with masked region | Filled image |
| Semantic-map → photo (pix2pix) | Class-coloured segmentation | Realistic photo |
| Sketch → photo | Outline drawing | Realistic photo |
| Text-to-image | Caption | Generated image |

In every case **multiple $y$'s are plausible** for the same $x$ — multiple legitimate colourings, multiple high-res reconstructions, multiple photos matching the same caption. The posterior captures that diversity; the task is to sample from it.

### Why MSE regression gives blurry outputs

Train an MLP to map $x \to y$ with MSE loss. It converges to $\mathbb{E}[y \mid x]$ — the *average* of all plausible answers. When the posterior is multimodal (red flower vs yellow flower), the mean is a desaturated grey-pink between them — neither one nor the other. The week-07 problem set drives this home with a $4 \times 4$ digit example: the optimal regression output for "draw a 1" is the *pixelwise average* of all training "1"s, full of fractional values like $1/3$ and $2/3$ that don't appear in any single training example.

A *generative* approach to conditional modelling samples from the posterior instead of averaging across it. Each sample is one plausible answer; running inference multiple times produces *diverse* outputs. See [[conditional-generative-model]].

### cGAN and pix2pix

The cGAN extension is a direct surgical edit of the GAN setup:

- **Generator** $G_\theta(z, x)$ — takes both noise and the condition.
- **Discriminator** $D_\phi(x, y)$ — takes both the condition and the candidate target; outputs probability that the *pair* is real.

Critically, $D$ sees the condition. Without it, $G$ could produce *any* realistic $y$ regardless of $x$ (always output a beautiful brown shoe regardless of the input sketch) and still fool $D$. With the condition, $D$ checks that the pair *matches* — punishing $G$ for ignoring $x$.

**pix2pix** (Isola et al. 2018) is the canonical image-to-image cGAN: U-Net generator + PatchGAN discriminator + L1+GAN hybrid loss. The L1 anchors the output near the ground truth; the GAN sharpens textures to *realistic* values rather than the L1's blurry conditional mean. Together they produce outputs that are both faithful to the input and crisp.

### Conditional models naturally express uncertainty

Because $G(z, x)$ depends on the random $z$, running it multiple times on the same $x$ produces different $y$'s. The variation across samples is an estimate of posterior uncertainty: high-confidence regions (real cell boundaries in a microscopy image) stay consistent; low-confidence regions (background noise) vary. This is impossible with a plain MSE-regression network.

> [!question]- A friend trains a colourisation network with MSE loss. Outputs are washed out and desaturated — the reds aren't red. What's wrong, and what should they switch to?
> They are seeing the MMSE-regression problem. A specific greyscale flower could be coloured red, yellow, or pink — multiple plausible colourings. MSE-regression converges to the conditional mean, the *pixelwise average* of all those plausible colourings — a desaturated neutral compromise. Switch to a [[conditional-generative-model|cGAN]] (pix2pix-style with L1+GAN loss) or a conditional diffusion model. The generative loss samples *one* plausible colouring at a time instead of averaging across all of them — outputs come out vivid and crisp, and you can re-sample to get diverse plausible colourings of the same image.

> [!question]- Why does the cGAN's discriminator need to see the condition $x$, not just the candidate $y$?
> Without the condition, $D$ only checks "is this $y$ realistic?" — and $G$ can fool it by producing *any* realistic image regardless of input (always output a beautiful handbag regardless of the sketch). With the condition, $D$ checks "is this $(x, y)$ pair realistic?" — a beautiful handbag generated from a shoe-sketch is a *mismatched pair* and gets flagged as fake. The condition forces $G$ to actually use $x$ as a constraint on the generation, turning a generic generator into a properly conditional one.

## Concepts introduced this week

- [[generative-model]] — the broad framing: density estimation $p_\theta \approx p_{data}$, the four families (latent variable, diffusion, autoregressive, normalising flows).
- [[generative-adversarial-network]] — generator + discriminator, BCE-on-mixed-batch loss, theoretical vs practical generator losses, min-max game, Jensen-Shannon proof, training algorithm, DCGAN.
- [[bayes-theorem]] — prior, likelihood, posterior; the penguin walkthrough; MMSE / why regression converges to the conditional mean.
- [[conditional-generative-model]] — cGAN, pix2pix, posterior sampling, why MSE regression gives blurry outputs, uncertainty quantification.

## Connections

- **Builds on** [[autoencoder]] / [[u-net]] — the encoder-decoder architecture pattern recurs (DCGAN's generator is a decoder; pix2pix's generator is a U-Net), but the *training signal* is no longer reconstruction. The GAN substitutes the decoder's reconstruction loss with a discriminator's adversarial loss.
- **Builds on** [[binary-cross-entropy]] and [[sigmoid-function-nc|sigmoid function]] — the discriminator is a standard binary classifier; the GAN losses are BCE in a min-max wrapper.
- **Builds on** [[backpropagation]] — gradients flow from $\mathcal{L}_G$ through $D$ back into $G$ via the chain rule; the generator's output is just an activation map consumed by the discriminator.
- **Sets up week 8 (diffusion)** — diffusion models are another approach to the same problem (sample from $p_{data}$ given only training samples) but with a fundamentally different training objective: predict the noise added at each step of a forward diffusion process. They have largely displaced GANs as the state of the art for image generation since 2020.
- **Sets up later weeks (autoregressive, multimodal)** — text-to-image (covered briefly here as a cGAN application) becomes a major topic via diffusion + CLIP-style contrastive embeddings; autoregressive models cover language modelling end-to-end.

## Open questions

- **Mode collapse** — GANs sometimes find one or a few outputs that consistently fool $D$ and produce only those (a perfect "7" every time). Symptom: low diversity. Fixes (minibatch discrimination, Wasserstein loss, careful capacity balancing) are partial. The deeper question of *why* the JS-divergence-minimising equilibrium isn't always the one SGD finds is still active research.
- **Why latent-space interpolation is smooth** — we observe smooth morphs $G(\alpha z_1 + (1-\alpha) z_2)$ in DCGAN/StyleGAN, but there's no theoretical reason this should hold. The implicit smoothness of $G$ is a happy empirical accident.
- **Why $z$ gets ignored in some cGANs** — pix2pix-style models often learn to ignore the noise input and produce nearly deterministic outputs. The condition dominates; the noise contributes little to output diversity. Fixing this is the motivation for noise-injection techniques in newer architectures and for switching to conditional diffusion (where noise plays a more structural role in the model).

## Problem-set lessons

- **Q1 (regression for image generation):** Train a regressor to map digit class $x \in \{0, 1\}$ to a $4 \times 4$ image $y$ with MSE loss and multiple training examples per digit. The optimum is the *pixelwise average* of training images — fractional pixel values like $1/3$ and $2/3$ that don't look like any specific digit. The MMSE estimator collapses multimodal posteriors into their mean. The generative-model fix: sample from the posterior instead of averaging it.
- **Q2 (cGAN tensor sizes):** A discriminator on a batch of 4 colour $256 \times 256$ images takes input of shape $(4, 3, 256, 256)$ and outputs $(4, 1)$ — one real-vs-fake probability per image. Standard CNN classifier on the input side; the only conditional twist is that the input *is* an image (real or generated), and in cGAN the condition is concatenated with it.
- **Q3 (true/false statements about GANs):**
  - "Discriminator only used during training" → **True**: discarded at inference.
  - "Generator maximises $D$'s ability to distinguish" → **False**: generator *minimises* it (fools $D$). It's the discriminator that maximises distinguishing ability.
  - "Discriminator minimises probability of correctly classifying real data" → **False**: it *maximises* correct classification on both real and fake.
  - "To compute discriminator gradients we first need generator gradients" → **False**: the two are updated in *separate* SGD steps. When updating $D$, $G$'s parameters are fixed (no $\theta$ gradients needed). When updating $G$, gradients flow *through* $D$ (chain rule) but $D$'s parameters $\phi$ are fixed and not updated.
