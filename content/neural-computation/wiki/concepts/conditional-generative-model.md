---
name: conditional-generative-model
description: A generative model that produces samples *given* a condition $x$ — colourise this image, super-resolve this photo, generate a face matching this description. Mathematically, learns the posterior $p_\theta(y \mid x) \approx p(y \mid x)$ and samples from it. cGAN, pix2pix, super-resolution, text-to-image, and inpainting are all instances.
type: concept
sources:
  - raw/week-07/w07-l15-transcript.txt
  - raw/week-07/w07-slides.pdf
  - raw/week-07/w07-problem-set.pdf
status: stable
updated: 2026-04-26
---

*Unconditional generative models produce a sample from nothing — "draw any face." Conditional generative models produce a sample tied to a specific input — "colourise this greyscale photo of a cat." The shift is small in architecture (feed the condition into both generator and discriminator) but huge in capability: every interesting image-to-image task — colourisation, super-resolution, semantic-map-to-photo, sketch-to-photo, inpainting, text-to-image — is a conditional generative problem. The probabilistic framing is exactly [[bayes-theorem|Bayesian posterior estimation]].*

## The posterior framing

An unconditional [[generative-model]] learns the marginal $p_\theta(x) \approx p_{data}(x)$.

A **conditional** generative model learns the posterior

$$\hat{p}_\theta(y \mid x) \approx p(y \mid x)$$

— the distribution over targets $y$ given a condition $x$. To generate: input a condition $x$, sample $\tilde{y} \sim \hat{p}_\theta(y \mid x)$.

This is direct [[bayes-theorem|Bayesian]] inference parameterised by a neural network. The condition $x$ is the observation; $y$ is the latent quantity we want to infer; the network learns the posterior.

## Tasks that fit this framing

| Task | Condition $x$ | Target $y$ |
|---|---|---|
| Image colourisation | Greyscale image | Colour image |
| Super-resolution | Low-res image | High-res image |
| Inpainting | Image with masked region | Filled image |
| Semantic-map → photo | Class-coloured segmentation mask | Realistic photo |
| Sketch → photo | Outline drawing | Realistic photo |
| Denoising | Noisy image | Clean image |
| Text-to-image | Caption (string) | Generated image |
| Class-conditional generation | Class label (one-hot) | Image of that class |
| Image-to-segmentation | Photo | Posterior over segmentations |

In every case, **multiple $y$'s are plausible for a given $x$.** A greyscale photo of a flower could be coloured red or yellow or pink. A low-res image has many compatible high-res reconstructions. The posterior $p(y \mid x)$ captures this multimodality; the task is to sample from it, not to produce a single point estimate.

## Why regression fails (the blurry-output problem)

The most natural-seeming approach: train a regression network $f_\theta : \mathcal{X} \to \mathcal{Y}$ to map $x$ to $y$, with [[loss-function|MSE loss]] between the predicted $\hat{y} = f_\theta(x)$ and the ground-truth $y$.

This **systematically fails** for tasks with multimodal posteriors. The reason is mathematical: MSE-regression converges to the conditional mean,

$$f_\theta^*(x) = \mathbb{E}_{p(y \mid x)}[y]$$

— the average of all plausible $y$'s for a given $x$. When the posterior has several modes (e.g. flower-could-be-red OR flower-could-be-yellow), the mean lands *between* them — a washed-out grey-pink that's neither, with low saturation and blurred edges.

> [!info] ASIDE — Worked example from the week-07 problem set
> Train a regressor to map digit $x \in \{0, 1\}$ to a $4 \times 4$ pixel image $y$ of that digit. The training set has multiple distinct hand-drawn examples per digit. After training to optimum (minimum MSE), what's the network's output for $x = 1$? **The pixelwise average of all training images of "1"** — fractional pixel values like $1/3$ and $2/3$ that don't appear in the training set, looking nothing like any specific "1" but compromising between all of them. Same for $x = 0$. The MMSE-optimal output is a blur of every plausible target, because that minimises the squared distance to all of them simultaneously. A conditional generative model picks *one* plausible $y$ instead of averaging.

The fix: use a **generative** approach. Sample from the posterior instead of averaging over it. Every sample is a single plausible $y$; running inference multiple times produces *diverse* outputs (different colourings, different reconstructions), each individually realistic.

## Conditional GAN (cGAN)

The most direct extension of the [[generative-adversarial-network|GAN]] framework to the conditional setting (Mirza & Osindero 2014; pix2pix by Isola et al. 2018 made it practical for image-to-image translation):

- **Generator** $G_\theta(z, x) : \mathcal{Z} \times \mathcal{X} \to \mathcal{Y}$ — takes both a latent noise vector $z \sim p(z)$ *and* the condition $x$, produces $\tilde{y} = G_\theta(z, x)$.
- **Discriminator** $D_\phi(x, y) : \mathcal{X} \times \mathcal{Y} \to [0, 1]$ — takes both the condition *and* the candidate $y$ (real or fake), outputs probability of being a real $(x, y)$ pair.

The training pairs are $(x, y_{real})$ from the dataset (real positive examples) and $(x, G_\theta(z, x))$ (fake examples sharing the same $x$). Critically, the discriminator sees the *condition*, not just the candidate target — it learns to detect mismatched pairs (e.g. a sketch of a shoe paired with a generated handbag) as fake.

Loss is the standard GAN objective with everything conditioned on $x$:

$$\mathcal{L}_D = -\mathbb{E}_{x, y \sim p_{data}}[\log D_\phi(x, y)] - \mathbb{E}_{x \sim p_{data}, z \sim p(z)}[\log(1 - D_\phi(x, G_\theta(z, x)))]$$

$$\mathcal{L}_G = -\mathbb{E}_{x \sim p_{data}, z \sim p(z)}[\log D_\phi(x, G_\theta(z, x))]$$

After training, generate by inputting a condition and sampling fresh $z$ for each desired output. Different $z$'s give different plausible $y$'s — capturing the posterior's multimodality.

### pix2pix (Isola et al. 2018)

A specific cGAN architecture for paired image-to-image translation:

- **Generator** is a [[u-net|U-Net]] encoder-decoder taking the condition image $x$ as input.
- **Discriminator** is a "PatchGAN" — classifies overlapping image patches as real-or-fake rather than the whole image, encouraging local texture realism.
- **Loss** combines the cGAN adversarial loss with an additional **L1** reconstruction loss between $\tilde{y}$ and the ground truth $y$. The L1 term keeps the output close to the target on average; the GAN term ensures the output looks *real* (sharp, plausible textures), not just average-close.

Pix2pix established the template for paired image-to-image translation: paired training data, U-Net generator, PatchGAN discriminator, hybrid L1+GAN loss. Later work (CycleGAN, etc.) extended this to unpaired translation.

## Why the L1 + GAN hybrid loss?

L1 alone (or L2) gives a blurry conditional-mean output — same problem as MSE regression. GAN alone produces realistic-looking but possibly off-target outputs (the generator might produce a beautiful red shoe when the input sketch demands a specific brown one). The combination:

- **L1 term** anchors the output near the ground truth (correct overall colour, layout).
- **GAN term** sharpens textures and details to *realistic* (not average) values.

The result: outputs that are both faithful to the input and crisp/realistic.

## Conditional models naturally capture uncertainty

A nice feature of the generative framing: because $G_\theta(z, x)$ depends on the random $z$, running the generator multiple times on the same condition $x$ produces *different* outputs $\tilde{y}_1, \tilde{y}_2, \ldots$. The variation across these samples is an estimate of the posterior's uncertainty.

In a microscopy-denoising example (Prakash et al. 2020), a conditional generative model trained to denoise low-photon images produces outputs where:

- High-confidence structures (real cell boundaries) are *consistent* across samples — the posterior is sharply peaked there.
- Low-confidence regions (background noise) *vary* across samples — the posterior is broad there.

Averaging the samples gives a smoother estimate; the variance gives an uncertainty map. This is impossible with a plain MSE-regression network, which collapses both into a single output.

## How $z$ behaves in conditional models

The latent $z$ in a cGAN plays a different role than in unconditional GANs. It's still sampled from a fixed prior, but in many cGAN setups (especially pix2pix) the network learns to *largely ignore $z$* and treat the conditioning $x$ as the dominant input — producing nearly the same output regardless of $z$. This makes the model effectively deterministic, which is fine for tasks like colourisation where we want a single high-quality output, but undermines the diversity benefit of the generative framing.

Modern conditional generative models (conditional diffusion, conditional VAEs) handle this differently — for example, by injecting noise at multiple layers or using stronger latent regularisation — but the core principle is the same: $x$ controls *what* gets generated, $z$ provides the variation needed to express the posterior.

> [!question]- A friend trains an MSE-regression network to colourise greyscale flower photos. Outputs come out muted and washed-out — the reds aren't red, the yellows aren't yellow. What's happening, and what should they switch to?
> They are seeing the [[bayes-theorem|MMSE-regression problem]] in action. A specific greyscale flower photo could be a red flower, a yellow flower, a pink flower — multiple plausible colourings, all equally consistent with the greyscale input. MSE-regression converges to the *conditional mean*, which is the pixelwise average of all those plausible colourings — a desaturated, neutral-toned compromise. The fix is a **conditional generative model** (cGAN with L1+GAN loss like pix2pix, or a conditional diffusion model). The generative loss samples *one* plausible colouring at a time instead of averaging across all of them, so individual outputs come out vivid and crisp. Running inference multiple times produces *diverse* colourings of the same input, each individually realistic — which is the right uncertainty behaviour for a problem with no unique answer.

> [!question]- The discriminator in a cGAN sees both the condition $x$ and the candidate $y$, instead of just $y$ alone. Why is this critical?
> Without the condition, the discriminator could only check "is this $y$ a realistic-looking image?" — the generator could then learn to produce *any* realistic image regardless of $x$ (e.g. always output a beautiful brown handbag, regardless of which sketch was input) and still fool the discriminator. With the condition, the discriminator checks "is this $(x, y)$ a realistic *pair*?" — punishing the generator for outputs that don't match the input. The condition forces the generator to actually use $x$ as a constraint, not just as a starting noise source. This is what turns a generative model into a *conditional* one in the proper sense.

## Sequence-to-sequence: the transformer as conditional GM

For sequences, the canonical conditional generative model is the **encoder-decoder [[transformer]]**. The condition $x$ is the input sequence (English sentence, audio waveform, source code, …); the target $y$ is the output sequence (French translation, text transcript, etc.). The decoder learns the autoregressive posterior

$$\hat{p}_\theta(y \mid x) = \prod_{i=1}^{|y|} \hat{p}_\theta(y_i \mid y_1, \ldots, y_{i-1}, x)$$

— the chain-rule factorisation of the conditional, with each step parameterised by the decoder. The encoder's latent code carries $x$; the decoder's [[cross-attention|cross-attention]] sub-layer is the conduit through which $x$ enters every output step. Generation is [[autoregressive-model|autoregressive]]: sample one token at a time conditioned on $x$ and previously sampled tokens.

This unifies machine translation, summarisation, speech-to-text (Whisper), and text-to-image text branches under one framing — all are "given $x$, sample $y$ from the learned posterior". The pix2pix-style cGAN handles the same problem for image $\to$ image; the transformer handles the same problem for sequence $\to$ sequence; conditional [[diffusion-model|diffusion]] handles it for noise $\to$ image conditioned on text. Different architectures, same conditional-generative framing.

## Connections

- **Built on** [[generative-adversarial-network]] — cGAN is a conditional extension of the GAN framework; same min-max game, same losses, same training algorithm, with $x$ added as input to both networks.
- **Built on** [[u-net]] — pix2pix's generator is a U-Net; the encoder-decoder + skip connections architecture is well-suited to image-to-image tasks where the output is structurally similar to the input.
- **Built on** [[bayes-theorem]] — conditional GMs learn $\hat{p}_\theta(y \mid x) \approx p(y \mid x)$, a parameterised posterior.
- **Implemented by** [[transformer]] for sequences — encoder-decoder transformers are the canonical conditional sequence-to-sequence GMs (translation, summarisation, speech-to-text).
- **Family member of** [[generative-model]] — the conditional branch.
- **Improvement over** [[loss-function|MSE regression]] for posterior estimation — regression converges to the conditional mean (blurry); generative models sample from the posterior (sharp, diverse).
