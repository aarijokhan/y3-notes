---
name: bayes-theorem
description: The relationship $p(y \mid x) = p(x \mid y) p(y) / p(x)$ that converts a forward (observation given truth) model into a backward (truth given observation) model. The probabilistic backbone of conditional generative models, classical statistical inference, and most "what's the answer given the evidence?" reasoning in ML.
type: concept
sources:
  - raw/week-07/w07-l15-transcript.txt
  - raw/week-07/w07-slides.pdf
status: stable
updated: 2026-04-26
---

*Bayes' theorem is the formula for inverting a probabilistic relationship. You have a model that predicts the noisy measurement $x$ given the true value $y$ — that's the **likelihood** $p(x \mid y)$. You have prior knowledge about how $y$ is distributed in the world — the **prior** $p(y)$. Bayes' theorem combines them to give you what you actually want — the **posterior** $p(y \mid x)$, which says "given that I observed $x$, what's the distribution over the true $y$?" The forward direction is mechanism; the backward direction is inference. Bayes goes from the first to the second.*

## Intuition pump: Steve the librarian (Kahneman & Tversky)

Before the formula, an intuition pump from Kahneman & Tversky:

> Steve is shy and withdrawn, invariably helpful but with very little interest in people or in the world of reality. A meek and tidy soul.

Is Steve more likely to be a librarian or a farmer?

Most people pick **librarian** — the description matches the stereotype. This is the wrong answer, and the reason it's wrong is the entire point of Bayes' theorem.

In the US there are roughly **20 farmers for every 1 librarian**. Even if Steve's description is, say, four times more *typical* of librarians than farmers, the sheer outnumbering still wins. Concretely:

- Imagine 210 people: 10 librarians, 200 farmers.
- Suppose 40% of librarians match the description: $10 \times 0.40 = 4$ matching librarians.
- Suppose 10% of farmers match the description: $200 \times 0.10 = 20$ matching farmers.
- Out of 24 matching people, **only 4 are librarians** — a posterior of $4/24 \approx 16.7\%$.

The description is *more characteristic* of librarians, but a Steve drawn from the population at random is *still much more likely to be a farmer*. The base rate (20-to-1) overwhelms the descriptive evidence.

The cognitive failure mode this exposes — judging probability by stereotype-fit while ignoring how rare the stereotype-fitting category is — is called **base-rate neglect**. It's the most common Bayesian mistake humans make, and the one Bayes' theorem is built to correct.

> [!warning] COMMON MISCONCEPTION — Base-rate neglect
> When given evidence that fits a hypothesis well, our intuition jumps to "the hypothesis is probably true." But "the evidence fits the hypothesis well" is the *likelihood* $p(\text{evidence} \mid \text{hypothesis})$ — and the question we actually want answered is the *posterior* $p(\text{hypothesis} \mid \text{evidence})$, which also depends on the *prior* $p(\text{hypothesis})$ — how rare the hypothesis is in the first place. Bayes' theorem is the correction: posterior ∝ likelihood × prior. Ignore the prior and you systematically overweight rare-but-stereotype-fitting explanations.

> [!tip] TIP — Concrete counts beat percentages
> Notice how much easier the Steve calculation became when phrased as "10 librarians, 200 farmers, 40% match, 10% match" rather than "$P(L) = 1/21, P(E \mid L) = 0.4, P(E \mid \neg L) = 0.1$." Humans reason about probability much better in concrete *counts* than in abstract probabilities. When working a Bayesian problem and your intuition rebels at the formula, drop to a hypothetical population of 100 or 1000, count the matching cases, and read off the answer.

## The geometry: restricted possibility space

The Steve calculation has a clean geometric reading. Picture all possibilities as a $1 \times 1$ square. Split it left/right by hypothesis: the left strip has width $P(H)$ (librarians), the right strip $P(\neg H)$ (farmers). Within each strip, shade the fraction matching the evidence: a tall band (40%) inside the librarian strip, a short band (10%) inside the farmer strip.

Now condition on the evidence. **Throw away everything unshaded.** What's left is two coloured rectangles: the librarian-and-evidence rectangle (small × tall = small area) and the farmer-and-evidence rectangle (large × short = larger area). The posterior $P(H \mid E)$ is the *librarian rectangle's share of the total remaining shaded area*.

> [!tip] TIP — Bayes is just the math of proportions
> The whole content of Bayes' theorem is geometric: restrict to the slice where the evidence is true, then read off the proportion where the hypothesis is also true. As Grant Sanderson puts it, *"the actual math of probability is really just the math of proportions, where turning to geometry is exceedingly helpful."* Every Bayesian calculation, no matter how complicated the formula looks, is a measurement of two areas in this restricted-space picture.

What changes belief is whether the evidence shrinks the two strips *unevenly*. If the evidence-band height is the same in both strips (40% of librarians match, 40% of farmers match), the proportion of librarians among matches equals the prior — the evidence is uninformative. If the bands are very different (40% vs 10%), the proportion shifts toward the strip with the taller band — but only by an amount that the prior widths permit.

> [!quote] From 3Blue1Brown
> *"Rationality is not about knowing facts. It's about recognizing which facts are relevant."* — Grant Sanderson, on what Bayes really teaches.

## The setup — formalising the geometry

The geometric picture above generalises to continuous variables. You have two random variables $x$ and $y$ — the "evidence" and the "hypothesis," to keep the Steve example in mind. The joint distribution $p(x, y)$ can be factored two ways:

$$p(x, y) = p(x \mid y) \, p(y) = p(y \mid x) \, p(x)$$

Equating the two factorisations and rearranging gives **Bayes' theorem**:

$$\boxed{p(y \mid x) = \frac{p(x \mid y) \, p(y)}{p(x)} = \frac{p(x \mid y) \, p(y)}{\int p(x \mid y') \, p(y') \, dy'}}$$

The denominator $p(x) = \int p(x \mid y') p(y') \, dy'$ (the **marginal likelihood** or **evidence**) is just the normalising constant that makes the posterior integrate to 1. For most practical purposes you can treat it as a constant and write the proportionality:

$$p(y \mid x) \propto p(x \mid y) \, p(y)$$

## The four named pieces

| Term | Name | Reading |
|---|---|---|
| $p(y)$ | **Prior** | What we believe about $y$ before seeing $x$ |
| $p(x \mid y)$ | **Likelihood** (or noise model) | How $x$ is produced given $y$ |
| $p(y \mid x)$ | **Posterior** | What we believe about $y$ after seeing $x$ |
| $p(x)$ | **Evidence** (marginal likelihood) | Normalising constant |

The directionality is the key insight: **likelihood goes forward (truth → observation), posterior goes backward (observation → truth)**. The forward direction is usually easy to model (you know how your sensor works); the backward direction is what you actually want at inference time.

## Worked example: penguin flipper length

A long-running ecological study of Antarctic penguins records each bird's flipper length. We have two measurements per bird:

- $y$: actual flipper length (mm), measured by hand.
- $x$: flipper length estimated from a computer-vision (CV) system on remote camera footage.

The CV system is noisy — it doesn't always nail the right answer. We want to deploy it without humans in the loop, but we need to understand how trustworthy its measurements are.

### Prior: $p(y)$

Histogram all the *true* flipper lengths $y$ in the dataset. You get something roughly Gaussian centred at $\sim 190$ mm. This is **prior knowledge** about the species — what penguin flipper lengths look like in the wild, regardless of any specific measurement.

The expected value is $\mathbb{E}_{p(y)}[y] \approx 190$ — your best guess about an unknown penguin's flipper length *before* any measurement.

### Likelihood: $p(x \mid y)$

Now condition on a particular true value, say $y = 192$. Filter the dataset to *only* the rows where $y = 192$, and histogram the corresponding $x$ values. You get a distribution centred at 192 with some spread — the CV system on average gets it right but with some noise.

This is the **likelihood** (also called the **noise model** or **forward model**): given the true value, what does the sensor produce? Typically zero-centred noise: $\mathbb{E}_{p(x \mid y)}[x] = y$ if the sensor is unbiased.

### Posterior: $p(y \mid x)$

Now the question we actually care about: the CV system measured $x = 201$. What's the distribution over the *true* $y$?

Filter the dataset to *only* rows where $x = 201$ (the CV measurement), histogram the corresponding $y$ values. The result is the **posterior** — not centred at 201, but somewhere between 201 and the prior mean of 190.

Why not centred at 201? Because two effects fight each other:

- The **likelihood** says "if $y = 201$, the most likely $x$ value is 201 — so our best guess given $x = 201$ is $y = 201$."
- The **prior** says "very few penguins have $y = 201$; many more have $y \approx 190$ — so most observations of $x = 201$ probably came from a true $y$ closer to 190 with noise that pushed the measurement up."

Bayes' theorem combines them. The posterior pulls the likelihood-only estimate (201) back toward the prior mean (190) — a phenomenon called **shrinkage** or **regression to the mean**. The exact location depends on the relative widths of prior and likelihood.

> [!info] ASIDE — Why $\mathbb{E}_{p(y \mid x)}[y] \neq x$
> Naively you'd think "the sensor reported 201, the noise has zero mean, so the best estimate is 201." That's the *maximum-likelihood* estimate — uses only the likelihood. Bayes does better when you have prior information: rare values of $y$ are *less likely* to have produced a given $x$ than common values, even if the noise model treats them symmetrically. The MAP estimate, $\arg\max_y p(y \mid x)$, takes the prior into account and pulls toward the prior mode.

## Why this matters for generative modelling

The link to [[conditional-generative-model|conditional generative models]] is direct: a conditional generative model learns

$$\hat{p}_\theta(y \mid x) \approx p(y \mid x)$$

— the **posterior** distribution. The neural network *is* a parameterised approximation to Bayes' theorem.

Examples that fit this template:

| Task | $x$ (condition) | $y$ (target) |
|---|---|---|
| Image colourisation | Greyscale image | Colour image |
| Super-resolution | Low-resolution image | High-resolution image |
| Inpainting | Image with hole | Filled image |
| Semantic-map → photo (pix2pix) | Segmentation mask | Photograph |
| Sketch → photo | Line sketch | Realistic image |
| Text-to-image | Caption | Generated image |

In every case, multiple $y$ values are plausible for a given $x$ — there's no unique colourisation of a greyscale photo, no unique high-res reconstruction of a low-res image. The posterior $p(y \mid x)$ captures this *distribution* of plausible answers, not a single point estimate.

## The MMSE estimate (and why MSE regression is blurry)

A classical exercise: given a set of noisy measurements $\mathbf{x} = (x_1, \ldots, x_n)$ of an unknown $\hat{x}$, find the estimate that minimises the squared-error loss

$$\hat{x}^* = \arg\min_{\hat{x}} \sum_i (\hat{x} - x_i)^2$$

Setting the derivative to zero gives the analytical solution: $\hat{x}^* = \frac{1}{n} \sum_i x_i$, the **sample mean**. The mean is the **MMSE estimate** (Minimum Mean Squared Error) — it's the estimator with the smallest expected squared error.

Generalising to the conditional setting: a regression network trained with MSE loss to predict $y$ from $x$ will, at convergence, output the **conditional mean** $\mathbb{E}_{p(y \mid x)}[y]$ — the average $y$ over all plausible explanations of the observed $x$.

This is what the generator loss in conditional [[generative-adversarial-network|GANs]] is fixing. If multiple $y$'s are plausible (e.g. multiple realistic colour assignments for a greyscale photo), the conditional mean is a *blurry compromise* between them — neither one nor the other, but their pixel-wise average. That's why an MSE-regression colouriser produces washed-out, low-saturation images, and why a cGAN does much better: it samples from the posterior instead of averaging over it.

> [!question]- A friend says: "We don't need Bayes' theorem — we have neural networks. Just train an MLP to map from $x$ to $y$ and you're done." Why is this a partial answer?
> Two reasons. First, **the MLP gives you a point estimate, not a distribution.** Trained with MSE it gives the conditional mean $\mathbb{E}[y \mid x]$; trained with cross-entropy it gives a categorical distribution over a fixed output set. Neither captures the full posterior $p(y \mid x)$ for continuous, structured outputs (images, audio). Second, **the MLP collapses uncertainty to a single output.** When multiple $y$'s are plausible (multiple colourings, multiple high-res reconstructions), the MLP averages them and produces a blurry compromise. A *generative* approach to conditional modelling — cGAN, conditional diffusion, conditional VAE — samples from the posterior, producing one plausible $y$ at a time and letting you generate diverse outputs by re-sampling. Bayes' theorem is the right framing because it makes the *uncertainty over $y$* a first-class object, not a thing to be averaged out.

> [!question]- "Steve is meek and tidy" sounds 4× more typical of librarians than farmers. So why is a randomly described Steve still much more likely to be a farmer than a librarian?
> Because the *prior* dominates. There are $\sim 20$ farmers per librarian in the population, so the farmer "strip" of possibility space is 20× wider than the librarian strip. The likelihood ratio (4× more typical of librarians) gets multiplied by the prior ratio (20× more farmers) — and the prior wins: $20 / 4 = 5$, so a Steve-matching person is still 5× more likely to be a farmer. The cognitive trap is to focus on "how well does the description fit?" (likelihood) and ignore "how rare is the category being described?" (prior). Bayes' theorem is the corrective — posterior ∝ likelihood × prior — and base-rate neglect is the failure mode it's designed to fix.

> [!question]- A penguin's true flipper length is $y = 192$ mm. The CV system measures $x$. What's $\mathbb{E}_{p(x \mid y=192)}[x]$, and why is this not the same question as $\mathbb{E}_{p(y \mid x=192)}[y]$?
> $\mathbb{E}_{p(x \mid y=192)}[x] = 192$ if the noise model is zero-centred (the sensor is unbiased) — given the truth, the average measurement equals the truth. But $\mathbb{E}_{p(y \mid x=192)}[y] \neq 192$ in general — given the measurement, the posterior over $y$ is shrunk toward the prior. The first is a property of the *forward* (likelihood) model; the second is the *backward* (posterior) inference and depends additionally on the prior $p(y)$. Confusing these two is one of the most common probabilistic errors — they live on opposite sides of Bayes' theorem.

## Connections

- [[conditional-generative-model]] — the direct application: conditional GMs learn $\hat{p}_\theta(y \mid x) \approx p(y \mid x)$, a parameterised posterior.
- [[generative-model]] — unconditional GMs learn $p_\theta(x) \approx p_{data}(x)$, the marginal; conditional GMs add a condition.
- [[maximum-likelihood-estimation-nc|maximum likelihood estimation]] — choosing $\theta$ to maximise $p(\text{data} \mid \theta)$ uses only the likelihood, ignoring the prior; the *non-Bayesian* counterpart of MAP estimation.
- [[loss-function]] — MSE regression converges to the conditional *mean* (an MMSE estimate, integrating over the posterior); generative models sample *from* the posterior, preserving multimodality. The choice of loss determines which.
