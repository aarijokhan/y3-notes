---
week: 6
topic: "Unsupervised Learning: Autoencoders, Representation, Contrastive"
deck: NeuralComputation::Week-06
---

TARGET DECK
NeuralComputation::Week-06

## Supervised vs unsupervised

> [!question]- What is the structural difference between **supervised** and **unsupervised** learning?
>
> - **Supervised:** learn $f_\theta : \mathcal{X} \to \mathcal{Y}$ from labelled pairs $(x, y)$. Labels supply the loss target.
> - **Unsupervised:** learn $f_\phi : \mathcal{X} \to \mathcal{Z}$ from inputs alone — no $y$. The challenge is constructing a training signal without labels.

> [!question]- Why is unsupervised learning **possible at all** despite the absence of labels?
> Real images occupy a **vanishingly thin manifold** in pixel space — a $28 \times 28$ greyscale image has $256^{784} \approx 10^{1888}$ possible pixel grids, but meaningful images (digits, faces) are an infinitesimal fraction of that ocean. Data has *structure*, and unsupervised learning is the search for that structure: find the manifold of "real" data, ignore the noise.

## Autoencoders

> [!question]- What is an **autoencoder**, and what is its loss?
>
> - **Encoder** $f_\phi : \mathcal{X} \to \mathcal{Z}$ compresses $x$ into a latent code $z$.
> - **Decoder** $g_\theta : \mathcal{Z} \to \mathcal{X}$ reconstructs $\hat{x} = g_\theta(f_\phi(x))$.
> - **Loss** is reconstruction MSE:
> $$\mathcal{L}_{rec} = \frac{1}{d} \sum_{j=1}^{d} (x^{(j)} - \hat{x}^{(j)})^2$$
> The data supervises itself — the input is the target.

> [!question]- What is the **identity-function trap** in autoencoders?
> If $\dim(\mathcal{Z}) \geq \dim(\mathcal{X})$, the loss is trivially minimised by $f_\phi(x) = x$ and $g_\theta(z) = z$ — the network just copies the input through. Loss is zero, but no useful structure has been extracted; the latent code is just a relabelling of the raw pixels.

> [!question]- How does the **bottleneck** force an autoencoder to learn something useful?
> Set $\dim(\mathcal{Z}) = v < d = \dim(\mathcal{X})$. The encoder cannot copy — it has to throw information away. Reconstruction loss penalises wrong pixel values, so the encoder keeps the few features that explain the *most pixels at once* (skin tone, face shape, stroke thickness) and discards fine details. The bottleneck width is a hyperparameter trade-off: wider $\to$ better reconstruction but more identity-like; narrower $\to$ more abstract features, blurrier reconstruction.

> [!question]- Why does an autoencoder **cluster** MNIST digits in latent space without ever seeing class labels?
> The decoder must produce different reconstructions for different digits. If the encoder mapped a "0" and a "1" to the same point $z$, the decoder would receive identical input from both and could not produce different outputs. To keep reconstructions unambiguous, the encoder must spread different inputs apart and pull similar ones together — clustering by class is a forced consequence, not an explicit goal.

> [!question]- What does **disentanglement** mean in the context of an autoencoder, and how is it observed?
> Different latent dimensions ending up controlling different semantic factors of variation. Probe by holding all but one dimension fixed and sweeping the remaining one through the decoder — observe which physical attribute changes. For 2-D MNIST 7s: $z_1$ may rotate the digit, $z_2$ change stroke thickness. The factors emerge for free; we don't choose them and re-training with a different seed can produce different ones.

## Pretext tasks

> [!question]- What is a **pretext task** in self-supervised learning?
> A task whose label is **fabricated from the data itself** but whose solution requires understanding image content. Examples: predict the rotation that was applied; fill in a removed patch; predict the spatial arrangement of two patches (Doersch et al. 2015). After training, the prediction head is discarded; the encoder's features are the deliverable.

> [!question]- What is the **Clever Hans effect**, and what is the canonical pretext-task example?
> A model achieves high accuracy via a **spurious shortcut** that correlates with the label in training but doesn't reflect the intended concept. Doersch et al.'s context-prediction network solved patch arrangement by reading **chromatic aberration** — a colour-shift artefact at lens edges that gives away absolute position — instead of looking at image content. Named after the early-1900s horse who appeared to count but was actually reading his trainer's micro-expressions.

## Contrastive learning

> [!question]- What is the **SimCLR** training procedure?
>
> 1. Sample a minibatch of $N$ images. For each $x$, apply two random augmentations $t, t' \sim \mathcal{T}$ to produce two views $\tilde{x}_i, \tilde{x}_j$. The batch now has $2N$ data points and $N$ positive pairs.
> 2. Encode each view: $h = f(\tilde{x})$ (base encoder, typically ResNet-50). Project: $z = g(h)$ (small MLP head).
> 3. Apply contrastive loss to pull positive pairs together and push all other pairs apart.

> [!question]- Write the **SimCLR contrastive loss** for a positive pair $(i, j)$ and explain its parts.
> $$\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\mathrm{sim}(z_i, z_k)/\tau)}$$
>
> - $\mathrm{sim}(u, v) = u^\top v / (\|u\|\|v\|)$ is **cosine similarity**.
> - $\tau$ is the **temperature** hyperparameter.
> - **Numerator (pull):** the positive pair $z_i, z_j$ should be similar.
> - **Denominator (push):** all other $2N - 2$ pairs (negatives) should be dissimilar.

> [!question]- In SimCLR, after training, do you keep $h$ or $z$ for downstream tasks? Why?
> **Keep $h$** (the encoder's output, before the projection head). Throw away the projection head $g$. The intuition is that $g$ absorbs information loss specific to the contrastive task (e.g. forgetting rotation, colour) so that $h$ retains richer, more general-purpose features. Note this is the **opposite** of an autoencoder, where the bottleneck $z$ is the keeper.

> [!question]- A friend says: "to learn a good image representation, we should reconstruct the image — that's the only way to be sure no information was lost." What's wrong, and what does SimCLR show?
> Reconstruction *forces* the network to spend capacity on every pixel, including irrelevant background details. Most pixel-level differences (lighting, exact colour, background texture) are useless for downstream tasks like classification. Contrastive learning instead asks "tell same vs different objects apart" — the encoder learns to discard whatever an augmentation can change (lighting, crop, colour) and keep what is invariant (the actual content). SimCLR matches fully-supervised ResNet-50 accuracy on ImageNet — proving you don't need pixel-perfect reconstruction or labels.

## Latent representation disambiguation

> [!question]- Both autoencoders and SimCLR have a vector called the "**latent representation**". Are they the same? What do you keep in each case?
> **No** — they refer to different vectors:
>
> - **Autoencoder:** $z$ is the bottleneck. Keep $z$ — it is the compressed representation.
> - **SimCLR:** $h$ comes out of the encoder $f$; $z$ comes out of the projection head $g$. Keep $h$, discard $g$ (and therefore $z$).
>
> When slides say "representation" they typically mean $h$; when they say "latent code" they mean $z$. Context matters.

## What ties the week together

> [!question]- The week-6 self-supervised recipe in four steps.
>
> 1. **Pick a target you can compute from the data alone** (the input itself / patch position / augmentation invariance).
> 2. **Train a network to predict that target.**
> 3. **Throw away the prediction head; keep the encoder.**
> 4. **Use the encoder for downstream tasks** — semi-supervised classification, clustering, retrieval, fine-tuning.
>
> The proxy task is disposable; the encoder's features are the prize.

> [!question]- Why would adding **U-Net-style skip connections** to an autoencoder *defeat* its purpose?
> Skip connections let information flow around the bottleneck — the decoder no longer needs to rely on a meaningful $z$ because high-resolution features arrive directly from the encoder. The reconstruction can be near-perfect while the bottleneck $z$ encodes nothing useful, returning us to the identity-function trap with extra parameters. Skips are right for U-Net (where the goal is per-pixel prediction with both context and detail); wrong for an AE (where the goal is a compressed useful $z$).
