---
name: data-augmentation
description: Synthesising additional training examples by applying label-preserving transformations to existing ones — flips, rotations, brightness changes, noise, crops. Cheap regularisation that fights overfitting by giving the model variety it would otherwise need to collect more data to see.
type: concept
sources:
  - raw/week-05/w05-l12-transcript.txt
  - raw/week-05/w05-slides.pdf
  - raw/week-05/w5-notion.zip
status: stable
updated: 2026-04-26
---

*Real datasets are rarely big enough for deep networks. Acquiring more data is slow and expensive — especially in domains like medical imaging. Data augmentation generates new training examples for free by transforming existing ones in ways that preserve their label, forcing the model to learn the underlying concept rather than memorising specific pixels.*

## The idea

Take one training image labelled "Chinstrap penguin". Apply a label-preserving transformation — flip it horizontally, rotate it slightly, brighten it, add noise — and you have a new training example with the same label. The network has never seen this exact image before, but the answer is still "Chinstrap penguin". One example becomes ten, fifty, a hundred.

![[data-augmentation-penguins.png]]

Each augmented version is *not* a duplicate; from the network's perspective, the pixel values are different. Repeated across the training set, augmentation turns one fixed dataset into an effectively much larger one, drawn from a richer distribution of viewing conditions.

## Common transformations

For images, the standard augmentations are:

| Transformation | What it changes | Label-preserving for... |
|---|---|---|
| **Horizontal flip** | Mirrors left/right | Most natural images |
| **Rotation** | Tilts the image by a random angle | Most objects (within a small range — don't flip a "6" into a "9") |
| **Translation / shift** | Moves the object within the frame | Always (same content, different position) |
| **Scale / crop** | Zooms in/out | Most images |
| **Brightness / contrast** | Changes pixel intensity globally | Lighting-invariant tasks |
| **Gaussian noise** | Adds per-pixel random noise | Most images; mimics sensor noise |
| **Colour jitter** | Random hue/saturation shift | Colour-invariant tasks |

Applied randomly per example, per epoch, so the network rarely sees the same image twice — even from a fixed dataset.

## Why it generalises

The model that sees only the original photo can memorise its exact pixel values. The model that also sees flipped, rotated, brightened, and noisy variants cannot rely on any single pixel — it has to learn features (edges, textures, shapes) that survive all the transformations. Those *invariant* features are exactly the ones that generalise to genuinely new images, where lighting and pose will also vary.

In other words: augmentation forces the network to extract the *concept* of a penguin (beak, tuxedo pattern, posture) rather than the *coordinates* of one specific photograph.

> [!tip] TIP — Augmentation is online, not offline
> In modern frameworks, augmentation happens *on-the-fly* during training: the dataloader applies a fresh random transformation each time it loads an image. So the network sees a different version of the same source image at every epoch. This is much more efficient than pre-computing and storing many augmented copies.

## When transformations stop being label-preserving

The whole game depends on the augmented example having the *same* label as the original. That breaks down in subtle cases:

- **Rotation of digits.** Rotating a "6" by 180° gives a "9" — different label. Limit rotation to a small range.
- **Horizontal flip of asymmetric scenes.** Flipping text or road signs is not label-preserving (a flipped "STOP" sign is no longer a stop sign).
- **Colour jitter on medical images.** A colour change might convert a benign lesion's appearance into a malignant one. The transformation is no longer label-invariant.

Choose augmentations that match what the *true* invariances of the task are, not just whatever transformations are available.

## Augmentation in domains beyond images

The principle generalises:

- **Audio:** time-stretching, pitch shifts, adding background noise.
- **Text (NLP):** synonym replacement, back-translation (translate to another language and back).
- **Tabular data:** less common; small perturbations of continuous features sometimes help.

But it's most powerful in computer vision, where natural invariances (translation, rotation, lighting, viewpoint) are well-understood and many transformations are obviously label-preserving.

## Augmentation as regularisation

Data augmentation is a form of [[regularization]] — it reduces [[overfitting]] without an explicit penalty term. The mechanism is not "more data" in any literal sense (you didn't collect more), but rather "more diverse data": every augmentation forces the model to find features that are stable under that transformation, which by construction are more likely to generalise.

In the regulariser hierarchy:

- **Weight decay (L2):** penalise large weights directly.
- **Early stopping:** stop training before overfitting starts.
- **Dropout:** force redundant pathways inside the network. See [[dropout]].
- **Data augmentation:** force the *features* to be invariant to nuisance variations.

These stack — most modern training pipelines use several at once.

## Related

- [[overfitting]] — the failure mode that augmentation combats
- [[regularization]] — the broader category augmentation belongs to
- [[dropout]] — another regulariser, attacking the problem from inside the network instead of from the data
- [[transfer-learning]] — paired with augmentation when data is severely limited
- [[shift-invariance-equivariance]] — translation augmentation reinforces what the architecture already approximately gives for free

## Active Recall

> [!question]- You have 500 labelled images of a rare skin disease. You apply data augmentation (flips, small rotations, brightness changes) to grow this to 5000 effective training examples. Explain why this is *not* equivalent to having actually collected 5000 images, and what it does instead.
> Five thousand augmented copies of 500 photos still come from 500 underlying scenes — they don't capture variation in patient demographics, anatomy, lighting set-up, or disease presentation that you'd get from 5000 actually distinct cases. What augmentation gives you is *robustness to nuisance variation*: the model is forced to recognise the disease regardless of orientation or brightness, instead of memorising specific pixel patterns. It reduces overfitting; it does not cure the data shortage.

> [!question]- Why is data augmentation considered a form of regularisation, even though it doesn't add a penalty term to the loss?
> Regularisation is anything that reduces the effective capacity the model has to memorise training data. Augmentation does this by guaranteeing the model never sees the *same* example twice (each epoch shows fresh random transformations), so it can't memorise specific pixel values. To minimise the loss across all augmented variants, the model is forced to learn features that are invariant to the augmentations — which is exactly what good generalisation requires. The mechanism is different from L2 weight decay, but the effect (smaller train/test gap) is the same.

> [!question]- Why is "rotate a digit by an arbitrary amount" a bad augmentation for training a digit classifier on MNIST?
> It isn't always label-preserving. A "6" rotated 180° becomes a "9"; a "9" rotated 180° becomes a "6". If the augmentation can change the correct label, training with it tells the network conflicting things ("this image is a 6", "the same image rotated is a 6") and the network either fails to learn or learns to ignore the meaningful orientation. Augmentations must respect the task's true invariances; for MNIST that means small rotations only (e.g. ±15°), not arbitrary ones.

> [!question]- Augmentation is typically done "online" rather than "offline". What does that mean and why is it preferred?
> Online: each time an image is loaded during training, the dataloader applies fresh random transformations on the fly, so the network sees a slightly different version of every image every epoch. Offline: pre-compute a fixed expanded dataset of augmented images once, store them all on disk, and train on that. Online wins because it gives effectively unlimited variety (you don't run out of new transformations) without disk overhead, and because the random sampling of augmentations acts as additional stochastic regularisation. Offline is only useful when augmentation is computationally expensive enough to be worth pre-computing.
