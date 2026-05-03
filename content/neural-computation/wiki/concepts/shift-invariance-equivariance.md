---
name: shift-invariance-equivariance
description: Two distinct symmetry properties of CNNs. Shift equivariance (from convolution) means features track input shifts; shift invariance (from pooling and FC layers) means the final output is unaffected by input shifts.
type: concept
sources:
  - raw/week-04/w04-l11-transcript.txt
  - raw/week-04/w04-slides.pdf
  - raw/week-04/w4-notion.zip
status: stable
updated: 2026-04-26
---

*A cat is still a cat regardless of where it sits in the frame. CNNs handle this gracefully because their architecture has built-in symmetries: convolution layers are shift-equivariant (features move with the object), and pooling + FC layers eventually convert that into shift invariance (the final output is the same).*

## The two properties

Let $f$ be the function the network computes (input → output) and $S_v$ be the operator that shifts an image by vector $v$.

**Shift invariance:** $f(S_v \mathbf{x}) = f(\mathbf{x})$ — shifting the input doesn't change the output. The network gives the same answer regardless of where in the frame the content sits.

**Shift equivariance:** $f(S_v \mathbf{x}) = S_v f(\mathbf{x})$ — shifting the input shifts the output by the *same amount*. The output structure tracks the input's spatial structure.

These are different. Invariance throws away spatial information; equivariance preserves it. **Both are useful, and CNNs exhibit each in different parts of the architecture.**

## When to want which

- **Classification** wants invariance. "Is this a cat?" should yield "yes" regardless of where the cat appears. The output (a class label) has no spatial structure, so the network should be insensitive to spatial shifts of the input.
- **Segmentation** wants equivariance. "Which pixels are cat?" must move when the cat moves — the output is a spatial map and has to align with the input. Invariance would be wrong (you'd lose the cat's position).
- **Object detection** wants both: the *class* should be invariant (cat is cat anywhere), but the *bounding box* should be equivariant (the box moves with the object).

## Convolution is shift-equivariant — by construction

The convolution operation uses the *same* kernel at every spatial position. If you shift the input, the kernel still finds the same patterns, just in different output positions. Formally:

$$\text{conv}(\mathbf{w}, S_v \mathbf{x}) = S_v \, \text{conv}(\mathbf{w}, \mathbf{x})$$

This isn't a learned property — it falls out of the architecture's weight sharing. Every conv layer has it. Stacking conv layers preserves it: a stack of equivariant layers is itself equivariant.

This is the technical reason why CNNs handle shifted objects well. An MLP, by contrast, has *no* such symmetry: every (input pixel, hidden unit) pair has its own weight, so the network learns position-specific detectors. Move a digit "7" from the centre to a corner and the MLP sees a totally different input. The CNN just sees the same features in different locations.

## Pooling adds shift invariance

If a feature shifts by 1 pixel within a $2 \times 2$ pooling window, the max value is unchanged — the pool output is identical regardless of the small shift. After a $2 \times 2$ max pool, the network is approximately invariant to shifts of 1 pixel in the original feature map.

Stack several pool layers, and the invariant range grows: each pool layer absorbs shifts within its window. After 4 pool layers (each $2 \times 2$), the network is invariant to shifts of up to roughly $2^4 = 16$ pixels in the original input — comparable to the spatial reduction the pools have produced together.

Average pooling provides the same kind of invariance: averaging over a window is approximately unchanged by small shifts inside it.

## The FC head completes the invariance

The flatten + fully connected layers at the end of a CNN have no notion of spatial position whatsoever. By the time the spatial feature representation has been pooled down to (say) $7 \times 7$ and flattened to a 25{,}088-dim vector, the FC layer just sees a long vector of feature activations. Whatever spatial information remained at that stage is now mixed across all positions.

So the architecture's overall flow is:

1. **Conv stack** — shift-equivariant feature extraction. Spatial positions of features track input.
2. **Pool stack** — adds local shift invariance step by step, while still keeping a coarse spatial map.
3. **FC head** — discards spatial position entirely; produces a single class score.

The output is approximately shift-invariant. The intermediate feature maps remain (approximately) shift-equivariant.

## Why this matters: generalisation

A network without these symmetries has to learn shift-handling from scratch — by being shown the same object in many positions during training. This is data-hungry and never quite works.

A network *with* these symmetries by design generalises automatically: train on cats in the centre, and the network correctly classifies cats at the edge too, because the feature detectors aren't position-specific. This is one of the reasons CNNs need much less data per class than MLPs would.

The general lesson is that **architectural symmetries reduce the effective hypothesis space**. The network can't learn position-dependent classifiers because its weights are shared across positions — and that's a *good* constraint, since position-dependence isn't what we want anyway.

## Beyond translation: other invariances

Translation (shift) is the cleanest invariance because it's a direct consequence of weight sharing. But classification systems benefit from invariance to other transformations too:

| Transformation | Invariance ideal | How CNNs partially achieve it |
|---|---|---|
| **Translation** | Cat is cat anywhere in frame | Convolution + pooling (architectural) |
| **Scale** | Cat is cat whether near or far | Multi-scale features (deeper layers see larger receptive fields); data augmentation (resize during training) |
| **Rotation** | Cat is cat right-side-up or tilted | Mostly *learned* via data augmentation (random rotations during training) — not architectural |
| **Lighting / colour** | Cat is cat in bright or shadow | Normalisation (batch norm, input normalisation); data augmentation |

**Translation is the only invariance the architecture builds in for free.** The others are achieved either by training-data tricks (augmentation) or by post-hoc normalisation. Some specialised architectures (group-equivariant CNNs, capsule networks) build other invariances directly into the weights, but they're not standard.

## Which operations preserve, build, or break equivariance

Different layers play different roles in the equivariance vs invariance trade-off:

| Operation | Effect on shift equivariance |
|---|---|
| **Convolution (stride 1)** | Preserves equivariance exactly |
| **ReLU / element-wise activations** | Preserves equivariance (commutes with shift) |
| **Batch normalisation** | Preserves equivariance (per-channel, position-independent) |
| **Strided convolution ($S > 1$)** | Approximately preserves; only exact for shifts that are multiples of the stride |
| **Max pooling** | Approximately invariant to small shifts within the window; breaks exact equivariance |
| **Fully connected layer** | Destroys spatial structure; produces invariance |
| **Global average pooling** | Fully invariant — collapses spatial dims away |

The architectural recipe is now visible: stack equivariance-preserving operations (conv + activation + maybe batch norm) to extract spatial features, then break equivariance with pool/FC/global-avg layers when the task no longer needs position information.

## Designing for equivariance vs invariance

The distinction has practical implications when designing CNNs for non-classification tasks.

**For segmentation**, you want shift equivariance throughout. Strategies:

- Use **fully convolutional networks (FCNs)** — replace FC layers with conv layers, so the output is a spatial map.
- Avoid pooling, or pair pooling with upsampling (encoder-decoder structure like U-Net) so the output is at the same resolution as the input.

**For classification**, you want invariance at the end and equivariance in the middle. Standard CNN with conv → pool → FC is fine.

**For detection**, the architecture has multiple heads: one for class (invariant), one for bounding box coordinates (equivariant — the box should move with the object). The shared backbone is equivariant; the heads diverge.

## A small caveat: discrete shifts

The textbook shift-invariance/equivariance definitions assume continuous translations. In practice, images are discrete, and convolutions with stride > 1 or pooling break exact equivariance — they only commute with shifts that are multiples of the stride/pool size. So real CNNs are *approximately* shift-equivariant, not exactly. The approximation is good enough that the basic intuition holds, but it's worth knowing.

## Related

- [[convolution]] — the operation that gives shift equivariance through weight sharing
- [[pooling]] — the operation that adds shift invariance
- [[convolutional-neural-network]] — the architecture combining both
- [[multi-layer-perceptron]] — the alternative with no built-in symmetries; has to learn shift handling from scratch

## Active Recall

> [!question]- Define shift invariance and shift equivariance, and give an example task that needs each.
> **Shift invariance:** $f(S_v \mathbf{x}) = f(\mathbf{x})$ — shifting the input doesn't change the output. Example: image classification ("is this a cat?" — yes regardless of position). **Shift equivariance:** $f(S_v \mathbf{x}) = S_v f(\mathbf{x})$ — shifting the input shifts the output by the same amount. Example: semantic segmentation (the cat-pixel map must move when the cat moves).

> [!question]- Where does shift equivariance come from in a CNN, and where does shift invariance come from?
> **Equivariance** comes from convolution layers, specifically from weight sharing: the same kernel applied at every position means that shifting the input shifts the output by the same amount. **Invariance** comes from pooling (max within a window absorbs small shifts) and from the FC head (which discards spatial position entirely after flattening).

> [!question]- Why does a CNN classify a cat correctly whether it's in the top-left or bottom-right of the frame, but an MLP often fails?
> A CNN's conv layers are shift-equivariant: the same feature detectors apply at every position, so a cat anywhere produces the same features (just in different output positions). Pooling and the FC head then make the final output invariant to those positional differences. An MLP has a unique weight for every (input pixel, hidden unit) pair, so it implicitly learns position-specific detectors — it has to encounter cats at *every* position during training to handle them, and never quite generalises perfectly.

> [!question]- For a fully convolutional segmentation network (FCN), why is shift equivariance throughout important, and shift invariance harmful?
> Segmentation outputs a spatial map: each output pixel says what class is at the corresponding input pixel. If the cat moves in the input, the cat-pixels must move in the output by the same amount — that's equivariance. Invariance would say "output the same map regardless of where the cat is", which is the opposite of what we want. So FCNs avoid the FC head (which destroys spatial position) and either skip pooling or pair it with upsampling.

> [!question]- An object detector has a CNN backbone that produces feature maps, then two heads: one predicts class, one predicts bounding box. Which head wants shift invariance and which wants equivariance?
> **Class head:** invariance. "Is this a cat?" should give "yes" regardless of where the cat is — the answer doesn't depend on position. **Box head:** equivariance. The box coordinates $(x, y)$ should change in lockstep with the object's position. Both heads share the equivariant conv backbone; the class head pools/flattens into invariance, while the box head preserves spatial information.

> [!question]- Why is shift equivariance only approximate in real CNNs (not exact)?
> Real images are on a discrete pixel grid, and operations like strided convolution and pooling only commute with shifts that are multiples of the stride or pool size. Shifting by half a pixel doesn't compose cleanly with a stride-2 conv. So real CNNs are *approximately* shift-equivariant, exact only for shifts that align with the stride pattern. The approximation is usually good enough that the conceptual story holds.

> [!question]- Translation invariance comes from the architecture; what about scale or rotation invariance?
> Translation is the only invariance built into the convolution architecture (via weight sharing). Scale, rotation, and lighting invariance are not architectural — they're achieved indirectly: scale partially via multi-scale receptive fields (deeper layers see larger areas); rotation almost entirely via **data augmentation** (rotating training images during training so the network sees objects at every orientation); lighting via normalisation (input normalisation, batch normalisation). Specialised architectures (group-equivariant CNNs) can build rotation equivariance into the weights, but they're not standard.

> [!question]- Categorise these operations by their effect on shift equivariance: stride-1 convolution, ReLU, max pooling, fully connected layer, global average pooling.
> **Preserve equivariance:** stride-1 convolution (by construction), ReLU (element-wise so commutes with shift). **Approximately preserve / locally invariant:** max pooling (small shifts within the window are absorbed). **Destroy equivariance / produce invariance:** fully connected layer (collapses spatial structure), global average pooling (averages away all spatial dimensions). The CNN architecture starts equivariant in the conv stack and gradually transitions to invariance in the FC head.
