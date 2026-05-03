---
name: u-net
description: An encoder-decoder CNN architecture for semantic segmentation. The encoder downsamples to capture context; the decoder upsamples to produce a per-pixel output. Skip connections concatenate high-resolution encoder features into the decoder, restoring spatial detail lost during pooling.
type: concept
sources:
  - raw/week-05/w05-l13-transcript.txt
  - raw/week-05/w05-slides.pdf
  - raw/week-05/w5-notion.zip
status: stable
updated: 2026-04-26
---

*The architecture that made dense prediction practical. Take an encoder that compresses the image into an abstract feature representation, mirror it with a decoder that grows back up to full resolution, and connect each encoder layer directly to its decoder counterpart. The skip connections give the decoder access to fine-grained spatial detail that pooling threw away. Originally designed for biomedical image segmentation, now the dominant template for any per-pixel vision task.*

## The problem U-Net solves

For classification, a CNN takes an image and produces a single class label. The network can pool and downsample aggressively, because the output is one number per image. For **semantic segmentation** the output is one class label *per pixel* — a full $H \times W$ label map for an $H \times W$ input.

This creates a tension:

- **Pooling helps the network see context** — global features, large-scale shapes, object identity — by pooling spatial detail into compact feature maps.
- **But segmentation needs fine spatial detail** — pixel-precise edges of objects, exactly where one tissue ends and another begins.
- **Without pooling, the network can't see the big picture** — it stays at full resolution but every neuron's receptive field is tiny, so it can't recognise larger structures.

You can't have it both ways with a single forward chain. U-Net resolves the tension by using *both* a path that pools (for context) and *another* path that preserves fine detail (for location), then combining them.

> [!tip] TIP — Forest *and* trees
> Standard CNNs make you choose: pool aggressively and you see the forest (the object as a whole) but lose the trees (where its edges are); refuse to pool and you see the trees but never the forest. Segmentation needs both — to know it's a tumour *and* to know exactly where its boundary is. U-Net's encoder sees the forest; its skip connections preserve the trees; its decoder combines the two views into a per-pixel answer.

## The U-shaped architecture

The architecture, drawn schematically, is a U:

- **Left arm (the encoder, "contracting path"):** a standard CNN. Each level applies a couple of $3 \times 3$ convolutions, then $2 \times 2$ max pooling to halve spatial size. As resolution drops, the channel count grows: $64 \to 128 \to 256 \to 512 \to 1024$. By the bottom of the U, the feature map is small spatially but rich semantically — it knows *what* is in the image.

- **Bottom of the U (the bottleneck):** the deepest, most abstract feature representation. Spatial size is small (say $32 \times 32$ from a $572 \times 572$ input), channel count is large (1024). This is where the network has the broadest view but the least spatial precision.

- **Right arm (the decoder, "expanding path"):** mirror image of the encoder. Each level uses [[upsampling]] (or transposed convolution) to double spatial size, then a couple of convolutions. As resolution grows, channel count drops back down: $1024 \to 512 \to 256 \to 128 \to 64$. By the top, the spatial size matches the input (or close to it) and channel count is small.

- **Final layer:** a $1 \times 1$ convolution to turn the final feature map into a per-pixel class score map of shape $H \times W \times C$, where $C$ is the number of classes. An argmax across the channel dimension gives the predicted class for each pixel.

## The skip connections

The genius of U-Net is the **skip connections from encoder to decoder**, drawn as horizontal arrows across the U.

When the decoder upsamples a feature map, the result is *blurry* — it knows the object is roughly here, but the edges are smeared because pooling discarded fine spatial detail in the encoder. The fix: copy the corresponding *high-resolution* feature map from the encoder side and **concatenate** it to the upsampled decoder feature map (along the channel dimension), then apply a couple of convolutions to mix them.

Now the decoder has, at every resolution level:

1. **Deep, abstract features** flowing up from the bottleneck — these tell it *what* the object is.
2. **High-resolution features** flowing across from the corresponding encoder level — these tell it *where exactly* the boundaries are.

Subsequent convolutions learn to combine the two sources into a sharper output. The result is segmentation that's both semantically correct (the network correctly identifies cells, tumours, road lanes) and spatially precise (the boundaries align with real edges in the image).

> [!info] ASIDE — Concatenation vs addition
> [[residual-connection|ResNet]]'s skip connections **add** the input to the output: $y = F(x) + x$. U-Net's skip connections **concatenate** the encoder feature map to the decoder feature map along the channel dimension: $y = [F(x), x]$. The role is similar (preserve information that would otherwise be lost), but the mechanism is different. Concatenation doubles the channel count and lets subsequent convolutions decide how to mix the sources; addition keeps the channel count fixed and forces the layers to learn a residual correction. Both are called "skip connections" in casual conversation; the distinction matters when reading architecture diagrams or implementing one.

## The encoder-decoder framing

U-Net is one example of a more general pattern: the **encoder-decoder** architecture.

- The **encoder** transforms the input into some compact, abstract representation — encoded features.
- The **decoder** transforms this representation back into a full-resolution output of the desired form — segmentation map, generated image, transcribed text.

This pattern recurs across deep learning:

- **Autoencoders** for unsupervised representation learning (encoder + decoder, target = input).
- **Variational autoencoders** for generative modelling.
- **Transformers** for sequence-to-sequence tasks (encoder + decoder, target = output sequence).
- **Image-to-image translation networks** for tasks like colourisation, style transfer, super-resolution.

In all cases the bottleneck is where the most abstract representation lives. In U-Net specifically, the addition of skip connections distinguishes it from a plain encoder-decoder by making the decoder's job easier.

## Original cropping and modern variants

The original U-Net (Ronneberger et al., 2015) used **valid convolutions** (no padding) throughout, which meant each convolution shrank the spatial size slightly. As a result, the output of the original U-Net was *smaller* than the input — only the central region of the prediction was used; the borders were discarded. The skip connections also required **cropping** the encoder feature maps to match the (slightly smaller) decoder spatial size before concatenation.

Modern U-Net variants typically use **same convolutions** (with padding) so that the output matches the input size exactly, and skip connections concatenate without cropping. The original cropping is a historical detail you'll see in some diagrams.

## U-Net in practice

The architecture has been wildly successful in:

- **Medical imaging** — its original domain. Cell nuclei segmentation, tumour boundaries, organ segmentation, retinal vessel extraction. Modest dataset sizes are typical here, and U-Net's combination of context (via the encoder) and detail (via skip connections) is well-suited.
- **Self-driving perception** — semantic segmentation of road, lane, vehicle, pedestrian classes per pixel.
- **Generative models** — diffusion models for image generation use U-Net as the noise-prediction network.
- **Earth observation** — segmenting satellite imagery into land-use categories.

Often paired with [[transfer-learning|transfer learning]]: use an ImageNet-pretrained CNN (e.g., ResNet) as the encoder, train the decoder from scratch on the target segmentation task. This dramatically reduces the data requirement.

## Related

- [[convolutional-neural-network]] — U-Net is a CNN architecture; the encoder is a standard CNN backbone
- [[upsampling]] — the operation used in the decoder to grow feature maps back up
- [[pooling]] — the operation used in the encoder to shrink feature maps
- [[residual-connection]] — different flavour of skip connection (addition, not concatenation), often combined with U-Net
- [[shift-invariance-equivariance]] — segmentation needs *equivariance* (output shifts with input), not just invariance, which is why fully convolutional designs work
- [[transfer-learning]] — pre-trained encoders are commonly used to bootstrap U-Net training on small medical datasets

## Active Recall

> [!question]- A standard classification CNN aggressively pools the image down to a small spatial size before its FC head. Why doesn't this work for semantic segmentation, and what's U-Net's solution?
> Pooling discards spatial precision — by the time you're at a $7 \times 7$ feature map, you've thrown away exactly the boundary information segmentation needs. Refusing to pool is also bad: without pooling, neurons have small receptive fields and can't see the big picture, so they can't recognise objects. U-Net's solution is to pool *and* preserve the discarded detail: the encoder pools (for context), the decoder upsamples back to full resolution (for output size), and skip connections from encoder to decoder feed the high-resolution detail directly into the decoder at each level. The decoder gets both context (from below) and location (from across).

> [!question]- What is the role of the skip connections in U-Net? Why are they essential for accurate segmentation?
> They preserve high-resolution spatial information that would otherwise be lost. The encoder discards spatial detail through repeated pooling — by the bottleneck, the network knows *what* is in the image but not *exactly where*. Skip connections feed the high-resolution feature maps from each encoder level directly to the corresponding decoder level. The decoder, when upsampling, has both the abstract semantic features from below and the spatially precise features from across. Without skip connections, the upsampled output would be semantically correct but spatially blurry — segmentation boundaries wouldn't align with real edges.

> [!question]- How are U-Net skip connections different from ResNet skip connections, and why might you choose one over the other?
> ResNet skip connections **add** the input to the layer output: $y = F(x) + x$. They preserve the channel count and force the layer to learn a residual correction. U-Net skip connections **concatenate** the encoder feature map onto the decoder feature map along the channel dimension: $y = [F(x), x]$. They double the channel count and let subsequent convolutions learn how to mix the two sources. ResNet's role is to ease optimisation in deep networks (gradient highway); U-Net's role is to merge two distinct kinds of information (deep + high-resolution). In segmentation, U-Net's concatenation is more flexible because the convolutions afterwards can choose any combination of the two streams; in classification, ResNet's addition is enough because there's only one stream of information.

> [!question]- Why is the term "encoder–decoder" appropriate for U-Net, and where else does this pattern appear?
> The encoder transforms the input into a compact, abstract representation (encoding); the decoder transforms that representation back into a desired output form (decoding). In U-Net specifically, the encoded form is a small spatial / large channel feature map; the decoder transforms it back into a per-pixel segmentation map of the original spatial size. The pattern recurs across deep learning: autoencoders for representation learning, transformers for sequence tasks, generative models like VAEs and diffusion models. U-Net is distinguished from a plain encoder-decoder by its skip connections, which directly bridge the two halves at every resolution level.

> [!question]- The original U-Net produced an output smaller than its input because it used valid (unpadded) convolutions. What does this mean for the segmentation map, and how do modern U-Net variants typically address it?
> With valid convolutions, every $3 \times 3$ convolution shrinks the spatial size by 2 (one pixel lost on each side). After many such operations, the decoder's output is smaller than the encoder's input — the original U-Net's $572 \times 572$ input produced a $388 \times 388$ output. Only the central region of the input is segmented; the borders are unlabelled. Modern variants use **same convolutions** (with padding) so spatial size is preserved through each conv, and the output matches the input exactly. This also simplifies the skip connections: no cropping required to match shapes between encoder and decoder.
