---
name: upsampling
description: "Increasing the spatial resolution of a feature map — the inverse of pooling. Three common methods: nearest neighbour (cheap, blocky), bilinear interpolation (smooth, fixed weights), and transposed convolution (learnable). Essential for segmentation networks like U-Net."
type: concept
sources:
  - raw/week-05/w05-l13-transcript.txt
  - raw/week-05/w05-slides.pdf
  - raw/week-05/w5-notion.zip
status: stable
updated: 2026-04-26
---

*The opposite of [[pooling]]. Pooling shrinks a feature map; upsampling grows it. Needed whenever the output of a network must match the spatial size of its input — most importantly in segmentation, where every pixel needs a prediction. Three standard methods sit on a spectrum from cheap-and-fixed to learnable-and-expensive.*

## Why upsample at all

A standard CNN downsamples through pooling: $224 \to 112 \to 56 \to 28 \to 14 \to 7$. Spatial resolution drops, channel count rises, and the final layers see a coarse, abstract representation of the input. Fine for classification (one label per image).

But for **dense prediction** — semantic segmentation, depth estimation, image-to-image translation — the network's output must be the *same size* as its input: a per-pixel label map of shape $H \times W$ rather than a single class score. After downsampling abstract features, the network has to climb back to full resolution.

That's the upsampling step. It takes a small spatial map (say, $32 \times 32$) and produces a larger one (say, $64 \times 64$ or $128 \times 128$), filling in the new pixels by some chosen rule.

## Nearest-neighbour upsampling

The simplest method: each pixel in the small map is *copied* into a $2 \times 2$ block in the larger map.

For an input $4 \times 4$ map upsampled $2\times$:

| Input | Output |
|---|---|
| $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ | $\begin{pmatrix} a & a & b & b \\ a & a & b & b \\ c & c & d & d \\ c & c & d & d \end{pmatrix}$ |

Each input value $a$ becomes a $2 \times 2$ block of $a$'s in the output. No computation, no parameters — just memory rearrangement.

**Effect:** the output is *blocky*. Sharp edges and visible "Lego" pixelation, because every $2 \times 2$ block has identical values. This is what you see when you naively zoom in on a small image in any image viewer.

## Bilinear interpolation

Smoother. The output values are computed by *interpolating between* the input values, weighted by distance.

### 1D case first: linear interpolation

Two known values $x_1$ at position 0 and $x_2$ at position 1. To estimate the value at some intermediate position $t \in [0, 1]$:

$$y = (1 - t) x_1 + t \, x_2 = d_2 \, x_1 + d_1 \, x_2$$

where $d_1 = t$ (distance from $x_1$) and $d_2 = 1 - t$ (distance from $x_2$). The closer the new point is to $x_1$ (small $d_1$), the more weight $x_1$ gets — note the **swap**: $x_1$ is multiplied by $d_2$ (the *far* distance), and vice versa. The intuition: if I'm right next to $x_1$, my value should be mostly $x_1$.

Concrete example: estimate a value 25% of the way from $x_1 = 60$ to $x_2 = 100$. Then $d_1 = 0.25$, $d_2 = 0.75$, and $y = 0.75 \cdot 60 + 0.25 \cdot 100 = 45 + 25 = 70$ — closer to 60 than to 100, as expected.

### 2D case: bilinear

"Bilinear" = "linear in each of two dimensions". Apply the 1D formula twice — once horizontally, then once vertically (or vice versa; the result is the same).

For each new pixel:

1. Use linear interpolation along the top edge to get a value above the new pixel.
2. Use linear interpolation along the bottom edge to get a value below it.
3. Use linear interpolation between those two values to get the new pixel.

The result is a smooth gradient — no blocky $2 \times 2$ patches, but a continuous transition from each input pixel's value to its neighbours'.

### Effect compared to nearest neighbour

| Method | Output character | Cost | Learnable? |
|---|---|---|---|
| Nearest neighbour | Blocky, "pixelated" | None | No |
| Bilinear | Smooth, slightly blurred | Small (per-pixel weighted average) | No |
| Transposed conv | Whatever the network learned | Moderate (parameters and FLOPs) | Yes |

Bilinear gives noticeably better results than nearest neighbour on natural images and is the default in most modern segmentation networks. It has *no learnable parameters* — the interpolation weights are fixed by geometry.

## Transposed convolution: learnable upsampling

The third option treats upsampling as the *transpose* of convolution. Instead of sliding a kernel across an input to produce a smaller output (regular convolution), slide a kernel that *expands* each input pixel into multiple output pixels, and the output is larger than the input.

The kernel weights are learnable, just like in regular convolution. The network can therefore learn whatever upsampling pattern best serves the loss — possibly something cleverer than nearest-neighbour or bilinear.

**Caveat:** transposed convolutions are notorious for producing **checkerboard artifacts** (regular grid-like patterns in the output), because of how the kernel's overlap aligns with the stride. Modern alternatives often pair bilinear upsampling with a regular convolution layer afterwards, getting learnability without the artefact.

This module focuses on bilinear and nearest-neighbour upsampling; transposed convolutions are mentioned for completeness but their full mechanics aren't required.

## Align-corners: a subtle option

Both bilinear and nearest-neighbour upsamplers in PyTorch take an `align_corners` argument that affects how the input grid is mapped to the output grid:

- **`align_corners=False` (default for neural networks).** Treats pixels as cells with extent, aligning their edges. The output corners are *not* exact copies of the input corners — the upsampling is uniform across the field but doesn't preserve corner values.
- **`align_corners=True` (default for image processing).** Treats pixels as points, aligning the corner pixels exactly. Better for visualisation; worse for learning because gradients near the edge get distorted.

Worth knowing exists; the exact semantics matter for reproducing results across libraries but rarely for correctness in standalone projects.

## Where upsampling appears

The dominant use case in this module: the **decoder** half of a [[u-net|U-Net]] or other encoder-decoder segmentation architecture. After the encoder downsamples through several pooling layers to a low-resolution / high-channel bottleneck, the decoder progressively upsamples back to full input resolution, while combining (via skip connections) high-resolution information from the corresponding encoder layers.

Outside segmentation, upsampling appears in:

- **Generative models** — generators in GANs, decoders in autoencoders.
- **Super-resolution** — taking a low-resolution input and producing a sharper, higher-resolution output.
- **Image-to-image translation** — colourisation, style transfer, depth estimation.

## Related

- [[pooling]] — the downsampling operation that upsampling inverts
- [[u-net]] — the canonical architecture using upsampling in its decoder
- [[convolution]] — transposed convolution is a learnable alternative to fixed-rule upsampling
- [[convolutional-neural-network]] — fully convolutional networks (FCNs) for segmentation are the architectural context

## Active Recall

> [!question]- Why does a network designed for semantic segmentation need an upsampling stage, while a classification network does not?
> Classification outputs a single label (or class probability vector) for the whole image — no spatial output, so the network can pool aggressively to a small bottleneck and end with FC layers. Segmentation outputs a *per-pixel* label map of the same size as the input — every input pixel needs a prediction. After the encoder pools the image down to a small abstract representation, the decoder must climb back up to full resolution, which is exactly what upsampling does.

> [!question]- A $4 \times 4$ input is upsampled $2\times$ using nearest neighbour. Describe the output and explain why the result looks "blocky".
> The output is $8 \times 8$. Each input pixel is copied into a $2 \times 2$ block of identical values in the output: input pixel $(i, j)$ becomes output pixels $(2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1)$, all with the same value. Because each $2 \times 2$ block is constant and the boundary between blocks is sharp, the visual effect is identical to enlarging an image by zooming in: the pixels become visible squares, hence "Lego" or "Minecraft" appearance.

> [!question]- Use the bilinear interpolation formula $y = d_2 x_1 + d_1 x_2$ to estimate the value at a point 30% of the way from $x_1 = 50$ to $x_2 = 150$. Explain why $x_1$ is multiplied by the *far* distance $d_2$.
> $d_1 = 0.3$ (distance from $x_1$), $d_2 = 0.7$ (distance from $x_2$). $y = 0.7 \cdot 50 + 0.3 \cdot 150 = 35 + 45 = 80$. The result is closer to $x_1$ than to $x_2$ because we're 30% of the way (closer to $x_1$). The "swap" makes intuitive sense: when the new point is *close* to $x_1$, $d_1$ is small but the influence of $x_1$ should be *large* — multiplying $x_1$ by the *far* distance $d_2$ (which is large when we're close to $x_1$) achieves this. Each value is weighted by its inverse-distance.

> [!question]- What's the difference between bilinear upsampling and transposed convolution? Why might you use one over the other?
> Bilinear has fixed (geometric) weights — no parameters to learn — and produces smooth, predictable interpolation. Transposed convolution has *learnable* kernel weights, so the network can in principle discover a better upsampling pattern for its specific task. Bilinear is cheap and reliable; transposed convolution is more flexible but introduces parameters (more compute, more risk of overfitting) and is prone to checkerboard artifacts due to kernel overlap with stride. Modern practice often uses bilinear upsampling followed by a regular conv layer — a learnable refinement step without the artifact issue.

> [!question]- Why are transposed convolutions often associated with "checkerboard artifacts" in their outputs?
> When a transposed convolution kernel's size is not divisible by its stride, some output pixels are written to by more kernel applications than their neighbours — they receive overlapping contributions, while neighbouring pixels receive only one. The result is a regular pattern of brighter and darker pixels in a grid (checkerboard). The standard fix is to use a kernel size that's an integer multiple of the stride (e.g. $4 \times 4$ kernel with stride 2), or to switch to bilinear upsampling + a separate convolution.
