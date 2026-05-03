---
name: image-representation
description: How digital images are stored as tensors of pixel intensities — bit depth, channels, RGB encoding — and why the resulting input size kills fully connected networks.
type: concept
sources:
  - raw/week-04/w04-l10-transcript.txt
  - raw/week-04/w04-l11-transcript.txt
  - raw/week-04/w04-slides.pdf
  - raw/week-04/w4-notion.zip
status: stable
updated: 2026-04-26
---

*An image is just a multi-dimensional array of numbers. The shape and the bit depth determine how much memory it occupies — and how badly an MLP scales when you try to feed it one.*

## Pixels are integers (mostly)

A digital image is a grid of pixels. Each pixel stores an intensity value as a fixed number of bits. The number of bits is called the **bit depth**, and it determines how many distinct intensities the pixel can represent.

| Bit depth | Bytes per pixel | Distinct values | Typical use |
|---|---|---|---|
| 8 bit | 1 | $2^8 = 256$ | Standard photos, web images |
| 16 bit | 2 | $2^{16} = 65{,}536$ | Scientific imaging, microscopy |
| 32 bit float | 4 | (continuous) | Computations, HDR, scientific |

For 8-bit images, intensities are integers in $[0, 255]$ — the canonical encoding for standard photographs. 16-bit gives much finer gradations, which matters in scientific imaging where small differences in intensity carry meaning. 32-bit floating point stores approximate real values using IEEE 754 — useful when computations push values outside the integer range or need fractional precision.

> [!info] ASIDE — Why 8 bits is enough for photos
> Human visual perception can only distinguish about 100–200 intensity levels in a single image (depending on lighting and contrast), so 256 levels comfortably exceeds what the eye notices. Scientific imaging uses 16-bit because instruments like microscope cameras can detect signal differences far smaller than the eye can — and you don't want to throw that information away during digitisation.

## Greyscale: a 2D matrix

A greyscale image of width $W$ and height $H$ is a 2D matrix of shape $H \times W$. Each entry is one pixel's intensity. Memory cost = $W \times H \times \text{bytes-per-pixel}$.

A $512 \times 512$ 16-bit greyscale image (a typical microscopy frame) is $512 \times 512 \times 2 = 524{,}288$ bytes ≈ 512 KB.

## Colour: stack greyscale images

Colour images use multiple **channels**. The standard is **RGB** — three channels for red, green, blue — stacked into a 3D tensor of shape $H \times W \times 3$. Each channel is essentially a greyscale image showing how much of that primary colour is at each position. Each channel-pixel is typically 8 bits, giving a "24-bit colour" image with $2^{24} \approx 16.8$ million possible colours per pixel.

> [!info] ASIDE — Why RGB? Trichromatic vision
> The choice of three channels isn't arbitrary — it matches the human eye, which has three types of cone cell sensitive to short (S, blue), medium (M, green), and long (L, red/yellow) wavelengths. Any visible colour can be approximated by mixing red, green, and blue light in the right proportions, which is why RGB monitors work. Other colour spaces exist (HSV, LAB, CMYK), but RGB is the default in computer graphics because it maps directly to display hardware.

A few useful RGB tuples:

| Colour | RGB | Notes |
|---|---|---|
| Black | $(0, 0, 0)$ | No light |
| White | $(255, 255, 255)$ | Maximum of every channel |
| Red | $(255, 0, 0)$ | Pure red |
| Yellow | $(255, 255, 0)$ | Red + green = yellow |
| Cyan | $(0, 255, 255)$ | Green + blue = cyan |
| Grey | $(128, 128, 128)$ | Equal mid-intensity in all channels |

## Why this matters for neural networks

The input to a neural network is the image flattened into a vector — or, in CNNs, the image kept as a 3D tensor. Either way, the *number of input components* is what matters for the first layer's parameter count.

For a small $28 \times 28$ greyscale digit (MNIST), a fully connected first layer with even just 1000 hidden units already needs $784 \cdot 1000 = 784{,}000$ weights — large but tractable.

Now scale up:

| Image size | Channels | Inputs | FC layer to 1M hidden units |
|---|---|---|---|
| $28 \times 28$ | 1 | 784 | $7.84 \times 10^8$ weights |
| $224 \times 224$ | 3 (RGB) | 150{,}528 | $1.5 \times 10^{11}$ weights |
| $1000 \times 1000$ | 3 | 3{,}000{,}000 | $3 \times 10^{12}$ weights |

The trillion-weight numbers aren't trainable — they exceed the memory of every modern accelerator and there isn't enough labelled data on the planet to fit them. **A fully connected MLP simply does not scale to image inputs.**

This is the bottleneck that motivates the entire CNN architecture: instead of one weight per (pixel, hidden unit) pair, share a small kernel of weights across every spatial location. See [[convolution]] for the operation, [[convolutional-neural-network]] for the resulting architecture.

## Image tensors in code

In PyTorch, images are typically stored as tensors of shape `(C, H, W)` or `(N, C, H, W)` for batches:

- `N` — batch size (number of images)
- `C` — channels (1 for greyscale, 3 for RGB)
- `H` — height
- `W` — width

Note the channel-first convention. Some libraries (TensorFlow's older API, NumPy in some contexts) use channel-last, `(N, H, W, C)`. When porting code, check which convention each library expects.

Pixel intensities for 8-bit images are integers $[0, 255]$ when loaded, but neural networks usually want floats normalised to $[0, 1]$ (divide by 255) or to mean 0, std 1 (subtract per-channel mean, divide by per-channel std). Normalisation is part of standard preprocessing.

## Related

- [[convolution]] — the operation that lets networks process images without exploding parameter counts
- [[convolutional-neural-network]] — the architecture built on top of convolution
- [[multi-layer-perceptron]] — the architecture that *cannot* handle images at scale
- [[perceptron]] — the underlying neuron, whose dot product gets reused at every spatial position in a convolution

## Active Recall

> [!question]- What is the bit depth of an image, and how many distinct intensity values does an 8-bit greyscale image's pixel encode?
> Bit depth is the number of bits used to represent each pixel's intensity. An 8-bit greyscale pixel can take any of $2^8 = 256$ distinct integer values, conventionally $[0, 255]$.

> [!question]- A $1024 \times 1024$ RGB image with 8-bit channels is loaded as a tensor. What is its shape and how much memory does it occupy as raw bytes?
> Shape is $1024 \times 1024 \times 3$ (or $3 \times 1024 \times 1024$ in channel-first form). Memory: $1024 \times 1024 \times 3 \times 1 = 3{,}145{,}728$ bytes ≈ 3 MB.

> [!question]- Why is a $1000 \times 1000$ image fundamentally a problem for a fully connected first layer?
> The image has $10^6$ pixels (or $3 \times 10^6$ if RGB). A fully connected first layer with $m$ hidden units needs $10^6 \cdot m$ weights. Even modest hidden-unit counts produce billions of parameters in just one layer — too many to store, too many to train. The fix is weight sharing via convolution: instead of a weight per (pixel, output) pair, share a small kernel across all spatial positions.

> [!question]- An RGB image is sometimes called "24-bit colour". Where does the 24 come from, and how many distinct colours can each pixel represent?
> Three channels (R, G, B) at 8 bits each = 24 bits per pixel. Each pixel can therefore be any of $2^{24} = 16{,}777{,}216$ ≈ 16.8 million distinct RGB combinations.

> [!question]- Why do scientific imaging applications often use 16-bit greyscale instead of 8-bit?
> Scientific instruments (e.g. fluorescence microscopes, X-ray detectors) can resolve intensity differences far finer than human vision can perceive. 8 bits (256 levels) would quantise away that information; 16 bits ($65{,}536$ levels) preserves it. The trade-off is double the memory per pixel, which is acceptable when individual images carry irreplaceable scientific data.
