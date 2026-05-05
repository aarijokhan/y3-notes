---
week: 4
topic: "Images and Convolutional Neural Networks"
deck: NeuralComputation::Week-04
---

TARGET DECK
NeuralComputation::Week-04

## Image representation

> [!question]- How is a **colour (RGB) image** represented as a tensor?
> A $W \times H$ RGB image is a tensor of shape $W \times H \times 3$ — three stacked matrices, one per channel (red, green, blue). Each channel-pixel is typically 8 bits, giving $2^{24} \approx 16.8$ million possible colours per pixel.

> [!question]- Why is the **bit depth** of an image significant when feeding it to a neural network?
> Bit depth determines the range of intensity values: 8-bit gives $[0, 255]$, 16-bit gives $[0, 65535]$. Large raw magnitudes propagate through the network and can cause activations to explode or sigmoids to saturate. Inputs are usually rescaled (e.g. divide by 255 to get $[0, 1]$, or Z-score normalise) before training.

## Why MLPs fail on images

> [!question]- Why is a fully connected MLP a bad architecture for images?
> Two reasons:
>
> 1. **Parameter explosion.** A $1000 \times 1000$ image fed into a fully connected hidden layer of $10^6$ units needs $10^{12}$ weights in just the first layer — far beyond memory, compute, and statistical feasibility.
> 2. **No translation structure.** The MLP treats nearby pixels as independent and would have to learn every spatial location separately. A "7" in the corner is a completely different input from a "7" in the centre.

## Convolution

> [!question]- What is the **convolution** operation in a CNN?
> $$G[i, j] = \sum_{u=-k}^{k} \sum_{v=-k}^{k} \mathbf{w}[u, v] \cdot X[i + u, j + v]$$
> Slide a small **kernel** $\mathbf{w}$ across the image $X$ and at each position compute the dot product between the kernel and the local image patch. The output map $G$ records the dot product at every spatial location. The same kernel weights are reused at every position — this is **weight sharing**.

> [!question]- Why can convolution be read as a **similarity / feature-detection** operation?
> Convolution is a sliding **dot product**, and the dot product measures similarity between two vectors. A high value of $G[i,j]$ means the local image patch at $(i, j)$ looks like the kernel pattern. So the kernel acts as a learnable template for a feature (edge, corner, texture), and the output map records *where in the image that feature appears*.

> [!question]- Why do CNNs need so many fewer parameters than MLPs for images?
> **Weight sharing.** A convolution layer uses the same kernel at every spatial position, so the parameter count scales with kernel size $\times$ depth $\times$ number of filters — not with image size. A $3 \times 3$ filter on a 3-channel input has $3 \cdot 3 \cdot 3 = 27$ weights regardless of whether the image is $32 \times 32$ or $1000 \times 1000$.

## Output size, padding, stride

> [!question]- State the **convolution output-size formula**.
> $$W_2 = \frac{W_1 - F + 2P}{S} + 1$$
> $W_1$ is the input width, $F$ is the kernel size, $P$ is the padding (zeros added at each border), and $S$ is the stride. The same formula applies to height. Output depth $D_2 = K$ equals the number of filters, regardless of input depth.

> [!question]- A $32 \times 32$ image is convolved with a $5 \times 5$ filter, stride 1, no padding. What size is the output?
> $W_2 = (32 - 5 + 0)/1 + 1 = 28$. Output is $28 \times 28$. Two pixels of border are lost on each side because the kernel cannot centre on boundary pixels.

> [!question]- For an odd $F \times F$ kernel, what padding $P$ gives **same convolution** (output spatial size = input)?
> $$P = \frac{F - 1}{2}$$
> So $F = 3$ needs $P = 1$, and $F = 5$ needs $P = 2$. This is why VGG uses $3 \times 3$ kernels with $P = 1$ everywhere.

> [!question]- What goes wrong if the convolution output-size formula yields a **non-integer**?
> The chosen $(F, S, P)$ combination does not tile the image cleanly — the kernel runs off the edge with leftover pixels that have no place to go. You either lose information at the borders or have to handle it specially. Always pick $(F, S, P)$ so the formula is integer.

## Convolution layer

> [!question]- Why must a convolution filter's **depth match the input's depth**?
> The filter performs a dot product across the entire local volume — both spatial and depth dimensions. If the input has $D$ channels, each spatial position holds a $D$-dimensional vector, so the filter needs one weight per (spatial offset, channel) triple. The spatial size $F \times F$ is a design choice; the depth is fixed by the input.

> [!question]- VGG16's first conv layer takes a $224 \times 224 \times 3$ input and applies 64 filters of size $3 \times 3$. How many learnable parameters?
> Each filter has $3 \times 3 \times 3 = 27$ weights plus 1 bias = 28 parameters. With 64 filters: $64 \times 28 = 1792$. Compare to a fully connected layer of the same output size, which would need $\sim 4.8 \times 10^{11}$ weights — five orders of magnitude more. That gap is the weight-sharing payoff.

## Activations

> [!question]- What is **ReLU**, and why does it dominate CNN hidden layers?
> $$\text{ReLU}(z) = \max(0, z)$$
> ReLU is non-saturating for $z > 0$ (gradient is 1, not vanishing), computationally trivial, and converges much faster than sigmoid (~6× in some benchmarks). Its only weakness is "dying ReLU" — neurons that always output 0 stop learning — which **leaky ReLU** $\max(0.01z, z)$ avoids by keeping a small slope on the negative side.

> [!question]- Compare **sigmoid**, **tanh**, **ReLU**, and **leaky ReLU** in one table.
>
> | Activation | Range | Saturates? |
> |---|---|---|
> | Sigmoid $1/(1+e^{-z})$ | $(0, 1)$ | Both ends |
> | Tanh | $(-1, 1)$ | Both ends |
> | ReLU $\max(0, z)$ | $[0, \infty)$ | Only at 0 (can "die") |
> | Leaky ReLU $\max(0.01z, z)$ | $\mathbb{R}$ | Never |

## Pooling

> [!question]- What is **max pooling**, and what is its default configuration?
> Slide an $F \times F$ window over the input and emit the **maximum** value within the window at each position. Default $F = 2, S = 2$ — halves spatial dimensions in both axes. Pooling has **no learnable parameters** and operates **independently per channel**, so depth is unchanged.

> [!question]- What two roles does pooling play in a CNN?
>
> 1. **Compute / memory savings.** Smaller spatial dimensions mean fewer ops in later layers and far fewer parameters in any subsequent FC layers.
> 2. **Shift invariance.** Small input translations leave the pooled output approximately unchanged, since the max over a window is robust to a feature shifting by a pixel or two.

## Shift equivariance vs invariance

> [!question]- What is the difference between **shift equivariance** and **shift invariance**, and which CNN component provides each?
>
> - **Shift equivariance** (convolution): if the input shifts by $\Delta$, the output of a convolutional layer shifts by $\Delta$ too — the features are detected the same way, just at moved positions.
> - **Shift invariance** (pooling + FC): even when a feature appears at a different location, the final classification stays the same.
>
> Convolution tracks *where* a feature is; pooling and FC eventually disregard *where* and ask only *what*.

## CNN architecture and parameter count intuition

> [!question]- Why do most VGG / modern CNN parameters live in the **fully connected** layers, not the convolutional ones?
> Convolution layers share weights across spatial positions, so each layer only stores kernel weights. The first FC layer must connect *every* element of the final feature map to every hidden unit — for VGG16 that is $7 \times 7 \times 512 = 25{,}088$ inputs into 4096 units, $\approx 10^8$ weights in one layer. Small kernels in the conv stack do most of the *work*; the FC layers carry most of the *parameters*.

> [!question]- Hierarchically, what kinds of features do **early**, **middle**, and **final** conv layers tend to learn?
>
> - **Early:** low-level — edges, corners, simple textures.
> - **Middle:** mid-level — eyes, wheels, leaves (compositions of edges).
> - **Final:** high-level — faces, cars, whole objects (compositions of parts).
>
> This hierarchy emerges naturally from training and matches what neuroscience reports about the visual cortex.
