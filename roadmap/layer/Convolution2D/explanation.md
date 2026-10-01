# 2D Convolution + 2D Pooling — Overview

## What it is
A layer that slides a small set of learned filters over a multi-channel 2D map — a spectrogram
(time × frequency) or a low-resolution image. Each filter looks at a `KH × KW` patch of all `Cin` input
channels, takes a weighted sum plus a bias, and produces one output channel; the patch then moves by the
stride and repeats. The result is a smaller map of `Cout` feature channels with
`(⌊(H − KH)/SH⌋ + 1) × (⌊(W − KW)/SW⌋ + 1)` positions and `KH·KW·Cin·Cout + Cout` parameters,
independent of the map size. The companion **2D pooling** layers shrink a map without learning anything:
each `PH × PW` patch of each channel becomes its **maximum** or its **mean**, and **global-average
pooling** collapses each whole channel to a single mean.

## Why it matters (embedded)
Keyword spotting and small vision tasks are the classic MCU workloads, and both are 2D. A 49-frame × 10
MFCC keyword window with 8 filters of 3×3 needs 80 parameters and about 27 000 multiply-adds, where a
`Dense` layer with the same input and output sizes would need about 1.5 million parameters. The layer
computes each output directly from the input, with no im2col scratch buffer, so inference with
flash-resident weights (N10) needs RAM only for the activation maps. Pooling by 2×2 cuts the next layer's RAM and work by about four, and
global-average pooling replaces a large flatten→Dense classifier head with one mean per channel.

## How it works (intuition)
Each output value is a dot product between a filter and the patch under it — a `Dense` row applied to a
small window of the map. Maps are stored channels-last (all channels of pixel (0,0), then (0,1), …, row
by row), so each row of a patch is one contiguous block of memory, and the output uses the same layout:
a pooling layer, another convolution or a `Dense` reads it directly without a Flatten step. As in Keras,
PyTorch and TFLite the filter is not flipped (strictly a cross-correlation), so trained weights import
without reversal. The backward pass sends each output's error back through the same filter to the
pixels its patch covered, adding up where patches overlap, and accumulates each filter's gradient as the
error times the input it saw. Max pooling remembers which pixel won each patch and routes the whole
gradient there (the first winner in row-major order on a tie); average pooling gives every pixel of the
patch an equal `1/(PH·PW)` share. Pixels no patch covers receive zero.

## Key parameters
- **Kernel `KH × KW`**: the patch each filter sees; may be non-square (e.g. tall in time, narrow in
  frequency).
- **Stride `SH × SW`**: how far the patch moves in each direction; `2 × 2` quarters the output and the cost.
- **Channels `Cin`, `Cout`**: input features per pixel and number of filters.
- **Pool `PH × PW`**: pooling patch; the stride defaults to the patch size (non-overlapping), as in Keras
  and PyTorch.
- **Padding**: "valid" only; trailing rows/columns that do not fill a patch are ignored. Pad the input
  yourself for "same". Dilation and grouped convolution are not supported (depthwise is N19).
- **Activation**: element-wise only (ReLU, LeakyReLU, Tanh, Sigmoid, Identity); not Softmax.

## Reference
Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-Based Learning Applied to Document Recognition,"
*Proc. IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998 (2D convolutional layers, subsampling, gradients).
V. Dumoulin, F. Visin, "A guide to convolution arithmetic for deep learning," arXiv:1603.07285, 2016
(output size, stride and the transposed convolution used by the backward pass).
Y.-L. Boureau, J. Ponce, Y. LeCun, "A Theoretical Analysis of Feature Pooling in Visual Recognition,"
*Proc. ICML*, pp. 111–118, 2010 (max vs average pooling).
M. Lin, Q. Chen, S. Yan, "Network In Network," *Proc. ICLR*, 2014, arXiv:1312.4400 (global-average
pooling as the classifier head).
L. Lai, N. Suda, V. Chandra, "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs,"
arXiv:1801.06601, 2018 (HWC layout and the im2col alternative on MCUs).
Y. Zhang, N. Suda, L. Lai, V. Chandra, "Hello Edge: Keyword Spotting on Microcontrollers,"
arXiv:1711.07128, 2017 (2D CNNs on MFCC maps under MCU memory limits).

## See also
`Dense` (the per-patch arithmetic), `TrainableLayerInterface` (N8, gradient accumulation and
parameter-free layers), `FlashResidentWeights` (N10), `Convolution1D` (N11, the `H = 1` case),
`Pooling1D` (N12), `WeightExport` (N13, kernel permutation and BN folding), `MiniBatchTraining` (N16),
`DepthwiseSeparableConvolution` (N19); the log-mel/MFCC front end (N15, upstream) produces the input map.
