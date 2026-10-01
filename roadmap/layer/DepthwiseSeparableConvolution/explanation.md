# Depthwise-Separable Convolution — Overview

## What it is
A convolution split into two cheap steps. The **depthwise** step gives every input channel its own
small `KH×KW` filter and filters that channel alone, so channels never mix. The **pointwise** step is a
`1×1` convolution: at every pixel it mixes the `Cin` filtered channels into `Cout` output channels with
a small `Cout×Cin` matrix, like a `Dense` layer applied pixel by pixel. Together they replace one
standard `KH×KW` convolution with `KH·KW·Cin + Cin·Cout` weights (plus biases) instead of
`KH·KW·Cin·Cout`. The toolbox provides the depthwise step as its own layer, the fused
depthwise-then-pointwise layer, and a 1D form of each for time series.

## Why it matters (embedded)
It is the building block of MobileNet and of DS-CNN, the keyword-spotting network that Zhang et al.
("Hello Edge") ran on Cortex-M microcontrollers. The saving is large: for a 3×3 layer with 64 channels
in and out on a 25×5 MFCC map, the parameters drop from 36 928 to 4 800 and the multiply-accumulates
from 2.54 M to 0.32 M, about 8× fewer. The general cost ratio is `1/Cout + 1/(KH·KW)`. Fewer weights
means less flash, and fewer MACs means less energy per inference. In the fused layer, an inference-only
build (N10) can also skip storing the intermediate map: it computes the `Cin` depthwise values of one
pixel and mixes them straight away.

## How it works (intuition)
A standard convolution does two jobs at once: it looks at a neighbourhood (spatial filtering) and it
combines channels (feature mixing). Depthwise-separable convolution does them one after the other. The
depthwise step slides one filter per channel over that channel's plane, with the same window, stride
and "valid" rule as `Convolution2D` (N18), so the output keeps `Cin` channels. An optional activation
follows (ReLU in MobileNet and DS-CNN, none in Keras `SeparableConv2D`). The pointwise step then
recombines the channels at each pixel and applies the output activation. Data stay in the channels-last
layout of N11/N18, so the channels of one pixel are next to each other in memory and both steps read
memory in order. Training runs the chain rule backwards: the pointwise step hands each pixel's error
back through its channel-mixing matrix, the depthwise activation scales it, and the depthwise step
scatters it back onto the input window of each channel, summing where windows overlap.

## Key parameters
- **Kernel `KH×KW`**: the neighbourhood each depthwise filter sees (typically 3×3; `1×K` in 1D).
- **Stride `SH×SW`**: applies to the depthwise step only; the pointwise step is always 1×1, stride 1.
- **Channels `Cin`, `Cout`**: depthwise filters (one per input channel) and pointwise outputs.
- **Two activations**: depthwise (use Identity, N1, for the Keras `SeparableConv2D` form) and pointwise.
  Both must be element-wise; not Softmax.
- **Padding**: "valid" only, as in N11/N18. Depth multiplier fixed at 1 (one filter per channel).

## Reference
A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, H. Adam, "MobileNets:
Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017
(depthwise + pointwise factorisation and its `1/N + 1/D_K²` cost ratio).
Y. Zhang, N. Suda, L. Lai, V. Chandra, "Hello Edge: Keyword Spotting on Microcontrollers,"
arXiv:1711.07128, 2017 (DS-CNN on Cortex-M).
F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," *Proc. IEEE CVPR*, 2017,
pp. 1251–1258.
L. Sifre, "Rigid-Motion Scattering for Image Classification," Ph.D. thesis, École Polytechnique, 2014
(origin of the depthwise-separable factorisation).

## See also
`Convolution2D` (N18, the standard layer this factorises), `Convolution1D` (N11, the 1D layout),
`TrainableLayerInterface` (N8, gradient accumulation), `FlashResidentWeights` (N10, fused inference
without the intermediate map), `Identity` (N1), `WeightExport` (N13, kernel layouts and BN folding),
`Pooling1D` (N12).
