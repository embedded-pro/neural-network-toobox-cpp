# 1D Convolution — Overview

## What it is
A layer that slides a small set of learned filters along a multi-channel time series. Each filter looks
at `K` consecutive frames of all `Cin` input channels, takes a weighted sum plus a bias, and produces one
output channel; the window then moves `S` frames and repeats. The result is a shorter series of `Cout`
feature channels with `⌊(L − K)/S⌋ + 1` positions. There are `K·Cin·Cout + Cout` parameters, independent
of the window length `L`. A companion streaming form computes the same filters one frame at a time.

## Why it matters (embedded)
Most MCU learning problems are time series: IMU gestures and activity, vibration for predictive
maintenance, ECG, current and pressure signatures, raw or MFCC audio. A convolution shares one small filter
bank across every position, so a 128-frame, 3-axis IMU window with 8 filters of length 5 needs 128
parameters and about 12 KB of RAM, where a `Dense` layer with the same input and output sizes would need
about 382 000 parameters. The streaming form keeps only the last `K − 1` frames, so a sensor pipeline
produces a new feature frame per sample instead of recomputing the whole window.

## How it works (intuition)
Each output value is a dot product between a filter and the window of input frames under it, like a
`Dense` row applied to a short slice of the signal. Frames are stored channels-last (all channels of
frame 0, then frame 1, …), so every window is one contiguous block of memory, and the output uses the
same layout, so a `Dense` or a pooling layer (N12) can read it directly without a Flatten step. As in
Keras and PyTorch the filter is not flipped (strictly a cross-correlation), so trained weights import
without reversal. The backward pass sends each output's error back through the same filter to the frames
that window covered, adding up where windows overlap, and accumulates each filter's gradient as the error
times the input it saw. The streaming form keeps a small ring buffer of past frames that starts at zero,
which is equivalent to padding the start of the signal with zeros ("causal" padding).

## Key parameters
- **Kernel size `K`**: how many frames each filter sees (its receptive field).
- **Stride `S`**: how far the window moves; `S > 1` shortens the output and the cost by about `S`.
- **Channels `Cin`, `Cout`**: input features per frame and number of filters.
- **Padding**: "valid" only (no padding); trailing frames that do not fill a window are ignored. Pad the
  input yourself for "same", or use the stream for causal padding. Dilation is not supported.
- **Activation**: element-wise only (ReLU, LeakyReLU, Tanh, Sigmoid, Identity); not Softmax.

## Reference
S. Kiranyaz, O. Avci, O. Abdeljaber, T. Ince, M. Gabbouj, D. J. Inman, "1D convolutional neural networks
and applications: A survey," *Mechanical Systems and Signal Processing*, vol. 151, 107398, 2021.
Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-Based Learning Applied to Document Recognition,"
*Proc. IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998 (convolutional layers and their gradients).
V. Dumoulin, F. Visin, "A guide to convolution arithmetic for deep learning," arXiv:1603.07285, 2016
(output length, stride and the transposed convolution used by the backward pass).
A. van den Oord et al., "WaveNet: A Generative Model for Raw Audio," arXiv:1609.03499, 2016 (causal
convolutions for streaming).

## See also
`Dense` (the per-window arithmetic), `TrainableLayerInterface` (N8, gradient accumulation),
`FlashResidentWeights` (N10), `Pooling1D` (N12), `WeightExport` (N13, kernel permutation and BN folding),
`Convolution2D` (N18), `DepthwiseSeparableConvolution` (N19).
