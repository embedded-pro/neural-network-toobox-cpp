# 1D Pooling (Max / Average / Global-Average) — Overview

## What it is
Parameter-free layers that shrink a multi-channel sequence along its length. A window of `K` positions
slides over the sequence in steps of `S`, and each channel of each window is replaced by one number:
its **maximum** (max pooling) or its **mean** (average pooling). **Global-average pooling** is the
special case where the window is the whole sequence, so every channel collapses to a single mean.

## Why it matters (embedded)
A 1D convolution (N11) over a sensor window or an audio frame keeps the full sequence length, and the
activations are where an MCU's RAM goes. Pooling by `K = 2` halves the length, and with it the RAM
and MACs of every layer that follows. Global-average pooling also replaces the usual flatten→Dense
classifier head: instead of `L·C·classes` weights the head needs `C·classes`, for example 650
parameters instead of 15 370 for `L = 24`, `C = 64` and 10 classes. The result no longer depends on
where in the window an event happened.

## How it works (intuition)
Max pooling asks "did this feature fire anywhere in the window?" and keeps the strongest response.
Average pooling asks "how much of this feature is there on average?" and smooths it. Both work on
each channel independently and learn nothing. The backward pass follows directly. For the maximum,
only the winning input influenced the output, so the whole upstream gradient goes back to it. Max
pooling remembers the winner's index during `Forward` for this; a tie is resolved to the first
winner. For the mean, every input contributed `1/K`, so each receives `1/K` of the gradient. When
windows overlap, an input collects the gradients of every window it belongs to. Positions that no
window covers (the tail of a sequence that does not divide evenly) receive zero.

## Key parameters
- **Pool size `K`**: window length. Must not exceed the sequence length.
- **Stride `S`**: step between windows. The default `S = K` gives non-overlapping windows (the Keras
  and PyTorch default).
- **Padding**: valid only. Output length is `⌊(L − K)/S⌋ + 1`, and the uncovered tail is dropped.
- **Layout**: channels-last, `x[t·C + c]`, shared with the 1D convolution (N11).

## Reference
Y.-L. Boureau, J. Ponce, Y. LeCun, "A Theoretical Analysis of Feature Pooling in Visual Recognition,"
*Proc. ICML*, pp. 111–118, 2010 (max vs average pooling).
M. Lin, Q. Chen, S. Yan, "Network In Network," *Proc. ICLR*, 2014, arXiv:1312.4400 (global-average
pooling as the classifier head).
I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press (2016), §9.3 "Pooling".

## See also
`Layer` (the contract), `TrainableLayerInterface` (N8, the parameter-free `Layer` specialisation),
`Convolution1D` (N11, the layer pooling usually follows), `Convolution2D` (N18, the 2D pooling
counterpart), `FlashResidentWeights` (N10, the inference-only variant without gradient buffers).
