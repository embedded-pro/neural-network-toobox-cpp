# Swish Family (SiLU / Hard-Sigmoid / Hard-Swish) — Overview

## What it is
Three related element-wise activations. **SiLU** (also called Swish) is `f(z) = z·σ(z)`: the input
gated by its own sigmoid. **Hard-Sigmoid** is `h(z) = ReLU6(z + 3)/6`, a three-piece linear stand-in
for the sigmoid that is `0` below `-3`, `1` above `3` and a straight line of slope `1/6` in between.
**Hard-Swish** is `q(z) = z·h(z)`: the same self-gating idea as SiLU, built on the hard sigmoid.

## Why it matters (embedded)
Modern compact vision and audio networks (the MobileNetV3 and EfficientNet families) use these instead
of ReLU, so a model trained with them can only be imported and run here once they exist. The hard
variants are the embedded-friendly ones: two comparisons and one multiply-add, no exponential, exact
saturation values and constant derivatives — cheap on an MCU and ready for a later fixed-point port.
SiLU costs one exponential per element, like Sigmoid.

## How it works (intuition)
ReLU passes positives and zeroes negatives with a hard corner at `0`. SiLU rounds that corner off: large
positive inputs pass almost unchanged, large negative inputs fade to `0`, and small negative inputs
produce a small negative dip (minimum `≈ -0.278` at `z ≈ -1.278`). The smooth, non-monotonic shape gives
useful gradients on both sides of zero; the slope is `1/2` at the origin and briefly exceeds `1`.
Hard-Swish copies that shape with straight lines and one parabola: zero below `-3`, identity above `3`,
and `z(z + 3)/6` in between (minimum `-0.375` at `z = -1.5`). It stays within `0.143` of SiLU, and the
hard sigmoid within `0.070` of the true sigmoid. They are still separate functions: run a network with
the activation it was trained with. Hard-Sigmoid on its own is the gate of MobileNetV3's
squeeze-and-excitation blocks.

## Key parameters
- **None.** All three are stateless; the vector length comes from the layer.
- **Knee convention (hard variants)** — at `z = ±3` the derivative takes the outer piece's value
  (`h' = 0`; `q'(-3) = 0`, `q'(3) = 1`).
- **Not included** — the trainable-slope Swish-β `z·σ(βz)` (needs the N8 gradient interface) and the
  older hard-sigmoid variants (`0.2z + 0.5`, `(z + 1)/2`), which use different knees.

## Reference
D. Hendrycks, K. Gimpel, "Gaussian Error Linear Units (GELUs)," arXiv:1606.08415, 2016 (introduces the
name SiLU); S. Elfwing, E. Uchibe, K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function
Approximation in Reinforcement Learning," *Neural Networks* 107, pp. 3–11, 2018; P. Ramachandran,
B. Zoph, Q. V. Le, "Searching for Activation Functions," arXiv:1710.05941, 2017 (Swish);
A. Howard et al., "Searching for MobileNetV3," *ICCV*, 2019 (h-sigmoid and h-swish).

## See also
`ActivationFunction` (the contract), `Sigmoid` (reused by SiLU), `ReLU` (the corner these smooth),
`Dense`, `WeightExport` (N13, importing MobileNetV3-class weights), `DepthwiseSeparableConvolution` (N19).
