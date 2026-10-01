# Per-Feature Affine Normalisation — Overview

## What it is
A layer that rescales and shifts every feature on its own: `y_i = a_i x_i + b_i`. It has one scale
and one offset per feature, `2N` parameters in total, and no mixing between features. Two helpers fill
it in. One is from the statistics of a trained batch-normalisation layer (frozen BN). The other is from
a per-feature mean and standard deviation (standardisation).

## Why it matters (embedded)
Raw sensor channels arrive in wildly different units: acceleration near `9.81 m/s²` with a spread of
`0.5`, angular rate in hundreds of `°/s`, a supply voltage near `3.3 V` varying by millivolts. Fed
directly into a `Dense` layer, the large-valued channels dominate the weighted sums, and the small ones are
effectively ignored. Gradient steps are also badly conditioned. Standardising each channel to zero mean
and unit spread fixes both problems. It costs one multiply-add per feature and `4N` floats of RAM,
instead of the `N² + N` parameters a `Dense` would need to do the same job.

## How it works (intuition)
At inference time, batch normalisation uses fixed statistics: `γ (x − μ)/√(σ² + ε) + β`. That is
just a per-feature straight line, so it collapses exactly into one scale `a = γ/√(σ² + ε)` and one
offset `b = β − a μ`. Standardisation is the same line with `γ = 1`, `β = 0`. Because the map is
affine, it can also be *folded* into a neighbouring linear layer by multiplying the weights by the
scales and absorbing the offsets into the bias. The weight exporter (N13) does this offline, so a
deployed network usually needs the runtime layer only for input standardisation, or when no fold is
possible. The backward pass is simple: the gradient passes through scaled by `a`. The scale and offset
gradients are `g ⊙ x` and `g` (exposed by N8).

## Key parameters
- **Scale `a` and offset `b`**: one pair per feature, stored as `[a…, b…]`, trainable like any other
  layer's parameters.
- **`ε` (BN helper)**: must match the training framework (Keras `1e-3`, PyTorch `1e-5`).
- **Large offsets**: when `|μ/σ|` is in the thousands, float32 loses about `1e-4` of absolute
  precision. Remove the offset upstream if that matters.
- **Not training-mode BN**: batch statistics need a batch, and this library's `Forward` sees one sample.
  Training-mode BN is deferred.

## Reference
S. Ioffe, C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift," *Proc. ICML*, PMLR 37, pp. 448–456, 2015 (Alg. 2, the inference transform).
B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only
Inference," *Proc. CVPR*, pp. 2704–2713, 2018 (folding BN into the preceding layer).
Y. LeCun, L. Bottou, G. B. Orr, K.-R. Müller, "Efficient BackProp," in *Neural Networks: Tricks of the
Trade*, Springer LNCS 1524, 1998 (why input normalisation speeds up training).

## See also
`Layer` (the contract), `Dense` (the layer it is usually folded into), `WeightExport` (N13, offline
BN folding), `TrainableLayerInterface` (N8, parameter gradients), `MiniBatchTraining` (N16,
fine-tuning the scale and offset on the device).
