# Weight Initialisation (Glorot, He, LeCun) + Deterministic PRNG — Overview

## What it is
A small toolkit that fills a new layer's weight matrix with random numbers of the right size. It has
three recipes: Glorot (Xavier), He (Kaiming) and LeCun, each in a uniform and a normal flavour. All
of them draw from a tiny, fully specified random generator (xorshift32), so the same seed gives the
same weights on every compiler and every target.

## Why it matters (embedded)
Training from scratch (N16) only works if the starting weights are the right scale. If they are too
large, tanh and sigmoid saturate and their gradients vanish. If they are too small, the signal
shrinks layer by layer. Today every caller of `Dense` must supply its own weights, and the simulator
uses `std::normal_distribution`, whose output differs between standard libraries. Unit tests and
field devices therefore cannot share reference values. The generator here needs 4 bytes of state and
six integer operations per draw, which is small enough to seed on the device from a serial number.

## How it works (intuition)
A weighted sum of `n` inputs has a variance of `n` times the weight variance times the input
variance. To keep the signal the same size from layer to layer, the weight variance must be about
`1/n`. Backpropagation sums over the layer's *outputs*, so keeping gradients the same size would
need `1/n_out`. Glorot splits the difference with `2/(n_in + n_out)` for tanh-like activations. ReLU
zeroes half its inputs and halves the signal energy, so He doubles the variance to `2/n_in`, or
`2/((1 + a²) n_in)` for LeakyReLU with slope `a`. LeCun uses `1/n_in`, which is what SELU's
self-normalisation assumes. All three are one formula, `Var[w] = scale / fan`. A uniform draw on
`[−L, L]` gets `L = √(3·scale/fan)`, and a normal draw gets `σ = √(scale/fan)` through the Box–Muller
transform. The seed is first scrambled with the MurmurHash3 finaliser, because raw xorshift makes
seeds 1 and 2 give streams that are bit-shifted copies of each other.

## Key parameters
- **Preset**: Glorot for tanh/sigmoid/softmax logits, He for ReLU (with the negative slope for
  LeakyReLU), LeCun for SELU (N6) and linear outputs.
- **Distribution**: uniform draws are bit-reproducible across toolchains. Normal draws depend on
  libm `log`/`cos` to about 1 ulp.
- **Fans**: `fan_in` = inputs per output, `fan_out` = outputs per input. For a Dense layer they are
  its input and output sizes. For a convolution, multiply each by the kernel size.
- **Seed**: any `uint32`. Successive calls on one initialiser continue the stream, so a model's layers
  get different weights in construction order.
- **Biases** stay zero.

## Reference
X. Glorot, Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks,"
*Proc. AISTATS*, PMLR 9, pp. 249–256, 2010.
K. He, X. Zhang, S. Ren, J. Sun, "Delving Deep into Rectifiers: Surpassing Human-Level Performance on
ImageNet Classification," *Proc. ICCV*, pp. 1026–1034, 2015.
Y. LeCun, L. Bottou, G. B. Orr, K.-R. Müller, "Efficient BackProp," in *Neural Networks: Tricks of the
Trade*, Springer LNCS 1524, 1998 (weight initialisation `σ = n_in^{−1/2}`).
G. Klambauer, T. Unterthiner, A. Mayr, S. Hochreiter, "Self-Normalizing Neural Networks," *Advances in
NeurIPS* 30, 2017 (LeCun-normal initialisation for SELU).
G. Marsaglia, "Xorshift RNGs," *J. Stat. Softw.* 8(14), pp. 1–6, 2003.
G. E. P. Box, M. E. Muller, "A Note on the Generation of Random Normal Deviates," *Ann. Math. Statist.*
29(2), pp. 610–611, 1958.
A. Appleby, *MurmurHash3* / SMHasher (2011), the `fmix32` finaliser used for seed scrambling.

## See also
`Dense` (consumes `Weights<Out, In>()` directly), `ExponentialLinearUnit` (N6, SELU needs LeCun
normal), `MiniBatchTraining` (N16, training from scratch), `Convolution1D` / `Convolution2D`
(N11/N18, kernel fans), `Lstm` (N21, adds its own forget-gate bias of 1), `WeightExport` (N13, the
path for trained weights, which are never re-initialised).
