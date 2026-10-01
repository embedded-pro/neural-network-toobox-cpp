# 2D Convolution + 2D Pooling — Implementation Pseudocode

> Roadmap ref: #N18 (Tier 3) · Target: `neural_network/layer` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```text
# sizes (all compile-time):
#   H = Height, W = Width, Cin = InChannels, Cout = OutChannels
#   KH × KW = KernelHeight × KernelWidth,  SH × SW = StrideHeight × StrideWidth
#   Hout = (H − KH) / SH + 1,   Wout = (W − KW) / SW + 1       (integer division, "valid" padding)
#   P    = KH·KW·Cin·Cout + Cout
#
# channels-last (HWC) flat layout, the 2D form of N11's convention:
#   input  element (i, j, c) at x[(i·W + j)·Cin + c],       i ∈ [0, H),    j ∈ [0, W),    c ∈ [0, Cin)
#   output element (r, s, o) at y[(r·Wout + s)·Cout + o],   r ∈ [0, Hout), s ∈ [0, Wout), o ∈ [0, Cout)
#   ⇒ one kernel row of the window at output (r, s) is the contiguous slice
#     x[((r·SH + p)·W + s·SW)·Cin, … + KW·Cin)   for p ∈ [0, KH)
#   ⇒ a Dense, a 2D pooling or another Convolution2D reads the Hout·Wout·Cout map directly: no Flatten
#   ⇒ H = 1, KH = SH = 1 is exactly N11's layout, with W playing the role of L

template<typename T, std::size_t Height, std::size_t Width, std::size_t InChannels, std::size_t OutChannels,
         std::size_t KernelHeight, std::size_t KernelWidth,
         std::size_t StrideHeight = 1, std::size_t StrideWidth = 1>
                                               # static_assert(std::is_floating_point_v<T>); instantiated for float
                                               # static_assert(InChannels > 0 && OutChannels > 0)
                                               # static_assert(KernelHeight > 0 && KernelHeight <= Height)
                                               # static_assert(KernelWidth  > 0 && KernelWidth  <= Width)
                                               # static_assert(StrideHeight > 0 && StrideWidth > 0)
class Convolution2D final : public Layer<T, H·W·Cin, Hout·Wout·Cout, P>:
    const ActivationFunction<T>& activation    # element-wise, applied to the whole Hout·Wout·Cout map (as Dense)
    ParameterVector parameters                 # θ = [F_0 … F_{Cout−1}, b]; F_o = filter o, KH·KW·Cin values in (p, q, c) order
                                               #     F[o, p, q, c] at o·KH·KW·Cin + (p·KW + q)·Cin + c,   b[o] at KH·KW·Cin·Cout + o
    ParameterVector parameterGradients         # G = Σ_samples ∇_θ L_s since the last ZeroGradients() (N8 contract)
    InputVector     input                      # x of the last Forward (needed for ∂L/∂F)
    OutputVector    preActivation              # z
    OutputVector    output                     # y = f(z)
    OutputVector    preActivationGradient      # δ = J_f(z)ᵀ g of the last Backward (member, not a stack temporary)
    InputVector     inputGradient              # ∂L/∂x of the last Backward (per sample, not accumulated)
    bool            forwardDone = false
    # the filter layout is TFLite's / CMSIS-NN's OHWI order: filter row F_o is a Dense row over the
    # KH·KW·Cin window, and θ keeps the "multiplicative block, then biases" order of Dense, N2 and N11.

# 2D pooling (parameter-free, the 2D form of N12). K = PH·PW window elements.
template<typename T, std::size_t Height, std::size_t Width, std::size_t Channels,
         std::size_t PoolHeight, std::size_t PoolWidth,
         std::size_t StrideHeight = PoolHeight, std::size_t StrideWidth = PoolWidth>
class MaxPooling2D final : public Layer<T, H·W·C, Hout·Wout·C, 0>:
    OutputVector                          output          # y(r, s, c) = max over the PH×PW window of channel c
    std::array<std::size_t, OutputSize>   argmax          # flat input index that produced y[k]
    InputVector                           inputGradient   # ∂L/∂x of the last Backward
    bool                                  forwardDone = false

template<… same parameters …>
class AveragePooling2D final : public Layer<T, H·W·C, Hout·Wout·C, 0>:
    static constexpr T inversePoolArea = T{ 1 } / static_cast<T>(PoolHeight · PoolWidth)
    OutputVector   output                     # y(r, s, c) = (1/K) Σ over the window
    InputVector    inputGradient
    bool           forwardDone = false        # no saved input: the average's Jacobian is constant

template<typename T, std::size_t Height, std::size_t Width, std::size_t Channels>
using GlobalAveragePooling2D = AveragePooling2D<T, Height, Width, Channels, Height, Width, 1, 1>
    # one window over the whole map: Hout = Wout = 1, output size = Channels

# pooling guards: static_assert(std::is_floating_point_v<T>), H, W, C > 0, 0 < PH ≤ H, 0 < PW ≤ W,
# SH, SW > 0. Hout, Wout as for the convolution with (KH, KW) → (PH, PW).
# Stride defaults to the pool size (non-overlapping windows), as Keras MaxPooling2D and PyTorch MaxPool2d.
```

## Interface

```text
# Convolution2D<T, H, W, Cin, Cout, KH, KW, SH, SW>
using KernelMatrix = math::Matrix<T, Cout, KH·KW·Cin>   # row o = filter o, column (p·KW + q)·Cin + c
static constexpr std::size_t OutputHeight = Hout
static constexpr std::size_t OutputWidth  = Wout

Convolution2D(const KernelMatrix& initialKernels, const ActivationFunction<T>& activation)   # biases = 0, as Dense

void                   Forward(const InputVector& input) override                  # hot path
const InputVector&     Backward(const OutputVector& outputGradient) override       # hot path; returns ∂L/∂x, accumulates G
const OutputVector&    Output() const override
const ParameterVector& Parameters() const override
void                   SetParameters(const ParameterVector& parameters) override   # does not touch G (N8)
const ParameterVector& ParameterGradients() const override                         # N8
void                   ZeroGradients() override                                    # N8
# inherited: using ValueType = T; virtual ~Layer() = default

# MaxPooling2D / AveragePooling2D<T, H, W, C, PH, PW, SH, SW>
static constexpr std::size_t OutputHeight = Hout
static constexpr std::size_t OutputWidth  = Wout
MaxPooling2D()  /  AveragePooling2D()                                               # no arguments
void                Forward(const InputVector& input) override                     # hot path
const InputVector&  Backward(const OutputVector& outputGradient) override          # hot path, returns ∂L/∂x
const OutputVector& Output() const override
# inherited from Layer<T, In, Out, 0> (N8): ValueType, InputSize, OutputSize, ParameterSize = 0;
# no Parameters/SetParameters
```

## Algorithm (pseudocode)

```text
function Convolution2D(kernels, activation):
    for o in 0..Cout-1:
        for j in 0..KH·KW·Cin-1:  parameters[o·KH·KW·Cin + j] = kernels.at(o, j)
    # biases and G value-initialised to 0

function Forward(x):                            # OPTIMIZE_FOR_SPEED; direct convolution, no im2col buffer
    input = x
    for r in 0..Hout-1:
        for s in 0..Wout-1:
            out = (r·Wout + s)·Cout
            for o in 0..Cout-1:
                acc = parameters[KH·KW·Cin·Cout + o]            # bias
                for p in 0..KH-1:
                    row = ((r·SH + p)·W + s·SW)·Cin            # start of a contiguous KW·Cin run
                    tap = o·KH·KW·Cin + p·KW·Cin
                    for j in 0..KW·Cin-1:
                        acc += parameters[tap + j] · x[row + j]
                preActivation[out + o] = acc
    activation.ForwardVector(output, preActivation)
    forwardDone = true

function Backward(g):                           # OPTIMIZE_FOR_SPEED; g = ∂L/∂y, size Hout·Wout·Cout
    really_assert(forwardDone)
    activation.BackwardVector(preActivationGradient, preActivation, output, g)   # δ = J_f(z)ᵀ g
    inputGradient = InputVector{}               # rows/columns no window covers stay 0
    for r in 0..Hout-1:
        for s in 0..Wout-1:
            for o in 0..Cout-1:
                d = preActivationGradient[(r·Wout + s)·Cout + o]
                for p in 0..KH-1:
                    row = ((r·SH + p)·W + s·SW)·Cin
                    tap = o·KH·KW·Cin + p·KW·Cin
                    for j in 0..KW·Cin-1:
                        inputGradient[row + j]      += parameters[tap + j] · d   # scatter: overlaps add up
                        parameterGradients[tap + j] += d · input[row + j]        # "+=": N8 accumulation
                parameterGradients[KH·KW·Cin·Cout + o] += d
    return inputGradient

function ZeroGradients():        parameterGradients = ParameterVector{}
function ParameterGradients():   return parameterGradients
function SetParameters(p):       parameters = p
function Parameters():           return parameters
function Output():               return output

function MaxPooling2D::Forward(x):             # OPTIMIZE_FOR_SPEED
    for r in 0..Hout-1:
        for s in 0..Wout-1:
            out   = (r·Wout + s)·C
            first = ((r·SH)·W + s·SW)·C
            for c in 0..C-1:
                output[out + c] = x[first + c];  argmax[out + c] = first + c
            for p in 0..PH-1:                   # row-major scan: p outer, q inner
                for q in 0..PW-1:
                    if p == 0 and q == 0: continue
                    base = ((r·SH + p)·W + s·SW + q)·C
                    for c in 0..C-1:            # contiguous inner loop (channels-last)
                        if x[base + c] > output[out + c]:        # strict ">": first maximum in scan order wins a tie
                            output[out + c] = x[base + c];  argmax[out + c] = base + c
    forwardDone = true

function MaxPooling2D::Backward(g):             # OPTIMIZE_FOR_SPEED
    really_assert(forwardDone)
    inputGradient = InputVector{}               # zero every call: "+=" below must not carry over
    for k in 0..Hout·Wout·C-1:
        inputGradient[argmax[k]] += g[k]        # "+=": with S < P one x can win several windows
    return inputGradient

function AveragePooling2D::Forward(x):         # OPTIMIZE_FOR_SPEED
    for r in 0..Hout-1:
        for s in 0..Wout-1:
            out = (r·Wout + s)·C
            for c in 0..C-1: output[out + c] = 0
            for p in 0..PH-1:
                for q in 0..PW-1:
                    base = ((r·SH + p)·W + s·SW + q)·C
                    for c in 0..C-1: output[out + c] += x[base + c]
            for c in 0..C-1: output[out + c] *= inversePoolArea
    forwardDone = true

function AveragePooling2D::Backward(g):         # OPTIMIZE_FOR_SPEED
    really_assert(forwardDone)
    inputGradient = InputVector{}
    for r in 0..Hout-1:
        for s in 0..Wout-1:
            out = (r·Wout + s)·C
            for p in 0..PH-1:
                for q in 0..PW-1:
                    base = ((r·SH + p)·W + s·SW + q)·C
                    for c in 0..C-1:
                        inputGradient[base + c] += g[out + c] · inversePoolArea
    return inputGradient
```

Math, convolution (one sample, filters `F[o,p,q,c]`, biases `b[o]`, upstream `g = ∂L/∂y`,
`[·]` = Iverson bracket):

```text
forward:           z[r,s,o] = b[o] + Σ_{p=0}^{KH−1} Σ_{q=0}^{KW−1} Σ_{c=0}^{Cin−1} F[o,p,q,c] · x[r·SH + p, s·SW + q, c]
                   y[r,s,o] = f(z[r,s,o])                                       (cross-correlation, kernel not flipped)
activation VJP:    δ = J_f(z)ᵀ g;  element-wise f ⇒ δ[r,s,o] = f'(z[r,s,o]) · g[r,s,o]
input Jacobian:    ∂z[r,s,o]/∂x[i,j,c] = F[o, i − r·SH, j − s·SW, c] · [0 ≤ i − r·SH < KH] · [0 ≤ j − s·SW < KW]
input gradient:    ∂L/∂x[i,j,c] = Σ_{r,s,o} δ[r,s,o] · F[o, i − r·SH, j − s·SW, c] · [0 ≤ i − r·SH < KH] · [0 ≤ j − s·SW < KW]
                   (transposed convolution; SH = SW = 1 ⇒ full 2D convolution of δ with F:
                    Σ_p Σ_q Σ_o F[o,p,q,c] · δ[i − p, j − q, o] over the valid (r, s) = (i − p, j − q))
                   i ≥ (Hout − 1)·SH + KH  or  j ≥ (Wout − 1)·SW + KW  (dropped by the floor) ⇒ ∂L/∂x[i,j,·] = 0
parameter grads:   ∂L/∂F[o,p,q,c] = Σ_{r,s} δ[r,s,o] · x[r·SH + p, s·SW + q, c]   (valid cross-correlation of x with δ)
                   ∂L/∂b[o]       = Σ_{r,s} δ[r,s,o]                                (bias shared over positions)
accumulation:      G ← G + ∇_θ L_s per Backward (N8); N16 divides by the batch size
```

Math, pooling (window `Ω(r,s) = {(r·SH + p, s·SW + q) : p < PH, q < PW}`, `K = PH·PW`):

```text
max forward:        y(r,s,c) = max_{(i,j) ∈ Ω(r,s)} x(i,j,c),   a(r,s,c) = first argmax in row-major (p, q) order
max Jacobian:       ∂y(r,s,c)/∂x(i,j,c') = [(i,j) = a(r,s,c)] · [c' = c]          (where the maximum is unique)
max backward:       ∂L/∂x(i,j,c) = Σ_{(r,s) : a(r,s,c) = (i,j)} g(r,s,c)

average forward:    y(r,s,c) = (1/K) Σ_{(i,j) ∈ Ω(r,s)} x(i,j,c)
average Jacobian:   ∂y(r,s,c)/∂x(i,j,c') = (1/K) · [(i,j) ∈ Ω(r,s)] · [c' = c]      (constant)
average backward:   ∂L/∂x(i,j,c) = (1/K) Σ_{(r,s) : (i,j) ∈ Ω(r,s)} g(r,s,c)

global average:     PH = H, PW = W:   y(c) = (1/(H·W)) Σ_{i,j} x(i,j,c),   ∂L/∂x(i,j,c) = g(c) / (H·W)
dropped tail:       (i,j) in no Ω(r,s) ⇒ ∂L/∂x(i,j,c) = 0
```

Max is not differentiable at a tie; routing the whole `g(r,s,c)` to the first maximum in scan order is a
valid subgradient (the gradient of the branch the strict `>` selects) and exact wherever the maximum is
unique.

## Complexity & memory

- `Convolution2D::Forward`: `Hout·Wout·Cout·KH·KW·Cin` MACs + the activation (`O(Hout·Wout·Cout)`). The
  innermost loop is a contiguous dot product of `KW·Cin` floats, repeated `KH` times per output; no index
  division, no scratch buffer.
- `Convolution2D::Backward`: `2·Hout·Wout·Cout·KH·KW·Cin` MACs (input and weight gradients, fused in one
  loop) + `Hout·Wout·Cout` adds (bias) + the activation VJP; `H·W·Cin` stores to clear `inputGradient`.
- RAM, trainable `Convolution2D`: `2P + 2·H·W·Cin + 3·Hout·Wout·Cout` floats + a reference, a `bool`
  and a vptr. Stack: `O(1)`.
  Example (keyword spotting, 49 frames × 10 MFCC, 8 filters 3×3, stride 1):
  `Convolution2D<float, 49, 10, 1, 8, 3, 3>` ⇒ `Hout = 47`, `Wout = 8`, `P = 80`, output `3008` floats,
  `27 072` MACs per `Forward`, `2·80 + 2·490 + 3·3008 = 10 164` floats (40 656 B). A `Dense` with the same
  `490 → 3008` I/O would need `1 476 928` parameters. Inference after N10: `2·3008 = 6016` floats.
  An im2col formulation would add a `Hout·Wout·KH·KW·Cin = 3384`-float scratch buffer (13.5 KB) on top.
  Stride 2 on a 28×28×1 input with 8 filters 3×3: `13 × 13 × 8` output, `12 168` MACs.
- `MaxPooling2D::Forward`: `Hout·Wout·C·(K − 1)` comparisons, no arithmetic. `Backward`: `H·W·C` zero
  stores + `Hout·Wout·C` adds.
- `AveragePooling2D::Forward`: `Hout·Wout·C·K` adds + `Hout·Wout·C` multiplies. `Backward`: `H·W·C` zero
  stores + `Hout·Wout·C·K` multiply-adds.
- RAM, pooling (floats; `size_t` is 4 B on Cortex-M): max `2·Hout·Wout·C + H·W·C`, average
  `Hout·Wout·C + H·W·C`, global average `C + H·W·C`; each + a `bool` and a vptr. Example: 2×2 max pool
  on the `47 × 8 × 8` map above ⇒ `23 × 4 × 8 = 736` outputs, `2·736 + 3008 = 4480` words (17.5 KiB);
  N10's inference-only variant drops `inputGradient` and `argmax`, leaving `736`.
- Parameters: pooling 0; the convolution's count is independent of `H` and `W`.

## Numerical / embedded notes

- **Cross-correlation, not convolution.** The kernel is not flipped, matching Keras `Conv2D`, PyTorch
  `nn.Conv2d`, TFLite and CMSIS-NN, so imported weights need no reversal.
- **Direct convolution, no im2col.** im2col copies each receptive field into a `KH·KW·Cin` scratch
  column so a matrix multiply can run (the full matrix is `Hout·Wout·KH·KW·Cin` floats; CMSIS-NN bounds
  it to two columns). With channels-last input a kernel row is already contiguous (`KW·Cin` floats), so
  the direct loop streams memory without the copy and without any scratch RAM. The loop order above (outputs outer, filter inner) needs one scalar
  accumulator; a Cout-inner order would need a `Cout`-float accumulator and is not required.
- **δ is a member buffer.** `Dense` and N11 keep `δ` as a stack temporary. A 2D map is larger (`3008`
  floats = 12 KB in the example above), which exceeds typical 2–8 KB MCU stacks, so `δ` lives in the
  object (`+Hout·Wout·Cout` floats, trainable variant only; N10 drops it).
- **Rounding.** Each output sums `KH·KW·Cin + 1` terms: worst-case relative error `≈ (KH·KW·Cin)·2⁻²⁴`
  (`3·3·8 = 72` ⇒ `≈ 4.3·10⁻⁶`). A weight gradient sums `Hout·Wout` terms per sample. Under
  `-ffast-math` the dot product may be reassociated or vectorised; tests use `EXPECT_NEAR`.
- **Activation scope.** `ForwardVector`/`BackwardVector` see the flat `Hout·Wout·Cout` map with no notion
  of channels. Use element-wise activations (ReLU, LeakyReLU, Tanh, Sigmoid, N1 Identity, N5, N6).
  Softmax here would normalise across positions and channels and is outside the contract.
- **Floor drops the tail.** Rows `i ≥ (Hout − 1)·SH + KH` and columns `j ≥ (Wout − 1)·SW + KW` are never
  read and get an exactly-zero input gradient; the same holds for the gaps when `S > K`. "same" padding
  is the caller's job (pad the input and size `H`, `W` accordingly); dilation and groups are deferred
  (depthwise is N19).
- **Pooling.** Max pooling only copies values, so it has no rounding at all. Average pooling sums first
  and multiplies once by the compile-time `1/K` (exact for powers of two, one rounding otherwise:
  `K = 6` ⇒ `0.16666667f`). The input gradient is zeroed then scattered on every `Backward`, never
  accumulated across calls.
- **Storage.** A `Model` holding a 2D layer is tens of KB (example above: ~40 KB trainable). Give it
  static storage duration, never an automatic (stack) variable on the target.
- Under `-ffast-math`, NaN/±∞ inputs or weights are outside the contract (the max-pool `>` is not
  reliable with NaN once finite math is assumed).
- Float-only: `static_assert(std::is_floating_point_v<T>)`; the generic `T` signature keeps a
  `Q15`/`Q31` (or affine int8) specialisation cheap to add later.

## Dependencies

- Requires **N8**: the accumulating `Backward`, `ParameterGradients()`/`ZeroGradients()`, the
  `Layer<T, In, Out, 0>` specialisation that pooling needs (`math::Vector<T, 0>` fails `Matrix`'s
  positive-dimension check) and the `if constexpr (ParameterSize > 0)` guards in `Model`. Also builds on
  `ActivationFunction` (`ForwardVector`/`BackwardVector`, exact derivatives), `numerical/math/Matrix.hpp`
  (`math::Vector`, `math::Matrix`, zero-filling default ctor), `numerical/math/CompilerOptimizations.hpp`,
  `infra/util/ReallyAssert.hpp`.
- **N11 / N12** (conventions, not code): the channels-last layout, the `θ` order and the valid-padding
  output formula. `Convolution2D<T, 1, L, Cin, Cout, 1, K, 1, S>` computes exactly
  `Convolution1D<T, L, Cin, Cout, K, S>` on the same `θ`, and `MaxPooling2D<T, 1, L, C, 1, K, 1, S>`
  equals `MaxPooling1D<T, L, C, K, S>`; once both exist the 1D batch classes may become aliases
  (N11's streaming class stays separate). N18 can deploy before or after them.
- **N10**: its flash storage policy applies to `Convolution2D` as to `Dense` (drops `G`, `input`,
  `preActivationGradient`, `inputGradient` and the RAM copy of `θ`) and yields inference-only pooling
  without `inputGradient`/`argmax`.
- Soft: **N1** Identity for a linear convolution output; **N3** initialisation uses
  `fan_in = KH·KW·Cin`, `fan_out = KH·KW·Cout` (Keras/PyTorch convention); **N15** produces the
  `frames × coefficients` map that feeds a `Convolution2D<float, T, M, 1, …>` with `H = T`, `W = M`.
- Used by: **N13** (export permutes Keras kernels `(KH, KW, Cin, Cout)` and PyTorch `(Cout, Cin, KH, KW)`
  to `[o][p][q][c]`; Keras `Flatten` after `Conv2D` already yields `(r·Wout + s)·Cout + o`, while a
  PyTorch NCHW flatten `o·Hout·Wout + r·Wout + s` needs the next Dense's columns permuted; frozen BN
  folds per output channel into `F_o`, `b[o]`), **N16** (training), **N19** (the depthwise stage reuses
  the window loops with one filter per channel; the pointwise stage is `KH = KW = 1`).
- Composes with `Model` through `make_layer<Convolution2D<float, …>>(kernels, activation)` and
  `make_layer<MaxPooling2D<float, …>>()`; `OutputSize = Hout·Wout·Cout` chains straight into a
  `Dense<float, Hout·Wout·Cout, …>`, pooling adds 0 to `TotalParameters`.

## Deployment

- Header: `neural_network/layer/Convolution2D.hpp` — `#pragma once` →
  `#pragma GCC optimize("O3","fast-math")` → includes (`neural_network/layer/TrainableLayerInterface.hpp`
  from N8, `neural_network/activation/ActivationFunction.hpp`, `numerical/math/Matrix.hpp`,
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`). `OPTIMIZE_FOR_SPEED` on
  `Forward`/`Backward`. Under `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`:
  `extern template class Convolution2D<float, 5, 4, 2, 2, 2, 3, 2, 1>;`
- Header: `neural_network/layer/Pooling2D.hpp` (both classes and the alias, next to N12's
  `Pooling1D.hpp`), same preamble plus `<array>`, `<cstddef>`; `OPTIMIZE_FOR_SPEED` on both `Forward`
  and both `Backward`. Under the coverage guard:
  `extern template class MaxPooling2D<float, 5, 4, 2, 2, 3, 2, 1>;`,
  `extern template class AveragePooling2D<float, 5, 4, 2, 2, 3, 2, 1>;`,
  `extern template class AveragePooling2D<float, 5, 4, 2, 5, 4, 1, 1>;` (the `GlobalAveragePooling2D<float, 5, 4, 2>` class).
- Coverage: `neural_network/layer/Convolution2D.cpp` →
  `template class Convolution2D<float, 5, 4, 2, 2, 2, 3, 2, 1>;` and
  `neural_network/layer/Pooling2D.cpp` → the same three pooling `template class …;` lines.
- Test: `neural_network/layer/test/TestConvolution2D.cpp` and `neural_network/layer/test/TestPooling2D.cpp`,
  plus one integration case in `neural_network/model/test/TestModel.cpp` (see `tests.md`).
- Doc: `doc/layer/Convolution2D.md` (per `doc/TEMPLATE.md`, one page covering 2D convolution and 2D
  pooling), with a row in `doc/layer/README.md`.
- CMake: `Convolution2D.hpp`, `Pooling2D.hpp` → `target_sources(neural_network.layer PRIVATE ...)`;
  `Convolution2D.cpp`, `Pooling2D.cpp` → `neural_network_add_coverage_sources(neural_network.layer ...)`;
  `TestConvolution2D.cpp`, `TestPooling2D.cpp` → `neural_network.layer_test` (already links `gmock_main`
  and has the QEMU hook).
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
