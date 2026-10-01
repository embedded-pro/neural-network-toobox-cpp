# Depthwise-Separable Convolution — Implementation Pseudocode

> Roadmap ref: #N19 (Tier 3) · Target: `neural_network/layer` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```text
# sizes (all compile-time):
#   H = Height, W = Width, Cin = InChannels (= Channels for the depthwise-only layer), Cout = OutChannels
#   KH × KW = KernelHeight × KernelWidth,  SH × SW = StrideHeight × StrideWidth   (depthwise stage only)
#   Hout = (H − KH) / SH + 1,   Wout = (W − KW) / SW + 1      (integer division, "valid" padding, as N18)
#   M    = Hout·Wout                                          (output pixels)
#   Pdw  = KH·KW·Cin + Cin                                    (depthwise taps + per-channel biases)
#   Ppw  = Cin·Cout + Cout                                    (pointwise weights + biases)
#   P    = Pdw + Ppw                                          (standard convolution: KH·KW·Cin·Cout + Cout)
#
# channels-last (HWC) layout of N11/N18:
#   input        (i, j, c) at x [(i·W + j)·Cin + c]
#   intermediate (r, s, c) at a1[(r·Wout + s)·Cin + c]         (depthwise output, still Cin channels)
#   output       (r, s, o) at y [(r·Wout + s)·Cout + o]
#   ⇒ the Cin values of one pixel are contiguous in x, a1 and in each depthwise tap of θ (below),
#     so every inner loop runs over c with unit stride
#   ⇒ H = 1, KH = SH = 1 is N11's sequence layout with W playing the role of L

template<typename T, std::size_t Height, std::size_t Width, std::size_t Channels,
         std::size_t KernelHeight, std::size_t KernelWidth, std::size_t StrideHeight, std::size_t StrideWidth>
struct detail::DepthwiseKernel:                # stateless; the only copy of the depthwise loops, used by both layers
    static constexpr std::size_t OutputHeight = Hout, OutputWidth = Wout
    static constexpr std::size_t TapCount     = KH·KW·C
    static constexpr std::size_t ParameterSize = KH·KW·C + C

template<typename T, std::size_t Height, std::size_t Width, std::size_t Channels,
         std::size_t KernelHeight, std::size_t KernelWidth,
         std::size_t StrideHeight = 1, std::size_t StrideWidth = 1>
                                               # static_assert(std::is_floating_point_v<T>); instantiated for float
                                               # static_assert(Channels > 0)
                                               # static_assert(KernelHeight > 0 && KernelHeight <= Height)
                                               # static_assert(KernelWidth  > 0 && KernelWidth  <= Width)
                                               # static_assert(StrideHeight > 0 && StrideWidth > 0)
class DepthwiseConvolution2D final : public Layer<T, H·W·C, M·C, KH·KW·C + C>:
    const ActivationFunction<T>& activation    # element-wise, applied to the whole M·C map (as Dense)
    ParameterVector parameters                 # θ = [D, b]
                                               #   D[p, q, c] at (p·KW + q)·C + c    (tap-major, channel innermost)
                                               #   b[c]       at KH·KW·C + c
    ParameterVector parameterGradients         # G = Σ_samples ∇_θ L_s since the last ZeroGradients() (N8 contract)
    InputVector     input                      # x of the last Forward (needed for ∂L/∂D)
    OutputVector    preActivation              # z
    OutputVector    output                     # y = f(z)
    OutputVector    preActivationGradient      # δ = J_f(z)ᵀ g (member, as N18: maps are too large for the stack)
    InputVector     inputGradient              # ∂L/∂x of the last Backward (per sample, not accumulated)
    bool            forwardDone = false

template<typename T, std::size_t Height, std::size_t Width, std::size_t InChannels, std::size_t OutChannels,
         std::size_t KernelHeight, std::size_t KernelWidth,
         std::size_t StrideHeight = 1, std::size_t StrideWidth = 1>
                                               # same static_asserts, plus OutChannels > 0
class DepthwiseSeparableConvolution2D final : public Layer<T, H·W·Cin, M·Cout, P>:
    using IntermediateVector = math::Vector<T, M·Cin>
    const ActivationFunction<T>& depthwiseActivation   # f1, element-wise on the M·Cin map (Identity (N1) for the Keras form)
    const ActivationFunction<T>& pointwiseActivation   # f2, element-wise on the M·Cout map
    ParameterVector    parameters              # θ = [D, bD, V, bP]
                                               #   D[p, q, c] at (p·KW + q)·Cin + c           bD[c] at KH·KW·Cin + c
                                               #   V[o, c]    at Pdw + o·Cin + c               bP[o] at Pdw + Cin·Cout + o
    ParameterVector    parameterGradients      # G, same layout (N8)
    InputVector        input                   # x
    IntermediateVector depthwisePreActivation  # z1
    IntermediateVector depthwiseOutput         # a1 = f1(z1)   (input of the pointwise stage, needed for ∂L/∂V)
    OutputVector       preActivation           # z2
    OutputVector       output                  # y = f2(z2)
    OutputVector       preActivationGradient   # δ2 = J_f2(z2)ᵀ g
    IntermediateVector depthwiseOutputGradient # g1 = ∂L/∂a1
    IntermediateVector depthwisePreActivationGradient   # δ1 = J_f1(z1)ᵀ g1
    InputVector        inputGradient           # ∂L/∂x
    bool               forwardDone = false
    # θ = [θ of DepthwiseConvolution2D ; θ of Convolution2D<T, Hout, Wout, Cin, Cout, 1, 1> (N18)]:
    # the pointwise block is N18's filter-major "weights, then biases" order with KH = KW = 1,
    # i.e. one Dense<T, Cin, Cout> row block shared by every pixel.

# 1D variants (no extra code; H = 1 reproduces N11's layout exactly):
template<typename T, std::size_t Length, std::size_t Channels, std::size_t KernelSize, std::size_t Stride = 1>
using DepthwiseConvolution1D = DepthwiseConvolution2D<T, 1, Length, Channels, 1, KernelSize, 1, Stride>
template<typename T, std::size_t Length, std::size_t InChannels, std::size_t OutChannels,
         std::size_t KernelSize, std::size_t Stride = 1>
using DepthwiseSeparableConvolution1D =
    DepthwiseSeparableConvolution2D<T, 1, Length, InChannels, OutChannels, 1, KernelSize, 1, Stride>
```

Why tap-major depthwise storage (unlike the filter-major kernels of N11/N18): one depthwise filter
touches a single channel, whose samples sit `Cin` floats apart in the channels-last input. Storing
`D[p, q, ·]` contiguously lets the inner loop run over `c` with unit stride in `D`, `x` and `z`
simultaneously. It is also exactly the flattened Keras `DepthwiseConv2D` kernel `(KH, KW, C, 1)` and the
TFLite / CMSIS-NN depthwise filter `(1, KH, KW, C)`.

## Interface

```text
# DepthwiseConvolution2D<T, H, W, C, KH, KW, SH, SW>
using KernelMatrix = math::Matrix<T, KH·KW, C>          # row p·KW + q, column c
static constexpr std::size_t OutputHeight = Hout
static constexpr std::size_t OutputWidth  = Wout

DepthwiseConvolution2D(const KernelMatrix& initialKernels, const ActivationFunction<T>& activation)   # biases = 0

void                   Forward(const InputVector& input) override                  # hot path
const InputVector&     Backward(const OutputVector& outputGradient) override       # hot path; returns ∂L/∂x, accumulates G
const OutputVector&    Output() const override
const ParameterVector& Parameters() const override
void                   SetParameters(const ParameterVector& parameters) override   # does not touch G (N8)
const ParameterVector& ParameterGradients() const override                         # N8
void                   ZeroGradients() override                                    # N8
# inherited: using ValueType = T; virtual ~Layer() = default

# DepthwiseSeparableConvolution2D<T, H, W, Cin, Cout, KH, KW, SH, SW>
using DepthwiseKernelMatrix = math::Matrix<T, KH·KW, Cin>   # as above
using PointwiseMatrix       = math::Matrix<T, Cout, Cin>    # row o = output channel (same shape idea as Dense::WeightMatrix)
static constexpr std::size_t OutputHeight = Hout
static constexpr std::size_t OutputWidth  = Wout

DepthwiseSeparableConvolution2D(const DepthwiseKernelMatrix& depthwiseKernels,
                                const ActivationFunction<T>& depthwiseActivation,
                                const PointwiseMatrix&       pointwiseWeights,
                                const ActivationFunction<T>& pointwiseActivation)   # bD = bP = 0
# same seven overrides as above

# detail::DepthwiseKernel<T, H, W, C, KH, KW, SH, SW>   (not part of the public API)
static void Forward(std::span<const T> theta, std::span<const T> x, std::span<T> z)            # OPTIMIZE_FOR_SPEED
static void Backward(std::span<const T> theta, std::span<const T> x, std::span<const T> delta,
                     std::span<T> thetaGradient, std::span<T> inputGradient)                   # OPTIMIZE_FOR_SPEED
# theta / thetaGradient span the first Pdw values of the owning layer's θ / G
```

## Algorithm (pseudocode)

```text
function DepthwiseKernel::Forward(θ, x, z):              # OPTIMIZE_FOR_SPEED; direct, no scratch
    for r in 0..Hout-1:
        for s in 0..Wout-1:
            out = (r·Wout + s)·C
            for c in 0..C-1: z[out + c] = θ[KH·KW·C + c]            # bias
            for p in 0..KH-1:
                for q in 0..KW-1:
                    base = ((r·SH + p)·W + s·SW + q)·C              # pixel (r·SH + p, s·SW + q)
                    tap  = (p·KW + q)·C
                    for c in 0..C-1:                                # unit stride in θ, x and z
                        z[out + c] += θ[tap + c] · x[base + c]

function DepthwiseKernel::Backward(θ, x, δ, G, gx):      # OPTIMIZE_FOR_SPEED
    gx = 0                                               # pixels no window covers stay 0
    for r in 0..Hout-1:
        for s in 0..Wout-1:
            out = (r·Wout + s)·C
            for p in 0..KH-1:
                for q in 0..KW-1:
                    base = ((r·SH + p)·W + s·SW + q)·C
                    tap  = (p·KW + q)·C
                    for c in 0..C-1:
                        gx[base + c] += θ[tap + c] · δ[out + c]     # scatter: overlapping windows add up
                        G[tap + c]   += δ[out + c] · x[base + c]    # "+=": N8 accumulation
            for c in 0..C-1: G[KH·KW·C + c] += δ[out + c]

# ---- DepthwiseConvolution2D ----
function DepthwiseConvolution2D(kernels, activation):
    for t in 0..KH·KW-1:
        for c in 0..C-1: parameters[t·C + c] = kernels.at(t, c)
    # biases and G value-initialised to 0

function Forward(x):                                     # OPTIMIZE_FOR_SPEED
    input = x
    DepthwiseKernel::Forward(parameters, x, preActivation)
    activation.ForwardVector(output, preActivation)
    forwardDone = true

function Backward(g):                                    # OPTIMIZE_FOR_SPEED; g = ∂L/∂y, size M·C
    really_assert(forwardDone)
    activation.BackwardVector(preActivationGradient, preActivation, output, g)          # δ = J_f(z)ᵀ g
    DepthwiseKernel::Backward(parameters, input, preActivationGradient, parameterGradients, inputGradient)
    return inputGradient

# ---- DepthwiseSeparableConvolution2D ----
function DepthwiseSeparableConvolution2D(dwKernels, f1, pwWeights, f2):
    for t in 0..KH·KW-1:
        for c in 0..Cin-1: parameters[t·Cin + c] = dwKernels.at(t, c)
    for o in 0..Cout-1:
        for c in 0..Cin-1: parameters[Pdw + o·Cin + c] = pwWeights.at(o, c)
    # bD, bP and G value-initialised to 0

function Forward(x):                                     # OPTIMIZE_FOR_SPEED
    input = x
    DepthwiseKernel::Forward(parameters[0, Pdw), x, depthwisePreActivation)
    depthwiseActivation.ForwardVector(depthwiseOutput, depthwisePreActivation)
    for m in 0..M-1:                                     # 1×1 convolution = one Dense per pixel
        for o in 0..Cout-1:
            acc = parameters[Pdw + Cin·Cout + o]         # bP[o]
            row = Pdw + o·Cin
            for c in 0..Cin-1:
                acc += parameters[row + c] · depthwiseOutput[m·Cin + c]
            preActivation[m·Cout + o] = acc
    pointwiseActivation.ForwardVector(output, preActivation)
    forwardDone = true

function Backward(g):                                    # OPTIMIZE_FOR_SPEED; g = ∂L/∂y, size M·Cout
    really_assert(forwardDone)
    pointwiseActivation.BackwardVector(preActivationGradient, preActivation, output, g)             # δ2
    depthwiseOutputGradient = IntermediateVector{}
    for m in 0..M-1:
        for o in 0..Cout-1:
            d   = preActivationGradient[m·Cout + o]
            row = Pdw + o·Cin
            for c in 0..Cin-1:
                depthwiseOutputGradient[m·Cin + c] += parameters[row + c] · d        # g1 = Vᵀ δ2 per pixel
                parameterGradients[row + c]        += d · depthwiseOutput[m·Cin + c] # ∂L/∂V, accumulated
            parameterGradients[Pdw + Cin·Cout + o] += d                              # ∂L/∂bP
    depthwiseActivation.BackwardVector(depthwisePreActivationGradient, depthwisePreActivation,
                                       depthwiseOutput, depthwiseOutputGradient)                   # δ1
    DepthwiseKernel::Backward(parameters[0, Pdw), input, depthwisePreActivationGradient,
                              parameterGradients[0, Pdw), inputGradient)
    return inputGradient

# both layers
function ZeroGradients():        parameterGradients = ParameterVector{}
function ParameterGradients():   return parameterGradients
function SetParameters(p):       parameters = p
function Parameters():           return parameters
function Output():               return output
```

No `BackwardVector` call aliases its result with an input span: the activation contract does not
promise in-place safety, hence the separate `g1` and `δ1` buffers.

Math, depthwise stage (one sample, taps `D[p,q,c]`, biases `bD[c]`, `[·]` = Iverson bracket):

```text
forward:           z1[r,s,c] = bD[c] + Σ_{p=0}^{KH−1} Σ_{q=0}^{KW−1} D[p,q,c] · x[r·SH + p, s·SW + q, c]
                   a1[r,s,c] = f1(z1[r,s,c])                       (per-channel cross-correlation, no flip)
activation VJP:    δ1 = J_f1(z1)ᵀ g1;  element-wise f1 ⇒ δ1[r,s,c] = f1'(z1[r,s,c]) · g1[r,s,c]
input Jacobian:    ∂z1[r,s,c]/∂x[i,j,c'] = [c' = c] · D[i − r·SH, j − s·SW, c] · [0 ≤ i − r·SH < KH] · [0 ≤ j − s·SW < KW]
input gradient:    ∂L/∂x[i,j,c] = Σ_{r,s} δ1[r,s,c] · D[i − r·SH, j − s·SW, c] · [0 ≤ i − r·SH < KH] · [0 ≤ j − s·SW < KW]
                   (per-channel transposed convolution: no sum over channels, unlike N18)
                   i ≥ (Hout − 1)·SH + KH  or  j ≥ (Wout − 1)·SW + KW  (dropped by the floor) ⇒ ∂L/∂x[i,j,·] = 0
parameter grads:   ∂L/∂D[p,q,c] = Σ_{r,s} δ1[r,s,c] · x[r·SH + p, s·SW + q, c]
                   ∂L/∂bD[c]    = Σ_{r,s} δ1[r,s,c]
```

For `DepthwiseConvolution2D` alone: `g1 = g`, `f1 = f`, `y = a1`.

Math, pointwise stage (weights `V[o,c]`, biases `bP[o]`, upstream `g = ∂L/∂y`):

```text
forward:           z2[r,s,o] = bP[o] + Σ_{c=0}^{Cin−1} V[o,c] · a1[r,s,c]
                   y[r,s,o]  = f2(z2[r,s,o])
activation VJP:    δ2[r,s,o] = f2'(z2[r,s,o]) · g[r,s,o]
intermediate grad: g1[r,s,c] = ∂L/∂a1[r,s,c] = Σ_o V[o,c] · δ2[r,s,o]          (Vᵀ δ2 at each pixel)
parameter grads:   ∂L/∂V[o,c] = Σ_{r,s} δ2[r,s,o] · a1[r,s,c]
                   ∂L/∂bP[o]  = Σ_{r,s} δ2[r,s,o]
chain:             g ──f2'──▶ δ2 ──Vᵀ──▶ g1 ──f1'──▶ δ1 ──Dᵀ (per channel)──▶ ∂L/∂x
accumulation:      G ← G + ∇_θ L_s per Backward (N8); N16 divides by the batch size
equivalence:       DepthwiseSeparableConvolution2D(θ) ≡ Model chain
                   DepthwiseConvolution2D<T, H, W, Cin, KH, KW, SH, SW>(θ[0, Pdw), f1)
                     → Convolution2D<T, Hout, Wout, Cin, Cout, 1, 1>(θ[Pdw, P), f2)          (N18)
                   in outputs, ∂L/∂x and G (up to rounding)
```

`f'` values come from the activations' exact derivatives (ReLU `1`/`0`, LeakyReLU `1`/`α`,
Sigmoid `σ(1 − σ)`, Tanh `1 − t²`).

## Complexity & memory

- Depthwise stage: `Forward` `M·Cin·KH·KW` MACs; `Backward` `2·M·Cin·KH·KW` MACs + `M·Cin` adds (bias) +
  `H·W·Cin` stores to clear `inputGradient`.
- Pointwise stage: `Forward` `M·Cin·Cout` MACs; `Backward` `2·M·Cin·Cout` MACs + `M·Cout` adds +
  `M·Cin` stores to clear `g1`. Plus one activation (and one VJP) per stage.
- Against `Convolution2D` (N18) with the same `KH, KW, SH, SW`: MACs `M·Cin·(KH·KW + Cout)` vs
  `M·Cin·Cout·KH·KW`, ratio `1/Cout + 1/(KH·KW)` (Howard et al., §3.1); parameters `P` vs
  `KH·KW·Cin·Cout + Cout`.
- RAM, trainable (floats; + references, a `bool` and a vptr; stack `O(1)`):
  `DepthwiseConvolution2D`: `2·Pdw + 2·H·W·C + 3·M·C`.
  `DepthwiseSeparableConvolution2D`: `2·P + 2·H·W·Cin + 4·M·Cin + 3·M·Cout`.
- Example (DS-CNN body block on a 25 × 5 × 64 map, 3×3, 64 → 64):
  `DepthwiseSeparableConvolution2D<float, 25, 5, 64, 64, 3, 3>` ⇒ `Hout = 23`, `Wout = 3`, `M = 69`,
  `P = 640 + 4160 = 4800` (standard: `36 928`), MACs `39 744 + 282 624 = 322 368` (standard:
  `2 543 616`, ratio `0.127 = 1/64 + 1/9`). Trainable RAM `56 512` floats (226 KB): on-device training of
  a DS-CNN body is not a target. Fused inference after N10: `M·Cout + Cin = 4480` floats (17.5 KiB)
  versus `M·(Cin + Cout) = 8832` for an unfused chain that stores the intermediate map.
- Example (1D, IMU window 128 × 3, `K = 5`, 16 outputs):
  `DepthwiseSeparableConvolution1D<float, 128, 3, 16, 5>` ⇒ `Lout = 124`, `P = 82` (standard `256`),
  `7812` MACs (standard `29 760`, ratio `0.2625`), trainable RAM `8372` floats (33 488 B).
- Test fixture `DepthwiseSeparableConvolution2D<float, 3, 5, 2, 3, 2, 2, 1, 2>`: `P = 19`, 56 MACs,
  166 floats.

## Numerical / embedded notes

- **Cross-correlation, not convolution.** Depthwise taps are not flipped (Keras `DepthwiseConv2D`,
  PyTorch `Conv2d(groups = Cin)`, TFLite, CMSIS-NN), so imported kernels need no reversal.
- **Channel-innermost loops.** Every inner loop runs over `c` with unit stride in `θ`, `x`, `z1`, `a1`,
  so the compiler can vectorise it (Helium/NEON) and no index division occurs. The depthwise stage
  accumulates straight into `z1`, the pointwise stage into one scalar: no scratch buffer.
- **Rounding.** A depthwise output sums `KH·KW + 1` terms (3×3 ⇒ `≈ 9·2⁻²⁴ ≈ 5.4·10⁻⁷` relative), a
  pointwise output `Cin + 1` terms. Weight gradients sum `M` terms per sample. Under `-ffast-math` the
  sums may be reassociated; tests use `EXPECT_NEAR`.
- **Redundant bias with a linear depthwise stage.** With `f1 = Identity` (Keras `SeparableConv2D`),
  `bD` folds into `bP' = bP + V·bD`; the parametrisation is redundant but the gradients stay exact.
  The exporter (N13) sets `bD = 0` in that case.
- **Activation scope.** Both activations see flat maps (`M·Cin`, `M·Cout`) with no notion of channels.
  Use element-wise activations (ReLU, LeakyReLU, Tanh, Sigmoid, N1 Identity, N5, N6); Softmax is outside
  the contract. MobileNet and DS-CNN use ReLU after both stages (BN folded in); implementations that
  use ReLU6 instead (e.g. Keras `MobileNet`) have no matching activation class here yet.
- **Floor drops the tail.** Rows `i ≥ (Hout − 1)·SH + KH` and columns `j ≥ (Wout − 1)·SW + KW` are never
  read and get an exactly-zero input gradient; so do the gaps when a stride exceeds its kernel extent.
- **Padding and depth multiplier.** "valid" only, as N11/N18. DS-CNN is usually trained with "same"
  padding; such a model cannot be imported layer by layer until a zero-padding layer exists (deferred).
  Train with "valid" padding instead. Depth multiplier `> 1` (Keras `depth_multiplier`, PyTorch
  `out_channels = k·Cin`) is deferred; `1` is what MobileNet and DS-CNN use.
- **Storage.** A `Model` holding these layers is tens of KB; give it static storage duration, never an
  automatic (stack) variable on the target.
- Under `-ffast-math` non-finite inputs or weights are outside the contract.
- Float-only: `static_assert(std::is_floating_point_v<T>)`; the generic `T` signature keeps a
  `Q15`/`Q31` (or affine int8, CMSIS-NN's `arm_depthwise_conv_s8`) specialisation cheap to add later.

## Dependencies

- Requires **N8**: the accumulating `Backward`, `ParameterGradients()`/`ZeroGradients()` and the
  `TrainableLayerInterface.hpp` header. Also builds on `ActivationFunction` (`ForwardVector`/
  `BackwardVector`, exact derivatives), `numerical/math/Matrix.hpp` (`math::Vector`, `math::Matrix`,
  zero-filling default ctor, contiguous iterators for `std::span`),
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`.
- **N11 / N18** (conventions, not code): channels-last layout, "valid" output formula and the pointwise
  block's `θ` order. No N18 code is reused, so N19 can deploy before or after N18; once both exist the
  equivalence above holds and the 1D aliases chain with `Convolution1D` and `Pooling1D` (N12).
- **N10**: flash storage for `θ`, and an inference-only fused variant that computes one pixel's `Cin`
  depthwise values into a `Cin`-float scratch and applies the pointwise stage immediately, so `z1`, `a1`
  are never stored (the reason the fused class exists besides chaining two layers).
- Soft: **N1** Identity for `f1` in the Keras form; **N3** initialisation: depthwise
  `fan_in = KH·KW` (each output reads `KH·KW` inputs; PyTorch's convention, Keras' generic rule gives
  `KH·KW·Cin`), pointwise `fan_in = Cin`, `fan_out = Cout`; **N15** produces the MFCC map for DS-CNN.
- Used by: **N13** (Keras `DepthwiseConv2D` kernel `(KH, KW, C, 1)` and TFLite `(1, KH, KW, C)` flatten
  to `D` directly; PyTorch `(C, 1, KH, KW)` is permuted to `(KH, KW, C)`; Keras `SeparableConv2D`
  pointwise `(1, 1, Cin, Cout)` is transposed to `V[o][c]`, PyTorch `(Cout, Cin, 1, 1)` maps directly;
  frozen BN folds per channel into `D[·,·,c]`, `bD[c]` and per output channel into `V[o,·]`, `bP[o]`),
  **N16** (training).
- Composes with `Model` through
  `make_layer<DepthwiseSeparableConvolution2D<float, …>>(dwKernels, relu, pwWeights, relu)` (both
  activations are lvalue references, kept by `make_layer`'s `std::reference_wrapper` handling);
  `OutputSize = M·Cout` chains straight into a `Dense<float, M·Cout, …>`.

## Deployment

- Header: `neural_network/layer/DepthwiseSeparableConvolution.hpp` (`detail::DepthwiseKernel`,
  `DepthwiseConvolution2D`, `DepthwiseSeparableConvolution2D`, both 1D aliases) — `#pragma once` →
  `#pragma GCC optimize("O3","fast-math")` → includes (`neural_network/layer/TrainableLayerInterface.hpp`
  from N8, `neural_network/activation/ActivationFunction.hpp`, `numerical/math/Matrix.hpp`,
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`, `<cstddef>`, `<span>`).
  `OPTIMIZE_FOR_SPEED` on both kernel functions and every `Forward`/`Backward`. Under
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`:
  `extern template class DepthwiseConvolution2D<float, 3, 5, 2, 2, 2, 1, 2>;` and
  `extern template class DepthwiseSeparableConvolution2D<float, 3, 5, 2, 3, 2, 2, 1, 2>;`
- Coverage: `neural_network/layer/DepthwiseSeparableConvolution.cpp` →
  `template class DepthwiseConvolution2D<float, 3, 5, 2, 2, 2, 1, 2>;`
  `template class DepthwiseSeparableConvolution2D<float, 3, 5, 2, 3, 2, 2, 1, 2>;`
- Test: `neural_network/layer/test/TestDepthwiseSeparableConvolution.cpp`, plus one integration case in
  `neural_network/model/test/TestModel.cpp` (see `tests.md`).
- Doc: `doc/layer/DepthwiseSeparableConvolution.md` (per `doc/TEMPLATE.md`), with a row in
  `doc/layer/README.md`.
- CMake: `DepthwiseSeparableConvolution.hpp` → `target_sources(neural_network.layer PRIVATE ...)`;
  `DepthwiseSeparableConvolution.cpp` → `neural_network_add_coverage_sources(neural_network.layer ...)`;
  `TestDepthwiseSeparableConvolution.cpp` → `neural_network.layer_test` (already links `gmock_main` and
  has the QEMU hook).
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
