# 1D Convolution — Implementation Pseudocode

> Roadmap ref: #N11 (Tier 2) · Target: `neural_network/layer` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```
# sizes (all compile-time):
#   L = Length, Cin = InChannels, Cout = OutChannels, K = KernelSize, S = Stride
#   Lout = (L − K) / S + 1          (integer division = ⌊(L − K)/S⌋ + 1, "valid" padding)
#   P    = K·Cin·Cout + Cout
#
# channels-last flat layout, shared by every spatial layer (N12, N18, N19):
#   input  element (t, c) at x[t·Cin + c],   t ∈ [0, L),    c ∈ [0, Cin)
#   output element (t, o) at y[t·Cout + o],  t ∈ [0, Lout), o ∈ [0, Cout)
#   ⇒ the window of output position t is the contiguous slice x[t·S·Cin, t·S·Cin + K·Cin)
#   ⇒ a Dense after the convolution reads the Lout·Cout map directly: no Flatten layer

template<typename T, std::size_t Length, std::size_t InChannels, std::size_t OutChannels,
         std::size_t KernelSize, std::size_t Stride = 1>
                                               # static_assert(std::is_floating_point_v<T>); instantiated for float
                                               # static_assert(InChannels > 0 && OutChannels > 0)
                                               # static_assert(KernelSize > 0 && KernelSize <= Length && Stride > 0)
class Convolution1D final : public Layer<T, L·Cin, Lout·Cout, P>:
    const ActivationFunction<T>& activation    # element-wise, applied to the whole Lout·Cout map (as Dense)
    ParameterVector parameters                 # θ = [W_0 … W_{Cout−1}, b]; W_o = filter o, K·Cin values in (k, c) order
                                               #     W[o, k, c] at o·K·Cin + k·Cin + c,   b[o] at K·Cin·Cout + o
    ParameterVector parameterGradients         # G = Σ_samples ∇_θ L_s since the last ZeroGradients() (N8 contract)
    InputVector     input                      # x of the last Forward (needed for ∂L/∂W)
    OutputVector    preActivation              # z
    OutputVector    output                     # y = f(z)
    InputVector     inputGradient              # ∂L/∂x of the last Backward (per sample, not accumulated)
    bool            forwardDone = false
    # filter row W_o over the window is exactly a Dense row over K·Cin inputs, so θ uses the same
    # "multiplicative block, then biases" order as Dense and AffineNormalization.

template<typename T, std::size_t InChannels, std::size_t OutChannels, std::size_t KernelSize,
         std::size_t Stride = 1>              # same static_asserts (no Length: the stream is unbounded)
class Convolution1DStream:                     # inference-only causal streaming form; not a Layer, not in Model
    const ParameterVector&       parameters    # borrowed, same θ layout; a Convolution1D's Parameters()
                                               # or a constexpr flash table (N10)
    const ActivationFunction<T>& activation
    std::array<T, (K − 1)·Cin>   history{}     # ring of the last K − 1 input frames; zeros = causal left padding
    std::size_t                  oldest = 0    # ring slot holding the oldest frame
    std::size_t                  phase  = 0    # frames since the last emitted output, mod S
    OutputFrame                  preActivation # z of the last emitted frame (Cout)
    OutputFrame                  output        # y of the last emitted frame (Cout)
    # std::array<T, 0> is legal, so K = 1 (pointwise) needs no special storage.
```

## Interface

```
# Convolution1D<T, L, Cin, Cout, K, S>
using KernelMatrix = math::Matrix<T, Cout, K·Cin>   # row o = filter o, column k·Cin + c (same shape idea as Dense::WeightMatrix)
static constexpr std::size_t OutputLength = Lout

Convolution1D(const KernelMatrix& initialKernels, const ActivationFunction<T>& activation)   # biases = 0, as Dense

void                   Forward(const InputVector& input) override                  # hot path
const InputVector&     Backward(const OutputVector& outputGradient) override       # hot path; returns ∂L/∂x, accumulates G
const OutputVector&    Output() const override
const ParameterVector& Parameters() const override
void                   SetParameters(const ParameterVector& parameters) override   # does not touch G (N8)
const ParameterVector& ParameterGradients() const override                         # N8
void                   ZeroGradients() override                                    # N8
# inherited: using ValueType = T; virtual ~Layer() = default

# Convolution1DStream<T, Cin, Cout, K, S>
using InputFrame      = math::Vector<T, Cin>
using OutputFrame     = math::Vector<T, Cout>
using ParameterVector = math::Vector<T, K·Cin·Cout + Cout>    # same type as Convolution1D<…>::ParameterVector

Convolution1DStream(const ParameterVector& parameters, const ActivationFunction<T>& activation)
bool               Push(const InputFrame& frame)   # hot path; true ⇔ Output() now holds a new output frame
const OutputFrame& Output() const
void               Reset()                          # history = 0, oldest = 0, phase = 0
```

## Algorithm (pseudocode)

```
function Convolution1D(kernels, activation):
    for o in 0..Cout-1:
        for j in 0..K·Cin-1:  parameters[o·K·Cin + j] = kernels.at(o, j)
    # biases and G value-initialised to 0

function Forward(x):                            # OPTIMIZE_FOR_SPEED
    input = x
    for t in 0..Lout-1:
        base = t·S·Cin                          # start of the contiguous window
        for o in 0..Cout-1:
            s = parameters[K·Cin·Cout + o]      # bias
            w = o·K·Cin
            for j in 0..K·Cin-1:
                s += parameters[w + j] · x[base + j]
            preActivation[t·Cout + o] = s
    activation.ForwardVector(output, preActivation)
    forwardDone = true

function Backward(g):                           # OPTIMIZE_FOR_SPEED; g = ∂L/∂y, size Lout·Cout
    really_assert(forwardDone)
    δ = OutputVector{}                          # stack temporary, Lout·Cout floats (as Dense)
    activation.BackwardVector(δ, preActivation, output, g)       # δ = J_f(z)ᵀ g
    inputGradient = InputVector{}               # rows that no window covers stay 0
    for t in 0..Lout-1:
        base = t·S·Cin
        for o in 0..Cout-1:
            d = δ[t·Cout + o]
            w = o·K·Cin
            for j in 0..K·Cin-1:
                inputGradient[base + j]  += parameters[w + j] · d   # scatter: overlapping windows add up
                parameterGradients[w + j] += d · input[base + j]    # "+=": N8 accumulation
            parameterGradients[K·Cin·Cout + o] += d
    return inputGradient

function ZeroGradients():        parameterGradients = ParameterVector{}
function ParameterGradients():   return parameterGradients
function SetParameters(p):       parameters = p
function Parameters():           return parameters
function Output():               return output

function Push(frame):                           # OPTIMIZE_FOR_SPEED
    emit = (phase == 0)
    if emit:
        for o in 0..Cout-1:
            s = parameters[K·Cin·Cout + o]
            w = o·K·Cin
            for r in 0..K-2:                    # past frames, oldest first
                slot = oldest + r
                if slot >= K-1: slot -= K-1     # compare-and-subtract, no modulo in the loop
                for c in 0..Cin-1:
                    s += parameters[w + r·Cin + c] · history[slot·Cin + c]
            for c in 0..Cin-1:                  # current frame is kernel tap K−1
                s += parameters[w + (K-1)·Cin + c] · frame[c]
            preActivation[o] = s
        activation.ForwardVector(output, preActivation)
    if constexpr (K > 1):
        history[oldest·Cin .. oldest·Cin + Cin) = frame     # overwrite the oldest frame with the newest
        oldest = (oldest + 1 == K-1) ? 0 : oldest + 1
    phase = (phase + 1 == S) ? 0 : phase + 1
    return emit

function Reset():
    history = {};  oldest = 0;  phase = 0
```

Math (one sample, `W[o,k,c]`, `b[o]`, upstream `g = ∂L/∂y`, `[·]` = Iverson bracket):

```
forward:           z[t,o] = b[o] + Σ_{k=0}^{K−1} Σ_{c=0}^{Cin−1} W[o,k,c] · x[t·S + k, c]       (cross-correlation)
                   y[t,o] = f(z[t,o])
activation VJP:    δ = J_f(z)ᵀ g;  element-wise f ⇒ δ[t,o] = f'(z[t,o]) · g[t,o]
input Jacobian:    ∂z[t,o]/∂x[i,c] = W[o, i − t·S, c] · [0 ≤ i − t·S < K]
input gradient:    ∂L/∂x[i,c] = Σ_{t=0}^{Lout−1} Σ_o δ[t,o] · W[o, i − t·S, c] · [0 ≤ i − t·S < K]
                   (transposed convolution; S = 1 ⇒ full convolution of δ with W:
                    Σ_k Σ_o W[o,k,c] δ[i − k, o] over valid t = i − k)
                   i ≥ (Lout − 1)·S + K  (tail dropped by the floor) ⇒ ∂L/∂x[i,·] = 0
parameter grads:   ∂L/∂W[o,k,c] = Σ_t δ[t,o] · x[t·S + k, c]         (valid cross-correlation of x with δ)
                   ∂L/∂b[o]     = Σ_t δ[t,o]                          (bias shared over positions)
accumulation:      G ← G + ∇_θ L_s per Backward (N8); N16 divides by the batch size
```

Streaming equivalence (what `Convolution1DStream` computes):

```
frames x_0, x_1, … pushed one per call; x_n = 0 for n < 0 (zero history)
Push(x_n) returns true ⇔ n mod S = 0, and then
    Output() = f( b + Σ_k Σ_c W[·,k,c] · x_{n − (K−1) + k}[c] )
⇔ the batch Convolution1D output position t = n / S on the sequence left-padded with K − 1 zero frames
  (for S = 1 this is Keras Conv1D(padding="causal")); an emitted frame with n ≥ K − 1 also equals the
  unpadded batch output t = (n − K + 1) / S whenever that is an integer (always for S = 1; for S > 1
  exactly when (K − 1) mod S = 0)
```

## Complexity & memory

- `Forward`: `Lout·Cout·K·Cin` MACs + the activation (`O(Lout·Cout)`). Inner loop is a contiguous dot
  product of length `K·Cin`, no index division.
- `Backward`: `2·Lout·Cout·K·Cin` MACs (input + weight gradients, fused in one loop) + `Lout·Cout` adds
  (bias) + the activation VJP; `L·Cin` stores to clear `inputGradient`.
- `Push`: `Cout·K·Cin` MACs + activation on emitting frames, `Cin` copies on every frame; amortised
  `Cout·K·Cin / S` MACs per frame.
- RAM, `Convolution1D` (trainable, Dense storage pattern): `2P + 2·L·Cin + 2·Lout·Cout` floats + a
  reference, a `bool` and a vptr; Backward adds a `Lout·Cout` stack temporary.
  Example `Convolution1D<float, 128, 3, 8, 5>` (IMU window, 3 axes, 8 filters): `Lout = 124`, `P = 128`,
  `2·128 + 2·384 + 2·992 = 3008` floats (12 032 B), `14 880` MACs per `Forward`. A `Dense` with the same
  `384 → 992` I/O would need `381 920` parameters. `S = 2`: `Lout = 62`, 2016 floats, 7440 MACs.
- RAM, `Convolution1DStream`: `(K − 1)·Cin + 2·Cout` floats + two indices + two references; weights are
  borrowed, not copied. Same example: `4·3 + 2·8 = 28` floats.
- After N10 an inference-only `Convolution1D` keeps only `preActivation` and `output` (`2·Lout·Cout`)
  plus a reference to flash weights.

## Numerical / embedded notes

- **Cross-correlation, not convolution.** The kernel is not flipped, matching Keras `Conv1D` and PyTorch
  `nn.Conv1d`, so imported weights need no reversal. `analysis::LinearConvolution` (numerical) flips the
  kernel, returns the full `L + K − 1` length and is single-channel on `BoundedVector`, so it is not reused.
- **Rounding.** Each output is a `K·Cin + 1` term sum: worst-case relative error `≈ (K·Cin)·2⁻²⁴`
  (`K·Cin = 15` ⇒ `≈ 1·10⁻⁶`). A weight gradient sums `Lout` terms per sample (`≈ Lout·2⁻²⁴`). Under
  `-ffast-math` the compiler may reassociate/vectorise the dot product, so batch and stream agree to
  rounding, not bit-exactly; tests use `EXPECT_NEAR`.
- **Activation scope.** `ForwardVector`/`BackwardVector` see the flat `Lout·Cout` map with no notion of
  channels. Use element-wise activations (ReLU, LeakyReLU, Tanh, Sigmoid, N1 Identity, N5, N6). Softmax
  here would normalise across positions and channels and is outside the contract.
- **Floor drops the tail.** When `(L − K) mod S ≠ 0` the last `(L − K) mod S` frames are never read and
  their input gradient is exactly 0; with `S > K` the gaps between windows behave the same way.
  "same" padding is the caller's job (pad the input and size `L` accordingly); dilation is deferred.
- **Streaming.** No modulo on the hot path (compare-and-subtract ring index). The ring holds the last
  `K − 1` frames; the current frame is read from the argument, never copied before use. Non-emitting
  pushes do no MACs and no activation call. `Reset()` must clear both history and `phase`, otherwise
  the first frame after a reset is dropped when `S > 1`.
- **Borrowed parameters.** The stream holds `const ParameterVector&`; the referent must outlive it.
  Borrowing a `Convolution1D`'s `Parameters()` makes later `SetParameters` calls visible to the stream
  (train with the batch layer, run the stream on the device).
- Under `-ffast-math` non-finite inputs or weights are outside the contract.
- Float-only: `static_assert(std::is_floating_point_v<T>)`; the generic `T` signature keeps a
  `Q15`/`Q31` (or affine int8) specialisation cheap to add later.

## Dependencies

- Requires **N8**: the accumulating `Backward`, `ParameterGradients()`/`ZeroGradients()` and the
  `Layer` header it introduces. Also builds on `ActivationFunction` (`ForwardVector`/`BackwardVector`,
  exact derivatives), `numerical/math/Matrix.hpp` (`math::Vector`, `math::Matrix`, zero-filling default
  ctor, constexpr ctor), `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`.
- **N10**: its flash storage policy applies to `Convolution1D` as it does to `Dense` (drops `G`,
  `input`, `inputGradient` and the RAM copy of `θ`). `Convolution1DStream` already borrows `θ` by
  reference, so a `constexpr` flash table works without N10.
- Soft: **N1** Identity for a linear convolution output; **N3** initialisation uses
  `fan_in = K·Cin`, `fan_out = K·Cout` (Keras/PyTorch convention).
- Used by: **N12** (pooling reads the channels-last layout; global-average pooling replaces the flattened
  Dense head), **N13** (export permutes Keras kernels `(K, Cin, Cout)` → `[o][k][c]` and PyTorch
  `(Cout, Cin, K)` → `[o][k][c]`; Keras `Flatten` after `Conv1D` already yields `t·Cout + o`, while a
  PyTorch channels-first flatten `o·Lout + t` needs the next Dense's columns permuted; frozen BN folds
  per output channel into `W_o`, `b[o]`), **N16** (training), **N18** (2D generalisation, `H×W×C`
  channels-last), **N19** (depthwise 1D variant reuses the layout and loops).
- Composes with `Model` through `make_layer<Convolution1D<float, …>>(kernels, activation)`; its
  `OutputSize = Lout·Cout` chains straight into a `Dense<float, Lout·Cout, …>`.

## Deployment

- Header: `neural_network/layer/Convolution1D.hpp` — `#pragma once` →
  `#pragma GCC optimize("O3","fast-math")` → includes (the N8 layer header, `ActivationFunction.hpp`,
  `numerical/math/Matrix.hpp`, `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`,
  `<array>`). `OPTIMIZE_FOR_SPEED` on `Forward`/`Backward`/`Push`. Under
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`:
  `extern template class Convolution1D<float, 6, 2, 2, 3, 2>;` and
  `extern template class Convolution1DStream<float, 2, 2, 3, 2>;`
- Coverage: `neural_network/layer/Convolution1D.cpp` →
  `template class Convolution1D<float, 6, 2, 2, 3, 2>; template class Convolution1DStream<float, 2, 2, 3, 2>;`
- Test: `neural_network/layer/test/TestConvolution1D.cpp`, plus one integration case in
  `neural_network/model/test/TestModel.cpp` (see `tests.md`).
- Doc: `doc/layer/Convolution1D.md` (per `doc/TEMPLATE.md`), with a row in `doc/layer/README.md`.
- CMake: `Convolution1D.hpp` → `target_sources(neural_network.layer PRIVATE ...)`;
  `Convolution1D.cpp` → `neural_network_add_coverage_sources(neural_network.layer ...)`;
  `TestConvolution1D.cpp` → `neural_network.layer_test` (already links `gmock_main`).
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
