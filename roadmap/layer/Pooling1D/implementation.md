# 1D Pooling (Max / Average / Global-Average) — Implementation Pseudocode

> Roadmap ref: #N12 (Tier 2) · Target: `neural_network/layer` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```
# Layout (shared with N11): a sequence of Length positions × Channels channels is stored channels-last
# in one flat math::Vector<T, Length·Channels>:   x(t, c) = x[t·Channels + c]
# Valid padding only:   OutputLength = ⌊(Length − PoolSize) / Stride⌋ + 1
# Positions after the last full window are dropped (they get a zero input gradient).

template<typename T, std::size_t Length, std::size_t Channels, std::size_t PoolSize, std::size_t Stride = PoolSize>
class MaxPooling1D final : public Layer<T, Length·Channels, OutputLength·Channels, 0>:
                                               # static_assert(std::is_floating_point_v<T>); instantiated for float
    OutputVector                           output          # y(o, c) = max_k x(o·S + k, c)
    std::array<std::size_t, OutputSize>    argmax          # flat input index that produced y[j]
    InputVector                            inputGradient   # ∂L/∂x of the last Backward
    bool                                   forwardDone = false

template<typename T, std::size_t Length, std::size_t Channels, std::size_t PoolSize, std::size_t Stride = PoolSize>
class AveragePooling1D final : public Layer<T, Length·Channels, OutputLength·Channels, 0>:
    static constexpr T inversePoolSize = T{ 1 } / static_cast<T>(PoolSize)
    OutputVector   output                      # y(o, c) = (1/K) Σ_k x(o·S + k, c)
    InputVector    inputGradient
    bool           forwardDone = false
    # no saved input: the average's Jacobian is constant

template<typename T, std::size_t Length, std::size_t Channels>
using GlobalAveragePooling1D = AveragePooling1D<T, Length, Channels, Length, 1>
    # one window spanning the whole sequence: OutputLength = 1, output size = Channels

# K = PoolSize, S = Stride, L = Length, C = Channels, Lₒ = OutputLength.
# ParameterSize = 0: both classes use N8's parameter-free Layer specialisation and own no θ.
```

Compile-time guards (every class):

```
static_assert(std::is_floating_point_v<T>, "<Name> requires a floating-point type")
static_assert(Length > 0 && Channels > 0)
static_assert(PoolSize > 0 && Stride > 0)
static_assert(PoolSize <= Length, "pooling window longer than the sequence")
```

## Interface

```
static constexpr std::size_t OutputLength = (Length − PoolSize) / Stride + 1

MaxPooling1D()                                  # default-constructible, no arguments
AveragePooling1D()

void                Forward(const InputVector& input) override               # hot path
const InputVector&  Backward(const OutputVector& outputGradient) override    # hot path, returns ∂L/∂x
const OutputVector& Output() const override
# inherited from Layer<T, In, Out, 0> (N8): using ValueType = T; InputSize, OutputSize,
# ParameterSize = 0; virtual ~InferenceLayer() = default. No Parameters/SetParameters.
```

## Algorithm (pseudocode)

```
function MaxPooling1D::Forward(x):             # OPTIMIZE_FOR_SPEED
    for o in 0..Lₒ-1:
        first = o·S·C
        for c in 0..C-1:
            output[o·C + c] = x[first + c]
            argmax[o·C + c] = first + c
        for k in 1..K-1:
            row = (o·S + k)·C
            for c in 0..C-1:                   # contiguous inner loop (channels-last)
                if x[row + c] > output[o·C + c]:          # strict ">": the first maximum wins a tie
                    output[o·C + c] = x[row + c]
                    argmax[o·C + c] = row + c
    forwardDone = true

function MaxPooling1D::Backward(g):             # OPTIMIZE_FOR_SPEED
    really_assert(forwardDone)
    inputGradient = InputVector{}               # zero every call: "+=" below must not carry over
    for j in 0..Lₒ·C-1:
        inputGradient[argmax[j]] += g[j]        # "+=": with S < K one x can win several windows
    return inputGradient

function AveragePooling1D::Forward(x):         # OPTIMIZE_FOR_SPEED
    for o in 0..Lₒ-1:
        for c in 0..C-1: output[o·C + c] = x[o·S·C + c]
        for k in 1..K-1:
            row = (o·S + k)·C
            for c in 0..C-1: output[o·C + c] += x[row + c]
        for c in 0..C-1: output[o·C + c] *= inversePoolSize
    forwardDone = true

function AveragePooling1D::Backward(g):         # OPTIMIZE_FOR_SPEED
    really_assert(forwardDone)                  # same Layer contract as Dense
    inputGradient = InputVector{}
    for o in 0..Lₒ-1:
        for k in 0..K-1:
            row = (o·S + k)·C
            for c in 0..C-1:
                inputGradient[row + c] += g[o·C + c] · inversePoolSize
    return inputGradient

function Output(): return output
```

Math (window `o`, channel `c`, upstream `g = ∂L/∂y`, `W(o) = {o·S, …, o·S + K − 1}`):

```
max forward:        y(o, c) = max_{t ∈ W(o)} x(t, c),       a(o, c) = first argmax
max Jacobian:       ∂y(o, c)/∂x(t, c') = [t = a(o, c)] · [c' = c]      (where the maximum is unique)
max backward:       ∂L/∂x(t, c) = Σ_{o : a(o, c) = t} g(o, c)
                    (0 for a position that is never the maximum; a sum when windows overlap)

average forward:    y(o, c) = (1/K) Σ_{t ∈ W(o)} x(t, c)
average Jacobian:   ∂y(o, c)/∂x(t, c') = (1/K) · [t ∈ W(o)] · [c' = c]    (constant, independent of x)
average backward:   ∂L/∂x(t, c) = (1/K) Σ_{o : t ∈ W(o)} g(o, c)

global average:     K = L, S = 1, Lₒ = 1:   y(c) = (1/L) Σ_{t=0}^{L−1} x(t, c),   ∂L/∂x(t, c) = g(c) / L

dropped tail:       t ≥ (Lₒ − 1)·S + K  ⇒  t ∉ any W(o)  ⇒  ∂L/∂x(t, c) = 0
```

Max is not differentiable at a tie. Routing the whole `g(o, c)` to the first maximum is a valid
subgradient (it is the gradient of the piecewise-linear branch selected by the strict `>`), and it is
exact everywhere the window maximum is unique.

## Complexity & memory

- `MaxPooling1D::Forward`: `Lₒ·C·(K − 1)` comparisons, no arithmetic. `Backward`: `L·C` zero stores +
  `Lₒ·C` adds.
- `AveragePooling1D::Forward`: `Lₒ·C·(K − 1)` adds + `Lₒ·C` multiplies. `Backward`: `L·C` zero stores +
  `Lₒ·C·K` multiply-adds.
- `GlobalAveragePooling1D`: `Forward` `L·C` adds + `C` multiplies; `Backward` `L·C` multiply-adds.
- RAM (floats; `size_t` is 4 B on Cortex-M, the same as a float):
  - Max: `Lₒ·C` (output) + `L·C` (inputGradient) + `Lₒ·C` indices = **`2·Lₒ·C + L·C`** + `bool` + vptr.
  - Average: `Lₒ·C + L·C` + `bool` + vptr. Global average: `C + L·C`.
  - Example `L = 49`, `C = 64`, max `K = S = 2` (`Lₒ = 24`): `1536 + 3136 + 1536 = 6208` words (24.3 KiB).
    The input-gradient buffer dominates; N10's inference-only variant drops it and the argmax table,
    leaving only the `1536`-float output.
- Parameters: 0. Global-average pooling replaces a flatten→Dense head: for `Lₒ = 24`, `C = 64` and
  10 classes, `Dense<24·64, 10>` needs `15 370` parameters, `GlobalAveragePooling1D` + `Dense<64, 10>`
  needs `650`.

## Numerical / embedded notes

- Max pooling is exact: it only copies values, so forward and backward have no rounding at all.
- Average pooling sums first and multiplies by the compile-time constant `1/K` once per output. `1/K`
  is exact for powers of two and rounded once otherwise (`K = 3`: `0.33333334f`). The float32 sum of
  `K` terms has a worst-case relative error of about `K·2⁻²⁴`; for global average over `L = 1024`
  that is `≤ 6.1·10⁻⁵`, which needs no compensated sum.
- Ties in max pooling go to the first (lowest-index) maximum. The forward value does not depend on the
  tie rule; only which input receives the gradient does.
- The input gradient is overwritten (zeroed, then scattered) on every `Backward`, never accumulated
  across calls. Overlapping windows (`S < K`) add inside one call only.
- Under `-ffast-math`, NaN/±∞ inputs are outside the contract (the `>` comparison with a NaN is not
  reliable once the compiler assumes finite math).
- `Stride` defaults to `PoolSize` (non-overlapping), the Keras and PyTorch default. `S > K` is allowed
  (positions between windows are skipped and get a zero gradient).
- Only valid padding. "Same" padding (max pads with `−∞`, average divides by the number of real
  elements) is deferred until a model needs it.
- Float-only: `static_assert(std::is_floating_point_v<T>)`; the generic `T` signature keeps a
  `Q15`/`Q31` specialisation cheap to add later.

## Dependencies

- Builds on **N8** (hard dependency): the parameter-free `Layer<T, In, Out, 0>` specialisation (today's
  `Layer` cannot be instantiated with `ParameterSize = 0`, because `math::Vector<T, 0>` fails `Matrix`'s
  positive-dimension check) and the `if constexpr (ParameterSize > 0)` guards N8 adds to `Model`'s
  parameter helpers, so pooling layers own no slice of `θ`. Also `numerical/math/Matrix.hpp`
  (`math::Vector`, zero-filling default ctor), `infra/util/ReallyAssert.hpp`.
- **N11** (layout only): the channels-last `x[t·C + c]` convention. Pooling states the same convention
  itself, so it can deploy before or after N11. A `Dense` output of size `N` is the `L = N`, `C = 1` case.
- Composes with `Model` through `make_layer<MaxPooling1D<float, L, C, K>>()`; it adds 0 to
  `TotalParameters` (N8 keeps `static_assert(TotalParameters > 0)` for the whole chain).
- Used by: N18 (2D pooling reuses the same forward/backward rules over `H×W` windows), N10 (an
  inference-only variant without `inputGradient`/`argmax`), N13 (nothing to export; Keras
  `MaxPooling1D`/`AveragePooling1D`/`GlobalAveragePooling1D` with the default
  `data_format="channels_last"` and `padding="valid"` map one-to-one).

## Deployment

- Header: `neural_network/layer/Pooling1D.hpp` (all three: two classes and the alias). Order:
  `#pragma once`, then `#pragma GCC optimize("O3","fast-math")`, then the includes
  `neural_network/layer/TrainableLayerInterface.hpp` (N8's successor of `Layer.hpp`),
  `numerical/math/CompilerOptimizations.hpp`, `numerical/math/Matrix.hpp`, `infra/util/ReallyAssert.hpp`,
  `<array>`, `<cstddef>`. Put `OPTIMIZE_FOR_SPEED` on both `Forward` and both `Backward`, and add under
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`:
  `extern template class MaxPooling1D<float, 6, 2, 3, 2>;`,
  `extern template class AveragePooling1D<float, 6, 2, 3, 2>;`,
  `extern template class AveragePooling1D<float, 6, 2, 6, 1>;` (the `GlobalAveragePooling1D<float, 6, 2>` class).
- Coverage: `neural_network/layer/Pooling1D.cpp` → the same three `template class …;` lines.
- Test: `neural_network/layer/test/TestPooling1D.cpp`, plus one integration case in
  `neural_network/model/test/TestModel.cpp` (see `tests.md`).
- Doc: `doc/layer/Pooling1D.md` (per `doc/TEMPLATE.md`), with a row in `doc/layer/README.md`.
- CMake: `Pooling1D.hpp` → `target_sources(neural_network.layer PRIVATE ...)`;
  `Pooling1D.cpp` → `neural_network_add_coverage_sources(neural_network.layer ...)`;
  `TestPooling1D.cpp` → `neural_network.layer_test`.
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
