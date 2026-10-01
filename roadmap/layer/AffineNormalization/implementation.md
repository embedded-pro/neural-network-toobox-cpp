# Per-Feature Affine Normalisation — Implementation Pseudocode

> Roadmap ref: #N2 (Tier 1) · Target: `neural_network/layer` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```
template<typename T, std::size_t Size>         # static_assert(std::is_floating_point_v<T>); instantiated for float
class AffineNormalization final : public Layer<T, Size, Size, 2 * Size>:
    ParameterVector parameters                 # [a_0 … a_{N-1}, b_0 … b_{N-1}]  (scales, then shifts)
    OutputVector    output                     # y = a ⊙ x + b
    InputVector     inputGradient              # ∂L/∂x = a ⊙ g
    bool            forwardDone = false
    # N = Size. The layout "multiplicative block, then additive block" matches Dense (weights, then biases).
    # No saved input and no parameter-gradient buffer yet: ∂L/∂x does not need x.
    # N8 adds both (N + 2N floats) when it exposes parameter gradients.
```

## Interface

```
using ScaleVector = math::Vector<T, Size>        # same type as InputVector / OutputVector

AffineNormalization()                                             # identity: a = 1, b = 0
AffineNormalization(const ScaleVector& scale, const ScaleVector& shift)

static AffineNormalization FromBatchNorm(const ScaleVector& gamma, const ScaleVector& beta,
                                         const ScaleVector& mean, const ScaleVector& variance,
                                         T epsilon)              # frozen-BN fold, epsilon from the source framework
static AffineNormalization FromStandardisation(const ScaleVector& mean,
                                               const ScaleVector& standardDeviation)

void                   Forward(const InputVector& input) override               # hot path
const InputVector&     Backward(const OutputVector& outputGradient) override    # hot path
const OutputVector&    Output() const override
const ParameterVector& Parameters() const override
void                   SetParameters(const ParameterVector& parameters) override
# inherited: using ValueType = T; virtual ~Layer() = default
# ParameterSize = 2N > 0, so the Matrix<T, 0, 1> limitation never applies
```

## Algorithm (pseudocode)

```
function AffineNormalization():
    for i in 0..N-1:
        parameters[i]     = T{1}
        parameters[N + i] = T{0}

function AffineNormalization(scale, shift):
    for i in 0..N-1:
        parameters[i]     = scale[i]
        parameters[N + i] = shift[i]

function FromBatchNorm(gamma, beta, mean, variance, epsilon):
    for i in 0..N-1:
        really_assert(variance[i] + epsilon > T{0})
        a[i] = gamma[i] / math::Sqrt(variance[i] + epsilon)
        b[i] = beta[i] - a[i] * mean[i]
    return AffineNormalization(a, b)

function FromStandardisation(mean, standardDeviation):
    for i in 0..N-1:
        really_assert(standardDeviation[i] > T{0})
        a[i] = T{1} / standardDeviation[i]
        b[i] = -mean[i] * a[i]
    return AffineNormalization(a, b)

function Forward(input):                        # OPTIMIZE_FOR_SPEED
    for i in 0..N-1:
        output[i] = parameters[i] * input[i] + parameters[N + i]
    forwardDone = true

function Backward(g):                           # OPTIMIZE_FOR_SPEED
    really_assert(forwardDone)                  # same Layer contract as Dense
    for i in 0..N-1:
        inputGradient[i] = parameters[i] * g[i]
    return inputGradient

function SetParameters(p):  parameters = p
function Parameters():      return parameters
function Output():          return output
```

Math (per feature `i`, upstream `g = ∂L/∂y`):

```
forward:            y_i = a_i x_i + b_i
Jacobian (input):   ∂y_i/∂x_j = a_i δ_ij                 # J = diag(a), independent of x
backward (input):   ∂L/∂x_i = a_i g_i
parameter grads:    ∂L/∂a_i = g_i x_i,   ∂L/∂b_i = g_i    # derived here; stored/exposed by N8
```

Frozen batch normalisation at inference (Ioffe & Szegedy, Alg. 2):

```
BN(x_i) = γ_i (x_i − μ_i) / √(σ²_i + ε) + β_i
        = a_i x_i + b_i   with   a_i = γ_i / √(σ²_i + ε),   b_i = β_i − a_i μ_i       # exact identity
```

Standardisation is the special case `γ = 1`, `β = 0`, `ε = 0`, `σ²_i = s_i²`: `a_i = 1/s_i`, `b_i = −μ_i/s_i`.

Folding (the offline form that N13 applies; stated here so the tests can pin it):

```
before a Dense:  W (a ⊙ x + b) + c = (W · diag(a)) x + (W b + c)     # W'_{kj} = W_{kj} a_j,  c'_k = c_k + Σ_j W_{kj} b_j
after a linear (pre-activation) output z = W x + c:
                 a ⊙ z + b = (diag(a) · W) x + (a ⊙ c + b)          # W'_{kj} = a_k W_{kj},  c'_k = a_k c_k + b_k
```

This library's `Dense` applies its activation internally, so "BN between the linear map and the
activation" (the usual Keras/PyTorch placement) cannot be expressed as a runtime layer here and must be
folded by the exporter (N13). A runtime `AffineNormalization` covers input standardisation and
post-activation BN (which can also be folded into the following Dense with the first identity).

## Complexity & memory

- `Forward`: `N` multiplies + `N` adds (one FMA per feature); `O(N)`, no temporaries.
- `Backward`: `N` multiplies; `O(N)`.
- `FromBatchNorm`: `N` square roots + `N` divides, once at set-up. `FromStandardisation`: `N` divides.
- RAM: `2N` (parameters) + `N` (output) + `N` (inputGradient) = **`4N` floats** + one `bool` + vptr.
  With N8 it grows by `N` (saved input) + `2N` (parameter gradients) = `7N` floats.
- For comparison, a `Dense<T, N, N>` doing the same scaling holds `N² + N` parameters.

## Numerical / embedded notes

- Forward and backward are exact up to one rounding per operation; there is no division or `sqrt`
  on the hot path. The only `sqrt`/divides run once in the factory helpers.
- **Cancellation for a large offset.** `a x + b` with `|μ/s| ≫ 1` subtracts two large, nearly equal
  numbers. Example: a pressure sensor with `μ = 101325 Pa`, `s = 50 Pa` has `a = 0.02` and `b = −2026.5`.
  In float32 the worst absolute error of the standardised output over `x ∈ [101200, 101450]` is about
  `1.0e-4` (for example `x = 101401` gives `1.5198975` instead of `1.52`). That is harmless for a network
  input. If more precision is needed, remove the offset upstream in integer or ADC units.
- `epsilon` must be the value the source framework used (Keras default `1e-3`, PyTorch default `1e-5`).
  It is a required argument so that no default can silently diverge from the trained model.
- `variance` is the framework's stored running variance. Using it as-is reproduces the framework's
  inference output exactly, whatever estimator (biased or unbiased) produced it.
- `a_i = 0` is legal: that feature's output is the constant `b_i` and its input gradient is `0`.
  Negative scales are legal (a negative `γ`).
- Under `-ffast-math`, non-finite parameters (NaN/±∞) are outside the contract. The `really_assert`s
  in the helpers reject a non-positive `σ² + ε` or `s`.
- Float-only: `static_assert(std::is_floating_point_v<T>)`; the generic `T` signature keeps a
  `Q15`/`Q31` specialisation cheap to add later.

## Dependencies

- Builds on: `Layer.hpp` (float-only `T`, `ValueType`, `virtual ~Layer() = default`,
  `const ParameterVector& Parameters() const`), `numerical/math/Math.hpp` (`math::Sqrt`),
  `numerical/math/Matrix.hpp` (`math::Vector`), `infra/util/ReallyAssert.hpp`.
  It uses no other N-item and needs no frozen or non-trainable parameter support.
- Composes with `Model` through `make_layer<AffineNormalization<float, N>>(scale, shift)`. Its
  `2N` parameters join the flat `θ` like any other layer's.
- Used by: N13 (the exporter folds BN offline into `Dense`/`Conv`, and uses this layer when a BN cannot
  be folded, or to emit input standardisation), N8 (adds the parameter-gradient accessor `∂L/∂a = g ⊙ x`,
  `∂L/∂b = g`), N16 (fine-tuning the scale and shift on the device), N11/N18 (a per-channel variant
  can reuse the same math on the channels-last layout, broadcasting `a` and `b` over positions).
- Deferred (not this item): training-mode BN, which needs batch statistics that a one-sample
  `Forward` cannot supply.

## Deployment

- Header: `neural_network/layer/AffineNormalization.hpp`. Order: `#pragma once`, then
  `#pragma GCC optimize("O3","fast-math")`, then the includes `neural_network/layer/Layer.hpp`,
  `numerical/math/Math.hpp`, `numerical/math/CompilerOptimizations.hpp` and `infra/util/ReallyAssert.hpp`.
  Put `OPTIMIZE_FOR_SPEED` on `Forward`/`Backward`, and add
  `extern template class AffineNormalization<float, 3>;` under `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`.
- Coverage: `neural_network/layer/AffineNormalization.cpp` → `template class AffineNormalization<float, 3>;`
- Test: `neural_network/layer/test/TestAffineNormalization.cpp`, plus one integration case in
  `neural_network/model/test/TestModel.cpp` (see `tests.md`).
- Doc: `doc/layer/AffineNormalization.md` (per `doc/TEMPLATE.md`). Add its row to the layer
  documentation tables as `roadmap/DEPLOYMENT.md` step 6 describes.
- CMake: `AffineNormalization.hpp` → `target_sources(neural_network.layer PRIVATE ...)`;
  `AffineNormalization.cpp` → `neural_network_add_coverage_sources(neural_network.layer ...)`;
  `TestAffineNormalization.cpp` → `neural_network.layer_test`.
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
