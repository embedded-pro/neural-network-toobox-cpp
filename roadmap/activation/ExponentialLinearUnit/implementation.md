# ELU / SELU Activation — Implementation Pseudocode

> Roadmap ref: #N6 (Tier 1) · Target: `neural_network/activation` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```
template<typename T>                           # static_assert(std::is_floating_point_v<T>); instantiated for float
class ExponentialLinearUnit final : public ActivationFunction<T>:
    static constexpr T defaultAlpha{ 1 }                                           # plain ELU
    static constexpr T seluAlpha{ static_cast<T>(1.6732632423543772848170429916717L) }   # α₀₁ (Klambauer 2017)
    static constexpr T seluScale{ static_cast<T>(1.0507009873554804934193349852946L) }   # λ₀₁

    T scale          # λ  (1 for ELU)
    T scaledAlpha    # λ·α, precomputed once: the negative saturation value and the x ≤ 0 derivative offset
    # no size template parameter: like ReLU/LeakyReLU, the vector length comes from the spans
```

One class covers both: ELU is `(α, λ) = (α, 1)`, SELU is `(α₀₁, λ₀₁)`. `α` itself is not stored — only
the product `λα` is ever used.

## Interface

```
explicit ExponentialLinearUnit(T alpha = defaultAlpha, T scale = T{1})   # really_assert(alpha > 0 and scale > 0)
static ExponentialLinearUnit Selu()                                      # returns ExponentialLinearUnit{ seluAlpha, seluScale }
T    Forward(T x) const                                                  # hot path
T    Backward(T x) const                                                 # exact f'(x); one exp for x ≤ 0
void ForwardVector(span<T> output, span<const T> input) const            # hot path
void BackwardVector(span<T> result, span<const T> preActivation,
                    span<const T> output, span<const T> outputGradient) const   # exp-free: reuses output
# all four override ActivationFunction<T>; the vector overrides remove the per-element virtual call
# inherited: virtual ~ActivationFunction() = default
```

## Algorithm (pseudocode)

```
constructor(alpha, scale):
    really_assert(alpha > 0 and scale > 0)
    this.scale       = scale
    this.scaledAlpha = scale * alpha

function Forward(x):                            # OPTIMIZE_FOR_SPEED
    if x > 0: return scale * x
    return scaledAlpha * (math::Exp(x) - 1)     # x ≤ 0 ⇒ Exp(x) ∈ (0, 1]: never overflows

function Backward(x):                           # OPTIMIZE_FOR_SPEED — exact derivative
    if x > 0: return scale
    return scaledAlpha * math::Exp(x)           # x = 0 takes this branch ⇒ f'(0) = λα

function ForwardVector(output, input):          # OPTIMIZE_FOR_SPEED
    really_assert(output.size() == input.size())
    for i in 0..N-1:
        output[i] = Forward(input[i])           # alias-safe: reads input[i] before writing output[i]

function BackwardVector(result, preActivation, output, outputGradient):   # OPTIMIZE_FOR_SPEED
    really_assert(result.size() == preActivation.size()
                  and result.size() == output.size()
                  and result.size() == outputGradient.size())
    for i in 0..N-1:
        if preActivation[i] > 0:                # branch on the pre-activation sign, value from the output
            result[i] = outputGradient[i] * scale
        else:
            result[i] = outputGradient[i] * (output[i] + scaledAlpha)   # = λα·e^z, no exp
```

`BackwardVector` relies on the existing `ActivationFunction` contract (already used by Sigmoid and Tanh):
`output` is the `ForwardVector` result for the same `preActivation`, which `Dense::Backward` guarantees.

Math (per element, `a = f(z)`, `λ > 0`, `α > 0`):

```
forward:   f(z) = λ z                 z > 0
           f(z) = λ α (eᶻ − 1)        z ≤ 0
derivative:
           f'(z) = λ                  z > 0
           f'(z) = λ α eᶻ             z ≤ 0
                 = f(z) + λ α         z ≤ 0            # since λα eᶻ = λα(eᶻ − 1) + λα
Jacobian:  ∂a_i/∂z_j = δ_ij f'(z_i)   # diagonal
backward:  ∂L/∂z_i = (∂L/∂a_i) · f'(z_i)
limits:    f(z) → −λα as z → −∞ (bounded below);  f'(z) → 0
at z = 0:  f(0) = 0; one-sided slopes λα (left) and λ (right)
           ELU (λ = 1, α = 1): both are 1 ⇒ f is C¹, no kink
           SELU: λα ≈ 1.7581 vs λ ≈ 1.0507 ⇒ kink; the x ≤ 0 branch is used, f'(0) = λα
           (same convention as LeakyReLU's f'(0) = α)
```

SELU constants: `(α₀₁, λ₀₁)` is the unique pair for which `z ~ N(0, 1)` gives `E[f(z)] = 0` and
`E[f(z)²] = 1`. Closed forms (with `Φ` the standard normal CDF):

```
E[f(z)]  = λ [ 1/√(2π) + α (e^{1/2} Φ(−1) − 1/2) ]                               = 0
E[f(z)²] = λ² [ 1/2 + α² (e² Φ(−2) − 2 e^{1/2} Φ(−1) + 1/2) ]                     = 1
```

Together with LeCun-normal weights (`Var w = 1/fan_in`, N3) and standardised inputs, the next layer's
pre-activations are again ≈ `N(0, 1)`, so the fixed point propagates through depth.

## Complexity & memory

- `Forward`: `O(1)` — one compare; `z > 0`: 1 multiply; `z ≤ 0`: 1 `Exp` + 1 subtract + 1 multiply.
- `Backward` (scalar): `O(1)` — one compare, at most 1 `Exp` + 1 multiply. Only reached through the
  base-class default path; `Dense` calls the vector API.
- `ForwardVector`: `O(N)` — at most `N` `Exp` calls (negative elements only), no temporaries.
- `BackwardVector`: `O(N)` — **zero `Exp`**: 1 compare + at most 1 add + 1 multiply per element.
- RAM: 2 floats of state (`scale`, `scaledAlpha`) + one vptr per instance; one instance can be shared by
  any number of `Dense` layers. `Dense` already stores `preActivation` and `output`, so reusing the
  output adds no buffer.

## Numerical / embedded notes

- Precompute `scaledAlpha = λα` in the constructor: forward and backward share the exact same constant,
  so a fully saturated element (`Exp(z)` underflowed to `0`) gives `output = −λα` and a reused
  derivative `output + λα = 0` **exactly**.
- Output-reuse accuracy: `output + λα` is accurate in absolute terms (`≤ ~λα·2⁻²³ ≈ 2e-7`) but not
  relative once `eᶻ ≪ 2⁻²³` (e.g. `z = −20`: true derivative `3.6e-9`, reused `0`). That is harmless for
  back-propagation; the scalar `Backward` keeps the exact `λα·Exp(z)` form.
- `Exp(z) − 1` loses relative precision for `z → 0⁻` (cancellation; `math::` has no `Expm1`). The
  absolute error stays at `~λα·2⁻²³`, well inside `math::Tolerance<float>()`. An upstream `math::Expm1`
  is a possible refinement, not a dependency.
- Overflow: impossible on the negative branch (`Exp(z) ≤ 1`); the positive branch `λz` is the same
  unbounded linear map as ReLU. `-0.0` returns `0` (`Exp(−0) − 1 = 0`).
- Under fast-math, `Exp` of very negative inputs may flush to zero (FTZ): the result is the saturated
  value above, which is the intended limit.
- SELU is self-normalising **only** with LeCun-normal initialisation (N3), standardised inputs and no
  ordinary dropout (Klambauer's alpha-dropout is out of scope). With He/Glorot init it is just a
  scaled ELU.
- Non-finite inputs (NaN, ±∞) are outside the contract of every activation under fast-math.
- Float-only: `static_assert(std::is_floating_point_v<T>)`; the generic `T` signature keeps a
  `Q15`/`Q31` specialisation cheap to add later (the unbounded positive branch would saturate there).

## Dependencies

- Builds on: `ActivationFunction.hpp` (float-only `T`, `virtual ~ActivationFunction() = default`, the
  `output` argument of `BackwardVector`); `math::Exp` from `numerical/math/Math.hpp`.
- Soft dependency: **N3** (LeCun-normal init) is required for SELU's self-normalising behaviour in
  training; the class and its tests do not depend on it.
- Used by: optional hidden-layer activation for N16 training examples (a deep MLP without
  normalisation layers). No other N-item depends on it.

## Deployment

- Header: `neural_network/activation/ExponentialLinearUnit.hpp` — `#pragma once` →
  `#pragma GCC optimize("O3","fast-math")`, `#include "neural_network/activation/ActivationFunction.hpp"`,
  `#include "numerical/math/Math.hpp"`, `OPTIMIZE_FOR_SPEED` on all four overrides, and
  `extern template class ExponentialLinearUnit<float>;` under `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`.
- Coverage: `neural_network/activation/ExponentialLinearUnit.cpp` → `template class ExponentialLinearUnit<float>;`
- Test: `neural_network/activation/test/TestExponentialLinearUnit.cpp`
- Doc: `doc/activation/ExponentialLinearUnit.md` (per `doc/TEMPLATE.md`); remove the "ELU" row of
  `doc/activation/Activation.md` → *Variants*, add ELU/SELU rows to `doc/activation/README.md`, add
  ELU/SELU to the activation row of `doc/README.md`, and add the Klambauer reference next to the
  existing Clevert one.
- CMake: `ExponentialLinearUnit.hpp` → `target_sources(neural_network.activation PRIVATE ...)`;
  `ExponentialLinearUnit.cpp` → `neural_network_add_coverage_sources(neural_network.activation ...)`;
  `TestExponentialLinearUnit.cpp` → `neural_network.activation_test`.
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
