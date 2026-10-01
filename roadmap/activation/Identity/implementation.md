# Identity (Linear) Activation — Implementation Pseudocode

> Roadmap ref: #N1 (Tier 1) · Target: `neural_network/activation` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```text
template<typename T>                           # static_assert(std::is_floating_point_v<T>); instantiated for float
class Identity final : public ActivationFunction<T>:
    # no members — stateless; sizeof == one vptr
    # no size template parameter: like ReLU/Sigmoid/Softmax, the vector length comes from the spans
```

## Interface

```text
Identity() = default
T    Forward(T x) const                                              # f(x) = x;  hot path
T    Backward(T x) const                                             # f'(x) = 1 (exact)
void ForwardVector(span<T> output, span<const T> input) const        # element-wise copy; hot path
void BackwardVector(span<T> result, span<const T> preActivation,
                    span<const T> output, span<const T> outputGradient) const   # result = outputGradient
# all four override ActivationFunction<T>; the vector overrides remove the per-element virtual call
# inherited: virtual ~ActivationFunction() = default
```

## Algorithm (pseudocode)

```text
function Forward(x):                            # OPTIMIZE_FOR_SPEED
    return x

function Backward(x):                           # OPTIMIZE_FOR_SPEED — argument unused
    return T{1}                                 # exactly 1, never 0.9999

function ForwardVector(output, input):          # OPTIMIZE_FOR_SPEED
    really_assert(output.size() == input.size())
    for i in 0..N-1:
        output[i] = input[i]                    # alias-safe when output.data() == input.data()

function BackwardVector(result, preActivation, output, outputGradient):   # OPTIMIZE_FOR_SPEED
    really_assert(result.size() == preActivation.size()
                  and result.size() == output.size()
                  and result.size() == outputGradient.size())
    for i in 0..N-1:
        result[i] = outputGradient[i]           # JVP: J = I  ⇒  ∂L/∂z = ∂L/∂a
```

Math (per element, `a = f(z)`):

```text
forward:   a_i = z_i
Jacobian:  ∂a_i/∂z_j = δ_ij                     # J = I_N (diagonal, all ones)
backward:  ∂L/∂z_i = Σ_j (∂L/∂a_j)(∂a_j/∂z_i) = ∂L/∂a_i
```

In a `Dense` layer this makes the output the raw affine map `a = W x + b`, so the layer backward reduces
to `∂L/∂x = Wᵀ g`, `∂L/∂W = g xᵀ`, `∂L/∂b = g` with `g = ∂L/∂a` (the parameter gradients stay private
until N8).

## Complexity & memory

- `Forward` / `Backward`: `O(1)`, zero FLOPs.
- `ForwardVector` / `BackwardVector`: `O(N)` copies, zero FLOPs, no temporaries.
- RAM: 0 floats of state (one vptr per instance); one instance can be shared by any number of `Dense` layers.
- `Dense<T, In, Out>` with Identity still stores both `preActivation` and `output` (`2·Out` floats, now
  equal). Eliding that duplicate belongs to N10 (inference-only layer), not to this item.

## Numerical / embedded notes

- Bit-exact: a copy has no rounding, so `-ffast-math` cannot change the result; `-0.0` and every finite
  value pass through unchanged. Derivative is the exact constant `1`.
- Output is **unbounded**: the reason this item is float-only (a `Q15`/`Q31` output would saturate at ±1).
- Pairings (per the code-fix contract):
  - regression: Identity → `MeanSquaredError` / `MeanAbsoluteError` (and Huber, N4);
  - multi-class: Identity → `CategoricalCrossEntropy`, which takes **logits** (fused softmax + CE);
    never Softmax → CCE, which applies softmax twice;
  - binary: current `BinaryCrossEntropy` takes **probabilities**, so keep `Sigmoid` there; Identity → BCE
    becomes valid only with the BCE-with-logits of N9.
- Non-finite inputs (NaN, ±∞) are outside the contract of every activation under fast-math.
- Float-only: `static_assert(std::is_floating_point_v<T>)`; the generic `T` signature keeps a
  `Q15`/`Q31` specialisation cheap to add later.

## Dependencies

- Builds on: `ActivationFunction.hpp` (float-only `T`, `virtual ~ActivationFunction() = default`).
  No other N-item; no numerical-toolbox API beyond `CompilerOptimizations.hpp`.
- Used by: N4 (Huber regression output), N9 (logit-input losses), N16 (training examples), and the
  simulator's regression demo (removes the sigmoid squash of sine targets).

## Deployment

- Header: `neural_network/activation/Identity.hpp` — `#pragma once` →
  `#pragma GCC optimize("O3","fast-math")`, `#include "neural_network/activation/ActivationFunction.hpp"`,
  `OPTIMIZE_FOR_SPEED` on all four overrides, and `extern template class Identity<float>;` under
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`.
- Coverage: `neural_network/activation/Identity.cpp` → `template class Identity<float>;`
- Test: `neural_network/activation/test/TestIdentity.cpp` (plus one integration case in
  `neural_network/layer/test/TestDense.cpp`, see `tests.md`).
- Doc: `doc/activation/Identity.md` (per `doc/TEMPLATE.md`); move the "Identity (linear)" row of
  `doc/activation/Activation.md` → *Variants* into the implemented list and add Identity to the
  activation row of `doc/README.md`.
- CMake: `Identity.hpp` → `target_sources(neural_network.activation PRIVATE ...)`; `Identity.cpp` →
  `neural_network_add_coverage_sources(neural_network.activation ...)`; `TestIdentity.cpp` →
  `neural_network.activation_test`.
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
