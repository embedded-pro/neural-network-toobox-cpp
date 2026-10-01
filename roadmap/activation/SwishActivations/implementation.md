# Swish Family (SiLU / Hard-Sigmoid / Hard-Swish) — Implementation Pseudocode

> Roadmap ref: #N5 (Tier 1) · Target: `neural_network/activation` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```text
template<typename T>                           # static_assert(std::is_floating_point_v<T>); instantiated for float
class SiLU final : public ActivationFunction<T>:
    Sigmoid<T> sigmoid                         # reused sign-branched stable sigmoid; Sigmoid is final ⇒ direct call

template<typename T>                           # static_assert(std::is_floating_point_v<T>); instantiated for float
class HardSigmoid final : public ActivationFunction<T>:
    static constexpr T lowerKnee{ -3 }
    static constexpr T upperKnee{  3 }
    static constexpr T slope{ T{1} / T{6} }    # multiply, never divide, on the hot path
    static constexpr T offset{ T{1} / T{2} }

template<typename T>                           # static_assert(std::is_floating_point_v<T>); instantiated for float
class HardSwish final : public ActivationFunction<T>:
    static constexpr T lowerKnee{ -3 }
    static constexpr T upperKnee{  3 }
    static constexpr T oneSixth{ T{1} / T{6} }
    static constexpr T oneThird{ T{1} / T{3} }
    static constexpr T oneHalf{  T{1} / T{2} }

# no size template parameter: like ReLU/Sigmoid/Identity, the vector length comes from the spans
# no trainable β (Swish-β = x·σ(βx)); a learnable slope needs the N8 gradient interface
```

## Interface

```text
# identical for SiLU<T>, HardSigmoid<T>, HardSwish<T>; all four override ActivationFunction<T>
T    Forward(T x) const                                              # hot path
T    Backward(T x) const                                             # exact derivative (see Math)
void ForwardVector(span<T> output, span<const T> input) const        # element-wise; hot path
void BackwardVector(span<T> result, span<const T> preActivation,
                    span<const T> output, span<const T> outputGradient) const   # result = g ⊙ f'(z)
# the vector overrides remove the per-element virtual call of the base-class defaults
# inherited: virtual ~ActivationFunction() = default; all three are default-constructible, stateless
```

## Algorithm (pseudocode)

```text
# ---------------- SiLU  f(x) = x·σ(x) ----------------
function SiLU.Forward(x):                       # OPTIMIZE_FOR_SPEED
    return x * sigmoid.Forward(x)               # σ via sign-branched form: no exp overflow for any finite x

function SiLU.Backward(x):                      # OPTIMIZE_FOR_SPEED
    s = sigmoid.Forward(x)
    return s * (T{1} + x * (T{1} - s))          # = σ + xσ(1−σ); stable form, see notes

function SiLU.ForwardVector(output, input):     # OPTIMIZE_FOR_SPEED
    really_assert(output.size() == input.size())
    for i in 0..N-1:
        output[i] = Forward(input[i])

function SiLU.BackwardVector(result, preActivation, output, outputGradient):   # OPTIMIZE_FOR_SPEED
    really_assert(all four sizes equal)
    for i in 0..N-1:
        result[i] = outputGradient[i] * Backward(preActivation[i])
        # σ is recomputed from z: it cannot be recovered from a = zσ(z) at z = 0,
        # and the output-reuse form a + σ(1 − a) cancels to 0 for z ≳ 1e8 (see notes)

# ---------------- Hard-Sigmoid  h(x) = ReLU6(x + 3)/6 ----------------
function HardSigmoid.Forward(x):                # OPTIMIZE_FOR_SPEED
    if x <= lowerKnee: return T{0}              # exact saturation constants
    if x >= upperKnee: return T{1}
    return x * slope + offset

function HardSigmoid.Backward(x):               # OPTIMIZE_FOR_SPEED
    return (lowerKnee < x and x < upperKnee) ? slope : T{0}      # open interval: 0 at x = ±3

function HardSigmoid.ForwardVector / BackwardVector:
    same loop shape as SiLU, calling the scalar Forward / Backward(preActivation[i]) inline

# ---------------- Hard-Swish  q(x) = x·ReLU6(x + 3)/6 ----------------
function HardSwish.Forward(x):                  # OPTIMIZE_FOR_SPEED
    if x <= lowerKnee: return T{0}
    if x >= upperKnee: return x                 # identity branch: exact, no overflow at large x
    return x * (x + T{3}) * oneSixth

function HardSwish.Backward(x):                 # OPTIMIZE_FOR_SPEED
    if x <= lowerKnee: return T{0}
    if x >= upperKnee: return T{1}
    return x * oneThird + oneHalf               # (2x + 3)/6

function HardSwish.ForwardVector / BackwardVector:
    same loop shape as SiLU; BackwardVector must read preActivation — q is non-monotonic on
    (−3, −1.5), so f' cannot be recovered from the output
```

Math (per element, `a = f(z)`, upstream `g = ∂L/∂a`; every Jacobian is diagonal):

```text
σ(z) = 1 / (1 + e^{−z}),   σ'(z) = σ(z)(1 − σ(z))

SiLU:         f(z)  = z σ(z)
              f'(z) = σ(z) + z σ(z)(1 − σ(z)) = σ(z)·(1 + z(1 − σ(z)))
              f'(0) = 1/2;  f' → 1 (z → +∞), f' → 0⁻ (z → −∞)
              range of f': [−0.0998, 1.0998] (extrema at z ≈ ∓2.399)
              global minimum at z* ≈ −1.2784645 (root of 1 + z(1 − σ(z)) = 0), f(z*) = z* + 1 ≈ −0.2784645

Hard-Sigmoid: h(z)  = min(max(z + 3, 0), 6) / 6 = clamp(z/6 + 1/2, 0, 1)
              h'(z) = 1/6 for −3 < z < 3,   0 otherwise (0 at the knees z = ±3)

Hard-Swish:   q(z)  = z · h(z) = { 0             z ≤ −3
                                 { z(z + 3)/6    −3 < z < 3
                                 { z             z ≥ 3
              q'(z) = { 0              z ≤ −3
                      { (2z + 3)/6     −3 < z < 3
                      { 1              z ≥ 3
              minimum at z = −3/2, q(−3/2) = −3/8; q is continuous but q' jumps at the knees
              (0 → −1/2 at z = −3, 3/2 → 1 at z = 3)

backward (all three):  ∂L/∂z_i = g_i · f'(z_i)
```

Knee convention: the middle piece owns the **open** interval `(−3, 3)`; at `z = ±3` exactly the outer
piece wins (`h' = 0`, `q'(−3) = 0`, `q'(3) = 1`). This is one rule for both hard functions and matches
the repo's ReLU choice (`f'(0) = 0`, the flat side). PyTorch uses the same rule for `hardsigmoid` but the
closed interval for `hardswish` (`q'(−3) = −1/2`, `q'(3) = 3/2`); the difference is on a measure-zero
set and does not affect imported weights.

## Complexity & memory

| Function     | `Forward` per element       | `Backward` per element      |
|--------------|-----------------------------|-----------------------------|
| SiLU         | 1 `exp` + 1 div + 2 add/mul | 1 `exp` + 1 div + 5 add/mul |
| Hard-Sigmoid | 2 compares + 1 FMA          | 2 compares                  |
| Hard-Swish   | 2 compares + 1 add + 2 mul  | 2 compares + 1 FMA          |

- `ForwardVector` / `BackwardVector`: `O(N)`, no temporaries, no heap, no recursion.
- RAM: 0 floats of state. `HardSigmoid` / `HardSwish` hold one vptr; `SiLU` holds its own vptr plus the
  embedded `Sigmoid`'s (two pointers). Constants are `static constexpr` (flash, not RAM). One instance can
  be shared by any number of `Dense` layers.
- `Dense<T, In, Out>` still stores `preActivation` and `output` (`2·Out` floats); unchanged by this item.

## Numerical / embedded notes

- **SiLU stability.** `Sigmoid<T>::Forward` is the sign-branched form, so `exp` never overflows; at
  `z = ±1e30` it returns `f = 1e30`, `f' = 1` and `f = −0`, `f' = −0` (no NaN).
- **SiLU backward form.** `σ(1 + z(1 − σ))` keeps `1 − σ` as a factor that rounds to exactly `0` once
  `σ` rounds to `1` (z ≳ 17), so `f' → 1` cleanly; its absolute float error is ≤ `1e-6` on `[−20, 20]`
  (measured against float64). The algebraically equal output-reuse form `a + σ(1 − a)` suffers
  catastrophic cancellation: at `z = 1e8` it returns `0` instead of `1`. Do not use it.
- **Hard variants are exp-free.** Only compares and one FMA: deterministic, branch-predictable, and
  exact at the saturated ends (`h ∈ {0, 1}`, `q ∈ {0, z}` are returned literally). `-ffast-math` cannot
  change the constant derivatives `0`, `1/6`, `1`.
- **Approximation quality.** `max |h − σ| ≈ 0.0692` (at `z ≈ ±1.317`) and `max |q − SiLU| ≈ 0.1423`
  (at `z = −3`). The hard functions are separate activations, not drop-in approximations: a network
  trained with SiLU must be run with SiLU (and MobileNetV3 weights with Hard-Swish).
- **Other "hard sigmoid" conventions.** Legacy Keras (`0.2z + 0.5` on `[−2.5, 2.5]`) and Courbariaux
  et al. (`clip((z + 1)/2, 0, 1)`) differ; this item implements the MobileNetV3 `ReLU6(z + 3)/6`, the
  definition also used by PyTorch `hardsigmoid` / `hardswish` and Keras 3.
- **Non-differentiable knees** (`z = ±3`): documented convention above; finite differences are not
  valid within `h` of a knee, so tests keep FD points away from them.
- Non-finite inputs (NaN, ±∞) are outside the contract of every activation under fast-math.
- Float-only: `static_assert(std::is_floating_point_v<T>)`; the generic `T` signature keeps a
  `Q15`/`Q31` specialisation cheap to add later (the hard variants are the natural first candidates:
  piecewise-linear, bounded constants).

## Dependencies

- Builds on: `ActivationFunction.hpp` (float-only `T`, `virtual ~ActivationFunction() = default`),
  `Sigmoid.hpp` (SiLU reuses its stable `Forward`), `numerical/math/CompilerOptimizations.hpp`.
  No other N-item is required.
- Used by: N13 (weight export/import of MobileNetV3-class models), N19 (depthwise-separable blocks,
  where Hard-Swish / Hard-Sigmoid gating is standard), N11/N12 (small CNN backbones). N8 may later make
  the activation a template parameter, removing the remaining virtual call per vector.

## Deployment

- Header: `neural_network/activation/SwishActivations.hpp` — `#pragma once` →
  `#pragma GCC optimize("O3","fast-math")`, `#include "neural_network/activation/ActivationFunction.hpp"`,
  `#include "neural_network/activation/Sigmoid.hpp"`, the three classes, `OPTIMIZE_FOR_SPEED` on all four
  overrides of each, and under `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`:
  `extern template class SiLU<float>; extern template class HardSigmoid<float>; extern template class HardSwish<float>;`
- Coverage: `neural_network/activation/SwishActivations.cpp` →
  `template class SiLU<float>; template class HardSigmoid<float>; template class HardSwish<float>;`
- Test: `neural_network/activation/test/TestSwishActivations.cpp`
- Doc: `doc/activation/SwishActivations.md` (per `doc/TEMPLATE.md`); move the "Swish / SiLU" row and the
  Hard-Sigmoid half of the "Hard Sigmoid / Hard Tanh" row of `doc/activation/Activation.md` → *Variants*
  into the implemented list, and add SiLU / Hard-Sigmoid / Hard-Swish rows to `doc/activation/README.md`.
- CMake: `SwishActivations.hpp` → `target_sources(neural_network.activation PRIVATE ...)`;
  `SwishActivations.cpp` → `neural_network_add_coverage_sources(neural_network.activation ...)`;
  `TestSwishActivations.cpp` → `neural_network.activation_test` (existing target, QEMU hook already present).
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
