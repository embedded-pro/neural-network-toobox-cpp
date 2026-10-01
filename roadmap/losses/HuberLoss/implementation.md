# Huber Loss — Implementation Pseudocode

> Roadmap ref: #N4 (Tier 1) · Target: `neural_network/losses` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```text
template<typename T, std::size_t NumberOfFeatures>   # static_assert(std::is_floating_point_v<T>) via Loss; instantiated for float
class HuberLoss : public Loss<T, NumberOfFeatures>:
    using Vector = typename Loss<T, NumberOfFeatures>::Vector      # math::Vector<T, N> (Matrix<T, N, 1>)
    static constexpr T inverseSize = T{1} / T(NumberOfFeatures)
    Vector  target                                                 # y, bound at construction (like MSE/MAE)
    T       delta                                                  # δ > 0, the quadratic/linear threshold
    regularization::Regularization<T, NumberOfFeatures>& regularization
    # delta is a runtime member, not an NTTP: it is a tuning knob users sweep, and the
    # coverage instantiation stays HuberLoss<float, 2> like the other losses
```

## Interface

```text
HuberLoss(const Vector& expectedTarget,
          regularization::Regularization<T, N>& regularizationTerm,
          T delta = T{1})                              # really_assert(delta > T{0}); default 1 as in Keras/PyTorch
T      Cost(const Vector& predictions) override        # ObjectiveFunction<T, N>; hot path
Vector Gradient(const Vector& predictions) override    # ObjectiveFunction<T, N>; hot path
# inherited: virtual ~ObjectiveFunction() = default
# argument = predictions ŷ of ONE sample (size N); the regulariser is added to the same argument,
# exactly as MSE/MAE/BCE/CCE do today (removed by N9)
```

## Algorithm (pseudocode)

Per element, residual `eᵢ = ŷᵢ − yᵢ`:

```text
ρ_δ(e) = ½ e²              if |e| ≤ δ
         δ (|e| − ½ δ)     otherwise

ψ_δ(e) = ρ_δ'(e) = clip(e, −δ, δ) = e if |e| ≤ δ, δ·sign(e) otherwise

J(ŷ)       = (1/N) Σᵢ ρ_δ(ŷᵢ − yᵢ) + R(ŷ)
∂J/∂ŷᵢ     = (1/N) clip(ŷᵢ − yᵢ, −δ, δ) + ∂R/∂ŷᵢ
```

Branch-free evaluation with `c = clip(e, −δ, δ)` (same `c` for cost and gradient):

```text
ρ_δ(e) = c · (e − ½ c)
    |e| ≤ δ:  c = e     ⇒ e·(e − ½e)  = ½e²
    e >  δ:   c = δ     ⇒ δ·(e − ½δ)  = δ(|e| − ½δ)
    e < −δ:   c = −δ    ⇒ −δ·(e + ½δ) = δ(−e − ½δ) = δ(|e| − ½δ)
```

```text
function Cost(predictions):                        # OPTIMIZE_FOR_SPEED
    sum = T{0}
    for i in 0..N-1:
        e = predictions[i] - target[i]
        c = e < -delta ? -delta : (e > delta ? delta : e)
        sum += c * (e - T{0.5} * c)
    return sum * inverseSize + regularization.Calculate(predictions)

function Gradient(predictions):                    # OPTIMIZE_FOR_SPEED
    regularizationGradient = regularization.Gradient(predictions)
    gradient = Vector{}
    for i in 0..N-1:
        e = predictions[i] - target[i]
        c = e < -delta ? -delta : (e > delta ? delta : e)
        gradient[i] = c * inverseSize + regularizationGradient[i]
    return gradient
```

Continuity at the threshold `|e| = δ`: both branches give `ρ = ½δ²` and `ψ = ±δ`, so `ρ_δ` is `C¹`
(the second derivative jumps from `1` to `0`, which is irrelevant to first-order training).

Limiting cases (the test references):

```text
δ ≥ maxᵢ |eᵢ|   ⇒ J − R = ½ · MSE          ∇ = ½ · ∇MSE = eᵢ/N
δ ≤ minᵢ |eᵢ|   ⇒ J − R = δ·MAE − ½δ²      ∇ = δ · ∇MAE = δ·sign(eᵢ)/N
δ → 0           ⇒ (J − R)/δ → MAE
```

Convention: this is the Keras / PyTorch `HuberLoss` scaling (quadratic part `½e²`), so for large `δ` it
is **half** of this library's MSE and the gradient is `e/N`, not `2e/N`. PyTorch `SmoothL1Loss` is the
same function divided by `δ`; not provided separately.

## Complexity & memory

- `Cost` / `Gradient`: `O(N)` — per element one subtraction, two compares, and 2 mul + 1 sub + 1 add
  (cost) or 1 mul + 1 add (gradient). No transcendental, no division at run time.
- RAM: `N + 1` floats of state (`target`, `delta`) plus one reference; `Gradient` returns one `N`-float
  vector and holds one `N`-float regulariser gradient on the stack. No heap.

## Numerical / embedded notes

- Robustness is also a **float range** property: MSE computes `e²`, which overflows `float` for
  `|e| > √FLT_MAX ≈ 1.84·10¹⁹`; Huber's linear branch stays finite for `|e| < FLT_MAX/δ` (≈ `2.27·10³⁸`
  at `δ = 1.5`). The running sum is finite while `Σ ρ < FLT_MAX` (summed before the `1/N` scale, as in MSE).
- Inside `|e| ≤ δ` the branch-free form is bit-identical to `½e²` (`½e` and `e − ½e` are exact in binary
  floating point), so no rounding difference versus the textbook branch.
- Gradient is bounded: `|∂J/∂ŷᵢ − ∂R/∂ŷᵢ| ≤ δ/N` for every input, which is the property that keeps a
  single corrupted sensor sample from blowing up an update.
- `sign(0)` issue of MAE does not arise: `c = 0` at `e = 0` and the function is smooth there.
- `delta` must be `> 0` (`really_assert` in the constructor); `δ` is in target units, so choose it
  from the expected inlier noise (`δ ≈ 1.345σ` gives 95 % asymptotic efficiency under Gaussian noise).
- Non-finite predictions/targets (NaN, ±∞) are outside the contract under `-ffast-math`, like every loss.
- Float-only: `static_assert(std::is_floating_point_v<T>)` (inherited from `Loss`); the generic `T`
  signature keeps a `Q15`/`Q31` specialisation cheap to add later.

## Dependencies

- Builds on: `Loss.hpp` (float-only `T`, `ObjectiveFunction` base with virtual destructor),
  `numerical/regularization/Regularization.hpp`, `numerical/math/CompilerOptimizations.hpp`,
  `infra/util/ReallyAssert.hpp`. Pattern copied from `MeanSquaredError` / `MeanAbsoluteError`.
- No hard N-item dependency. Best paired with N1 (`Identity` output) for unbounded regression targets;
  a `Sigmoid`/`Tanh` output bounds `e` and defeats the purpose.
- Migrated by N9 to the per-sample contract `Cost/Gradient(prediction, target)` without the in-loss
  regulariser; consumed by N16 (mini-batch training). `Model::Train` is unchanged by this item.

## Deployment

- Header: `neural_network/losses/HuberLoss.hpp` — `#pragma once` →
  `#pragma GCC optimize("O3","fast-math")`, includes `neural_network/losses/Loss.hpp`,
  `numerical/math/CompilerOptimizations.hpp`, `numerical/regularization/Regularization.hpp`,
  `infra/util/ReallyAssert.hpp`; `OPTIMIZE_FOR_SPEED` on `Cost`/`Gradient`, and
  `extern template class HuberLoss<float, 2>;` under `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`.
- Coverage: `neural_network/losses/HuberLoss.cpp` → `template class HuberLoss<float, 2>;`
- Test: `neural_network/losses/test/TestHuberLoss.cpp` (reuses `test/LossTestSupport.hpp`:
  `RegularizationMock`, `CentralDifferenceGradient`).
- Doc: `doc/losses/HuberLoss.md` (per `doc/TEMPLATE.md`); add its row to `doc/losses/README.md`,
  drop the "Huber loss" row from *Variants & Generalizations* in `doc/losses/Loss.md` and link the new
  page from its *Applications* regression bullet.
- CMake: `HuberLoss.hpp` → `target_sources(neural_network.losses PRIVATE ...)`; `HuberLoss.cpp` →
  `neural_network_add_coverage_sources(neural_network.losses ...)`; `TestHuberLoss.cpp` →
  `neural_network.losses_test` (existing target, already QEMU-wired; `infra.util` already linked).
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
