# Per-Sample Loss Contract + Logit-Input BCE — Implementation Pseudocode

> Roadmap ref: #N9 (Tier 2) · Target: `neural_network/losses` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```
template<typename T, std::size_t NumberOfFeatures, typename Target = math::Vector<T, NumberOfFeatures>>
class Loss:                                          # replaces Loss.hpp; NOT an ObjectiveFunction any more
    static_assert(std::is_floating_point_v<T>, "Loss requires a floating-point type")
    static_assert(NumberOfFeatures > 0, "Loss requires at least one feature")
    using ValueType  = T
    using Vector     = math::Vector<T, NumberOfFeatures>          # prediction ŷ (or logits z) and gradient
    using TargetType = Target                                     # Vector (dense / soft labels) or std::size_t (class index)
    static constexpr std::size_t Size = NumberOfFeatures
    # no data members: the target arrives per call, the regulariser is gone

# concrete losses after N9 (all stateless except Huber's δ):
MeanSquaredError<T, N>                    : Loss<T, N>                # no members
MeanAbsoluteError<T, N>                   : Loss<T, N>                # no members
BinaryCrossEntropy<T, N>                  : Loss<T, N>                # probability input; static constexpr T epsilon = 1e-7
BinaryCrossEntropyWithLogits<T, N>        : Loss<T, N>                # NEW, logit input; no members
CategoricalCrossEntropy<T, N, Target = Vector>
                                          : Loss<T, N, Target>        # logit input; Target ∈ { Vector, std::size_t }
HuberLoss<T, N>                           : Loss<T, N>                # T delta (N4)
```

The regulariser member and the `(target, regularization&)` constructors are removed from every loss.
Weight decay is a property of θ, not of one sample's prediction, so it moves to the N16 training step.

## Interface

```
# Loss<T, N, Target> — pure interface, mirrors Layer's special-member pattern
virtual ~Loss() = default
virtual T      Cost    (const Vector& prediction, const TargetType& target) const = 0   # hot path
virtual Vector Gradient(const Vector& prediction, const TargetType& target) const = 0   # ∂Cost/∂prediction; hot path
protected: Loss() = default; copy/move ctor and assignment = default

# concrete constructors
MeanSquaredError()                      MeanAbsoluteError()
BinaryCrossEntropy()                    BinaryCrossEntropyWithLogits()
CategoricalCrossEntropy()               HuberLoss(T delta = T{1})       # really_assert(delta > 0)

# every concrete class is `final` and overrides both methods with OPTIMIZE_FOR_SPEED
```

- `prediction` is one sample's model output (size `N`): probabilities for `BinaryCrossEntropy`, raw logits
  for `BinaryCrossEntropyWithLogits` / `CategoricalCrossEntropy`, anything for MSE/MAE/Huber.
- `Gradient` is exactly the `outputGradient` that `Model::Backward` takes; N16 averages it over `B` samples.
- `CategoricalCrossEntropy<T, N, std::size_t>` is the integer-label (sparse) form: `target` is the class
  index `k ∈ [0, N)`, the same `std::size_t` class index as N7's `ArgMax` / `ConfusionMatrix`.
- `Model::Train` keeps its behaviour; its loss parameter changes from `Loss<T, TotalParameters>&` to
  `optimization::ObjectiveFunction<T, TotalParameters>&`, the type it already forwards to
  `optimizer.Minimize`. N16 removes `Train`.

## Algorithm (pseudocode)

Notation: `ŷ` prediction, `y` target, `N` size, `inv = 1/N` (`static constexpr`), `Σ` over `i = 0..N−1`.

### MSE, MAE, Huber (migrated, formulas unchanged minus the regulariser)

```
MSE:    J = inv · Σ (ŷᵢ − yᵢ)²                      ∂J/∂ŷᵢ = 2(ŷᵢ − yᵢ) · inv
MAE:    J = inv · Σ |ŷᵢ − yᵢ|                       ∂J/∂ŷᵢ = sign(ŷᵢ − yᵢ) · inv,  sign(0) = 0
Huber:  eᵢ = ŷᵢ − yᵢ,  cᵢ = clip(eᵢ, −δ, δ)
        J = inv · Σ cᵢ (eᵢ − ½cᵢ)                   ∂J/∂ŷᵢ = cᵢ · inv                      (N4)
```

### BinaryCrossEntropy — probability input (migrated)

```
p̃ᵢ = clamp(ŷᵢ, ε, 1 − ε),  ε = T{1e-7}
J       = −inv · Σ [ yᵢ log p̃ᵢ + (1 − yᵢ) log(1 − p̃ᵢ) ]
∂J/∂ŷᵢ  = (p̃ᵢ − yᵢ) / (p̃ᵢ (1 − p̃ᵢ)) · inv          # evaluated at the clamped p̃ (kept for Sigmoid outputs)
```

### BinaryCrossEntropyWithLogits — NEW

Per element with logit `z` and label `y ∈ [0, 1]` (soft labels allowed), `σ(z) = 1/(1 + e^{−z})`:

```
ℓ(z, y) = −[ y log σ(z) + (1 − y) log(1 − σ(z)) ]
        = (1 − y) z + log(1 + e^{−z})                     # log σ = −log(1+e^{−z}),  log(1−σ) = −z − log(1+e^{−z})
        = softplus(z) − y z
        = max(z, 0) − y z + log(1 + e^{−|z|})             # softplus(z) = max(z,0) + log(1+e^{−|z|}); stable form

∂ℓ/∂z   = (1 − y) − e^{−z}/(1 + e^{−z}) = (1 − y) − (1 − σ(z)) = σ(z) − y

J        = inv · Σ ℓ(zᵢ, yᵢ)
∂J/∂zᵢ   = (σ(zᵢ) − yᵢ) · inv
```

`max(z,0)` and `|z|` each have a kink at `z = 0`, but their combination is `softplus(z) − yz`, which is
`C^∞`; the gradient is continuous (`½ − y` at `z = 0`).

```
function Cost(z, y):                               # OPTIMIZE_FOR_SPEED
    sum = T{0}
    for i in 0..N-1:
        a = z[i] < 0 ? -z[i] : z[i]
        e = math::Exp(-a)                          # ∈ (0, 1], never overflows
        sum += (z[i] > 0 ? z[i] : T{0}) - z[i] * y[i] + math::Log(T{1} + e)
    return sum * inv

function Gradient(z, y):                           # OPTIMIZE_FOR_SPEED
    g = Vector{}
    for i in 0..N-1:
        a = z[i] < 0 ? -z[i] : z[i]
        e = math::Exp(-a)
        s = z[i] >= 0 ? T{1} / (T{1} + e) : e / (T{1} + e)       # stable σ, no e^{+|z|}
        g[i] = (s - y[i]) * inv
    return g
```

Chain-rule consistency with the probability form (test reference): with `p = σ(z)` unclamped,
`BCE(p, y) = BCEWithLogits(z, y)` and `∂BCE/∂p · σ'(z) = (p − y)/(p(1−p)) · p(1−p) · inv = (σ(z) − y) · inv`.

### CategoricalCrossEntropy — logit input, log-sum-exp (kept) + class-index target (NEW)

Fused softmax + cross-entropy; the model's last layer emits logits (N1 `Identity`), never a `Softmax` output.

```
m  = maxⱼ zⱼ
eᵢ = exp(zᵢ − m)          ∈ (0, 1],  e_argmax = 1
s  = Σ eᵢ                 ∈ [1, N]   ⇒ log s ≥ 0, never log(0)
L  = log s                            # logsumexp(z) = m + L
softmaxᵢ = eᵢ / s
```

Dense target `y` (one-hot, soft, or unnormalised), `S = Σ yᵢ`:

```
J       = S · logsumexp(z) − Σ yᵢ zᵢ = Σ yᵢ (L − (zᵢ − m))       # m cancels; no large-m cancellation
∂J/∂zᵢ  = S · softmaxᵢ − yᵢ
          # ∂/∂zᵢ logsumexp(z) = softmaxᵢ;  S = 1 for a normalised target ⇒ softmax − y
```

Class-index target `k` (`really_assert(k < N)`), equivalent to the one-hot `y = e_k`, `S = 1`:

```
J       = L − (z_k − m) = logsumexp(z) − z_k  ≥ 0
∂J/∂zᵢ  = softmaxᵢ − [i == k]
```

```
function Cost(z, target):                          # OPTIMIZE_FOR_SPEED
    m = MaxLogit(z)
    L = math::Log(Σᵢ math::Exp(z[i] - m))
    if constexpr (Target is std::size_t):
        return L - (z[target] - m)
    else:
        return Σᵢ target[i] * (L - (z[i] - m))

function Gradient(z, target):                      # OPTIMIZE_FOR_SPEED
    m = MaxLogit(z)
    g = Vector{};  s = T{0}
    for i: g[i] = math::Exp(z[i] - m);  s += g[i]  # exponentials written straight into the result
    if constexpr (Target is std::size_t):
        r = T{1} / s
        for i: g[i] *= r
        g[target] -= T{1}
    else:
        r = TargetSum(target) / s                  # S / s: one division for the whole vector
        for i: g[i] = g[i] * r - target[i]
    return g
```

`Target` is restricted with `static_assert(std::is_same_v<Target, Vector> || std::is_same_v<Target, std::size_t>)`.

## Complexity & memory

| Loss                          | `Cost`                          | `Gradient`                          |
|-------------------------------|---------------------------------|-------------------------------------|
| MSE / MAE / Huber             | `O(N)`, no transcendental       | `O(N)`, no transcendental           |
| BCE (probability)             | `N` clamps + `2N` log           | `N` clamps + `N` div                |
| BCE with logits               | `N` exp + `N` log               | `N` exp + `N` div                   |
| CCE (dense or class index)    | `N` exp + 1 log + `N`-compare max | `N` exp + 1 div + `N`-compare max |

- Object size: `0` data floats (one vtable pointer); Huber `1` float (`δ`). The target is no longer
  copied into the loss, saving `N` floats per loss object versus today.
- Stack: `Gradient` returns one `N`-float vector; CCE writes its exponentials into that vector, so no
  second `N`-float buffer. No heap, no recursion.
- The class-index CCE target is one `std::size_t` instead of `N` floats per stored sample — the RAM win
  for N16 datasets of `K`-class samples.

## Numerical / embedded notes

- **Why logits for BCE.** Probability BCE saturates: for `z ≥ 16.64` (`e^{−z} < 2⁻²⁴`) `σ(z)` rounds to
  exactly `1` in `float`, the clamp returns `1 − ε`, and the cost caps at `−log(1.19·10⁻⁷) = 15.94` —
  e.g. `z = 20, y = 0` gives `15.94` instead of `20`. The gradient `(p̃ − y)/(p̃(1−p̃))/N` then reaches
  `~2·10⁶`. The logit form returns `20` and a gradient bounded by `1/N` for every finite `z`.
- `log(1 + e)` uses `math::Log(T{1} + e)` (no `Log1p` upstream): for `|z| > 16.64`, `1 + e` rounds to `1`
  and the term is `0` instead of `e < 6·10⁻⁸` — an absolute error below `2⁻²⁴` on a loss that is `≥ 0`.
  Swap in `Log1p` if numerical-toolbox adds it; the tests do not depend on it.
- `e = exp(−|z|) ∈ (0, 1]` and CCE's `exp(zᵢ − m) ∈ (0, 1]`: no overflow for any finite input; the
  sums stay in `[1, N]` so no `log(0)`. Every gradient is finite for every finite `z` (`|∂J/∂zᵢ| ≤ 1/N`
  for BCE-with-logits, `≤ S` for CCE with a non-negative target, `≤ 1` for a class index). The BCE-with-logits cost is finite while `Σ|zᵢ| < FLT_MAX`;
  class-index CCE for `|z| ≤ 1e38` (`z_k − m ≥ −2·10³⁸`); dense CCE while `S · 2 maxᵢ|zᵢ| < FLT_MAX`
  (e.g. `|z| ≤ 1e30` with `S = O(1)`).
- `exp(zᵢ − m)` of very negative arguments underflows to denormal/`0` (flush-to-zero on Cortex-M FPUs);
  the result is correct to absolute precision.
- `Softmax` is **not** placed before `CategoricalCrossEntropy` or `BinaryCrossEntropyWithLogits`:
  both apply the squashing internally. Pairing them applies softmax/sigmoid twice.
- `-ffast-math` leaks into every TU that includes these headers (`#pragma GCC optimize`), so
  `std::isfinite`/`std::isnan` may fold to constants; tests use exact values or `math::IsFinite`
  (bit-based) instead.
- Non-finite inputs (NaN, ±∞) are outside the contract, like every other component under fast-math.
- Float-only: `static_assert(std::is_floating_point_v<T>)` in `Loss`; the generic `T` signature keeps a
  `Q15`/`Q31` specialisation cheap to add later.

## Dependencies

- Builds on: current `Loss.hpp` (float-only `T`, `N > 0` asserts), the float-only mean losses
  (MSE, MAE, ε-clamped BCE) and the logit-input max-shift CCE from the code-fix pass;
  `numerical/math/Matrix.hpp`, `numerical/math/Math.hpp` (`Exp`, `Log`),
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`.
- N1 (`Identity`): the only way for a `Dense` output to be raw logits for the two logit losses.
- N4 (`HuberLoss`): if it lands first it migrates here (ctor `HuberLoss(T delta)`); otherwise it is
  written directly against this contract.
- Consumed by: N16 (per-sample `Forward → Gradient(ŷ, y) → Backward`, weight decay on θ via the
  upstream `regularization` module). N7 shares the `std::size_t` class-index convention.
- Independent of N8: the loss gradient is w.r.t. predictions, which `Model::Backward` already accepts.
- Out of scope: batch reduction (N16 averages), class weights / `pos_weight`, label smoothing, focal loss.

## Deployment

- Header: `neural_network/losses/LossContract.hpp` (replaces `Loss.hpp`, which is deleted) —
  `#pragma once`, `numerical/math/Matrix.hpp`, `<cstddef>`, `<type_traits>`; the interface above, and
  `extern template class Loss<float, 2>;` and `extern template class Loss<float, 2, std::size_t>;` under
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`.
- New loss: `neural_network/losses/BinaryCrossEntropyWithLogits.hpp` — `#pragma GCC optimize("O3","fast-math")`,
  `OPTIMIZE_FOR_SPEED` on `Cost`/`Gradient`, `extern template class BinaryCrossEntropyWithLogits<float, 2>;`.
- Migrated: `MeanSquaredError.hpp`, `MeanAbsoluteError.hpp`, `BinaryCrossEntropy.hpp`,
  `CategoricalCrossEntropy.hpp` (adds the `Target` parameter and the `std::size_t` branch), `HuberLoss.hpp`
  (if present): include `LossContract.hpp`, drop `numerical/regularization` and the target/regulariser
  members, make `Cost`/`Gradient` `const` two-argument overrides.
- Coverage: `neural_network/losses/LossContract.cpp` → `template class Loss<float, 2>;`
  `template class Loss<float, 2, std::size_t>;`; `BinaryCrossEntropyWithLogits.cpp` →
  `template class BinaryCrossEntropyWithLogits<float, 2>;`; `CategoricalCrossEntropy.cpp` adds
  `template class CategoricalCrossEntropy<float, 2, std::size_t>;`.
- Model: `Model.hpp` includes `numerical/optimization/ObjectiveFunction.hpp` instead of `Loss.hpp`;
  `Train(Optimizer&, optimization::ObjectiveFunction<T, TotalParameters>&, const ParameterVector&)`.
  `neural_network.model` drops its `neural_network.losses` link.
- Test: `neural_network/losses/test/TestLossContract.cpp` (new) plus the mechanical migration of the
  four existing `Test<Loss>.cpp` files and `LossTestSupport.hpp` (see `tests.md`); `TestModel.cpp`'s
  `LossMock` derives from `optimization::ObjectiveFunction<float, Size>`.
- Doc: `doc/losses/LossContract.md` (per `doc/TEMPLATE.md`); add its row to `doc/losses/README.md`;
  in `doc/losses/Loss.md` drop the regularisation term from every formula, add a *Binary
  Cross-Entropy with Logits* section and the class-index CCE form, and fix *Pitfalls* (BCE saturation).
- CMake: `LossContract.hpp` + `BinaryCrossEntropyWithLogits.hpp` → `target_sources(neural_network.losses
  PRIVATE ...)` (remove `Loss.hpp`); `LossContract.cpp` + `BinaryCrossEntropyWithLogits.cpp` →
  `neural_network_add_coverage_sources(neural_network.losses ...)`; `TestLossContract.cpp` →
  `neural_network.losses_test` (existing, already QEMU-wired); `neural_network.losses` drops
  `numerical.regularization` and `numerical.optimization` from `target_link_libraries`.
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
