# Classification Metrics (ArgMax, Accuracy, Confusion Matrix) — Implementation Pseudocode

> Roadmap ref: #N7 (Tier 1) · Target: `neural_network/metrics` (new module) · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```text
template<typename T>                                   # static_assert(std::is_floating_point_v<T>); instantiated for float
class AccuracyCounter:
    std::uint32_t correct = 0                          # samples with actual == predicted
    std::uint32_t total   = 0                          # samples seen
    # 2 words (8 B). T is only the type of the returned ratio.

template<typename T, std::size_t NumberOfClasses>      # static_assert(std::is_floating_point_v<T>); instantiated for float
class ConfusionMatrix:                                 # static_assert(NumberOfClasses >= 2)
    using ScoreVector = math::Vector<T, NumberOfClasses>          # = Model::OutputVector of a K-class head
    std::array<std::uint32_t, NumberOfClasses * NumberOfClasses> counts{}   # row-major, C[a·K + p]
    std::uint32_t correct = 0                          # = trace(C), kept incrementally
    std::uint32_t total   = 0                          # = Σ C, kept incrementally
```

Convention (Sokolova & Lapalme; also scikit-learn `confusion_matrix`): **row = actual class,
column = predicted class**. For class `k`:

```text
TP_k = C[k][k]
FN_k = Σ_{j≠k} C[k][j]          support_k   = Σ_j C[k][j] = TP_k + FN_k     (row sum)
FP_k = Σ_{i≠k} C[i][k]          predicted_k = Σ_i C[i][k] = TP_k + FP_k     (column sum)
```

Counts are **integers, not `T`**: a `float` counter stops at `2²⁴ = 16 777 216`
(`2²⁴ + 1` rounds back to `2²⁴`), about 46 h of inference at 100 Hz. `std::uint32_t` lasts `2³² − 1`
samples (≈ 1.36 years at 100 Hz). `math::Matrix` is not used for the counts because it is a
floating-point/`QNumber` container.

## Interface

```text
# free functions
template<typename T, std::size_t NumberOfClasses>
std::size_t ArgMax(const math::Vector<T, NumberOfClasses>& scores)   # lowest index of the largest score; hot path
                                                                     # static_assert(NumberOfClasses >= 2): a 1-output head uses BinaryLabel
template<typename T>
std::size_t BinaryLabel(T score, T threshold)                        # 1 if score > threshold else 0; no default threshold
                                                                     # (0.5 for a Sigmoid probability, 0 for a logit)

# AccuracyCounter<T>
void               Add(std::size_t actual, std::size_t predicted)   # hot path
void               Reset()
std::uint32_t      Correct() const
std::uint32_t      Total() const
std::optional<T>   Accuracy() const                                  # nullopt when Total() == 0

# ConfusionMatrix<T, K>
void               Add(std::size_t actual, std::size_t predicted)   # really_assert(actual < K && predicted < K); hot path
std::size_t        AddScores(std::size_t actual, const ScoreVector& scores)   # Add(actual, ArgMax(scores)); returns the prediction; hot path
void               Reset()
std::uint32_t      Count(std::size_t actual, std::size_t predicted) const
std::uint32_t      Total() const
std::optional<T>   Accuracy() const                                  # trace / total;           nullopt if total == 0
std::optional<T>   Precision(std::size_t k) const                    # TP / (TP + FP);           nullopt if predicted_k == 0
std::optional<T>   Recall(std::size_t k) const                       # TP / (TP + FN);           nullopt if support_k == 0
std::optional<T>   F1(std::size_t k) const                           # 2TP / (2TP + FP + FN);    nullopt if support_k + predicted_k == 0
std::optional<T>   BalancedAccuracy() const                          # mean Recall over classes with support_k > 0
std::optional<T>   MacroF1() const                                   # mean F1 over classes with support_k + predicted_k > 0
```

Both `Add` overloads keep the scikit-learn argument order `(actual, predicted)`; the score overload has
its own name so an integer literal can never be converted into a `ScoreVector` by accident. Undefined
ratios are `std::optional` (no exceptions, no NaN).

Typical use with the existing `Model` (its `Forward` returns `math::Vector<T, OutputSize>`):

```text
ConfusionMatrix<float, K> validation{}
for (input, label) in validation set:
    validation.AddScores(label, model.Forward(input))
accuracy = validation.Accuracy().value_or(0)
```

## Algorithm (pseudocode)

```text
function ArgMax(scores):                           # OPTIMIZE_FOR_SPEED
    best = 0
    for i in 1..K-1:
        if scores[i] > scores[best]:               # strict: ties keep the lowest index (numpy/PyTorch argmax)
            best = i
    return best

function BinaryLabel(score, threshold):            # OPTIMIZE_FOR_SPEED
    return score > threshold ? 1 : 0               # strict, as Keras BinaryAccuracy: score == threshold ⇒ 0

function AccuracyCounter::Add(actual, predicted):  # OPTIMIZE_FOR_SPEED
    really_assert(total != UINT32_MAX)
    total   += 1
    correct += (actual == predicted) ? 1 : 0

function AccuracyCounter::Accuracy():
    if total == 0: return nullopt
    return T(correct) / T(total)

function ConfusionMatrix::Add(actual, predicted):  # OPTIMIZE_FOR_SPEED
    really_assert(actual < K and predicted < K)
    really_assert(total != UINT32_MAX)             # every C[a][p] ≤ total, so no cell can overflow first
    counts[actual * K + predicted] += 1
    total   += 1
    correct += (actual == predicted) ? 1 : 0

function ConfusionMatrix::AddScores(actual, scores):   # OPTIMIZE_FOR_SPEED
    predicted = ArgMax(scores)
    Add(actual, predicted)
    return predicted

function Reset():
    counts.fill(0);  correct = 0;  total = 0

function RowSum(k):     return Σ_j counts[k * K + j]      # support_k
function ColumnSum(k):  return Σ_i counts[i * K + k]      # predicted_k

function Accuracy():
    if total == 0: return nullopt
    return T(correct) / T(total)

function Precision(k):
    really_assert(k < K)
    column = ColumnSum(k)
    if column == 0: return nullopt
    return T(counts[k * K + k]) / T(column)

function Recall(k):
    really_assert(k < K)
    row = RowSum(k)
    if row == 0: return nullopt
    return T(counts[k * K + k]) / T(row)

function F1(k):                                    # 2TP + FP + FN = support_k + predicted_k
    really_assert(k < K)
    denominator = std::uint64_t(RowSum(k)) + ColumnSum(k)       # ≤ 2·total: may exceed uint32
    if denominator == 0: return nullopt
    return T{2} * T(counts[k * K + k]) / T(denominator)

function BalancedAccuracy():
    sum = T{0};  classes = 0
    for k in 0..K-1:
        row = RowSum(k)
        if row > 0:  sum += T(counts[k * K + k]) / T(row);  classes += 1
    if classes == 0: return nullopt
    return sum / T(classes)

function MacroF1():
    sum = T{0};  classes = 0
    for k in 0..K-1:
        denominator = std::uint64_t(RowSum(k)) + ColumnSum(k)
        if denominator > 0:  sum += T{2} * T(counts[k * K + k]) / T(denominator);  classes += 1
    if classes == 0: return nullopt
    return sum / T(classes)
```

Math:

```text
Accuracy          = Σ_k TP_k / Σ_{a,p} C[a][p]
Precision_k       = TP_k / (TP_k + FP_k)
Recall_k          = TP_k / (TP_k + FN_k)
F1_k              = 2 P_k R_k / (P_k + R_k) = 2TP_k / (2TP_k + FP_k + FN_k)
                    (the count form equals the harmonic mean whenever P_k, R_k are defined and P_k + R_k > 0,
                     and is also defined — as 0 — when TP_k = 0 but only one of P_k, R_k exists)
BalancedAccuracy  = (1/|S|) Σ_{k∈S} R_k,          S = { k : support_k > 0 }
MacroF1           = (1/|U|) Σ_{k∈U} F1_k,         U = { k : support_k + predicted_k > 0 }
```

`S` and `U` reproduce scikit-learn's `balanced_accuracy_score` (classes absent from `y_true` are
dropped) and `f1_score(average="macro")` with default labels (the union of `y_true` and `y_pred`).
`MacroF1` is the **mean of per-class F1**; Sokolova & Lapalme's macro F-score is instead the F-score of
macro-averaged precision and recall. The two differ in general; only the former is provided.

Forward / backward. `ArgMax` (and therefore every metric here) is piecewise constant in the scores:
`∂ArgMax/∂z = 0` almost everywhere and undefined on ties, so there is **no `Backward`** and no gradient
check. These are evaluation metrics, not training objectives. The differentiable surrogate for
accuracy is the existing `CategoricalCrossEntropy` on logits, `∂J/∂z = softmax(z)·Σy − y`.

Invariance that justifies taking `ArgMax` on logits: with `m = maxⱼ zⱼ` and `S = Σⱼ e^{zⱼ−m} > 0`,
`softmax(z)ᵢ = e^{zᵢ−m}/S` is strictly increasing in `zᵢ` with `S` shared, so
`ArgMax(softmax(z)) = ArgMax(z)` in exact arithmetic. The same holds for any strictly increasing
element-wise output (`Sigmoid`, `Tanh`, `LeakyReLU`); `ReLU` maps every negative score to `0` and
creates ties. For a 1-output head `σ(z) > ½ ⇔ z > 0`, so `BinaryLabel(p, ½)` and `BinaryLabel(z, 0)`
agree in exact arithmetic.

## Complexity & memory

- `ArgMax`: `K − 1` compares, no arithmetic. `BinaryLabel`: 1 compare.
- `Add`: `O(1)` — 3 integer increments and 1 index compare. `AddScores`: `O(K)` (the `ArgMax`).
- `Accuracy`: `O(1)`, 1 division. `Precision`/`Recall`/`F1`: `O(K)` integer adds + 1 division.
- `BalancedAccuracy`/`MacroF1`: `O(K²)` integer adds + `K` divisions; no scratch buffer (row and column
  sums are recomputed per class). They run once per evaluation, never per inference.
- RAM: `ConfusionMatrix` = `K² + 2` 32-bit words (the same size as `K² + 2` floats):
  `K = 2` → 24 B, `K = 10` → 408 B, `K = 12` → 584 B, `K = 35` → 4 908 B. `AccuracyCounter` = 2 words
  (8 B), the choice when `K²` words do not fit. No heap, no recursion.

## Numerical / embedded notes

- **Take `ArgMax` on logits, not on softmax outputs.** Besides saving `K` exps and a divide, float
  softmax can merge near-ties: for `z = (0, 10⁻⁸)` (distinct in `float`), IEEE `float32` softmax gives
  `(0.5, 0.5)` because `e^{−10⁻⁸}` rounds to `1`, so `ArgMax(softmax) = 0` while `ArgMax(z) = 1`.
  Likewise `σ(10⁻⁸)` rounds to exactly `0.5` in `float32`, so `BinaryLabel` on the logit (`> 0`) keeps
  a decision that the probability (`> 0.5`) loses. Under the current loss contract a `K`-class head
  already emits logits (CCE takes logits and must not follow a `Softmax` layer).
- **Tie-breaking** is lowest-index (strict `>`), matching `numpy.argmax` / `torch.argmax`, so on-device
  and reference predictions agree bit-for-bit on exact ties such as saturated outputs.
- **NaN** scores are outside the contract (`-ffast-math` lets the compiler assume none). `±∞` is fine:
  `(−∞, −5, −∞, −7)` ⇒ `1`.
- **Ratios**: `T(count)` is exact up to `2²⁴`; above it each conversion has relative error `≤ 2⁻²⁴`, so a
  ratio is within a few ulp — irrelevant for a metric. `2TP + FP + FN ≤ 2·total` can exceed `2³² − 1`,
  hence the `std::uint64_t` denominator in `F1`/`MacroF1`.
- **Overflow**: `really_assert(total != UINT32_MAX)` in `Add`. Every cell is `≤ total`, so guarding
  `total` guards all counters. Call `Reset()` per evaluation window.
- **Class indices** are `std::size_t` and range-checked with `really_assert` in `Add`, `Count`,
  `Precision`, `Recall`, `F1` (not unit-tested, like the other precondition asserts).
- Float-only: `static_assert(std::is_floating_point_v<T>)` in `ArgMax`, `BinaryLabel`,
  `AccuracyCounter` and `ConfusionMatrix`; the generic `T` signature keeps a `Q15`/`Q31` specialisation
  cheap to add later (the counters are already integer).

## Dependencies

- Builds on: `numerical/math/Matrix.hpp` (`math::Vector<T, K>` scores, `operator[]`),
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`, `<array>`, `<cstdint>`,
  `<optional>`. The regression counterparts (`math::MeanSquaredError`, `MeanAbsoluteError`,
  `RSquaredScore` in `numerical/math/Statistics.hpp`) stay upstream; this item adds only the
  classification side and does not modify them.
- No hard N-item dependency: it consumes any `math::Vector<T, K>` such as `Model::Forward`'s output.
  N1 (`Identity`) gives the cleanest input, raw logits. N9 passes integer class labels at run time;
  it should use the same `std::size_t` class index as `actual` here.
- Used by: N16 (per-epoch validation accuracy / early stopping), N13 (checking on the device that an
  imported model reproduces its reference accuracy).
- Not in scope: top-k accuracy, ROC/AUC and threshold sweeps, multi-label metrics, Sokolova's
  macro F-score and micro averages (micro-P = micro-R = micro-F1 = `Accuracy` for single-label data).

## Deployment

- Header: `neural_network/metrics/ClassificationMetrics.hpp`. Order: `#pragma once`, then
  `#pragma GCC optimize("O3","fast-math")`, then `numerical/math/Matrix.hpp`,
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`, `<array>`, `<cstddef>`,
  `<cstdint>`, `<optional>`, `<type_traits>`. `OPTIMIZE_FOR_SPEED` on `ArgMax`, `BinaryLabel`,
  `Add`, `AddScores`. Under `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`:
  `extern template class ConfusionMatrix<float, 4>;`, `extern template class AccuracyCounter<float>;`,
  `extern template std::size_t ArgMax<float, 4>(const math::Vector<float, 4>&);`,
  `extern template std::size_t BinaryLabel<float>(float, float);`.
- Coverage: `neural_network/metrics/ClassificationMetrics.cpp` → the same four explicit
  instantiations without `extern` (`<float, 4>` matches the test fixture).
- Test: `neural_network/metrics/test/TestClassificationMetrics.cpp`.
- Doc: `doc/metrics/ClassificationMetrics.md` (per `doc/TEMPLATE.md`) in a new `doc/metrics/` folder
  with its `README.md`, plus a `metrics` row in `doc/README.md` (`roadmap/DEPLOYMENT.md` step 6).
- CMake (new module): `neural_network/metrics/CMakeLists.txt` with
  `neural_network_add_header_library(neural_network.metrics)`, the same `target_include_directories`
  as `losses`, links `infra.util` + `numerical.math`; `ClassificationMetrics.hpp` →
  `target_sources(neural_network.metrics PRIVATE ...)`; `ClassificationMetrics.cpp` →
  `neural_network_add_coverage_sources(neural_network.metrics ...)`; `add_subdirectory(test)`.
  `test/CMakeLists.txt`: `neural_network.metrics_test` (`emil_build_for` on
  `NEURAL_NETWORK_TOOLBOX_BUILD_TESTS`, `emil_add_test`, `neural_network_link_qemu_runtime`), linking
  `gmock_main` and `neural_network.metrics`. Register `add_subdirectory(metrics)` in
  `neural_network/CMakeLists.txt`, append `neural_network.metrics` to `NEURAL_NETWORK_TOOLBOX_INSTALL_TARGETS`
  in the root `CMakeLists.txt`, and append `neural_network.metrics_test` to the `targets` of both
  `qemu-cortex-m4-RelWithDebInfo` and `qemu-cortex-m7-RelWithDebInfo` in `CMakePresets.json`.
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
