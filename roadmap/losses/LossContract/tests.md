# Per-Sample Loss Contract + Logit-Input BCE — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixtures

```
class TestLossContract : public ::testing::Test:
    static constexpr std::size_t size = 4
    using Vector = neural_network::Loss<float, size>::Vector
    neural_network::MeanSquaredError<float, size> mse                 # stateless: no ctor arguments
    const neural_network::Loss<float, size>& contract = mse           # every call goes through the const interface
    const Vector predictions = { 1.0, -1.5, 2.5, 1.0 }
    const Vector target      = { 0.5, -1.0, 2.0, 0.0 }

class TestBinaryCrossEntropyWithLogits : public ::testing::Test:
    static constexpr std::size_t size = 4
    using Vector = neural_network::BinaryCrossEntropyWithLogits<float, size>::Vector
    neural_network::BinaryCrossEntropyWithLogits<float, size> loss
    const Vector logits = { 2.0, -1.0, 0.5, -3.0 }
    const Vector target = { 1.0,  0.0, 0.25, 1.0 }                    # one soft label
    const Vector sigmoidOfLogits = { 0.8807971, 0.2689414, 0.6224593, 0.0474259 }

class TestCategoricalCrossEntropyClassIndex : public ::testing::Test:
    static constexpr std::size_t size = 3
    using Vector = neural_network::CategoricalCrossEntropy<float, size, std::size_t>::Vector
    neural_network::CategoricalCrossEntropy<float, size, std::size_t> loss
    const Vector logits = { 1.0, 2.0, 3.0 }                           # same logits as TestCategoricalCrossEntropy

# each case below is a TEST_F(<fixture>, <name>)
# no collaborator remains after the regulariser is removed ⇒ no mocks in this file
# finite differences use the migrated test_support::CentralDifferenceGradient(loss, point, target), h = 1e-3
# Tolerance = math::Tolerance<float>() = 1e-3 unless stated
```

Compile-time checks at the top of `TestLossContract.cpp` (not tests):

```
static_assert(std::is_base_of_v<Loss<float, 4>, MeanSquaredError<float, 4>>)
static_assert(std::is_base_of_v<Loss<float, 4>, BinaryCrossEntropyWithLogits<float, 4>>)
static_assert(std::is_base_of_v<Loss<float, 3, std::size_t>, CategoricalCrossEntropy<float, 3, std::size_t>>)
static_assert(!std::is_base_of_v<optimization::ObjectiveFunction<float, 4>, MeanSquaredError<float, 4>>)
```

## Test cases (Arrange / Act / Assert)

```
TestLossContract.TargetIsSuppliedPerCallAndNotRetained:
    Act:     c1 = contract.Cost(predictions, target)
             c2 = contract.Cost(predictions, predictions)          # a different target on the same object
             c3 = contract.Cost(predictions, target)
             g1 = contract.Gradient(predictions, target)
             g2 = contract.Gradient(predictions, predictions)
    Assert:  EXPECT_NEAR(c1, 0.4375f, Tolerance);  EXPECT_FLOAT_EQ(c3, c1)
             EXPECT_FLOAT_EQ(c2, 0.0f)
             EXPECT_NEAR(g1[i], { 0.25, -0.25, 0.25, 0.5 }[i], Tolerance)
             EXPECT_FLOAT_EQ(g2[i], 0.0f)

TestBinaryCrossEntropyWithLogits.CostIsMeanSoftplusFormAndMatchesProbabilityBce:
    Arrange: BinaryCrossEntropy<float, 4> probability
    Act:     cost = loss.Cost(logits, target)
    Assert:  EXPECT_NEAR(cost, 1.0844635f, Tolerance)
             # ℓ = (0.1269280, 0.3132617, 0.8490770, 3.0485874)
             EXPECT_NEAR(cost, probability.Cost(sigmoidOfLogits, target), Tolerance)   # 1.0844636

TestBinaryCrossEntropyWithLogits.GradientIsSigmoidMinusTargetOverSize:
    Arrange: BinaryCrossEntropy<float, 4> probability
    Act:     gradient = loss.Gradient(logits, target)
             chained[i] = probability.Gradient(sigmoidOfLogits, target)[i]
                          * sigmoidOfLogits[i] * (1 - sigmoidOfLogits[i])            # ∂J/∂p · σ'(z)
    Assert:  EXPECT_NEAR(gradient[i], { -0.0298007, 0.0672354, 0.0931148, -0.2381435 }[i], Tolerance)
             EXPECT_NEAR(gradient[i], chained[i], Tolerance)

TestBinaryCrossEntropyWithLogits.GradientMatchesFiniteDifferenceOfCost:
    Arrange: zeros = Vector{ 0, 0, 0, 0 }                          # kink of max(z,0) and |z|; J smooth there
    Act:     for point in { logits, zeros }:
                 numeric  = CentralDifferenceGradient(loss, point, target)
                 analytic = loss.Gradient(point, target)
    Assert:  EXPECT_NEAR(analytic[i], numeric[i], Tolerance) for both points
             EXPECT_NEAR(loss.Cost(zeros, target), 0.6931472f, Tolerance)        # ln 2 for every label
             # analytic at zeros = (½ − y)/4 = (-0.125, 0.125, 0.0625, -0.125)

TestBinaryCrossEntropyWithLogits.ExtremeLogitsGiveExactCostAndBoundedGradient:
    Arrange: extreme  = Vector{ 100.0, -100.0, 1e30, -1e30 }
             labels   = Vector{   1.0,    0.0,  0.0,   1.0 }        # two confident-right, two confident-wrong
    Act:     cost = loss.Cost(extreme, labels);  gradient = loss.Gradient(extreme, labels)
    Assert:  EXPECT_FLOAT_EQ(cost, 5e29f)                           # (0 + 0 + 1e30 + 1e30)/4
             EXPECT_NEAR(gradient[i], { 0.0, 0.0, 0.25, -0.25 }[i], Tolerance)
             # probability BCE on σ(extreme) would report 8.01512 and a 2.5e6 gradient

TestCategoricalCrossEntropyClassIndex.ClassIndexEqualsOneHotTarget:
    Arrange: CategoricalCrossEntropy<float, 3> dense
             oneHot = Vector{ 0, 1, 0 }
    Act:     cost = loss.Cost(logits, 1);  gradient = loss.Gradient(logits, 1)
    Assert:  EXPECT_NEAR(cost, 1.4076060f, Tolerance)
             EXPECT_NEAR(gradient[i], { 0.0900306, -0.7552715, 0.6652410 }[i], Tolerance)
             EXPECT_NEAR(cost, dense.Cost(logits, oneHot), Tolerance)
             EXPECT_NEAR(gradient[i], dense.Gradient(logits, oneHot)[i], Tolerance)

TestCategoricalCrossEntropyClassIndex.GradientMatchesFiniteDifferenceOfCost:
    Arrange: point = Vector{ 0.4, -1.2, 2.5 };  k = 0
    Act:     numeric  = CentralDifferenceGradient(loss, point, k)
             analytic = loss.Gradient(point, k)
    Assert:  EXPECT_NEAR(analytic[i], numeric[i], Tolerance)
             # analytic = (-0.8932544, 0.0215516, 0.8717028), cost 2.2373067

TestCategoricalCrossEntropyClassIndex.HugeLogitsGiveExactCostAndGradient:
    Arrange: huge = Vector{ 1e30, 0.0, -1e30 }
    Act:     wrong = loss.Cost(huge, 2);  wrongGradient = loss.Gradient(huge, 2)
             right = loss.Cost(huge, 0);  rightGradient = loss.Gradient(huge, 0)
    Assert:  EXPECT_FLOAT_EQ(wrong, 2e30f)                          # logsumexp − z₂ = 1e30 + 1e30
             EXPECT_NEAR(wrongGradient[i], { 1.0, 0.0, -1.0 }[i], Tolerance)
             EXPECT_FLOAT_EQ(right, 0.0f)
             EXPECT_NEAR(rightGradient[i], 0.0f, Tolerance)
             # unshifted exp(1e30f) = inf
```

## Migration of existing tests (same cases, regulariser removed)

`LossTestSupport.hpp`: delete `RegularizationMock`; generalise the helper to
`CentralDifferenceGradient(const Loss<float, Size, Target>& loss, const Vector& point, const Target& target)`.
Each existing fixture drops its `StrictMock<RegularizationMock>` member and constructs the loss with no
arguments; every call passes the fixture target. Cases are renamed only to drop "PlusRegularization":

| File                                | Case                                        | New reference (was, with `R = 0.1`, `∇R = (0.01, 0.02, 0.03, 0.04)`) |
|-------------------------------------|---------------------------------------------|-------------------------------------------------------------------------|
| `TestMeanSquaredError.cpp`          | `CostIsMeanSquaredError`                    | `0.4375` (was `0.5375`)                                                |
|                                     | `GradientIsScaledError`                     | `(0.25, -0.25, 0.25, 0.5)` (was `(0.26, -0.23, 0.28, 0.54)`)           |
| `TestMeanAbsoluteError.cpp`         | `CostIsMeanAbsoluteError`                   | `0.625` (was `0.725`)                                                  |
|                                     | `GradientIsSignOverSize`                    | `(0.25, -0.25, 0.25, 0.25)` (was `(0.26, -0.23, 0.28, 0.29)`)          |
| `TestBinaryCrossEntropy.cpp`        | `CostIsMeanCrossEntropy`                    | `0.2990012` (was `0.3990012`)                                          |
|                                     | `GradientMatchesReference`                  | `(-0.2777778, 0.3125, -0.4166667, 0.3571429)`                          |
| `TestCategoricalCrossEntropy.cpp`   | `CostIsNegativeLogSoftmaxOfTarget`          | `1.4076060` (was `1.5076060`)                                          |
|                                     | `GradientIsSoftmaxMinusTarget`              | `(0.0900306, -0.7552715, 0.6652410)`                                   |
| `TestHuberLoss.cpp` (if N4 landed)  | `CostIsMeanHuber` / `GradientIsClippedResidualOverSize` | `1.375` / `(0.125, -0.125, 0.375, -0.375)`                  |

The finite-difference, zero-error, saturated-BCE (`8.01512`), large-logit CCE (`1000`, `(1, -1, 0)`) and
Huber limiting-case/outlier cases keep their values; only the mock expectations go. `TestModel.cpp`'s
`LossMock` becomes a `StrictMock` of `optimization::ObjectiveFunction<float, Size>` with the same
expectations (Model's `Train` behaviour is unchanged).

## Reference vectors

Computed by `scratchpad/specs/LossContract/reference.py` (numpy: `float32` mirror of the C++ loops,
cross-checked in `float64` against the textbook `−[y log σ + (1−y) log(1−σ)]`, `np.logaddexp`, and
max-shifted log-sum-exp; output in `reference.out`):

| Quantity                                                    | Value                                                                  |
|-------------------------------------------------------------|------------------------------------------------------------------------|
| BCE-logits `z = (2, -1, 0.5, -3)`, `y = (1, 0, 0.25, 1)`: `σ(z)` | `(0.8807971, 0.2689414, 0.6224593, 0.0474259)`                    |
| per-element `ℓ`                                             | `(0.1269280, 0.3132617, 0.8490770, 3.0485874)`                         |
| cost (float32 / float64 textbook / prob-BCE of `σ(z)`)      | `1.0844635` / `1.0844635086` / `1.0844636`                             |
| gradient `(σ − y)/4` (= prob-BCE gradient `· σ(1−σ)`)       | `(-0.0298007, 0.0672354, 0.0931148, -0.2381435)`                       |
| FD at `z` / at `0`: max error                               | `4.7e-5` / `9.1e-6`                                                    |
| at `z = 0`: cost / gradient                                 | `0.6931472` / `(-0.125, 0.125, 0.0625, -0.125)`                        |
| extreme `z = (100, -100, 1e30, -1e30)`, `y = (1, 0, 0, 1)`   | cost `5e29` (bit-equal `5e29f`), gradient `(0, 9.8e-45, 0.25, -0.25)`  |
| same input through prob-BCE on `σ(z)` (ε-clamped)           | cost `8.01512`, gradient `(-0.25, 0.25, 2.097e6, -2.5e6)`              |
| `z = 20, y = 0`: logits / prob-BCE                          | `20.0` / `15.942385`                                                   |
| CCE index `z = (1, 2, 3)`, `k = 1`: cost / gradient         | `1.4076060` / `(0.0900306, -0.7552715, 0.6652410)` (= one-hot dense)   |
| CCE index `z = (0.4, -1.2, 2.5)`, `k = 0`: cost / gradient  | `2.2373067` / `(-0.8932544, 0.0215516, 0.8717028)`, FD max err `1.0e-4` |
| CCE index `z = (1e30, 0, -1e30)`: `k = 2` / `k = 0`         | cost `2e30`, grad `(1, 0, -1)` / cost `0`, grad `(0, 0, 0)`            |
| MSE `ŷ = (1, -1.5, 2.5, 1)`, `y = (0.5, -1, 2, 0)`          | cost `0.4375`, gradient `(0.25, -0.25, 0.25, 0.5)`                     |

All FD errors are far below `math::Tolerance<float>() = 1e-3`.

## Edge cases

- `z = 0` (kink of `max(z,0)` and `|z|` individually): covered by the FD point `zeros`.
- `|z| > 16.64`: `1 + e^{−|z|}` rounds to `1`, the log term is `0` (error `< 2⁻²⁴`); covered implicitly by
  the extreme case (`±100`) and not asserted to more than `Tolerance`.
- Confident-wrong `σ(−100)` gradient is the denormal `9.8e-45` (or `0` with flush-to-zero): assert with
  `EXPECT_NEAR`, never `EXPECT_FLOAT_EQ`.
- Do **not** FD-check at `|z| ≥ 1e30`: `z ± 1e-3 == z` in `float`.
- Class index `k ≥ N`: `really_assert` (not unit-tested, consistent with the other precondition asserts).
- No separate soft-label, unnormalised-target or `N = 1` case: the soft label sits in the BCE-logits
  fixture, the unnormalised dense-CCE FD case already exists, and `N` is a template parameter with no
  special path.
- No `std::isfinite` assertions: the fast-math pragma reaches the test TU; the exact `EXPECT_FLOAT_EQ`
  values already fail on `inf`/`NaN`.
