# Huber Loss — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
class TestHuberLoss : public ::testing::Test:
    static constexpr std::size_t size = 4
    using Vector = neural_network::HuberLoss<float, size>::Vector
    ::testing::StrictMock<neural_network::test_support::RegularizationMock<size>> regularization
    const Vector target      = { 0.5, -1.0, 2.0,  0.0 }
    const Vector predictions = { 1.0, -1.5, 4.0, -3.0 }     # e = (0.5, -0.5, 2.0, -3.0)
    neural_network::HuberLoss<float, size> loss{ target, regularization, 1.5f }
# each case below is a TEST_F(TestHuberLoss, <name>)
# δ = 1.5 puts two residuals in the quadratic branch and one on each side of the linear branch,
# and δ ≠ δ² so a δ/δ² mix-up is caught
# collaborator = Regularization ⇒ StrictMock<RegularizationMock<4>> from test/LossTestSupport.hpp;
# every Cost() calls Calculate once, every Gradient() calls Gradient once
# finite differences reuse test_support::CentralDifferenceGradient (h = 1e-3, float)
```

## Test cases (Arrange / Act / Assert)

```
CostIsMeanHuberPlusRegularization:
    Arrange: EXPECT_CALL(regularization, Calculate(_)).WillOnce(Return(0.1f))
    Act:     cost = loss.Cost(predictions)
    Assert:  EXPECT_NEAR(cost, 1.475f, Tolerance)
             # ρ = (0.125, 0.125, 1.875, 3.375), mean 1.375, + 0.1

GradientIsClippedResidualOverSizePlusRegularizationGradient:
    Arrange: EXPECT_CALL(regularization, Gradient(_)).WillOnce(Return(Vector{ 0.01, 0.02, 0.03, 0.04 }))
    Act:     gradient = loss.Gradient(predictions)
    Assert:  EXPECT_NEAR(gradient[i], { 0.135, -0.105, 0.405, -0.335 }[i], Tolerance)
             # clip(e, ±1.5)/4 = (0.125, -0.125, 0.375, -0.375), + regulariser gradient

GradientMatchesFiniteDifferenceOfCost:
    Arrange: EXPECT_CALL(regularization, Calculate(_)).WillRepeatedly(Return(0.0f))
             EXPECT_CALL(regularization, Gradient(_)).Times(2).WillRepeatedly(Return(Vector{}))
             boundary = target;  boundary[0] = target[0] + 1.5f          # e₀ = δ exactly (C¹ seam)
    Act:     for point in { predictions, boundary }:
                 numeric  = CentralDifferenceGradient(loss, point)
                 analytic = loss.Gradient(point)
    Assert:  EXPECT_NEAR(analytic[i], numeric[i], Tolerance) for both points

ReducesToHalfMseForLargeDeltaAndScaledMaeForSmallDelta:
    Arrange: EXPECT_CALL(regularization, Calculate(_)).WillRepeatedly(Return(0.0f))
             EXPECT_CALL(regularization, Gradient(_)).WillRepeatedly(Return(Vector{}))
             HuberLoss<float,4>         wide  { target, regularization, 10.0f }    # δ ≥ max|e| = 3
             HuberLoss<float,4>         narrow{ target, regularization, 0.25f }    # δ ≤ min|e| = 0.5
             MeanSquaredError<float,4>  mse   { target, regularization }
             MeanAbsoluteError<float,4> mae   { target, regularization }
    Act:     evaluate Cost and Gradient of all four at predictions
    Assert:  EXPECT_NEAR(wide.Cost,   0.5 * mse.Cost,                 Tolerance)   # 1.6875 = ½·3.375
             EXPECT_NEAR(wide.Gradient[i],   0.5 * mse.Gradient[i],   Tolerance)   # (0.125,-0.125,0.5,-0.75)
             EXPECT_NEAR(narrow.Cost, 0.25 * mae.Cost - 0.5 * 0.25²,  Tolerance)   # 0.34375 = 0.25·1.5 − 0.03125
             EXPECT_NEAR(narrow.Gradient[i], 0.25 * mae.Gradient[i],  Tolerance)   # (0.0625,-0.0625,0.0625,-0.0625)

LargeOutlierKeepsCostFiniteAndGradientBoundedByDelta:
    Arrange: EXPECT_CALL(regularization, Calculate(_)).WillOnce(Return(0.0f))
             EXPECT_CALL(regularization, Gradient(_)).WillOnce(Return(Vector{}))
             outlier = target;  outlier[0] = target[0] + 1e20f                # e = (1e20, 0, 0, 0)
    Act:     cost = loss.Cost(outlier);  gradient = loss.Gradient(outlier)
    Assert:  EXPECT_TRUE(std::isfinite(cost))
             EXPECT_FLOAT_EQ(cost, 3.75e19f)                                  # 1.5·(1e20 − 0.75)/4 (MSE here is +inf)
             EXPECT_FLOAT_EQ(gradient[0], 0.375f)                             # δ/N
             EXPECT_FLOAT_EQ(gradient[i], 0.0f) for i = 1..3                  # e = 0 ⇒ exactly 0
```

`ReducesToHalfMseForLargeDeltaAndScaledMaeForSmallDelta` includes `MeanSquaredError.hpp` and
`MeanAbsoluteError.hpp`; both live in the same `neural_network.losses` target, so no new link.

## Reference vectors

Computed by `scratchpad/specs/HuberLoss/reference.py` (numpy `float32`, same branch-free formula and FD
step as the helper; the quadratic/linear piecewise form in `float64` agrees on every cost):

| Quantity                                                  | Value                                                         |
|-----------------------------------------------------------|---------------------------------------------------------------|
| residuals `e = ŷ − y`                                     | `(0.5, -0.5, 2.0, -3.0)`                                      |
| `ρ_1.5(e)`                                                | `(0.125, 0.125, 1.875, 3.375)`                                |
| `Cost`, δ = 1.5, `R = 0.1`                                | `1.375 + 0.1 = 1.475`                                         |
| `Gradient`, δ = 1.5, `∇R = (0.01, 0.02, 0.03, 0.04)`      | `(0.125, -0.125, 0.375, -0.375) + ∇R = (0.135, -0.105, 0.405, -0.335)` |
| FD at `predictions`, δ = 1.5                              | `(0.1250505, -0.1250505, 0.3750324, -0.3750324)` (max err `5.1e-5`)    |
| boundary `e = (1.5, 0, 0, 0)`: cost / analytic / FD       | `0.28125` / `(0.375, 0, 0, 0)` / `(0.3749281, 0, 0, 0)` (err `7.2e-5`) |
| `MSE` / `MAE` at `predictions`                            | `3.375` / `1.5`                                               |
| Huber δ = 10: cost / gradient                             | `1.6875` / `(0.125, -0.125, 0.5, -0.75)` (FD max err `5.4e-5`) |
| Huber δ = 0.25: cost / gradient                           | `0.34375` / `(0.0625, -0.0625, 0.0625, -0.0625)` (FD max err `4.5e-6`) |
| outlier `e₀ = 1e20`, δ = 1.5: cost / gradient             | `3.75e19` / `(0.375, 0, 0, 0)`; float32 MSE on the same input = `inf`  |

All FD errors are far below `math::Tolerance<float>() = 1e-3`; the analytic FD truncation error at the
seam `|e| = δ` is `h/(4N) = 6.25e-5`.

## Edge cases

- `|e| = δ` exactly: both branches agree (`½δ²`, `±δ`); covered by the boundary FD point.
- `e = 0`: cost and gradient exactly `0` (no MAE-style sub-gradient choice); covered by the outlier case.
- `|e| ≳ 1.84e19`: MSE overflows to `inf`, Huber stays finite; do **not** FD-check there — `x ± 1e-3 == x`.
- `delta ≤ 0`: `really_assert` fires in the constructor (not unit-tested, consistent with the other
  size/precondition asserts).
- No separate zero-error or sign-symmetry case: the fixture already has residuals of both signs in both
  branches, and `e = 0` is hit by the outlier case.
