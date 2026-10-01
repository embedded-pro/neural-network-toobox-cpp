# ELU / SELU Activation — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
class TestExponentialLinearUnit : public ::testing::Test:
    neural_network::ExponentialLinearUnit<float> elu                                                   # α = 1, λ = 1
    neural_network::ExponentialLinearUnit<float> selu{ neural_network::ExponentialLinearUnit<float>::Selu() }
# each case below is a TEST_F(TestExponentialLinearUnit, <name>)
# no collaborators ⇒ no mocks; the base-class default vector paths are already covered by
# TestActivationFunction (StrictMock<ActivationFunctionMock>)
# finite differences reuse test/ActivationFiniteDifference.hpp (h = 1e-3, float, via base reference)
# tolerance: math::Tolerance<float>() (1e-3) unless stated
```

## Test cases (Arrange / Act / Assert)

```
EluForwardIsLinearForPositiveAndExponentialForNonPositive:
    Act:    elu.Forward(x) for x ∈ {-3.0, -1.0, -0.5, 0.0, 2.0}
    Assert: EXPECT_NEAR(.., {-0.950213, -0.6321206, -0.3934693, 0.0, 2.0}, Tolerance)

SeluForwardVectorScalesBothBranches:
    Arrange: input = {-2.0, -0.5, 0.0, 0.7, 1.5}
    Act:     selu.ForwardVector(output, input)
    Assert:  output ≈ {-1.520167, -0.6917582, 0.0, 0.7354907, 1.576051}

EluBackwardIsExactAndMatchesFiniteDifference:
    Assert: EXPECT_NEAR(elu.Backward(x), {0.1353353, 0.6065307, 1.0}, Tolerance)   for x ∈ {-2.0, -0.5, 0.0}
            EXPECT_FLOAT_EQ(elu.Backward(1.5f), 1.0f)                              # exact constant, not 0.9999
    Assert: EXPECT_NEAR(CentralDifference(elu, x), elu.Backward(x), Tolerance)     for x ∈ {-2.0, -0.5, 0.0, 0.7}
                                                                                   # x = 0 included: ELU is C¹ there

SeluBackwardHasKinkAtZeroAndMatchesFiniteDifferenceElsewhere:
    Assert: EXPECT_NEAR(selu.Backward(x), {0.2379329, 1.066341, 1.758099}, Tolerance)   for x ∈ {-2.0, -0.5, 0.0}
            EXPECT_FLOAT_EQ(selu.Backward(0.5f), ExponentialLinearUnit<float>::seluScale)
    Assert: EXPECT_NEAR(CentralDifference(selu, x), selu.Backward(x), Tolerance)   for x ∈ {-2.0, -0.5, 0.7}
                                                                                   # never at the kink x = 0

BackwardVectorReusesOutputAndMatchesFiniteDifference:
    Arrange: input    = {-2.0, -0.5, 0.7, 1.5}
             upstream = { 0.3, -1.2, 0.8, -0.4}
    Act:     for activation ∈ {elu, selu}:
                 analytic = AnalyticGradient(activation, input, upstream)
                 numeric  = CentralDifferenceGradient(activation, input, upstream)
    Assert:  elu  analytic ≈ {0.04060058, -0.7278368, 0.8, -0.4}
             selu analytic ≈ {0.07137984, -1.279609, 0.8405609, -0.4202804}
             EXPECT_NEAR(analytic[i], numeric[i], Tolerance)          for both

SaturatesAtMinusScaleTimesAlphaWithoutOverflow:
    Assert: EXPECT_FLOAT_EQ(elu.Forward(-100.0f), -1.0f)
            EXPECT_FLOAT_EQ(selu.Forward(-100.0f), -1.7580993f)          # −λα
            EXPECT_NEAR(selu.Backward(-100.0f), 0.0f, Tolerance)
            EXPECT_FLOAT_EQ(selu.Forward(1e30f), 1.050701e30f)           # λz, finite, no clamp
    Arrange: input = {-100.0, 1e30}, output = selu.ForwardVector(input), upstream = {1.0, 1.0}
    Act:     selu.BackwardVector(result, input, output, upstream)
    Assert:  EXPECT_FLOAT_EQ(result[0], 0.0f)                             # output + λα cancels exactly
             EXPECT_FLOAT_EQ(result[1], ExponentialLinearUnit<float>::seluScale)

SeluMapsStandardNormalToZeroMeanUnitVariance:
    Arrange: midpoint rule on [-10, 10], 2000 cells, step h = 0.01, no arrays:
             for k in 0..1999:
                 z = -10 + (k + 0.5)·h
                 w = h · Exp(-z²/2) / √(2π)
                 y = selu.Forward(z)
                 mean += w·y;  second += w·y²
    Assert:  EXPECT_NEAR(mean,   0.0f, Tolerance)
             EXPECT_NEAR(second, 1.0f, Tolerance)
```

`SeluMapsStandardNormalToZeroMeanUnitVariance` is the only test of the defining SELU property (the
`(α₀₁, λ₀₁)` fixed point); the same quadrature with `elu` gives mean `0.1605`, second moment `0.6449`,
so it would fail for a wrong constant pair.

## Reference vectors

Computed by `scratchpad/specs/ExponentialLinearUnit/reference.py` (float32 emulation of the
pseudocode and of `ActivationFiniteDifference.hpp`, cross-checked against double `expm1`/`exp`):

| Quantity                                                    | Value                                                           |
|-------------------------------------------------------------|-----------------------------------------------------------------|
| float32 SELU constants `α₀₁`, `λ₀₁`, `λα`                   | `1.6732632`, `1.0507010`, `1.7580993`                           |
| `elu.Forward` at `{-3, -1, -0.5, 0, 2}`                     | `{-0.950213, -0.6321206, -0.3934693, 0, 2}`                     |
| `selu.ForwardVector` at `{-2, -0.5, 0, 0.7, 1.5}`           | `{-1.520167, -0.6917582, 0, 0.7354907, 1.576051}`               |
| `elu.Backward` at `{-2, -0.5, 0, 1.5}`                      | `{0.1353353, 0.6065307, 1, 1}`                                  |
| `elu` scalar FD at `{-2, -0.5, 0.7}`                        | `{0.1353323, 0.6065071, 0.9999871}` (max err `2.4e-5`)          |
| `elu` scalar FD at `0`                                      | `0.9997552` (err `2.4e-4`, from the `f''` jump; < `1e-3`)       |
| `selu.Backward` at `{-2, -0.5, 0, 0.5}`                     | `{0.2379329, 1.066341, 1.758099, 1.050701}`                     |
| `selu` scalar FD at `{-2, -0.5, 0.7}`                       | `{0.2379417, 1.066297, 1.050681}` (max err `4.4e-5`)            |
| `elu` `BackwardVector` analytic / FD                        | `{0.04060058, -0.7278368, 0.8, -0.4}` / `{0.04062056, -0.7278025, 0.8000135, -0.4000067}` (max diff `3.4e-5`) |
| `selu` `BackwardVector` analytic / FD                       | `{0.07137984, -1.279609, 0.8405609, -0.4202804}` / `{0.07137656, -1.279563, 0.8405446, -0.4203319}` (max diff `5.1e-5`) |
| `Forward(-100)` ELU / SELU                                  | `-1` / `-1.7580993`                                             |
| `Forward(1e30)` ELU / SELU                                  | `1e30` / `1.050701e30`                                          |
| `BackwardVector` at `-100` (reused output)                  | `0` exactly (ELU and SELU)                                      |
| SELU moments, closed form (double)                          | mean `1.7e-16`, second moment `1 − 2e-16`                       |
| SELU moments, float32 quadrature as specified               | mean `-1.2e-6`, second moment `0.9999996`                       |
| ELU moments, same quadrature                                | mean `0.1605203`, second moment `0.6449451`                     |

## Edge cases

- `x = 0`: `Forward` returns exactly `0`; `Backward` takes the `x ≤ 0` branch (`λα`: `1` for ELU,
  `1.758099` for SELU). Never finite-difference SELU at `0` — the central difference averages the two
  slopes (`1.40397`).
- `x = -100`: `Exp` underflows (or flushes to zero under fast-math); output saturates at `−λα`,
  derivative `0`, no NaN.
- `|x| = 1e30`: do **not** FD-check there — `x ± 1e-3 == x` in float32, so the difference collapses to `0`.
- Reused-output derivative at `z = −10`: `7.98702e-5` vs exact `7.98176e-5` (abs err `5.3e-8`); fine for
  training, documented rather than tested.
- `output` aliasing `input` in `ForwardVector`: each element is read before it is written, safe.
- `alpha ≤ 0`, `scale ≤ 0`, size-mismatched spans: `really_assert` fires (not unit-tested, consistent
  with the other activations).
