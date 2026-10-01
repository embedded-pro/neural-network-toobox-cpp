# Swish Family (SiLU / Hard-Sigmoid / Hard-Swish) — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
class TestSiLU : public ::testing::Test:
    neural_network::SiLU<float> activation
class TestHardSigmoid : public ::testing::Test:
    neural_network::HardSigmoid<float> activation
class TestHardSwish : public ::testing::Test:
    neural_network::HardSwish<float> activation
# all three fixtures in one anonymous namespace in TestSwishActivations.cpp;
# each case below is a TEST_F(<Fixture>, <name>)
# no injected collaborators ⇒ no mocks (SiLU's Sigmoid is a concrete final value member, already
# covered by TestSigmoid; the base-class default vector paths are covered by TestActivationFunction)
# finite differences reuse test/ActivationFiniteDifference.hpp (h = 1e-3, float, via base reference)
# tolerance: math::Tolerance<float>() (1e-3) unless EXPECT_FLOAT_EQ is stated
```

## Test cases (Arrange / Act / Assert)

```
# ---------------- TestSiLU ----------------
ForwardMatchesReferenceValues:
    Act:    Forward(x) for x ∈ {-2.0, 0.0, 1.0, 3.0}
    Assert: EXPECT_NEAR ≈ {-0.2384058, 0.0, 0.7310586, 2.857722}

BackwardMatchesReferenceValues:
    Assert: EXPECT_FLOAT_EQ(Backward(0.0f), 0.5f)                   # σ(0)·(1 + 0) exactly
            EXPECT_NEAR(Backward(2.0f),  1.090784)                  # > 1: SiLU overshoots slope 1
            EXPECT_NEAR(Backward(-2.0f), -0.09078425)               # negative: non-monotonic region
            EXPECT_NEAR(Backward(-1.2784645f), 0.0)                 # stationary point z*
            EXPECT_NEAR(Forward(-1.2784645f), -0.2784645)           # global minimum f(z*) = z* + 1

BackwardMatchesFiniteDifference:
    Assert: EXPECT_NEAR(Backward(x), CentralDifference(activation, x), Tolerance)
            for x ∈ {-3.0, -0.5, 1.0, 4.0}

BackwardVectorMatchesFiniteDifference:
    Arrange: input    = {-3.0, -0.5, 1.0, 4.0}
             upstream = { 0.3, -1.2, 0.8, -0.4}
    Act:     analytic = AnalyticGradient(activation, input, upstream)
             numeric  = CentralDifferenceGradient(activation, input, upstream)
    Assert:  EXPECT_NEAR(analytic[i], {-0.02643123, -0.3120466, 0.7421364, -0.4210659}[i], Tolerance)
             EXPECT_NEAR(analytic[i], numeric[i], Tolerance)

SaturatesWithoutOverflowOrCancellation:
    Assert: EXPECT_FLOAT_EQ(Forward(1e30f), 1e30f);   EXPECT_FLOAT_EQ(Backward(1e30f), 1.0f)
            EXPECT_FLOAT_EQ(Forward(-1e30f), 0.0f);   EXPECT_FLOAT_EQ(Backward(-1e30f), 0.0f)
    Arrange: input = {1e8, 1e30, -1e30},  upstream = {1, 1, 1}
    Act:     result = AnalyticGradient(activation, input, upstream)
    Assert:  EXPECT_FLOAT_EQ(result[i], {1.0, 1.0, 0.0}[i])
             # guards BackwardVector against the output-reuse form a + σ(1 − a), which returns 0 at 1e8

# ---------------- TestHardSigmoid ----------------
ForwardMatchesPiecewiseLinearReference:
    Act:    Forward(x) for x ∈ {-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0}
    Assert: EXPECT_FLOAT_EQ for the saturated points  {-4, -3} → 0,  {3, 4} → 1
            EXPECT_NEAR     for the linear points     {-1.5, 0, 1.5} → {0.25, 0.5, 0.75}

BackwardIsExactSlopeInsideKneesAndMatchesFiniteDifference:
    Assert: EXPECT_FLOAT_EQ(Backward(x), 1.0f / 6.0f)  for x ∈ {0.0, 2.9}
            EXPECT_FLOAT_EQ(Backward(x), 0.0f)         for x ∈ {-4.0, -3.0, 3.0, 4.0}   # knees → 0
            EXPECT_NEAR(Backward(x), CentralDifference(activation, x), Tolerance)
                                                        for x ∈ {-2.0, 0.0, 1.0, 2.5}

BackwardVectorMatchesFiniteDifference:
    Arrange: input    = {-4.0, -1.5, 0.7, 3.5}          # both saturated sides + linear region
             upstream = { 0.3, -1.2, 0.8, -0.4}
    Assert:  EXPECT_NEAR(analytic[i], {0.0, -0.2, 0.1333333, 0.0}[i], Tolerance)
             EXPECT_NEAR(analytic[i], numeric[i], Tolerance)

# ---------------- TestHardSwish ----------------
ForwardMatchesPiecewiseReference:
    Act:    Forward(x) for x ∈ {-4.0, -3.0, -1.5, 0.0, 1.0, 3.0, 5.0}
    Assert: EXPECT_NEAR ≈ {0.0, 0.0, -0.375, 0.0, 0.6666667, 3.0, 5.0}   # -0.375 is the global minimum
            EXPECT_FLOAT_EQ(Forward(1e30f), 1e30f)       # identity branch; x(x+3)/6 would overflow to +inf
            EXPECT_FLOAT_EQ(Forward(-1e30f), 0.0f)

BackwardMatchesReferenceAndKneeConventionAndFiniteDifference:
    Assert: EXPECT_FLOAT_EQ(Backward(x), expected) for
                x        = {-4.0, -3.0, 3.0, 5.0}
                expected = { 0.0,  0.0, 1.0, 1.0}        # knees: outer piece wins (open middle interval)
            EXPECT_NEAR(Backward(x), expected, Tolerance) for
                x        = {-1.5, 0.0, 1.0}
                expected = { 0.0, 0.5, 0.8333334}        # (2x + 3)/6
            EXPECT_NEAR(Backward(x), CentralDifference(activation, x), Tolerance)
                for x ∈ {-2.0, -0.5, 1.0, 2.5}

BackwardVectorMatchesFiniteDifference:
    Arrange: input    = {-4.0, -1.0, 0.7, 3.5}
             upstream = { 0.3, -1.2, 0.8, -0.4}
    Assert:  EXPECT_NEAR(analytic[i], {0.0, -0.2, 0.5866667, -0.4}[i], Tolerance)
             EXPECT_NEAR(analytic[i], numeric[i], Tolerance)
```

## Reference vectors

Computed by [`reference.py`](reference.py) (numpy `float32`, same operation order as
the pseudocode and the same `h = 1e-3` step and float accumulation as `ActivationFiniteDifference.hpp`;
float64 cross-check agrees to 7 significant digits):

| Quantity                                             | Value                                                          |
|------------------------------------------------------|----------------------------------------------------------------|
| SiLU `Forward` `{-2, 0, 1, 3}`                       | `{-0.2384058, 0, 0.7310586, 2.857722}`                         |
| SiLU `Backward` `{0, 2, -2}`                         | `{0.5, 1.090784, -0.09078425}`                                 |
| SiLU stationary point `z*`, `f(z*)`                  | `-1.278464543`, `-0.278464543` (`= z* + 1`), `f'(z*) ≈ 0`      |
| SiLU scalar FD `{-3, -0.5, 1, 4}` analytic           | `{-0.0881041, 0.2600388, 0.9276705, 1.052665}`                 |
| SiLU scalar FD `{-3, -0.5, 1, 4}` numeric            | `{-0.08811056, 0.2600327, 0.9276866, 1.052499}` (max err `1.7e-4`) |
| SiLU `BackwardVector` analytic                       | `{-0.02643123, -0.3120466, 0.7421364, -0.4210659}`             |
| SiLU `BackwardVector` FD                             | `{-0.02640486, -0.3120303, 0.7421374, -0.4210472}` (max err `2.6e-5`) |
| SiLU at `±1e30`                                      | `f = 1e30, f' = 1`;  `f = -0, f' = -0` (no NaN)                |
| SiLU `f'` at `1e8`: chosen form / output-reuse form  | `1` / `0` (cancellation)                                       |
| Hard-Sigmoid `Forward` `{-4,-3,-1.5,0,1.5,3,4}`      | `{0, 0, 0.25, 0.5, 0.75, 1, 1}`                                |
| Hard-Sigmoid `Backward` `{-4,-3,0,2.9,3,4}`          | `{0, 0, 0.1666667, 0.1666667, 0, 0}`                           |
| Hard-Sigmoid scalar FD `{-2,0,1,2.5}`                | max err `1.8e-5`                                               |
| Hard-Sigmoid `BackwardVector` analytic / FD          | `{0, -0.2, 0.1333333, 0}` / `{0, -0.2000034, 0.1333207, 0}` (max err `1.3e-5`) |
| Hard-Swish `Forward` `{-4,-3,-1.5,0,1,3,5}`          | `{0, 0, -0.375, 0, 0.6666667, 3, 5}`                           |
| Hard-Swish `Backward` `{-4,-3,-1.5,0,1,3,5}`         | `{0, 0, 0, 0.5, 0.8333334, 1, 1}`                              |
| Hard-Swish scalar FD `{-2,-0.5,1,2.5}` analytic      | `{-0.1666667, 0.3333333, 0.8333334, 1.333333}`                 |
| Hard-Swish scalar FD `{-2,-0.5,1,2.5}` numeric       | `{-0.1666546, 0.3333389, 0.8333325, 1.333237}` (max err `9.7e-5`) |
| Hard-Swish `BackwardVector` analytic                 | `{0, -0.2, 0.5866667, -0.4}`                                   |
| Hard-Swish `BackwardVector` FD                       | `{0, -0.2000034, 0.5866885, -0.4000067}` (max err `2.2e-5`)    |

## Edge cases

- Knees `z = ±3` of the hard functions: derivative pinned by the convention (outer piece), asserted
  exactly; **no** FD check within `h` of a knee (one-sided slopes differ: `q'` jumps `0 → -0.5` at `-3`
  and `1.5 → 1` at `3`, so the central difference averages them).
- `|z| = 1e30`: SiLU forward/backward stay finite via the sign-branched sigmoid; Hard-Swish must take the
  identity branch before evaluating `z(z + 3)`; do **not** FD-check there (`z ± 1e-3 == z` in float).
- SiLU at `z = 0`: `f = 0`, `f' = 0.5` exactly — the reason `BackwardVector` recomputes `σ` from `z`
  instead of dividing the output by `z`.
- `output` aliasing `input` in `ForwardVector`: element-wise read-then-write, safe.
- Size-mismatched spans: `really_assert` fires (not unit-tested, consistent with the other activations).
