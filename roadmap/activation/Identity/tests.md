# Identity (Linear) Activation — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
class TestIdentity : public ::testing::Test:
    neural_network::Identity<float> activation
# each case below is a TEST_F(TestIdentity, <name>)
# no collaborators ⇒ no mocks; the base-class default vector paths are already covered by
# TestActivationFunction (StrictMock<ActivationFunctionMock>)
# finite differences reuse test/ActivationFiniteDifference.hpp (h = 1e-3, float, via base reference)
```

## Test cases (Arrange / Act / Assert)

```
ForwardReturnsInputUnchanged:
    Act:    Forward(x) for x ∈ {-2.5, 0.0, 3.75, 1e30}
    Assert: EXPECT_FLOAT_EQ(Forward(x), x)             # bit-exact, no clamp at large |x|

BackwardIsExactlyOneAndMatchesFiniteDifference:
    Assert: EXPECT_FLOAT_EQ(Backward(x), 1.0f)        for x ∈ {-1e30, 0.0, 7.5}
    Assert: EXPECT_NEAR(CentralDifference(activation, x), Backward(x), Tolerance)
                                                      for x ∈ {-2.0, 0.0, 0.7}

ForwardVectorCopiesInput:
    Arrange: input = {-1.5, 0.0, 0.25, 4.0}
    Act:     ForwardVector(output, input)
    Assert:  output[i] == input[i]  (EXPECT_FLOAT_EQ)

BackwardVectorPassesUpstreamThroughAndMatchesFiniteDifference:
    Arrange: input    = {-2.0, -0.5, 0.7, 1.5}
             upstream = { 0.3, -1.2, 0.8, -0.4}
    Act:     analytic = AnalyticGradient(activation, input, upstream)
             numeric  = CentralDifferenceGradient(activation, input, upstream)
    Assert:  EXPECT_FLOAT_EQ(analytic[i], upstream[i])
             EXPECT_NEAR(analytic[i], numeric[i], Tolerance)
```

Integration case, deployed as `TEST_F(TestDense, IdentityActivationEmitsRawAffineOutput)` in
`neural_network/layer/test/TestDense.cpp` (the activation test target does not link `neural_network.layer`):

```
IdentityActivationEmitsRawAffineOutput:
    Arrange: Dense<float,3,2> layer{ weights, identity }
             weights = {{0.1, -0.2, 0.3}, {0.4, 0.5, -0.6}}
             SetParameters({0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.2, -0.1})   # biases (0.2, -0.1)
    Act:     layer.Forward({1, 2, 3});  grad = layer.Backward({0.8, -1.3})
    Assert:  Output ≈ (0.8, -0.5)                  # negative output survives (LeakyReLU 0.1 would give -0.05)
             grad   ≈ (-0.44, -0.81, 1.02)          # = Wᵀ g
```

## Reference vectors

Computed by [`reference.py`](reference.py) (float32-rounded, same step as the helper):

| Quantity                                         | Value                                                     |
|--------------------------------------------------|-----------------------------------------------------------|
| `Forward(x)` for `{-2.5, 0, 3.75, 1e30}`         | `{-2.5, 0, 3.75, 1e30}` (bit-exact)                       |
| `Backward(x)` for `{-1e30, 0, 7.5}`              | `1, 1, 1`                                                 |
| scalar FD at `{-2, 0, 0.7}`                      | `0.9999871, 1.0, 0.9999871` (max err `1.3e-5` < `1e-3`)   |
| `BackwardVector` analytic                        | `{0.3, -1.2, 0.8, -0.4}`                                  |
| `BackwardVector` FD                              | `{0.2999902, -1.1999607, 0.8000135, -0.4000067}` (max err `3.9e-5`) |
| Dense + Identity output, `x = (1,2,3)`, `b = (0.2,-0.1)` | `(0.8, -0.5)`                                     |
| Dense + Identity input gradient, `g = (0.8,-1.3)`        | `(-0.44, -0.81, 1.02)` (FD agrees)                |

## Edge cases

- `|x| = 1e30`: forward passes through, derivative stays exactly `1`; do **not** FD-check there — the
  float32 central difference collapses to `0` because `x ± 1e-3 == x`.
- `x = ±0.0`: returned unchanged, derivative `1` (no kink, unlike ReLU).
- `output` aliasing `input` in `ForwardVector`: a plain element copy, safe.
- Size-mismatched spans: `really_assert` fires (not unit-tested, consistent with the other activations).
