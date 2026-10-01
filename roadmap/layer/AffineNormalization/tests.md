# Per-Feature Affine Normalisation — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
class TestAffineNormalization : public ::testing::Test:
    using Normalization   = neural_network::AffineNormalization<float, 3>
    using Vector3         = Normalization::InputVector
    using ParameterVector = Normalization::ParameterVector

    const Vector3 scale{ 2.0, -0.5, 0.25 }
    const Vector3 shift{ 1.0,  0.5, -3.0 }
    const Vector3 input{ 0.3, -0.7, 1.1 }
    const Vector3 upstream{ 0.8, -1.3, 0.4 }

    float ProjectedOutput(Normalization& layer, const Vector3& x, const Vector3& g):
        layer.Forward(x)
        return Σ_i g[i] · layer.Output()[i]
# each case below is a TEST_F(TestAffineNormalization, <name>)
# no collaborators ⇒ no mocks (same as TestDense)
# finite difference: central, h = 1e-3 (float), tolerance math::Tolerance<float>() = 1e-3
```

## Test cases (Arrange / Act / Assert)

```
DefaultConstructedIsIdentity:
    Arrange: Normalization layer{}
    Act:     layer.Forward(input)
    Assert:  Parameters() ≈ {1, 1, 1, 0, 0, 0}
             Output()     ≈ input = {0.3, -0.7, 1.1}

ParametersAreScalesFollowedByShifts:
    Arrange: const Normalization layer{ scale, shift }
    Assert:  Parameters() ≈ {2.0, -0.5, 0.25, 1.0, 0.5, -3.0}

ForwardAppliesPerFeatureScaleAndShift:
    Arrange: Normalization layer{ scale, shift }
    Act:     layer.Forward(input)
    Assert:  Output() ≈ {1.6, 0.85, -2.725}                 # negative scale and negative output survive

SetParametersUsesSameLayoutForForward:
    Arrange: layer{ scale, shift };  p = {1.5, 0.0, -4.0, -1.0, 2.0, 0.5}
    Act:     layer.SetParameters(p);  layer.Forward(input)
    Assert:  Output()     ≈ {-0.55, 2.0, -3.9}              # a₁ = 0 ⇒ constant b₁ = 2
             Parameters() ≈ p

BackwardInputGradientIsScaledUpstreamAndMatchesFiniteDifference:
    Arrange: layer{ scale, shift }
             numeric[j] = (ProjectedOutput(x + h eⱼ) − ProjectedOutput(x − h eⱼ)) / (2h),  x = input
    Act:     layer.Forward(input);  analytic = layer.Backward(upstream)
    Assert:  analytic ≈ {1.6, 0.65, 0.1}                    # = a ⊙ g
             EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>())

FromBatchNormMatchesFrozenBatchNormFormula:
    Arrange: gamma = {1.5, 0.8, 2.0},  beta = {0.1, -0.2, 0.0}
             mean  = {0.5, -1.0, 3.0}, variance = {4.0, 0.25, 0.01},  epsilon = 1e-3
    Act:     layer = Normalization::FromBatchNorm(gamma, beta, mean, variance, epsilon)
             layer.Forward({1.0, -0.5, 3.2})
    Assert:  Parameters() ≈ {0.7499063, 1.5968096, 19.069252, -0.2749531, 1.3968096, -57.207756}
             Output()     ≈ {0.4749531, 0.5984048, 3.8138504}    # = γ (x − μ)/√(σ² + ε) + β

FromStandardisationMapsSensorUnitsToUnitScale:
    Arrange: mean = {9.81, -1.2, 3.3}   (m/s², °/s, V),  standardDeviation = {0.5, 35.0, 0.02}
    Act:     layer = Normalization::FromStandardisation(mean, standardDeviation)
             Forward(mean)              → out0
             Forward(mean + sd)         → out1        # {10.31, 33.8, 3.32}
             Forward(mean − 2·sd)       → out2        # {8.81, -71.2, 3.26}
    Assert:  Parameters() ≈ {2.0, 0.028571429, 50.0, -19.62, 0.034285717, -165.0}
             out0 ≈ {0, 0, 0},  out1 ≈ {1, 1, 1},  out2 ≈ {-2, -2, -2}
```

Integration case, deployed as `TEST_F(TestModel, AffineNormalizationComposesAndFoldsIntoDense)` in
`neural_network/model/test/TestModel.cpp`. The layer test target does not link `neural_network.model`,
and the model target is where layer chaining and parameter concatenation are exercised:

```
AffineNormalizationComposesAndFoldsIntoDense:
    Arrange: Model<float, 3, 1, AffineNormalization<float, 3>, Dense<float, 3, 1>> model{
                 make_layer<AffineNormalization<float, 3>>(scale, shift),
                 make_layer<Dense<float, 3, 1>>(W, tanh) }     # W = {{0.4, -0.3, 0.2}}
             model.SetParameters({2.0, -0.5, 0.25, 1.0, 0.5, -3.0, 0.4, -0.3, 0.2, 0.1})   # TotalParameters = 6 + 4
             folded = Dense<float, 3, 1>{ W, tanh };  folded.SetParameters({0.8, 0.15, 0.05, -0.25})
                                                               # W' = W·diag(a), c' = c + W b
    Act:     y = model.Forward({0.3, -0.7, 1.1});  yf = folded.Forward(same input)
    Assert:  y[0] ≈ -0.0599281  (= tanh(-0.06))
             y[0] ≈ yf[0]                                      # offline fold is exact (N13)
             model.GetParameters() round-trips the 10 values in order
```

## Reference vectors

Computed by `scratchpad/specs/AffineNormalization/reference.py`. It uses float32 rounding at every
operation and the same central-difference step as the tests:

| Quantity                                                         | Value                                                               |
|------------------------------------------------------------------|---------------------------------------------------------------------|
| Forward, `a = (2, -0.5, 0.25)`, `b = (1, 0.5, -3)`, `x = (0.3, -0.7, 1.1)` | `(1.6, 0.85, -2.725)`                                     |
| Forward after `SetParameters(1.5, 0, -4, -1, 2, 0.5)`            | `(-0.55, 2.0, -3.9)`                                                |
| `Backward(g = (0.8, -1.3, 0.4))` analytic `a ⊙ g`                | `(1.6, 0.65, 0.1)`                                                  |
| `Backward` finite difference                                     | `(1.5999674, 0.6499886, 0.1000166)` (max err `3.3e-5`)              |
| `FromBatchNorm` scales `a`                                       | `(0.7499063, 1.5968096, 19.069252)`                                 |
| `FromBatchNorm` shifts `b`                                       | `(-0.2749531, 1.3968096, -57.207756)`                               |
| BN output at `x = (1, -0.5, 3.2)` (direct and folded agree)      | `(0.4749531, 0.5984048, 3.8138504)`                                 |
| `FromStandardisation` `a`                                        | `(2.0, 0.028571429, 50.0)`                                          |
| `FromStandardisation` `b`                                        | `(-19.62, 0.034285717, -165.0)`                                     |
| Standardised `mean` / `mean + sd` / `mean − 2 sd`                | `0` / `1` / `-2` per feature (float32 max err `1.2e-7`)             |
| Model `Affine → Dense(tanh)` pre-activation and output           | `-0.06` → `-0.0599281`; folded Dense `W' = (0.8, 0.15, 0.05)`, `c' = -0.25` gives the same value |
| N8 parameter gradients (not asserted here)                       | `∂L/∂a = g ⊙ x = (0.24, 0.91, 0.44)`, `∂L/∂b = g = (0.8, -1.3, 0.4)`; FD `(0.2400279, 0.9099841, 0.4400015)` / `(0.8000731, -1.2999772, 0.4000067)` |

## Edge cases

- Zero scale `a_i = 0`: the output is the constant `b_i` and the gradient is `0`. This is covered by
  `SetParametersUsesSameLayoutForForward` (the forward part). The backward part is `a_i · g_i = 0` by construction.
- Negative scale (a negative `γ`): it flips the sign of the feature and of its gradient. This is covered
  by `a₁ = -0.5` in the fixture.
- Large offset `|μ/s| ≈ 2·10³` (pressure `μ = 101325`, `s = 50`): about `1e-4` absolute cancellation
  error. This is documented in `implementation.md` and not unit-tested, because it is a property of
  float32 and not of this layer.
- `variance + epsilon ≤ 0` or `standardDeviation ≤ 0`: `really_assert` fires. This is not unit-tested,
  consistent with the other layers' size and precondition asserts.
- `Backward` before any `Forward`: `really_assert(forwardDone)` fires. This is not unit-tested, the same
  as `Dense`.
- Parameter-gradient finite-difference checks (`∂L/∂a`, `∂L/∂b`) belong to N8's test plan, once the
  gradients are exposed. The reference values are listed above.
