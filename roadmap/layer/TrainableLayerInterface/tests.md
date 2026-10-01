# Trainable-Layer Interface — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
class TestTrainableLayerInterface : public ::testing::Test:
    using DenseLayer      = neural_network::Dense<float, 3, 2>
    using InputVector     = DenseLayer::InputVector
    using OutputVector    = DenseLayer::OutputVector
    using ParameterVector = DenseLayer::ParameterVector

    using Vector3         = math::Vector<float, 3>

    # the mock and the two concepts are declared in the anonymous namespace, next to the fixture
    class ZeroParameterLayerMock : public neural_network::Layer<float, 3, 3, 0>:
        MOCK_METHOD(void, Forward, (const Vector3& input), (override))
        MOCK_METHOD(const Vector3&, Output, (), (const, override))
        MOCK_METHOD(const Vector3&, Backward, (const Vector3& outputGradient), (override))

    template<typename L> concept HasBackward       = requires(L& l, const typename L::OutputVector& g) { l.Backward(g); }
    template<typename L> concept HasParameterApi   = requires(L& l) { l.Parameters(); l.ParameterGradients(); l.ZeroGradients(); }
    # concepts, because a requires-expression with an invalid requirement outside a template is ill-formed

    neural_network::Tanh<float> tanhActivation
    const DenseLayer::WeightMatrix weights{ {0.1, -0.2, 0.3}, {0.4, 0.5, -0.6} }
    const ParameterVector theta{ 0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.2, -0.1 }   # W row-major, b = (0.2, -0.1)
    const InputVector  x1{ 0.3, -0.7, 1.1 };   const OutputVector g1{ 0.8, -1.3 }
    const InputVector  x2{ 1.0,  2.0, 3.0 };   const OutputVector g2{ -0.5, 0.25 }
    DenseLayer layer{ weights, tanhActivation }                                 # SetParameters(theta) in the cases that use b

    float ProjectedOutput(DenseLayer& l, const ParameterVector& p, const InputVector& x, const OutputVector& g):
        l.SetParameters(p);  l.Forward(x)
        return g[0]·l.Output()[0] + g[1]·l.Output()[1]

    ParameterVector NumericParameterGradient(const InputVector& x, const OutputVector& g):
        DenseLayer probe{ weights, tanhActivation }                              # separate instance: FD never touches `layer`
        for k in 0..7:
            numeric[k] = (ProjectedOutput(probe, theta + h e_k, x, g) − ProjectedOutput(probe, theta − h e_k, x, g)) / (2h)
        return numeric
# each case below is a TEST_F(TestTrainableLayerInterface, <name>)
# the only collaborator is the ActivationFunction; Tanh is concrete and already tested ⇒ no activation mock
# ZeroParameterLayerMock is used as StrictMock<ZeroParameterLayerMock> only
# finite difference: central, h = 1e-3 (float), tolerance math::Tolerance<float>() = 1e-3
```

## Test cases (Arrange / Act / Assert)

```
ParameterGradientsStartAtZeroAndForwardLeavesThemUntouched:
    Arrange: layer.SetParameters(theta)
    Act:     layer.Forward(x1)
    Assert:  ParameterGradients()[k] == 0 for k in 0..7     # EXPECT_FLOAT_EQ; value-initialised buffer

BackwardParameterGradientsMatchAnalyticAndFiniteDifference:
    Arrange: layer.SetParameters(theta);  numeric = NumericParameterGradient(x1, g1)
    Act:     layer.Forward(x1);  layer.Backward(g1)
    Assert:  ParameterGradients() ≈ (0.1523375, -0.3554542, 0.5585709, -0.1662969, 0.3880261, -0.6097553,
                                     0.5077917, -0.554323)   # = [δ xᵀ ; δ], δ = (1 − a²) ⊙ g
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..7

BackwardAccumulatesParameterGradientsAcrossSamples:
    Arrange: layer.SetParameters(theta)
    Act:     Forward(x1); Backward(g1); Forward(x2); Backward(g2)
    Assert:  ParameterGradients() ≈ (-0.12719, -0.9145093, -0.2800118, 0.030315, 0.7812499, -0.01991957,
                                     0.2282642, -0.3577111)  # = grad(x1,g1) + grad(x2,g2)
             # overwriting (pre-N8 Dense) would leave grad(x2,g2) = (-0.2795275, -0.5590551, …) instead

BackwardReturnsPerSampleInputGradient:
    Arrange: layer.SetParameters(theta);  Forward(x1); Backward(g1)
             numeric[j] = (P(x2 + h e_j) − P(x2 − h e_j)) / (2h),  P(x) = Σ_i g2_i · a_i(x)
    Act:     layer.Forward(x2);  analytic = layer.Backward(g2)
    Assert:  analytic ≈ (0.05069201, 0.1542115, -0.2018254)       # = Wᵀ δ₂ only, not added to Wᵀ δ₁ = (-0.17095, -0.3787199, 0.4849313)
             EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>())

ZeroGradientsClearsAccumulationAndRestartsFromZero:
    Arrange: layer.SetParameters(theta);  Forward(x1); Backward(g1); Forward(x2); Backward(g2)
    Act:     layer.ZeroGradients()
    Assert:  ParameterGradients()[k] == 0 for k in 0..7                                     # EXPECT_FLOAT_EQ
    Act:     Forward(x2); Backward(g2)
    Assert:  ParameterGradients() ≈ (-0.2795275, -0.5590551, -0.8385826, 0.1966119, 0.3932238, 0.5898358,
                                     -0.2795275, 0.1966119)  # single-sample gradient of (x2, g2)

SetParametersKeepsAccumulatedGradients:
    Arrange: layer.SetParameters(theta);  Forward(x1); Backward(g1)
    Act:     layer.SetParameters({1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.5, -1.0})
    Assert:  ParameterGradients() ≈ grad(x1, g1) values from BackwardParameterGradientsMatchAnalyticAndFiniteDifference
             Parameters()         ≈ {1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.5, -1.0}

DenseIsUsableThroughInferenceLayerBase:
    Arrange: static_assert(std::is_base_of_v<InferenceLayer<float, 3, 2>, DenseLayer>)
             static_assert(!HasBackward<InferenceLayer<float, 3, 2>> && HasBackward<DenseLayer>)
             InferenceLayer<float, 3, 2>& base{ layer }                                   # default θ: biases 0
    Act:     base.Forward(x2)
    Assert:  base.Output() ≈ (0.5370496, -0.3799491)          # tanh(0.6), tanh(-0.4)

ZeroParameterSpecialisationHasNoParameterApiAndDispatchesBackward:
    Arrange: static_assert(Layer<float, 3, 3, 0>::ParameterSize == 0)
             static_assert(std::is_base_of_v<InferenceLayer<float, 3, 3>, Layer<float, 3, 3, 0>>)
             static_assert(!HasParameterApi<Layer<float, 3, 3, 0>> && HasParameterApi<DenseLayer>)
             static_assert(sizeof(Layer<float, 3, 3, 0>) == sizeof(void*))               # vptr only, no parameter storage
             StrictMock<ZeroParameterLayerMock> mock;  Layer<float, 3, 3, 0>& base{ mock }
             const Vector3 downstream{ 0.5, -1.0, 2.0 };  const Vector3 upstream{ 1.0, 0.0, -1.0 }
             EXPECT_CALL(mock, Backward(Ref(upstream))).WillOnce(ReturnRef(downstream))
    Act:     const auto& result = base.Backward(upstream)
    Assert:  &result == &downstream
```

Integration cases, deployed as `TEST_F(TestModel, …)` in `neural_network/model/test/TestModel.cpp`.
They reuse the existing `TestModel` fixture: `Dense<2,3>` (LeakyReLU 0.1, `W₁ = {{0.5, -1}, {1.5, 0.25}, {-0.5, 0.75}}`)
→ `Dense<3,1>` (tanh, `W₂ = {{1, -0.5, 2}}`), zero biases, `input = (2, 0.5)`, output `-0.8298019`.

```
GradientsConcatenateLayerGradientsAndMatchFiniteDifference:
    Arrange: θ = model.GetParameters()
             numeric[k] = (Out(θ + h e_k) − Out(θ − h e_k)) / (2h),  Out(θ) = SetParameters(θ); Forward(input)[0]
             model.SetParameters(θ)
    Act:     model.Forward(input);  model.Backward(OutputVector{ 1.0 });  G = model.Gradients()
    Assert:  G ≈ (0.6228576, 0.1557144, -0.3114288, -0.0778572, 0.1245715, 0.03114288,     # ∂/∂W₁ (row-major)
                  0.3114288, -0.1557144, 0.06228576,                                        # ∂/∂b₁
                  0.1557144, 0.973215, -0.0194643,                                          # ∂/∂W₂
                  0.3114288)                                                                # ∂/∂b₂
             EXPECT_NEAR(G[k], numeric[k], math::Tolerance<float>())   for k in 0..12

ZeroGradientsClearsEveryLayer:
    Arrange: model.Forward(input);  model.Backward(OutputVector{ 1.0 })
    Act:     model.ZeroGradients()
    Assert:  model.Gradients()[k] == 0 for k in 0..12                                        # EXPECT_FLOAT_EQ
    Act:     model.Forward(input);  model.Backward(OutputVector{ 1.0 })
    Assert:  model.Gradients()[0] ≈ 0.6228576,  model.Gradients()[10] ≈ 0.973215              # not doubled (1.2457152, 1.94643)
```

Conditional case, only if the optional compile-time activation from `implementation.md` is deployed
(it goes in `TestDense.cpp`):

```
ConcreteActivationTemplateMatchesInterfaceActivation:
    Arrange: Dense<float, 3, 2, Tanh<float>> byValue{ weights, Tanh<float>{} };  byValue.SetParameters(theta)
             layer.SetParameters(theta)
    Act:     both: Forward(x1); Backward(g1)
    Assert:  byValue.Output() ≈ (0.6043678, -0.7573623) ≈ layer.Output()
             byValue.ParameterGradients()[k] ≈ layer.ParameterGradients()[k]   for k in 0..7
```

## Reference vectors

Computed by [`reference.py`](reference.py), with float32 rounding at every
operation and the same central-difference step (`h = 1e-3`) as the tests. The closed forms were
cross-checked in float64.

| Quantity                                                               | Value                                                                                           |
|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Sample 1: `z₁`, `a₁ = tanh(z₁)`                                        | `(0.7, -0.99)`, `(0.6043678, -0.7573623)`                                                       |
| Sample 1: `δ₁ = (1 − a₁²) ⊙ g₁`                                        | `(0.5077917, -0.554323)`                                                                        |
| Sample 1: `∇θ` analytic                                                | `(0.1523375, -0.3554542, 0.5585709, -0.1662969, 0.3880261, -0.6097553, 0.5077917, -0.554323)`    |
| Sample 1: `∇θ` finite difference                                       | `(0.1523495, -0.3554225, 0.5585551, -0.1662969, 0.3880262, -0.6097555, 0.507772, -0.5543232)`, max err `3.2e-5` |
| Sample 1: `∂L/∂x = Wᵀδ₁` (analytic / FD)                               | `(-0.17095, -0.3787199, 0.4849313)` / max err `7.2e-5`                                          |
| Sample 2: `z₂`, `a₂`                                                   | `(0.8, -0.5)`, `(0.6640368, -0.4621173)`                                                        |
| Sample 2: `δ₂`                                                         | `(-0.2795275, 0.1966119)`                                                                       |
| Sample 2: `∇θ` analytic                                                | `(-0.2795275, -0.5590551, -0.8385826, 0.1966119, 0.3932238, 0.5898358, -0.2795275, 0.1966119)`   |
| Sample 2: `∇θ` finite difference                                       | `(-0.2795309, -0.5590618, -0.8385777, 0.1966059, 0.3932267, 0.5898178, -0.2795309, 0.1966059)`, max err `1.8e-5` |
| Sample 2: `∂L/∂x = Wᵀδ₂` (analytic / FD)                               | `(0.05069201, 0.1542115, -0.2018254)` / `(0.05069375, 0.1542121, -0.2018213)`                   |
| Accumulated `G = ∇θ₁ + ∇θ₂`                                            | `(-0.12719, -0.9145093, -0.2800118, 0.030315, 0.7812499, -0.01991957, 0.2282642, -0.3577111)` (float32 sum exact to the last bit) |
| Default `θ` (zero biases), `x₂`: `z`, `tanh(z)`                        | `(0.6, -0.4)`, `(0.5370496, -0.3799491)`                                                        |
| Model: hidden `z`, LeakyReLU `a`, output `z`, `y`                      | `(0.5, 3.125, -0.625)`, `(0.5, 3.125, -0.0625)`, `-1.1875`, `-0.8298019`                        |
| Model: `δ_out = 1 − y²`, `W₂ᵀδ_out`, `δ_hidden`                        | `0.3114288`, `(0.3114288, -0.1557144, 0.6228576)`, `(0.3114288, -0.1557144, 0.06228576)`        |
| Model `Gradients()` analytic                                           | `(0.6228576, 0.1557144, -0.3114288, -0.0778572, 0.1245715, 0.03114288, 0.3114288, -0.1557144, 0.06228576, 0.1557144, 0.973215, -0.0194643, 0.3114288)` |
| Model `Gradients()` finite difference                                  | `(0.6228387, 0.1556873, -0.3114343, -0.07784367, 0.1245439, 0.03117323, 0.3114045, -0.1556873, 0.06228685, 0.1556873, 0.9731948, -0.01943111, 0.3114045)`, max err `3.3e-5` |
| Model input gradient (already asserted by the existing FD case)        | `(-0.1090001, -0.303643)`                                                                       |

## Edge cases

- Kinks: every LeakyReLU pre-activation in the model fixture (`0.5, 3.125, -0.625`) is at least `0.5`
  from `0`, so a `±1e-3` perturbation never crosses a kink and the finite differences stay valid.
- `Backward` twice after one `Forward` adds the same sample twice. This is the documented contract
  and is not unit-tested.
- `Backward` before any `Forward`: `really_assert(forwardDone)` fires. It is not unit-tested, the same
  as for `Dense` today.
- A model made only of zero-parameter layers is rejected by `static_assert(TotalParameters > 0)` at
  compile time. It is not a runtime test.
- A zero-parameter layer inside a `Model` (the `if constexpr` skip in `Get/SetParameters` and
  `Gradients`) needs a concrete, movable parameter-free layer. Mocks cannot be stored in `Model`'s
  tuple. That case is asserted by N12's first pooling layer (`TestModel`), not here.
- A NaN upstream gradient poisons `G` until `ZeroGradients()`. This is outside the `-ffast-math`
  contract, and N16 guards it before stepping.
