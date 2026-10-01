# 1D Pooling (Max / Average / Global-Average) — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```text
class TestPooling1D : public ::testing::Test:
    using MaxNonOverlapping = neural_network::MaxPooling1D<float, 6, 2, 2>          # K = S = 2 ⇒ Lₒ = 3
    using MaxOverlapping    = neural_network::MaxPooling1D<float, 6, 2, 3, 2>       # K = 3, S = 2 ⇒ Lₒ = 2, t = 5 dropped
    using Average           = neural_network::AveragePooling1D<float, 6, 2, 3, 2>
    using GlobalAverage     = neural_network::GlobalAveragePooling1D<float, 6, 2>
    using SequenceVector    = MaxOverlapping::InputVector                          # 12 floats, x[t·2 + c]

    # channel 0 over t = 0..5: ( 0.5, -1.2, 2.0,  0.3, -0.4, 1.7)
    # channel 1 over t = 0..5: (-0.8,  0.9, 0.1, -2.1,  1.4, 0.6)
    # every window maximum beats its runner-up by ≥ 0.2 ≫ h, so the FD checks never cross a tie
    const SequenceVector input{ 0.5, -0.8, -1.2, 0.9, 2.0, 0.1, 0.3, -2.1, -0.4, 1.4, 1.7, 0.6 }
    const math::Vector<float, 4> upstream{ 0.8, -1.3, 0.4, 0.25 }                   # g(o, c) for Lₒ = 2, C = 2

    template<typename PoolingLayer>
    float ProjectedOutput(PoolingLayer& layer, const SequenceVector& x, const typename PoolingLayer::OutputVector& g):
        layer.Forward(x)
        return Σ_j g[j] · layer.Output()[j]

    template<typename PoolingLayer>
    SequenceVector NumericInputGradient(PoolingLayer& layer, const typename PoolingLayer::OutputVector& g):
        numeric[j] = (ProjectedOutput(layer, input + h eⱼ, g) − ProjectedOutput(layer, input − h eⱼ, g)) / (2h)
# each case below is a TEST_F(TestPooling1D, <name>)
# no collaborators (pooling calls nothing) ⇒ no mocks, the same as TestDense
# finite difference: central, h = 1e-3 (float), tolerance math::Tolerance<float>() = 1e-3
```

## Test cases (Arrange / Act / Assert)

```text
OutputLengthFollowsValidPaddingFormula:
    Assert:  MaxNonOverlapping::OutputSize == 6                   # ⌊(6−2)/2⌋+1 = 3 positions × 2 channels
             MaxOverlapping::OutputSize    == 4                   # ⌊(6−3)/2⌋+1 = 2 × 2
             MaxPooling1D<float, 7, 1, 3, 2>::OutputSize == 3     # windows end exactly at t = 6
             MaxPooling1D<float, 5, 1, 5, 1>::OutputSize == 1     # K = L
             GlobalAverage::OutputSize == 2, GlobalAverage::ParameterSize == 0
             # EXPECT_EQ on the static constexpr members

MaxPoolingForwardTakesWindowMaximumPerChannel:
    Arrange: MaxNonOverlapping layer{}
    Act:     layer.Forward(input)
    Assert:  Output() ≈ {0.5, 0.9, 2.0, 0.1, 1.7, 1.4}            # EXPECT_FLOAT_EQ: max copies values exactly

MaxPoolingBackwardRoutesToArgmaxAccumulatesOverlapAndMatchesFiniteDifference:
    Arrange: MaxOverlapping layer{};  numeric = NumericInputGradient(layer, upstream)
    Act:     layer.Forward(input);  first = copy of layer.Backward(upstream);  second = layer.Backward(upstream)
    Assert:  Output() ≈ {2.0, 0.9, 2.0, 1.4}                      # x(2,0) = 2.0 wins both channel-0 windows
             first ≈ {0, 0, 0, -1.3, 1.2, 0, 0, 0, 0, 0.25, 0, 0}
                                                                  # x(2,0): 0.8 + 0.4; x(5,0) = 1.7 is dropped ⇒ 0
             EXPECT_NEAR(first[j], numeric[j], math::Tolerance<float>())
             second == first                                      # the buffer is zeroed per call, not accumulated

MaxPoolingTieRoutesGradientToFirstMaximum:
    Arrange: MaxPooling1D<float, 4, 1, 2> layer{};  x = {0.7, 0.7, -0.3, -0.3}
    Act:     layer.Forward(x);  gradient = layer.Backward({1.0, 2.0})
    Assert:  Output() ≈ {0.7, -0.3}
             gradient ≈ {1.0, 0.0, 2.0, 0.0}                      # subgradient convention; no FD at a tie

AveragePoolingForwardTakesWindowMeanPerChannel:
    Arrange: Average layer{}
    Act:     layer.Forward(input)
    Assert:  Output() ≈ {0.43333334, 0.06666666, 0.6333333, -0.2}

AveragePoolingBackwardSpreadsGradientOverWindowAndMatchesFiniteDifference:
    Arrange: Average layer{};  numeric = NumericInputGradient(layer, upstream)
    Act:     layer.Forward(input);  analytic = layer.Backward(upstream)
    Assert:  analytic ≈ {0.26666668, -0.43333334, 0.26666668, -0.43333334, 0.4, -0.35,
                         0.13333334,  0.08333334, 0.13333334,  0.08333334, 0.0,  0.0}
                                                                  # t = 2 in both windows: (g₀ + g₁)/3; t = 5 dropped
             EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>())

GlobalAveragePoolingReducesEachChannelToItsMeanAndMatchesFiniteDifference:
    Arrange: GlobalAverage layer{};  g = {0.8, -1.3};  numeric = NumericInputGradient(layer, g)
    Act:     layer.Forward(input);  analytic = layer.Backward(g)
    Assert:  Output()  ≈ {0.48333335, 0.01666667}                 # 2.9 / 6, 0.1 / 6
             analytic[t·2 + 0] ≈ 0.13333334, analytic[t·2 + 1] ≈ -0.21666667 for t in 0..5    # g / L
             EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>())
```

Integration case, deployed as `TEST_F(TestModel, PoolingLayersComposeWithDenseAndBackpropagate)` in
`neural_network/model/test/TestModel.cpp`. It exercises `Model` with parameter-free layers: the
`TotalParameters` count, the `SetParameters` offsets that skip them, and the backward chain through them:

```text
PoolingLayersComposeWithDenseAndBackpropagate:
    Arrange: Model<float, 12, 1,
                   MaxPooling1D<float, 6, 2, 2>,                  # 12 → 6
                   GlobalAveragePooling1D<float, 3, 2>,           #  6 → 2
                   Dense<float, 2, 1>> model{                     #  2 → 1
                 make_layer<MaxPooling1D<float, 6, 2, 2>>(),
                 make_layer<GlobalAveragePooling1D<float, 3, 2>>(),
                 make_layer<Dense<float, 2, 1>>(W, tanh) }        # W = {{0.5, -0.25}}
             model.SetParameters({0.5, -0.25, 0.1})               # TotalParameters == 3 (Dense only)
             numeric[j] = (model.Forward(input + h eⱼ)[0] − model.Forward(input − h eⱼ)[0]) / (2h)
    Act:     y = model.Forward(input);  dx = model.Backward({1.0})
    Assert:  y[0] ≈ 0.5370496                                     # max → (0.5, 0.9, 2.0, 0.1, 1.7, 1.4),
                                                                  # GAP → (1.4, 0.8), z = 0.6, tanh(0.6)
             dx ≈ {0.1185963, 0, 0, -0.05929815, 0.1185963, -0.05929815,
                   0, 0, 0, -0.05929815, 0.1185963, 0}
             EXPECT_NEAR(dx[j], numeric[j], math::Tolerance<float>())
```

## Reference vectors

Computed by [`reference.py`](reference.py). It uses float32 rounding at every operation
and the same central-difference step as the tests:

| Quantity                                                       | Value                                                                                                                         |
|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Max `K = S = 2` forward                                        | `(0.5, 0.9, 2.0, 0.1, 1.7, 1.4)`                                                                                              |
| Max `K = 3, S = 2` forward / argmax flat indices               | `(2.0, 0.9, 2.0, 1.4)` / `(4, 3, 4, 9)`                                                                                       |
| Max `K = 3, S = 2` backward, `g = (0.8, -1.3, 0.4, 0.25)`      | `(0, 0, 0, -1.3, 1.2, 0, 0, 0, 0, 0.25, 0, 0)`                                                                                |
| … finite difference                                            | `(0, 0, 0, -1.2999772, 1.1999011, 0, 0, 0, 0, 0.24998187, 0, 0)` (max err `9.9e-5`)                                           |
| Tie `x = (0.7, 0.7, -0.3, -0.3)`, `K = 2`, `g = (1, 2)`        | forward `(0.7, -0.3)`, backward `(1, 0, 2, 0)`                                                                                |
| Average `K = 3, S = 2` forward                                 | `(0.43333334, 0.06666666, 0.6333333, -0.19999997)`                                                                            |
| Average `K = 3, S = 2` backward                                | `(0.26666668, -0.43333334, 0.26666668, -0.43333334, 0.40000004, -0.35, 0.13333334, 0.08333334, 0.13333334, 0.08333334, 0, 0)` |
| … finite difference                                            | max err `1.3e-5`                                                                                                              |
| Global average forward / backward (`g = (0.8, -1.3)`)          | `(0.48333335, 0.01666667)` / `(0.13333334, -0.21666667)` per position                                                         |
| … finite difference                                            | max err `1.9e-5`                                                                                                              |
| Model `Max → GAP → Dense(tanh)`, `W = (0.5, -0.25)`, `b = 0.1` | pooled `(1.4, 0.8)`, `z = 0.6`, `y = 0.5370496`, `1 − y² = 0.7115778`                                                         |
| Model `Backward(1)`                                            | `(0.1185963, 0, 0, -0.05929815, 0.1185963, -0.05929815, 0, 0, 0, -0.05929815, 0.1185963, 0)`; FD max err `1.7e-5`             |

## Edge cases

- Tie inside a max window: the gradient goes to the first maximum (`MaxPoolingTieRoutesGradientToFirstMaximum`).
  The finite difference is not defined there, so that case asserts the convention, not an FD match.
- Overlapping windows (`S < K`): one input receives the sum of several window gradients. Covered for
  max (`x(2, 0)`) and average (`t = 2`).
- Dropped tail (`(L − K) mod S ≠ 0`): the uncovered positions get a zero gradient. Covered by `t = 5`
  in the `K = 3, S = 2` cases.
- Repeated `Backward` after one `Forward` returns the same vector (zeroed buffer). Covered in the max
  overlap case; the average `Backward` shares the same zero-then-scatter structure.
- `K = 1, S = 1` is the identity and `K = L` is global pooling. Not separately tested beyond
  `OutputLengthFollowsValidPaddingFormula`, because they follow from the same loops.
- `PoolSize > Length`, `PoolSize = 0` or `Stride = 0`: rejected by `static_assert` at compile time.
  Not unit-tested (a compile failure).
- `Backward` before any `Forward`: `really_assert(forwardDone)` fires. Not unit-tested, the same as `Dense`.
- NaN/±∞ inputs: outside the `-ffast-math` contract; not tested.
