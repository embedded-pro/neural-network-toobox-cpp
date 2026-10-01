# 2D Convolution + 2D Pooling — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixtures

```
class TestConvolution2D : public ::testing::Test:
    using Convolution     = neural_network::Convolution2D<float, 5, 4, 2, 2, 2, 3, 2, 1>
                            # H=5, W=4, Cin=2, Cout=2, KH=2, KW=3, SH=2, SW=1 ⇒ Hout=2 (row 4 dropped), Wout=2 (columns overlap), P=26
    using InputVector     = Convolution::InputVector          # 40 = 5·4·2, x[(i·4 + j)·2 + c]
    using OutputVector    = Convolution::OutputVector         # 8  = 2·2·2, y[(r·2 + s)·2 + o]
    using ParameterVector = Convolution::ParameterVector      # 26

    class ActivationFunctionMock : public neural_network::ActivationFunction<float>:   # anonymous namespace
        MOCK_METHOD(float, Forward, (float x), (const, override))
        MOCK_METHOD(float, Backward, (float x), (const, override))
        MOCK_METHOD(void, ForwardVector, (std::span<float>, std::span<const float>), (const, override))
        MOCK_METHOD(void, BackwardVector, (std::span<float>, std::span<const float>, std::span<const float>, std::span<const float>), (const, override))

    neural_network::LeakyReLU<float> leakyRelu{ 0.1 }
    neural_network::Tanh<float>      tanhActivation
    const Convolution::KernelMatrix kernels{ { 0.2, -0.1, 0.4, 0.3, -0.5, 0.1,   0.3, 0.2, -0.2, 0.1, 0.1, -0.4 },   # F_0: p = 0 row, p = 1 row; (q, c) inside
                                             {-0.3, 0.2, 0.1, -0.4, 0.6, 0.05,  -0.1, 0.3, 0.2, -0.2, 0.4, 0.1  } }  # F_1
    const ParameterVector theta{ kernels row 0, kernels row 1, 0.1, -0.2 }     # 24 filter values, then b = (0.1, -0.2)
    const InputVector x{  0.5, -1.0,   0.25,  0.75,  -0.5,  1.5,    1.0, -0.25,      # row i = 0, columns j = 0..3, (c0, c1)
                          0.0,  2.0,  -1.5,   0.5,    0.75, -0.5,   1.25, 0.25,      # row 1
                         -0.25, 1.0,   2.0,  -0.75,   0.5,  0.5,   -1.0,  1.5,       # row 2
                          1.5, -0.5,  -0.75,  0.25,   0.25, -1.25, -0.25, 1.0,       # row 3
                         -2.0,  0.75,  1.0,  -1.5,    0.0,  0.5,    0.75, -1.0 }     # row 4 (never read: dropped by the floor)
    const OutputVector g { 0.8, -1.3, 0.4, 0.6, -0.5, 0.25, 1.0, -0.75 }
    InputVector  x2:  x2[k] = 0.1f · float((7k mod 11) − 5),  k in 0..39            # second sample, filled in the fixture ctor
    const OutputVector g2{ 0.3, 0.9, -1.1, 0.2, 0.7, -0.4, -0.6, 1.2 }

    float ProjectedOutput(Convolution& layer, const ParameterVector& p, const InputVector& in, const OutputVector& up):
        layer.SetParameters(p);  layer.Forward(in)
        return Σ_k up[k] · layer.Output()[k]

    ParameterVector NumericParameterGradient(const InputVector& in, const OutputVector& up):
        Convolution probe{ kernels, tanhActivation }                         # separate instance: FD never touches G
        numeric[k] = (ProjectedOutput(probe, theta + h e_k, in, up) − ProjectedOutput(probe, theta − h e_k, in, up)) / (2h),  k in 0..25
# each case below is a TEST_F(TestConvolution2D, <name>)
# the concrete activations are already tested (as in TestDense); the mock is only used as StrictMock<ActivationFunctionMock>
# finite difference: central, h = 1e-3 (float), tolerance math::Tolerance<float>() = 1e-3

class TestPooling2D : public ::testing::Test:
    using MaxNonOverlapping = neural_network::MaxPooling2D<float, 5, 4, 2, 2, 2>            # 2×2, default stride 2×2 ⇒ 2×2 out, row 4 dropped
    using MaxOverlapping    = neural_network::MaxPooling2D<float, 5, 4, 2, 2, 3, 2, 1>      # 2×3, stride 2×1 ⇒ 2×2 out, columns overlap, row 4 dropped
    using Average           = neural_network::AveragePooling2D<float, 5, 4, 2, 2, 3, 2, 1>
    using GlobalAverage     = neural_network::GlobalAveragePooling2D<float, 5, 4, 2>
    using MapVector         = MaxOverlapping::InputVector                                   # 40 floats, x[(i·4 + j)·2 + c]
    const MapVector input{ same 40 values as TestConvolution2D::x }
        # every window maximum beats its runner-up by ≥ 0.25 ≫ h, so the FD checks never cross a tie
    const math::Vector<float, 8> upstream{ 0.8, -1.3, 0.4, 0.25, -0.6, 1.1, 0.5, -0.9 }    # g(r, s, c)

    template<typename PoolingLayer>
    MapVector NumericInputGradient(PoolingLayer& layer, const typename PoolingLayer::OutputVector& g):
        numeric[k] = (Σ g ⊙ layer(input + h e_k) − Σ g ⊙ layer(input − h e_k)) / (2h),  k in 0..39
# each case below is a TEST_F(TestPooling2D, <name>)
# no collaborators (pooling calls nothing) ⇒ no mocks
```

## Test cases (Arrange / Act / Assert)

`TestConvolution2D` (`neural_network/layer/test/TestConvolution2D.cpp`):

```
SizesFollowValidStridedWindowCount:
    Assert: Convolution::InputSize == 40, OutputSize == 8, ParameterSize == 26
            Convolution::OutputHeight == 2, Convolution::OutputWidth == 2
            Convolution2D<float, 5, 4, 2, 2, 2, 3, 1, 2>::OutputHeight == 4, ::OutputWidth == 1   # strides swapped
            Convolution2D<float, 28, 28, 1, 8, 3, 3, 2, 2>::OutputHeight == 13, ::OutputWidth == 13
            Convolution2D<float, 4, 3, 1, 1, 4, 3>::OutputSize == 1                               # kernel = whole map
            Convolution2D<float, 4, 3, 1, 1, 1, 1>::OutputSize == 12                              # pointwise
            # EXPECT_EQ on the static constexpr members

ConstructorStoresFilterMajorKernelsAndZeroBiases:
    Arrange: const Convolution layer{ kernels, leakyRelu }
    Assert:  Parameters() ≈ {row 0 of kernels, row 1 of kernels, 0, 0}

ForwardComputesStridedChannelsLastCrossCorrelation:
    Arrange: Convolution layer{ kernels, leakyRelu };  layer.SetParameters(theta)
    Act:     layer.Forward(x)
    Assert:  Output()     ≈ {2.05, -0.06, -0.0725, 0.8875, 1.375, 0.375, 0.8, -0.1175}
                            # z = (2.05, -0.6, -0.725, 0.8875, 1.375, 0.375, 0.8, -1.175); LeakyReLU on z < 0
             Parameters() ≈ theta
             # a flipped kernel, a channels-first kernel or swapped strides give different values (Reference vectors)

HeightOneMatchesConvolution1DLayout:
    Arrange: Convolution2D<float, 1, 6, 2, 2, 1, 3, 1, 2> layer{ N11 fixture kernels, leakyRelu }
             layer.SetParameters(N11 fixture theta)        # {0.2, -0.1, 0.4, 0.3, -0.5, 0.1, -0.3, 0.2, 0.1, -0.4, 0.6, 0.05, 0.1, -0.2}
    Act:     layer.Forward(N11 fixture x)                  # {0.5, -1.0, 0.25, 0.75, -0.5, 1.5, 1.0, -0.25, 0.0, 2.0, -1.5, 0.5}
    Assert:  Output() ≈ {1.025, -0.105, 0.375, 0.55}       # = Convolution1D<float, 6, 2, 2, 3, 2> output (N11 tests.md)
             # pins the shared θ / channels-last contract so 1D weights and exporters carry over

ActivationReceivesWholePreActivationMapOncePerPass:
    Arrange: StrictMock<ActivationFunctionMock> activation;  Convolution layer{ kernels, activation };  layer.SetParameters(theta)
             EXPECT_CALL(activation, ForwardVector(_, _)).WillOnce: assert in.size() == 8, in ≈ z, copy in → out
             EXPECT_CALL(activation, BackwardVector(_, _, _, _)).WillOnce: assert preActivation ≈ z, output ≈ z,
                                                                            outputGradient ≈ g (all size 8); result = outputGradient
    Act:     layer.Forward(x);  layer.Backward(g)
    Assert:  Output() ≈ z = {2.05, -0.6, -0.725, 0.8875, 1.375, 0.375, 0.8, -1.175}
             (StrictMock) no scalar Forward/Backward calls: one vector call per pass over the whole map

BackwardInputGradientMatchesAnalyticAndFiniteDifference:
    Arrange: Convolution layer{ kernels, tanhActivation };  layer.SetParameters(theta)
             numeric[k] = (Σ g ⊙ layer(x + h e_k) − Σ g ⊙ layer(x − h e_k)) / (2h),  k in 0..39
    Act:     layer.Forward(x);  analytic = layer.Backward(g)
    Assert:  analytic ≈ {0.2877789, -0.190142, -0.1119517, 0.4202729, -0.4524522, -0.08621092, 0.05532312, 0.03949448,
                         0.1079004, -0.2672518, -0.1511419, 0.3286123, -0.3546557, -0.1478893, 0.1435868, -0.06875598,
                         -0.087959, 0.05487425, 0.1599564, -0.2246405, 0.3869897, 0.2626985, -0.4226012, 0.04398272,
                         -0.05567525, 0.04277801, 0.2577317, -0.01460002, -0.08363949, 0.1705672, -0.03947689, -0.2474677,
                         0, 0, 0, 0, 0, 0, 0, 0}
             # columns 1–2 sum both overlapping windows; row 4 is never read ⇒ exactly 0 (EXPECT_FLOAT_EQ)
             EXPECT_NEAR(analytic[k], numeric[k], math::Tolerance<float>())

BackwardParameterGradientsMatchAnalyticAndFiniteDifference:
    Arrange: Convolution layer{ kernels, tanhActivation };  layer.SetParameters(theta)
             numeric = NumericParameterGradient(x, g)
    Act:     layer.Forward(x);  layer.Backward(g)
    Assert:  ParameterGradients() ≈ {1.233568, -0.3988803, -0.05666873, 0.7720925, -0.3949487, 0.7975226,
                                     -0.9580824, 0.421996, 0.3321829, -0.8245192, 0.1782883, 0.7361474,
                                     -0.9195597, 1.544843, -0.06340403, -0.5303278, 1.107336, -1.710665,
                                     0.05957477, -1.869961, 1.387596, -0.258685, -0.2079424, 0.02605237,
                                     0.7436619, -0.6482056}      # [Σ_{r,s} δ[r,s,o] x[r·SH+p, s·SW+q, c] ; Σ_{r,s} δ[r,s,o]]
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())

BackwardAccumulatesAcrossSamplesUntilZeroGradients:
    Arrange: Convolution layer{ kernels, tanhActivation };  layer.SetParameters(theta)
    Act:     layer.Forward(x);  layer.Backward(g);  layer.Forward(x2);  layer.Backward(g2)
    Assert:  ParameterGradients() ≈ {1.071563, -0.3113069, -0.3801179, 0.9937703, -0.2535605, 0.5278882,
                                     -1.173902, 1.475115, -0.0450809, -0.9865241, 0.2658617, 0.4126983,
                                     -1.323569, 1.380783, 0.01625719, -0.1126793, 0.7820522, -1.792227,
                                     -0.1832111, -2.048987, 1.628481, -0.6626945, -0.3720023, 0.1057136,
                                     0.2055147, 0.9640299}       # = G(x, g) + G(x2, g2)
             ParameterGradients()[k] ≈ NumericParameterGradient(x, g)[k] + NumericParameterGradient(x2, g2)[k]
    Act:     layer.ZeroGradients()
    Assert:  ParameterGradients()[k] == 0 for k in 0..25        # EXPECT_FLOAT_EQ
```

`TestPooling2D` (`neural_network/layer/test/TestPooling2D.cpp`):

```
OutputSizeFollowsValidPaddingFormula:
    Assert: MaxNonOverlapping::OutputHeight == 2, ::OutputWidth == 2, ::OutputSize == 8
            MaxOverlapping::OutputHeight == 2, ::OutputWidth == 2
            MaxPooling2D<float, 47, 8, 8, 2, 2>::OutputSize == 736          # 23 × 4 × 8
            GlobalAverage::OutputSize == 2, GlobalAverage::ParameterSize == 0
            # EXPECT_EQ on the static constexpr members

MaxPoolingForwardTakesWindowMaximumPerChannel:
    Arrange: MaxNonOverlapping layer{}
    Act:     layer.Forward(input)
    Assert:  Output() == {0.5, 2.0, 1.25, 1.5, 2.0, 1.0, 0.5, 1.5}          # EXPECT_FLOAT_EQ: max copies values exactly

MaxPoolingBackwardRoutesToArgmaxAccumulatesOverlapAndMatchesFiniteDifference:
    Arrange: MaxOverlapping layer{};  numeric = NumericInputGradient(layer, upstream)
    Act:     layer.Forward(input);  first = copy of layer.Backward(upstream);  second = layer.Backward(upstream)
    Assert:  Output() == {0.75, 2.0, 1.25, 1.5, 2.0, 1.0, 2.0, 1.5}
             first[5] ≈ 0.25, first[9] ≈ -1.3, first[12] ≈ 0.8, first[14] ≈ 0.4,
             first[17] ≈ 1.1, first[18] ≈ -0.1, first[23] ≈ -0.9, every other entry == 0
                    # x(2,1,0) = 2.0 wins windows (1,0) and (1,1): −0.6 + 0.5; row 4 (k ≥ 32) dropped ⇒ 0
             EXPECT_NEAR(first[k], numeric[k], math::Tolerance<float>())
             second == first                                                  # zeroed per call, not accumulated

MaxPoolingTieRoutesGradientToFirstMaximumInRowMajorOrder:
    Arrange: MaxPooling2D<float, 2, 4, 1, 2, 2> layer{};  x = {-0.3, 0.7, 0.2, 0.2,     # row 0
                                                              0.7, 0.1, -0.5, 0.2}     # row 1
    Act:     layer.Forward(x);  gradient = layer.Backward({1.0, 2.0})
    Assert:  Output() ≈ {0.7, 0.2}
             gradient ≈ {0, 1.0, 2.0, 0, 0, 0, 0, 0}
             # window 0 ties (0,1) and (1,0): row-major picks (0,1) (a column-major scan would pick (1,0));
             # window 1 ties three 0.2s: (0,2) wins. Subgradient convention; no FD at a tie.

AveragePoolingForwardTakesWindowMeanPerChannel:
    Arrange: Average layer{}
    Act:     layer.Forward(input)
    Assert:  Output() ≈ {-0.08333334, 0.5416667, 0.2083333, 0.375, 0.5416667, -0.125, 0.125, 0.2083333}

AveragePoolingBackwardSpreadsGradientOverWindowAndMatchesFiniteDifference:
    Arrange: Average layer{};  numeric = NumericInputGradient(layer, upstream)
    Act:     layer.Forward(input);  analytic = layer.Backward(upstream)
    Assert:  analytic ≈ {0.1333333, -0.2166667, 0.2, -0.175, 0.2, -0.175, 0.06666667, 0.04166667,        # row 0
                         0.1333333, -0.2166667, 0.2, -0.175, 0.2, -0.175, 0.06666667, 0.04166667,        # row 1
                         -0.1, 0.1833333, -0.01666667, 0.03333333, -0.01666667, 0.03333333, 0.08333334, -0.15,   # row 2
                         -0.1, 0.1833333, -0.01666667, 0.03333333, -0.01666667, 0.03333333, 0.08333334, -0.15,   # row 3
                         0, 0, 0, 0, 0, 0, 0, 0}                                                           # row 4 dropped
             # columns 1–2 are in both windows of a row: (g(r,0,c) + g(r,1,c)) / 6
             EXPECT_NEAR(analytic[k], numeric[k], math::Tolerance<float>())

GlobalAveragePoolingReducesEachChannelToItsMeanAndMatchesFiniteDifference:
    Arrange: GlobalAverage layer{};  gc = {0.8, -1.3};  numeric = NumericInputGradient(layer, gc)
    Act:     layer.Forward(input);  analytic = layer.Backward(gc)
    Assert:  Output() ≈ {0.175, 0.1875}                                        # 3.5 / 20, 3.75 / 20
             analytic[(i·4 + j)·2 + 0] ≈ 0.04, analytic[(i·4 + j)·2 + 1] ≈ -0.065 for every (i, j)   # g / (H·W)
             EXPECT_NEAR(analytic[k], numeric[k], math::Tolerance<float>())
```

Integration case, deployed as `TEST_F(TestModel, Convolution2DAndGlobalAveragePoolingFeedDenseAndBackpropagate)`
in `neural_network/model/test/TestModel.cpp` (the layer test target does not link `neural_network.model`).
It exercises `Model` with a 2D convolution, a parameter-free 2D pooling layer and the backward chain
through both:

```
Convolution2DAndGlobalAveragePoolingFeedDenseAndBackpropagate:
    Arrange: Model<float, 40, 1,
                   Convolution2D<float, 5, 4, 2, 2, 2, 3, 2, 1>,          # 40 → 8
                   GlobalAveragePooling2D<float, 2, 2, 2>,                #  8 → 2
                   Dense<float, 2, 1>> model{                             #  2 → 1
                 make_layer<Convolution2D<float, 5, 4, 2, 2, 2, 3, 2, 1>>(kernels, leakyRelu),
                 make_layer<GlobalAveragePooling2D<float, 2, 2, 2>>(),
                 make_layer<Dense<float, 2, 1>>(W, tanh) }                # W = {{0.5, -0.25}}
             model.SetParameters(theta ++ {0.5, -0.25, 0.1})             # TotalParameters == 26 + 0 + 3
             numeric[k] = (model.Forward(x + h e_k)[0] − model.Forward(x − h e_k)[0]) / (2h)
    Act:     y = model.Forward(x);  dx = model.Backward({1.0})
    Assert:  y[0] ≈ 0.5014566                   # conv → (2.05, -0.06, -0.0725, 0.8875, 1.375, 0.375, 0.8, -0.1175),
                                                # GAP → (1.038125, 0.27125), z = 0.55125, tanh(z)
             dx ≈ {0.02011705, -0.01029244, 0.05286574, 0.01964921, -0.05052654, 0.03064341, -0.03274868, -0.001403515,
                   0.02853814, 0.01731002, -0.0121638, -0.001871355, -0.003742707, -0.02760246, -0.01777786, -0.00842109,
                   0.03274868, -0.01871353, 0.05286573, 0.03649139, -0.0378949, 0.03695923, -0.04959086, 0.009122848,
                   0.03274868, 0.004678383, 0.0004678402, 0.03602355, -0.02900598, -0.03181301, 0.007485413, -0.0378949,
                   0, 0, 0, 0, 0, 0, 0, 0}
             EXPECT_NEAR(dx[k], numeric[k], math::Tolerance<float>())
             model.GetParameters() round-trips the 29 values in order
```

## Reference vectors

Computed by `scratchpad/specs/Convolution2D/reference.py` (output in `reference.out`) with float32
rounding at every operation and the same central-difference step as the tests (`h = 1e-3`):

| Quantity                                                               | Value                                                                  |
|------------------------------------------------------------------------|------------------------------------------------------------------------|
| `(Hout, Wout)` for `(H, W, KH, KW, SH, SW)` = (5,4,2,3,2,1) / (5,4,2,3,1,2) / (28,28,3,3,2,2) / (4,3,4,3,1,1) / (4,3,1,1,1,1) | `(2,2) / (4,1) / (13,13) / (1,1) / (4,3)` |
| Pre-activation `z`, fixture `θ`, `x`                                   | `(2.05, -0.6, -0.725, 0.8875, 1.375, 0.375, 0.8, -1.175)` (float64 agrees) |
| `z` with biases 0 (constructor state)                                  | `(1.95, -0.4, -0.825, 1.0875, 1.275, 0.575, 0.7, -0.975)`             |
| Forward, LeakyReLU(0.1)                                                | `(2.05, -0.06, -0.0725, 0.8875, 1.375, 0.375, 0.8, -0.1175)`          |
| Wrong-layout `z`: flipped kernel / channels-first kernel / swapped strides | `(0.675, -0.275, 1.5, -1.525, -1.4, 0.825, 0.525, 1.4375)` / `(0.975, -0.8, 1.225, -1.3875, -0.625, 1.9125, -0.05, -0.6875)` / `(2.05, -0.6, -1.475, 1.4, 1.375, 0.375, -1.025, 0.1375)` |
| Height-one conv on the N11 fixture, LeakyReLU                          | `z = (1.025, -1.05, 0.375, 0.55)`, `y = (1.025, -0.105, 0.375, 0.55)`  |
| Forward, Tanh                                                          | `(0.9673949, -0.5370496, -0.6199968, 0.7101567, 0.8798267, 0.3583574, 0.6640368, -0.8258684)` |
| `δ = (1 − y²) ⊙ g`                                                     | `(0.0513176, -0.9250511, 0.2462416, 0.2974065, -0.1129525, 0.217895, 0.5590552, -0.238456)` |
| `∂L/∂x` analytic                                                       | see `BackwardInputGradientMatchesAnalyticAndFiniteDifference`; finite difference max err `1.5e-4` |
| `∂L/∂θ` analytic                                                       | see `BackwardParameterGradientsMatchAnalyticAndFiniteDifference`; finite difference max err `1.6e-4` |
| `x2` / `z` for `x2`                                                    | `(-0.5, 0.2, -0.2, 0.5, 0.1, -0.3, 0.4, 0, -0.4, 0.3, …)` period 11 / `(-0.02, 0.085, -0.42, 0.48, -0.25, 0.715, 0.23, -0.65)` |
| `∂L/∂θ` for `(x2, g2)`                                                 | `(-0.162005, 0.08757342, -0.3234491, 0.2216778, 0.1413881, -0.2696344, -0.2158197, 1.053119, -0.3772638, -0.162005, 0.08757342, -0.3234491, -0.4040094, -0.1640598, 0.07966122, 0.4176485, -0.3252834, -0.08156233, -0.2427859, -0.179027, 0.2408848, -0.4040094, -0.1640598, 0.07966122, -0.5381472, 1.612236)` |
| Accumulated `G(x,g) + G(x2,g2)`                                        | see `BackwardAccumulatesAcrossSamplesUntilZeroGradients`; FD max err `2.0e-4` |
| Max 2×2 / stride 2×2: forward, argmax flat indices                     | `(0.5, 2, 1.25, 1.5, 2, 1, 0.5, 1.5)`, `(0, 9, 14, 5, 18, 17, 20, 23)` |
| Max 2×3 / stride 2×1: forward, argmax flat indices                     | `(0.75, 2, 1.25, 1.5, 2, 1, 2, 1.5)`, `(12, 9, 14, 5, 18, 17, 18, 23)`; min window margin `0.25` |
| … backward, `g = (0.8, -1.3, 0.4, 0.25, -0.6, 1.1, 0.5, -0.9)`         | nonzero: `[5] 0.25, [9] -1.3, [12] 0.8, [14] 0.4, [17] 1.1, [18] -0.1, [23] -0.9`; FD max err `9.0e-5` |
| Tie `2×4×1`, `x = (-0.3, 0.7, 0.2, 0.2, 0.7, 0.1, -0.5, 0.2)`, `g = (1, 2)` | forward `(0.7, 0.2)`, argmax `(1, 2)`, backward `(0, 1, 2, 0, 0, 0, 0, 0)` |
| Average 2×3 / stride 2×1 forward                                       | `(-0.08333334, 0.5416667, 0.2083333, 0.375, 0.5416667, -0.125, 0.125, 0.2083333)` |
| … backward                                                             | see `AveragePoolingBackward…`; FD max err `6.3e-5`                     |
| Global average forward / backward (`g = (0.8, -1.3)`)                  | `(0.175, 0.1875)` / `(0.04, -0.065)` per position; FD max err `1.6e-5` |
| Model `Conv2D(LeakyReLU) → GAP → Dense(tanh)`, `W = (0.5, -0.25)`, `b = 0.1` | pooled `(1.038125, 0.27125)`, `z = 0.55125`, `y = 0.5014566`; `∂L/∂pooled = (0.3742707, -0.1871353)` |
| Model `Backward(1)`                                                    | see the integration case; FD max err `4.2e-5`; min `|z|` of the conv layer `0.375` ≫ h (no LeakyReLU kink crossed) |

## Edge cases

- Rows dropped by the floor (`(H − KH) mod SH ≠ 0`): never read, gradient exactly 0 — row 4 in
  `BackwardInputGradientMatchesAnalyticAndFiniteDifference` and in both pooling backward cases.
- Overlapping windows (`SW < KW`): input-gradient contributions add — columns 1–2 in the same cases,
  and `x(2,1,0)` winning two max windows.
- Height/width or `SH`/`SW` transposed in the index math: caught by `H ≠ W`, `KH ≠ KW`, `SH ≠ SW` in the
  fixture (the wrong-layout `z` rows in Reference vectors) and the swapped-stride size assertion.
- `KH = H`, `KW = W` (single window) and `KH = KW = 1` (pointwise): compile-time sizes covered by
  `SizesFollowValidStridedWindowCount`; they follow from the same loops.
- `H = 1` degenerate case equals N11: `HeightOneMatchesConvolution1DLayout`.
- `SetParameters` leaves `G` untouched: an N8 contract tested in N8's own suite; here the
  finite-difference probe uses a separate instance so it never depends on it (as in N11).
- `Kernel > map`, zero stride or zero channels: rejected by `static_assert` (not unit-tested).
- `Backward` before any `Forward`: `really_assert(forwardDone)` fires (not unit-tested, same as `Dense`).
- Softmax as the convolution's activation and NaN/±∞ inputs: outside the contract, documented only.
