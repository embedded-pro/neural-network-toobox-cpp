# Depthwise-Separable Convolution — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
class TestDepthwiseSeparableConvolution : public ::testing::Test:
    using Depthwise   = neural_network::DepthwiseConvolution2D<float, 3, 5, 2, 2, 2, 1, 2>
                        # H=3, W=5, C=2, 2×2 taps, stride (1, 2) ⇒ Hout = Wout = 2, output 8, P = 10
                        # rows overlap (row 1 is read by both output rows), column 4 is never read
    using Separable   = neural_network::DepthwiseSeparableConvolution2D<float, 3, 5, 2, 3, 2, 2, 1, 2>
                        # same depthwise stage + Cout = 3 ⇒ output 12, P = 10 + 9 = 19
    using Separable1D = neural_network::DepthwiseSeparableConvolution1D<float, 6, 2, 3, 3, 2>
                        # L=6, Cin=2, Cout=3, K=3, S=2 ⇒ Lout = 2, output 6, P = 8 + 9 = 17

    class ActivationFunctionMock : public neural_network::ActivationFunction<float>:   # anonymous namespace
        MOCK_METHOD(float, Forward, (float x), (const, override))
        MOCK_METHOD(float, Backward, (float x), (const, override))
        MOCK_METHOD(void, ForwardVector, (std::span<float>, std::span<const float>), (const, override))
        MOCK_METHOD(void, BackwardVector, (std::span<float>, std::span<const float>, std::span<const float>, std::span<const float>), (const, override))

    neural_network::LeakyReLU<float> leakyRelu{ 0.1 }
    neural_network::Tanh<float>      tanhActivation
    neural_network::Sigmoid<float>   sigmoidActivation

    const Depthwise::KernelMatrix depthwiseKernels{ { 0.5, -0.3 },     # tap (0,0), channels 0, 1
                                                    {-0.2,  0.4 },     # tap (0,1)
                                                    { 0.1,  0.6 },     # tap (1,0)
                                                    { 0.3, -0.25} }    # tap (1,1)
    const Separable::PointwiseMatrix pointwiseWeights{ { 0.6, -0.4 },
                                                       {-0.3,  0.8 },
                                                       { 0.2,  0.5 } }
    const Depthwise::ParameterVector thetaDepthwise{ 0.5, -0.3, -0.2, 0.4, 0.1, 0.6, 0.3, -0.25,
                                                     0.05, -0.1 }                          # D, then bD
    const Separable::ParameterVector thetaSeparable{ 0.5, -0.3, -0.2, 0.4, 0.1, 0.6, 0.3, -0.25,
                                                     0.05, -0.1,
                                                     0.6, -0.4, -0.3, 0.8, 0.2, 0.5,
                                                     0.1, -0.05, 0.2 }                     # D, bD, V, bP

    const Depthwise::InputVector x{  0.4, -0.6,   1.0, 0.2,  -0.8, 0.5,   0.3, -1.2,   0.7, 0.9,    # row 0, 5 pixels × 2 channels
                                    -0.5,  1.1,   0.25, -0.4, 0.6, 0.8,  -1.0, -0.3,   0.2, 1.5,    # row 1
                                     0.9, -0.2,  -0.7, 0.35,  1.2, -0.9,  0.45, 0.1,  -0.15, 0.65 } # row 2
    const Depthwise::InputVector x2{ -0.3, 0.8,   0.6, -0.1,   0.2, 0.4,  -0.9, 0.5,   1.1, -0.7,
                                      0.35, -0.25, -0.6, 1.3,  0.15, 0.05, 0.7, -0.45, -1.1, 0.2,
                                      0.5, 0.9,  -0.2, -0.8,   0.4, 0.6,  -0.35, 1.0,  0.75, -0.5 }
    const Depthwise::OutputVector gDepthwise { 0.7, -1.1, 0.3, 0.5, -0.4, 0.9, 1.2, -0.6 }
    const Depthwise::OutputVector g2Depthwise{ -0.5, 0.4, 1.0, -0.8, 0.6, 0.2, -0.3, 0.9 }
    const Separable::OutputVector gSeparable { 0.5, -0.8, 0.3, 1.1, -0.2, 0.6, -0.7, 0.4, 0.9, 0.25, -1.0, 0.15 }
    const Separable::OutputVector g2Separable{ -0.4, 0.7, 0.2, -0.9, 0.5, -0.3, 1.0, -0.6, 0.35, 0.8, -0.15, 0.45 }

    float ProjectedOutput(Layer& layer, const ParameterVector& p, const InputVector& in, const OutputVector& up):
        layer.SetParameters(p);  layer.Forward(in)
        return Σ_i up[i] · layer.Output()[i]

    NumericInputGradient(layer, θ, in, up):      numeric[j] = (ProjectedOutput(layer, θ, in + h e_j, up) − ProjectedOutput(layer, θ, in − h e_j, up)) / (2h)
    NumericParameterGradient(probe, θ, in, up):  numeric[k] = (ProjectedOutput(probe, θ + h e_k, in, up) − ProjectedOutput(probe, θ − h e_k, in, up)) / (2h)
        # probe is a separate instance with the same activations: FD never touches the tested G
# each case below is a TEST_F(TestDepthwiseSeparableConvolution, <name>)
# gradient cases use Tanh (depthwise) and Sigmoid (pointwise): smooth, and distinct so a swapped
# activation order fails; min |z1| = 0.075 keeps LeakyReLU's kink away from the forward cases
# finite difference: central, h = 1e-3 (float), tolerance math::Tolerance<float>() = 1e-3
```

## Test cases (Arrange / Act / Assert)

```
SizesFollowValidStridedWindowCountAndSeparableParameterCount:
    Assert: Depthwise::InputSize == 30, Depthwise::OutputSize == 8,  Depthwise::ParameterSize == 10
            Depthwise::OutputHeight == 2, Depthwise::OutputWidth == 2
            Separable::InputSize == 30, Separable::OutputSize == 12, Separable::ParameterSize == 19
            Separable1D::InputSize == 12, Separable1D::OutputSize == 6, Separable1D::ParameterSize == 17
            DepthwiseConvolution1D<float, 6, 2, 3, 2>::OutputSize == 4, ::ParameterSize == 8
            DepthwiseSeparableConvolution2D<float, 25, 5, 64, 64, 3, 3>::ParameterSize == 4800
                ::OutputHeight == 23, ::OutputWidth == 3            # standard 3×3 conv: 36 928 parameters

DepthwiseConstructorStoresTapMajorKernelsAndZeroBiases:
    Arrange: const Depthwise layer{ depthwiseKernels, leakyRelu }
    Assert:  Parameters() ≈ {0.5, -0.3, -0.2, 0.4, 0.1, 0.6, 0.3, -0.25, 0, 0}

DepthwiseForwardFiltersEachChannelIndependently:
    Arrange: Depthwise layer{ depthwiseKernels, leakyRelu };  layer.SetParameters(thetaDepthwise)
    Act:     layer.Forward(x)
    Assert:  Output() ≈ {0.075, 0.92, -0.065, -0.0175, -0.037, -0.07975, 0.805, -0.1025}
             # z = (0.075, 0.92, -0.65, -0.175, -0.37, -0.7975, 0.805, -1.025); LeakyReLU on the negatives
             # a flipped kernel gives z₀ = 0.495, a channel-major (c, p, q) read gives z₀ = 0.65 ⇒ both fail

DepthwiseBackwardInputGradientMatchesAnalyticAndFiniteDifference:
    Arrange: Depthwise layer{ depthwiseKernels, tanhActivation };  layer.SetParameters(thetaDepthwise)
             numeric = NumericInputGradient(layer, thetaDepthwise, x, gDepthwise)
    Act:     layer.Forward(x);  analytic = layer.Backward(gDepthwise)
    Assert:  analytic ≈ {0.3480386, 0.1561141, -0.1392155, -0.2081521, 0.100979, -0.1454984, -0.04039161, 0.1939979, 0, 0,
                         -0.1053303, -0.4636745, 0.2787984, 0.3320235, 0.3534042, 0.3637489, -0.07269594, -0.2182513, 0, 0,
                         -0.03498759, 0.3028927, -0.1049628, -0.1262053, 0.06664168, -0.145504, 0.1999251, 0.06062666, 0, 0}
             # row 1 sums both output rows' windows; column 4 (indices 8, 9, 18, 19, 28, 29) is never read ⇒ exactly 0
             # channel c of ∂L/∂x depends only on δ[·,·,c]: no cross-channel terms
             EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>())

DepthwiseBackwardParameterGradientsMatchAnalyticAndFiniteDifference:
    Arrange: Depthwise layer{ depthwiseKernels, tanhActivation };  layer.SetParameters(thetaDepthwise)
             numeric = NumericParameterGradient(probe, thetaDepthwise, x, gDepthwise)
    Act:     layer.Forward(x);  layer.Backward(gDepthwise)
    Assert:  ParameterGradients() ≈ {0.6916525, 0.9160236, 0.002778828, -0.8152462, 0.2579481, -0.06713068,
                                     0.516862, 0.2150904, 1.214576, 0.226929}   # [Σ δ[r,s,c] x[r+p, 2s+q, c] ; Σ δ[r,s,c]]
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())

DepthwiseBackwardAccumulatesAcrossSamplesUntilZeroGradients:
    Arrange: Depthwise layer{ depthwiseKernels, tanhActivation };  layer.SetParameters(thetaDepthwise)
    Act:     layer.Forward(x);  layer.Backward(gDepthwise);  layer.Forward(x2);  layer.Backward(g2Depthwise)
    Assert:  ParameterGradients() ≈ {1.11584, 0.7959129, -1.462806, -1.546723, 0.3656613, 0.4152863,
                                     1.299051, 1.663394, 1.760726, 0.5737541}   # = G(x, g) + G(x2, g2)
             ParameterGradients()[k] ≈ NumericParameterGradient(x, g)[k] + NumericParameterGradient(x2, g2)[k]
    Act:     layer.ZeroGradients()
    Assert:  ParameterGradients()[k] == 0 for k in 0..9                          # EXPECT_FLOAT_EQ

SeparableConstructorConcatenatesDepthwiseThenPointwiseBlocks:
    Arrange: const Separable layer{ depthwiseKernels, leakyRelu, pointwiseWeights, leakyRelu }
    Assert:  Parameters() ≈ {0.5, -0.3, -0.2, 0.4, 0.1, 0.6, 0.3, -0.25, 0, 0,
                             0.6, -0.4, -0.3, 0.8, 0.2, 0.5, 0, 0, 0}

SeparableForwardAppliesDepthwiseActivationBeforePointwise:
    Arrange: Separable layer{ depthwiseKernels, leakyRelu, pointwiseWeights, leakyRelu }
             layer.SetParameters(thetaSeparable)
    Act:     layer.Forward(x)
    Assert:  Output() ≈ {-0.0223, 0.6635, 0.675, 0.068, -0.00445, 0.17825,
                         0.1097, -0.01027, 0.152725, 0.624, -0.03735, 0.30975}
             # a1 = depthwise output of DepthwiseForwardFilters…; z2 = (-0.223, 0.6635, 0.675, 0.068, -0.0445, 0.17825,
             #      0.1097, -0.1027, 0.152725, 0.624, -0.3735, 0.30975)
             # skipping f1 gives y[3] = -0.022; reading V as [c][o] gives y[1] = -0.0403 ⇒ both fail

SeparableBackwardInputGradientMatchesAnalyticAndFiniteDifference:
    Arrange: Separable layer{ depthwiseKernels, tanhActivation, pointwiseWeights, sigmoidActivation }
             layer.SetParameters(thetaSeparable)
             numeric = NumericInputGradient(layer, thetaSeparable, x, gSeparable)
    Act:     layer.Forward(x);  analytic = layer.Backward(gSeparable)
    Assert:  analytic ≈ {0.07194697, 0.02345931, -0.02877879, -0.03127908, 0.07026808, 0.02158366, -0.02810723, -0.02877822, 0, 0,
                         -0.02419481, -0.09010164, 0.05860187, 0.07712679, 0.04236495, -0.02264992, 0.03083632, -0.009370159, 0, 0,
                         -0.007716842, 0.08636604, -0.02315052, -0.03598585, 0.005662267, -0.04103482, 0.0169868, 0.01709784, 0, 0}
             EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>())

SeparableBackwardParameterGradientsMatchAnalyticAndFiniteDifference:
    Arrange: Separable layer{ depthwiseKernels, tanhActivation, pointwiseWeights, sigmoidActivation }
             layer.SetParameters(thetaSeparable)
             numeric = NumericParameterGradient(probe, thetaSeparable, x, gSeparable)
    Act:     layer.Forward(x);  layer.Backward(gSeparable)
    Assert:  ParameterGradients() ≈ {0.01768646, 0.1145705, 0.11014, 0.03363517, 0.01087036, -0.1108104,
                                     -0.02506458, 0.0964038, 0.2638843, -0.0745912,               # ∂L/∂D, ∂L/∂bD
                                     -0.04959245, 0.1171257, -0.1578377, -0.02943434, -0.1344315, -0.1523542,
                                     0.2765898, -0.3513142, 0.4792622}                            # ∂L/∂V, ∂L/∂bP
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())

SeparableBackwardAccumulatesAcrossSamplesUntilZeroGradients:
    Arrange: Separable layer{ depthwiseKernels, tanhActivation, pointwiseWeights, sigmoidActivation }
             layer.SetParameters(thetaSeparable)
    Act:     layer.Forward(x);  layer.Backward(gSeparable);  layer.Forward(x2);  layer.Backward(g2Separable)
    Assert:  ParameterGradients() ≈ {0.1043872, 0.265168, 0.1756504, 0.05620104, 0.1137983, -0.204523,
                                     -0.1587471, 0.1500742, 0.3786798, 0.07424293,
                                     -0.06004707, 0.3338869, -0.1947358, -0.2397041, -0.172101, -0.1502282,
                                     0.4082839, -0.2393738, 0.6462871}           # = G(x, g) + G(x2, g2)
             ParameterGradients()[k] ≈ NumericParameterGradient(x, g)[k] + NumericParameterGradient(x2, g2)[k]
    Act:     layer.ZeroGradients()
    Assert:  ParameterGradients()[k] == 0 for k in 0..18                         # EXPECT_FLOAT_EQ

SeparableActivationsSeeWholeMapsInChainOrder:
    Arrange: StrictMock<ActivationFunctionMock> depthwiseActivation, pointwiseActivation
             Separable layer{ depthwiseKernels, depthwiseActivation, pointwiseWeights, pointwiseActivation }
             InSequence sequence
             EXPECT_CALL(depthwiseActivation, ForwardVector(SizeIs(8), SizeIs(8)))
             EXPECT_CALL(pointwiseActivation, ForwardVector(SizeIs(12), SizeIs(12)))
             EXPECT_CALL(pointwiseActivation, BackwardVector(SizeIs(12), SizeIs(12), SizeIs(12), SizeIs(12)))
             EXPECT_CALL(depthwiseActivation, BackwardVector(SizeIs(8), SizeIs(8), SizeIs(8), SizeIs(8)))
    Act:     layer.Forward(x);  layer.Backward(gSeparable)
    Assert:  (StrictMock) each activation called once per pass on the whole map, f1 before f2 forward and
             f2 before f1 backward; no per-element Forward/Backward calls

OneDimensionalAliasReadsChannelsLastSequence:
    Arrange: Separable1D layer{ {{0.2, -0.1}, {0.4, 0.3}, {-0.5, 0.1}}, leakyRelu, pointwiseWeights, leakyRelu }
             layer.SetParameters({0.2, -0.1, 0.4, 0.3, -0.5, 0.1,  0.05, -0.1,
                                  0.6, -0.4, -0.3, 0.8, 0.2, 0.5,  0.1, -0.05, 0.2})
    Act:     layer.Forward({0.5, -1.0,  0.25, 0.75,  -0.5, 1.5,  1.0, -0.25,  0.0, 2.0,  -1.5, 0.5})   # N11's fixture x
    Assert:  Output() ≈ {0.25, 0.1, 0.4875, 0.315, -0.0165, 0.26375}
             # z1 = (0.5, 0.375, 0.35, -0.125): frames 0–2 and 2–4, kernel rows = taps k (Keras DepthwiseConv1D (K, C, 1))
```

Integration case, deployed as `TEST_F(TestModel, DepthwiseSeparableConvolutionFeedsDenseThroughChannelsLastLayout)`
in `neural_network/model/test/TestModel.cpp` (the layer test target does not link `neural_network.model`):

```
DepthwiseSeparableConvolutionFeedsDenseThroughChannelsLastLayout:
    Arrange: Model<float, 30, 1, Separable, Dense<float, 12, 1>> model{
                 make_layer<Separable>(depthwiseKernels, leakyRelu, pointwiseWeights, leakyRelu),
                 make_layer<Dense<float, 12, 1>>(Wd, tanh) }
                 # Wd = {{0.3, -0.2, 0.5, 0.1, 0.4, -0.6, -0.25, 0.2, 0.35, 0.15, -0.1, 0.45}}
             model.SetParameters(thetaSeparable ++ Wd ++ {0.0})        # TotalParameters = 19 + 13 = 32
    Act:     y = model.Forward(x)
    Assert:  y[0] ≈ 0.3424605                                        # = tanh(0.3568773), Wd · (separable output above)
             model.GetParameters() round-trips the 32 values in order
```

## Reference vectors

Computed by `scratchpad/specs/DepthwiseSeparableConvolution/reference.py` with float32 rounding at every
operation, the loop order of `implementation.md` and the same central-difference step as the tests
(`h = 1e-3`); float64 forward values agree to the printed digits.

| Quantity                                                          | Value |
|-------------------------------------------------------------------|-------|
| `(Hout, Wout)` for fixture / 1D fixture / DS-CNN block / 49×10 3×3 | `(2, 2)` / `(1, 2)` / `(23, 3)` / `(47, 8)` |
| Depthwise `z1`, fixture `θ`, `x`                                  | `(0.075, 0.92, -0.65, -0.175, -0.37, -0.7975, 0.805, -1.025)` |
| Depthwise `z1` with `bD = 0` (constructor state)                  | `(0.025, 1.02, -0.7, -0.075, -0.42, -0.6975, 0.755, -0.925)` |
| Depthwise `z1`, flipped taps / channel-major read (must not match) | `(0.495, 0.73, -0.78, -0.535, …)` / `(0.65, 0.62, -1.1, -0.165, …)` |
| Depthwise Tanh `y`                                                | `(0.07485969, 0.7258974, -0.5716699, -0.1732352, -0.3539917, -0.6626369, 0.6668228, -0.7718952)` |
| Depthwise `δ = (1 − y²) ⊙ g`                                      | `(0.6960772, -0.5203803, 0.201958, 0.4849948, -0.349876, 0.5048211, 0.6664168, -0.2425066)` |
| Depthwise `∂L/∂x` FD                                              | `(0.3480315, 0.156045, -0.1392365, -0.2081394, 0.1009703, -0.1454949, -0.04035234, 0.1940131, 0, 0, -0.1053214, -0.4637241, 0.2788305, 0.331968, 0.3533959, 0.3637671, -0.07265806, -0.2182126, 0, 0, -0.03498793, 0.302881, -0.1049638, -0.126183, 0.0666976, -0.1454949, 0.199914, 0.06061792, 0, 0)` (max err `6.9e-5`) |
| Depthwise `∂L/∂θ` FD                                              | `(0.6915926, 0.9159743, 0.002712011, -0.8152425, 0.2580285, -0.06708503, 0.5168915, 0.2150535, 1.214564, 0.2270043)` (max err `8.0e-5`) |
| Depthwise `∂L/∂θ` for `(x2, g2)`                                  | `(0.4241874, -0.1201106, -1.465585, -0.7314765, 0.1077132, 0.482417, 0.7821891, 1.448304, 0.5461495, 0.3468251)`; accumulated FD max err `1.2e-4` |
| Separable (LeakyReLU, LeakyReLU) `z2`                             | `(-0.223, 0.6635, 0.675, 0.068, -0.0445, 0.17825, 0.1097, -0.1027, 0.152725, 0.624, -0.3735, 0.30975)` |
| Separable without `f1` / with `V` read as `[c][o]` (must not match) | `(-0.0223, 0.6635, 0.675, -0.022, …)` / `(-0.0131, -0.0403, 0.72, …)` |
| Separable (Tanh, Sigmoid) `y`                                     | `(0.4637032, 0.6243985, 0.6405888, 0.4566819, 0.4957283, 0.4997621, 0.538091, 0.3836907, 0.4496421, 0.6918648, 0.2957616, 0.4868573)` |
| `δ2 = y(1 − y) ⊙ g`                                               | `(0.1243413, -0.18762, 0.06907044, 0.2729359, -0.04999635, 0.15, -0.1739843, 0.09458887, 0.2227177, 0.05329698, -0.2082867, 0.03747409)` |
| `g1 = Vᵀ δ2` per pixel                                            | `(0.1447049, -0.1652973, 0.2087604, -0.07417145, -0.08822374, 0.2566237, 0.101959, -0.1692111)` |
| `δ1 = (1 − a1²) ⊙ g1`                                             | `(0.1438939, -0.07819769, 0.1405362, -0.07194554, -0.07716841, 0.1439434, 0.05662267, -0.06839136)` |
| Separable `∂L/∂x` FD                                              | `(0.07194281, 0.02342462, -0.02878904, -0.03123283, 0.07027388, 0.02157688, -0.02807379, -0.02878904, 0, 0, -0.02425909, -0.09006261, 0.05859136, 0.07718801, 0.0424087, -0.02270937, 0.0308752, -0.009447336, 0, 0, -0.007688999, 0.08630752, -0.0231862, -0.0359416, 0.005722045, -0.04103779, 0.01701713, 0.01713634, 0, 0)` (max err `7.7e-5`) |
| Separable `∂L/∂θ` FD                                              | `(0.01767278, 0.1145005, 0.1100898, 0.03364682, 0.01081824, -0.1107752, -0.02509355, 0.0963807, 0.2637804, -0.07462502, -0.04953146, 0.1171231, -0.1578331, -0.02938509, -0.1343787, -0.1523793, 0.2765656, -0.3513097, 0.4793107)` (max err `1.0e-4`) |
| Separable `∂L/∂θ` for `(x2, g2)`                                  | `(0.08670071, 0.1505975, 0.06551041, 0.02256588, 0.1029279, -0.0937126, -0.1336825, 0.05367035, 0.1147954, 0.1488341, -0.01045462, 0.2167613, -0.03689801, -0.2102697, -0.03766945, 0.00212607, 0.131694, 0.1119403, 0.167025)`; accumulated FD max err `1.4e-4` |
| 1D alias `z1` / `z2`                                              | `(0.5, 0.375, 0.35, -0.125)` / `(0.25, 0.1, 0.4875, 0.315, -0.165, 0.26375)` |
| Model Separable → Dense(tanh)                                     | pre-activation `0.3568773`, output `0.3424605` |
| DS-CNN block `<25, 5, 64, 64, 3, 3>`                              | `P = 4800` (standard `36 928`), MACs `322 368` (standard `2 543 616`), ratio `0.1267 = 1/64 + 1/9` |

## Edge cases

- Overlapping windows (`SH < KH`) add their input-gradient contributions: row 1 in both
  `…InputGradientMatchesAnalyticAndFiniteDifference` cases.
- Tail columns never read (`(W − KW) mod SW ≠ 0`): column 4, gradient exactly 0 in the same cases.
- `KH = KW = 1` (depthwise reduces to a per-channel scale + bias) and `KH = H`, `KW = W` (one window):
  compile-time sizes only, no separate runtime case.
- The two activations are distinct objects and may be the same object (e.g. one `ReLU` passed twice):
  both are `const` and stateless, so no test is needed.
- Channels never mix in the depthwise stage: pinned by the forward and gradient reference values (a
  cross-channel term would change them), not a separate case.
- `KernelHeight > Height`, zero strides or zero channels: rejected by `static_assert` (not unit-tested).
- `Backward` before any `Forward`: `really_assert(forwardDone)` fires (not unit-tested, same as `Dense`).
- Softmax as either activation: outside the contract (normalises across pixels), documented only.
