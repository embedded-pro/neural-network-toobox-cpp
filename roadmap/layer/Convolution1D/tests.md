# 1D Convolution — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
class TestConvolution1D : public ::testing::Test:
    using Convolution     = neural_network::Convolution1D<float, 6, 2, 2, 3, 2>   # L=6, Cin=2, Cout=2, K=3, S=2 ⇒ Lout=2, P=14
    using Stream          = neural_network::Convolution1DStream<float, 2, 2, 3, 2>
    using InputVector     = Convolution::InputVector          # 12 = 6 frames × 2 channels, channels-last
    using OutputVector    = Convolution::OutputVector         # 4  = 2 positions × 2 channels
    using ParameterVector = Convolution::ParameterVector      # 14
    using Frame           = Stream::InputFrame                # 2

    class ActivationFunctionMock : public neural_network::ActivationFunction<float>:   # anonymous namespace
        MOCK_METHOD(float, Forward, (float x), (const, override))
        MOCK_METHOD(float, Backward, (float x), (const, override))
        MOCK_METHOD(void, ForwardVector, (std::span<float>, std::span<const float>), (const, override))
        MOCK_METHOD(void, BackwardVector, (std::span<float>, std::span<const float>, std::span<const float>, std::span<const float>), (const, override))

    neural_network::LeakyReLU<float> leakyRelu{ 0.1 }
    neural_network::Tanh<float>      tanhActivation
    const Convolution::KernelMatrix kernels{ { 0.2, -0.1, 0.4,  0.3, -0.5, 0.1  },     # filter 0, (k, c) order
                                             {-0.3,  0.2, 0.1, -0.4,  0.6, 0.05 } }    # filter 1
    const ParameterVector theta{ 0.2, -0.1, 0.4, 0.3, -0.5, 0.1,
                                 -0.3, 0.2, 0.1, -0.4, 0.6, 0.05,
                                 0.1, -0.2 }                                   # kernels, then b = (0.1, -0.2)
    const InputVector  x { 0.5, -1.0,  0.25, 0.75,  -0.5, 1.5,  1.0, -0.25,  0.0, 2.0,  -1.5, 0.5 }
    const OutputVector g { 0.8, -1.3, 0.4, 0.6 }
    const InputVector  x2{ -0.2, 0.4,  1.2, -0.6,  0.3, 0.9,  -1.1, 0.7,  0.5, -0.3,  0.8, -0.4 }
    const OutputVector g2{ -0.5, 0.25, 1.0, -0.75 }

    float ProjectedOutput(Convolution& layer, const ParameterVector& p, const InputVector& in, const OutputVector& up):
        layer.SetParameters(p);  layer.Forward(in)
        return Σ_i up[i] · layer.Output()[i]

    ParameterVector NumericParameterGradient(const InputVector& in, const OutputVector& up):
        Convolution probe{ kernels, tanhActivation }                       # separate instance: FD never touches G
        for k in 0..13:
            numeric[k] = (ProjectedOutput(probe, theta + h e_k, in, up) − ProjectedOutput(probe, theta − h e_k, in, up)) / (2h)
        return numeric
# each case below is a TEST_F(TestConvolution1D, <name>)
# the concrete activations are already tested (as in TestDense); the mock is used only as StrictMock<ActivationFunctionMock>
# finite difference: central, h = 1e-3 (float), tolerance math::Tolerance<float>() = 1e-3
```

## Test cases (Arrange / Act / Assert)

```
SizesFollowValidStridedWindowCount:
    Assert: Convolution::InputSize == 12, Convolution::OutputSize == 4, Convolution::ParameterSize == 14
            Convolution1D<float, 7, 1, 1, 3, 2>::OutputLength == 3      # (7 − 3)/2 + 1, last window ends at frame 6
            Convolution1D<float, 8, 1, 1, 3, 2>::OutputLength == 3      # floor drops frame 7
            Convolution1D<float, 5, 1, 1, 5, 1>::OutputLength == 1      # K = L
            Convolution1D<float, 5, 1, 1, 1, 1>::OutputLength == 5      # K = 1, pointwise

ConstructorStoresFilterMajorKernelsAndZeroBiases:
    Arrange: const Convolution layer{ kernels, leakyRelu }
    Assert:  Parameters() ≈ {0.2, -0.1, 0.4, 0.3, -0.5, 0.1, -0.3, 0.2, 0.1, -0.4, 0.6, 0.05, 0, 0}

ForwardComputesStridedChannelsLastCrossCorrelation:
    Arrange: Convolution layer{ kernels, leakyRelu };  layer.SetParameters(theta)
    Act:     layer.Forward(x)
    Assert:  Output()     ≈ {1.025, -0.105, 0.375, 0.55}    # z = (1.025, -1.05, 0.375, 0.55); LeakyReLU on z₁
             Parameters() ≈ theta
             # asymmetric kernels ⇒ a flipped (true-convolution) kernel would fail these values

BackwardInputGradientMatchesAnalyticAndFiniteDifference:
    Arrange: Convolution layer{ kernels, tanhActivation };  layer.SetParameters(theta)
             numeric[j] = (Σ g ⊙ layer(x + h e_j) − Σ g ⊙ layer(x − h e_j)) / (2h),  j in 0..11
    Act:     layer.Forward(x);  analytic = layer.Backward(g)
    Assert:  analytic ≈ {0.2162922, -0.1334167, 0.07879564, 0.2991676, -0.5300985, 0.06213795,
                         0.1844216, -0.0752855, 0.09549667, 0.05734759, 0, 0}
             # frame 2 (indices 4, 5) sums both overlapping windows; frame 5 is never read ⇒ exactly 0
             EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>())

BackwardParameterGradientsMatchAnalyticAndFiniteDifference:
    Arrange: Convolution layer{ kernels, tanhActivation };  layer.SetParameters(theta)
             numeric = NumericParameterGradient(x, g)
    Act:     layer.Forward(x);  layer.Backward(g)
    Assert:  ParameterGradients() ≈ {-0.01264492, 0.1996059, 0.4294676, 0.1553486, -0.1616711, 1.182277,
                                     -0.4775501, 1.179944, 0.3233346, -0.4914812, 0.2527062, 0.1412569,
                                     0.6719742, -0.05572465}          # [Σ_t δ[t,o] x[tS+k, c] ; Σ_t δ[t,o]]
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())

BackwardAccumulatesAcrossSamplesUntilZeroGradients:
    Arrange: Convolution layer{ kernels, tanhActivation };  layer.SetParameters(theta)
    Act:     layer.Forward(x);  layer.Backward(g);  layer.Forward(x2);  layer.Backward(g2)
    Assert:  ParameterGradients() ≈ {0.329558, 0.7585474, -1.043521, 1.016178, 0.112478, 0.5127075,
                                     -0.7308673, 0.61202, 1.341777, -1.108158, -0.04787174, 0.5289932,
                                     1.033206, -0.5800695}            # = G(x, g) + G(x2, g2)
             ParameterGradients()[k] ≈ NumericParameterGradient(x, g)[k] + NumericParameterGradient(x2, g2)[k]
    Act:     layer.ZeroGradients()
    Assert:  ParameterGradients()[k] == 0 for k in 0..13              # EXPECT_FLOAT_EQ

StreamingMatchesBatchOnCausallyPaddedSequence:
    Arrange: Stream stream{ theta, leakyRelu }
             Convolution1D<float, 8, 2, 2, 3, 2> padded{ kernels, leakyRelu };  padded.SetParameters(theta)
             padded.Forward({0, 0, 0, 0} ++ x)                           # K − 1 = 2 zero frames in front
    Act:     for n in 0..5: emitted[n] = stream.Push(frame n of x);  if emitted[n]: out[n] = stream.Output()
    Assert:  emitted == {true, false, true, false, true, false}       # n mod S == 0
             out[0] ≈ (-0.025, 0.05),  out[2] ≈ (1.025, -0.105),  out[4] ≈ (0.375, 0.55)
             out[0], out[2], out[4] ≈ padded.Output() = (-0.025, 0.05, 1.025, -0.105, 0.375, 0.55)
             # out[2], out[4] also equal the unpadded batch output of ForwardComputes… ((K − 1) mod S == 0)

StreamingResetRestoresZeroHistoryAndPhase:
    Arrange: Stream stream{ theta, leakyRelu }
             stream.Push({3, -2});  stream.Push({1, 1});  stream.Push({-4, 0.5})   # history ≠ 0, phase = 1
    Act:     stream.Reset();  emitted = stream.Push({0.5, -1.0})                    # frame 0 of x
    Assert:  emitted == true
             Output() ≈ (-0.025, 0.05)                              # same as the very first output above

StreamingInvokesActivationOnlyOnEmittedFrames:
    Arrange: StrictMock<ActivationFunctionMock> activation;  Stream stream{ theta, activation }
             EXPECT_CALL(activation, ForwardVector(_, _)).Times(2)
    Act:     4 × stream.Push(frame n of x),  n in 0..3
    Assert:  (StrictMock) no Forward/Backward/BackwardVector calls; ForwardVector exactly on frames 0 and 2
```

Integration case, deployed as `TEST_F(TestModel, Convolution1DFeedsDenseThroughChannelsLastLayout)` in
`neural_network/model/test/TestModel.cpp` (the layer test target does not link `neural_network.model`):

```
Convolution1DFeedsDenseThroughChannelsLastLayout:
    Arrange: Model<float, 12, 1, Convolution1D<float, 6, 2, 2, 3, 2>, Dense<float, 4, 1>> model{
                 make_layer<Convolution1D<float, 6, 2, 2, 3, 2>>(kernels, leakyRelu),
                 make_layer<Dense<float, 4, 1>>(W, tanh) }        # W = {{0.5, -0.25, 0.75, 1.0}}
             model.SetParameters(theta ++ {0.5, -0.25, 0.75, 1.0, 0.0})   # TotalParameters = 14 + 5
    Act:     y = model.Forward(x)
    Assert:  y[0] ≈ 0.8786922                                       # = tanh(0.5·1.025 − 0.25·(−0.105) + 0.75·0.375 + 0.55) = tanh(1.37)
             model.GetParameters() round-trips the 19 values in order
```

## Reference vectors

Computed by [`reference.py`](reference.py) with float32 rounding at every operation and
the same central-difference step as the tests (`h = 1e-3`):

| Quantity                                                        | Value                                                                  |
|-----------------------------------------------------------------|------------------------------------------------------------------------|
| `Lout` for `(L, K, S)` = (6,3,2) / (7,3,2) / (8,3,2) / (5,5,1) / (5,1,1) | `2 / 3 / 3 / 1 / 5`                                           |
| Pre-activation `z`, fixture `θ`, `x`                            | `(1.025, -1.05, 0.375, 0.55)` (float64 agrees)                          |
| Forward, LeakyReLU(0.1)                                         | `(1.025, -0.105, 0.375, 0.55)`                                          |
| Forward, Tanh                                                   | `(0.7718952, -0.7818063, 0.3583574, 0.5005202)`                         |
| `δ = (1 − y²) ⊙ g`, `g = (0.8, -1.3, 0.4, 0.6)`                 | `(0.3233422, -0.5054124, 0.348632, 0.4496877)`                          |
| `∂L/∂x` analytic                                                | `(0.2162922, -0.1334167, 0.07879564, 0.2991676, -0.5300985, 0.06213795, 0.1844216, -0.0752855, 0.09549667, 0.05734759, 0, 0)` |
| `∂L/∂x` finite difference                                       | `(0.2162456, -0.1333952, 0.07879733, 0.2992153, -0.5300045, 0.06210804, 0.1844168, -0.07534027, 0.09548663, 0.05733966, 0, 0)` (max err `9.4e-5`) |
| `∂L/∂θ` analytic                                                | `(-0.01264492, 0.1996059, 0.4294676, 0.1553486, -0.1616711, 1.182277, -0.4775501, 1.179944, 0.3233346, -0.4914812, 0.2527062, 0.1412569, 0.6719742, -0.05572465)` |
| `∂L/∂θ` finite difference                                       | `(-0.01275539, 0.1996755, 0.429511, 0.1554489, -0.1616478, 1.182199, -0.4775524, 1.179934, 0.3232956, -0.4914999, 0.2527237, 0.1413822, 0.6719827, -0.05567073)` (max err `1.3e-4`) |
| `∂L/∂θ` for `(x2, g2)`                                          | `(0.3422029, 0.5589416, -1.472988, 0.8608289, 0.2741491, -0.6695697, -0.2533172, -0.5679241, 1.018443, -0.616677, -0.3005779, 0.3877364, 0.3612314, -0.5243449)` |
| Accumulated `G(x,g) + G(x2,g2)`                                 | `(0.329558, 0.7585474, -1.043521, 1.016178, 0.112478, 0.5127075, -0.7308673, 0.61202, 1.341777, -1.108158, -0.04787174, 0.5289932, 1.033206, -0.5800695)`; FD max err `1.3e-4` |
| Stream (S = 2, LeakyReLU) emitted frames 0 / 2 / 4              | `(-0.025, 0.05)` / `(1.025, -0.105)` / `(0.375, 0.55)`                  |
| Batch `L = 8` on zero-padded `x`                                | `(-0.025, 0.05, 1.025, -0.105, 0.375, 0.55)` (pre-activation `z₀ = -0.25`) |
| Stream before reset (frames `(3,-2)`, `(1,1)`, `(-4,0.5)`)       | emits `(-0.16, 1.5)` and `(3.65, -0.4175)`; after `Reset` + frame 0: `(-0.025, 0.05)` |
| Model Conv → Dense(tanh), `W = (0.5, -0.25, 0.75, 1.0)`, `c = 0` | pre-activation `1.37`, output `0.8786922`                             |

## Edge cases

- Tail frames beyond `(Lout − 1)·S + K − 1` and gaps when `S > K`: never read, gradient exactly 0 —
  covered by frame 5 in `BackwardInputGradientMatchesAnalyticAndFiniteDifference`.
- Overlapping windows (`S < K`): input-gradient contributions add — covered by frame 2 in the same case.
- `K = 1` (pointwise, empty ring) and `K = L` (single window): compile-time sizes covered by
  `SizesFollowValidStridedWindowCount`; `std::array<T, 0>` history needs no runtime test.
- Stride-1 stream: emits every frame; reference outputs `(-0.025, 0.05), (-0.005, 0.4375), (1.025, -0.105),
  (-0.02, -0.01875), (0.375, 0.55), (1.725, -0.2225)` equal the stride-1 batch on the padded input
  (listed for the implementer, not a separate case — the stride-2 case already exercises emission and ring wrap).
- `KernelSize > Length`, `Stride == 0`, zero channels: rejected by `static_assert` (not unit-tested).
- `Backward` before any `Forward`: `really_assert(forwardDone)` fires (not unit-tested, same as `Dense`).
- Softmax as the activation: outside the contract (normalises across positions), documented only.
