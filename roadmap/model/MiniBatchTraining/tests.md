# Mini-Batch SGD Training Step — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```text
class TestMiniBatchTraining : public ::testing::Test:
    using HiddenLayer     = neural_network::Dense<float, 2, 3>
    using OutputLayer     = neural_network::Dense<float, 3, 1>
    using ModelType       = neural_network::Model<float, 2, 1, HiddenLayer, OutputLayer>
    using InputVector     = ModelType::InputVector
    using OutputVector    = ModelType::OutputVector
    using ParameterVector = ModelType::ParameterVector                    # P = 13: [W₁ (6), b₁ (3), W₂ (3), b₂ (1)]
    template<std::size_t B> using Trainer = neural_network::MiniBatchTraining<float, B, ModelType>
    using Configuration   = neural_network::TrainingConfiguration<float>
    using Status          = neural_network::TrainingStatus

    # declared in the anonymous namespace, next to the fixture
    class LossMock : public neural_network::Loss<float, 1>:
        MOCK_METHOD(float, Cost, (const Vector& prediction, const Vector& target), (const, override))
        MOCK_METHOD(Vector, Gradient, (const Vector& prediction, const Vector& target), (const, override))
    MATCHER_P(FirstElementNear, value, "") { return std::abs(arg[0] - value) <= math::Tolerance<float>(); }
    template<typename M> concept HasTrain = requires(M& m) { &M::Train; }

    # same network as TestModel (and N8's model integration cases)
    neural_network::LeakyReLU<float> leakyRelu{ 0.1f }
    neural_network::Tanh<float>      tanhActivation
    const HiddenLayer::WeightMatrix  hiddenWeights{ {0.5, -1.0}, {1.5, 0.25}, {-0.5, 0.75} }     # b₁ = 0
    const OutputLayer::WeightMatrix  outputWeights{ {1.0, -0.5, 2.0} }                            # b₂ = 0
    ModelType model{ make_layer<HiddenLayer>(hiddenWeights, leakyRelu), make_layer<OutputLayer>(outputWeights, tanhActivation) }
    ModelType probe{ make_layer<HiddenLayer>(hiddenWeights, leakyRelu), make_layer<OutputLayer>(outputWeights, tanhActivation) }
    const ParameterVector theta0 = model.GetParameters()

    neural_network::MeanSquaredError<float, 1> mse                      # N9: stateless, L = (ŷ − y)², ∂L/∂ŷ = 2(ŷ − y)
    ::testing::StrictMock<LossMock> lossMock

    const InputVector  x1{  2.0, 0.5 };   const OutputVector y1{ -0.5 }
    const InputVector  x2{ -1.0, 1.0 };   const OutputVector y2{ 0.25 }
    const InputVector  x3{  0.5, -1.5 };  const OutputVector y3{ 0.75 }
    const Configuration sgd{ .learningRate = 0.1f }                       # λ = 0, no clipping

    float BatchObjective(const ParameterVector& θ, float λ):             # J(θ) on {(x1,y1),(x2,y2)} + L2 oracle
        probe.SetParameters(θ)
        J = (mse.Cost(probe.Forward(x1), y1) + mse.Cost(probe.Forward(x2), y2)) / 2
        return J + regularization::L2<float, 13>{ λ }.Calculate(θ)      # λ/2 ‖θ‖²

    ParameterVector NumericGradient(float λ):                            # central difference on `probe`, h = 1e-3
        numeric[k] = (BatchObjective(theta0 + h e_k, λ) − BatchObjective(theta0 − h e_k, λ)) / (2h)

    ParameterVector EffectiveGradient(float η):                          # what the step applied: (θ₀ − θ)/η
        return (theta0 − model.GetParameters()) / η

    void ExpectParameters(const ParameterVector& expected):              # EXPECT_NEAR, Tolerance, all 13
# each case below is a TEST_F(TestMiniBatchTraining, <name>)
# collaborators: the Loss (StrictMock<LossMock> where the call contract is asserted), the Model (concrete,
#   already tested; mocks cannot be stored in Model's tuple), the activations (concrete, already tested)
# Tolerance = math::Tolerance<float>() = 1e-3; finite difference h = 1e-3 on the separate `probe` model
```

Compile-time checks at the top of `TestMiniBatchTraining.cpp` (not tests):

```text
static_assert(!HasTrain<ModelType>)                                     # Model::Train is gone
static_assert(ModelType::NumberOfLayers == 2)
static_assert(std::is_same_v<Trainer<2>::LossType, neural_network::Loss<float, 1, OutputVector>>)
```

## Test cases (Arrange / Act / Assert)

```text
ConstructorDiscardsStaleGradients:
    Arrange: model.Forward(x1);  model.Backward(OutputVector{ 1.0 })          # G ≠ 0 before the trainer exists
    Act:     Trainer<2> trainer{ model, mse, sgd }
    Assert:  model.Gradients()[k] == 0 for k in 0..12                          # EXPECT_FLOAT_EQ

AccumulateFeedsLossGradientIntoBackwardWithoutStepping:
    Arrange: Trainer<2> trainer{ model, lossMock, sgd }
             EXPECT_CALL(lossMock, Cost(FirstElementNear(-0.8298019f), Ref(y1))).WillOnce(Return(0.3f))
             EXPECT_CALL(lossMock, Gradient(FirstElementNear(-0.8298019f), Ref(y1))).WillOnce(Return(OutputVector{ 1.0 }))
    Act:     r = trainer.Accumulate(x1, y1)
    Assert:  r.status == Accumulating;  r.samples == 1;  trainer.PendingSamples() == 1
             model.GetParameters()[k] == theta0[k]                            # EXPECT_FLOAT_EQ: no step yet
             model.Gradients() ≈ (0.6228576, 0.1557144, -0.3114288, -0.0778572, 0.1245715, 0.03114288,
                                  0.3114288, -0.1557144, 0.06228576, 0.1557144, 0.973215, -0.0194643, 0.3114288)
             # = ∇θ ŷ at x1: exactly the mocked ∂L/∂ŷ = 1 back-propagated (N8's reference vector)

FullBatchStepsWithMeanGradientAndMatchesFiniteDifference:
    Arrange: Trainer<2> trainer{ model, mse, sgd };  numeric = NumericGradient(0)
    Act:     r1 = trainer.Accumulate(x1, y1);  r2 = trainer.Accumulate(x2, y2)
    Assert:  r1.status == Accumulating
             r2.status == Stepped;  r2.samples == 2
             EXPECT_NEAR(r2.meanLoss, 0.3238175f, Tolerance)                   # (0.1087693 + 0.5388657)/2
             EXPECT_NEAR(r2.gradientNorm, 0.4411865f, Tolerance)
             ExpectParameters((0.5207739, -0.9950964, 1.489613, 0.2475482, -0.4912527, 0.7463882,
                               0.01003904, -0.005019518, -0.00258471, 1.005483, -0.4676133, 1.996459, 0.007951528))
             EXPECT_NEAR(EffectiveGradient(0.1)[k], numeric[k], Tolerance)     for k in 0..12
             model.Gradients()[k] == 0 for k in 0..12;  trainer.PendingSamples() == 0   # reset for the next batch

FlushAveragesOverPendingSamplesOnly:
    Arrange: Trainer<4> trainer{ model, mse, sgd };  trainer.Accumulate(x1, y1)          # 1 of 4
    Act:     r = trainer.Flush()
    Assert:  r.status == Stepped;  r.samples == 1
             EXPECT_NEAR(r.meanLoss, 0.1087693f, Tolerance);  EXPECT_NEAR(r.gradientNorm, 0.8664723f, Tolerance)
             ExpectParameters((0.5410839, -0.989729, 1.479458, 0.2448645, -0.4917832, 0.7520542,
                               0.02054196, -0.01027098, 0.004108393, 1.010271, -0.4358064, 1.998716, 0.02054196))
             # = θ₀ − 0.1 · ∇θL₁: divided by k = 1, not by B = 4 (which would give a step 4× smaller)

FlushWithoutPendingSamplesIsIdle:
    Arrange: Trainer<2> trainer{ model, lossMock, sgd }                       # StrictMock: any loss call fails
    Act:     r = trainer.Flush()
    Assert:  r.status == Idle;  r.samples == 0
             model.GetParameters()[k] == theta0[k]                            # EXPECT_FLOAT_EQ

FrozenLayerKeepsParametersAndIsExcludedFromNorm:
    Arrange: Trainer<2> trainer{ model, mse, sgd };  trainer.SetTrainableLayers({ false, true })
    Act:     trainer.Accumulate(x1, y1);  r = trainer.Accumulate(x2, y2)
    Assert:  r.status == Stepped
             EXPECT_NEAR(r.gradientNorm, 0.3398141f, Tolerance)               # output-layer slice only (0.4411865 unmasked)
             model.GetParameters()[k] == theta0[k] for k in 0..8              # EXPECT_FLOAT_EQ: hidden layer untouched
             model.GetParameters()[9..12] ≈ (1.005483, -0.4676133, 1.996459, 0.007951528)   # same as the unmasked step
             model.Gradients()[k] == 0 for k in 0..12                          # frozen G cleared too

WeightDecayAddsLambdaThetaAndMatchesFiniteDifferenceOfPenalisedObjective:
    Arrange: Trainer<2> trainer{ model, mse, { .learningRate = 0.1f, .weightDecay = 0.1f } }
             numeric = NumericGradient(0.1)                                  # ∇(J + λ/2 ‖θ‖²), L2 oracle
    Act:     trainer.Accumulate(x1, y1);  r = trainer.Accumulate(x2, y2)
    Assert:  r.status == Stepped;  EXPECT_NEAR(r.gradientNorm, 0.4411865f, Tolerance)   # data term only
             ExpectParameters((0.5157739, -0.9850965, 1.474613, 0.2450482, -0.4862527, 0.7388882,
                               0.01003904, -0.005019518, -0.00258471, 0.9954834, -0.4626133, 1.976459, 0.007951528))
             EXPECT_NEAR(EffectiveGradient(0.1)[k], numeric[k], Tolerance)     for k in 0..12
             # effective gradient = ḡ + λθ₀ = (-0.1577389, -0.1490355, 0.2538693, …, -0.07951528);
             # biases are 0 at θ₀, so their entries equal the undecayed step; weights move by up to 0.02

GlobalNormClippingRescalesToThreshold:
    Arrange: Trainer<2> trainer{ model, mse, { .learningRate = 0.1f, .maxGradientNorm = 0.25f } }
    Act:     trainer.Accumulate(x1, y1);  r = trainer.Accumulate(x2, y2)
    Assert:  r.status == Clipped
             EXPECT_NEAR(r.gradientNorm, 0.4411865f, Tolerance)               # reported before clipping
             EXPECT_NEAR(‖EffectiveGradient(0.1)‖₂, 0.25f, Tolerance)
             ExpectParameters((0.5117716, -0.9972214, 1.494114, 0.2486107, -0.4950433, 0.7479534,
                               0.005688658, -0.002844329, -0.001464635, 1.003107, -0.4816479, 1.997993, 0.004505763))
             # = θ₀ − 0.1 · 0.5666538 · ḡ: same direction as the unclipped step

ClippingThresholdAboveNormLeavesStepUnchanged:
    Arrange: Trainer<2> trainer{ model, mse, { .learningRate = 0.1f, .maxGradientNorm = 1.0f } }
    Act:     trainer.Accumulate(x1, y1);  r = trainer.Accumulate(x2, y2)
    Assert:  r.status == Stepped                                              # 0.4411865 ≤ 1
             ExpectParameters(θ₁ of FullBatchStepsWithMeanGradientAndMatchesFiniteDifference)

ClippingExcludesWeightDecayTerm:
    Arrange: Trainer<2> trainer{ model, mse, { .learningRate = 0.1f, .weightDecay = 0.1f, .maxGradientNorm = 0.25f } }
    Act:     trainer.Accumulate(x1, y1);  r = trainer.Accumulate(x2, y2)
    Assert:  r.status == Clipped
             ExpectParameters((0.5067716, -0.9872214, 1.479114, 0.2461107, -0.4900433, 0.7404534,
                               0.005688658, -0.002844329, -0.001464635, 0.9931072, -0.4766479, 1.977993, 0.004505763))
             # = θ₀ − 0.1 · (0.5666538 · ḡ + 0.1 · θ₀); clipping ḡ + λθ₀ as a whole would give
             # (0.5065537, -0.9938079, 1.489452, …, 0.003303673), up to 0.0122 away

NonFiniteGradientSkipsStepAndClearsGradients:
    Arrange: Trainer<1> trainer{ model, lossMock, sgd }
             EXPECT_CALL(lossMock, Cost(_, Ref(y1))).WillOnce(Return(0.25f))
             EXPECT_CALL(lossMock, Gradient(_, Ref(y1))).WillOnce(Return(OutputVector{ std::numeric_limits<float>::quiet_NaN() }))
    Act:     r = trainer.Accumulate(x1, y1)
    Assert:  r.status == SkippedNonFinite;  r.samples == 1;  EXPECT_FALSE(math::IsFinite(r.gradientNorm))
             model.GetParameters()[k] == theta0[k]                            # EXPECT_FLOAT_EQ
             model.Gradients()[k] == 0 for k in 0..12;  trainer.PendingSamples() == 0

SetLearningRateAppliesToNextStep:
    Arrange: Trainer<2> trainer{ model, mse, sgd }
    Act:     trainer.SetLearningRate(0.05f);  trainer.Accumulate(x1, y1);  trainer.Accumulate(x2, y2)
    Assert:  ExpectParameters((0.5103869, -0.9975482, 1.494807, 0.2487741, -0.4956264, 0.7481941,
                               0.005019518, -0.002509759, -0.001292355, 1.002742, -0.4838066, 1.998229, 0.003975764))

TrainEpochStepsEveryBatchAndFlushesRemainder:
    Arrange: Trainer<2> trainer{ model, mse, sgd }
             const std::array<InputVector, 3>  inputs{ x1, x2, x3 }
             const std::array<OutputVector, 3> targets{ y1, y2, y3 }
    Act:     e = trainer.TrainEpoch(inputs, targets)
    Assert:  e.steps == 2;  e.skippedSteps == 0;  trainer.PendingSamples() == 0
             EXPECT_NEAR(e.meanLoss, 0.2207218f, Tolerance)                   # (0.1087693 + 0.5388657 + 0.01453056)/3
             ExpectParameters((0.5178389, -0.9862914, 1.490978, 0.2434533, -0.4918354, 0.7481365,
                               0.004169009, -0.002289585, -0.003750245, 0.9951906, -0.4697644, 1.997257, 0.002113514))
             # x3 is evaluated at θ₁ (after the first step) and flushed alone (k = 1);
             # a gradient left over from batch 1 would give a different θ₂

RepeatedEpochsFitTwoSampleRegression:
    Arrange: Trainer<2> trainer{ model, mse, sgd }
             const std::array<InputVector, 2> inputs{ x1, x2 };  const std::array<OutputVector, 2> targets{ y1, y2 }
    Act:     first = trainer.TrainEpoch(inputs, targets)
             repeat 39 times: last = trainer.TrainEpoch(inputs, targets)
    Assert:  EXPECT_NEAR(first.meanLoss, 0.3238175f, Tolerance)
             EXPECT_LT(last.meanLoss, 1e-6f)                                   # reference: 4.1e-15 at epoch 40
             EXPECT_NEAR(model.Forward(x1)[0], -0.5f, Tolerance);  EXPECT_NEAR(model.Forward(x2)[0], 0.25f, Tolerance)
```

## Reference vectors

Computed by [`reference.py`](reference.py) with float32 rounding at every operation,
the same `h = 1e-3` central difference as the tests, and the trainer algorithm above. `ḡ` was
cross-checked in float64 (agrees to `2·10⁻⁷`). The 40-epoch trajectory was reproduced by an independent
float64 finite-difference gradient descent (`J = 2.6·10⁻⁶` at epoch 22 in both).

| Quantity                                                      | Value                                                                                                                                                                               |
|---------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sample 1: `ŷ`, `L`, `∂L/∂ŷ`, hidden `z`                       | `-0.8298019`, `0.1087693`, `-0.6596038`, `(0.5, 3.125, -0.625)`                                                                                                                     |
| Sample 2: `ŷ`, `L`, `∂L/∂ŷ`, hidden `z`                       | `0.9840747`, `0.5388657`, `1.468149`, `(-1.5, -1.25, 1.25)`                                                                                                                         |
| Sample 3 at `θ₀`: `ŷ`, `L`, hidden `z`                        | `0.8584704`, `0.01176582`, `(1.75, 0.375, -1.375)`                                                                                                                                  |
| `∇θ ŷ` at `x1` (mocked `∂L/∂ŷ = 1`)                           | `(0.6228576, 0.1557144, -0.3114288, -0.0778572, 0.1245715, 0.03114288, 0.3114288, -0.1557144, 0.06228576, 0.1557144, 0.973215, -0.0194643, 0.3114288)`                              |
| `∇θ L₁`                                                       | `(-0.4108392, -0.1027098, 0.2054196, 0.0513549, -0.08216785, -0.02054196, -0.2054196, 0.1027098, -0.04108392, -0.1027098, -0.6419363, 0.01283873, -0.2054196)`, `‖·‖ = 0.8664723`   |
| `∇θ L₂`                                                       | `(-0.004638906, 0.004638906, 0.002319453, -0.002319453, -0.09277812, 0.09277812, 0.004638906, -0.002319453, 0.09277812, -0.006958359, -0.005798632, 0.05798632, 0.04638906)`        |
| `ḡ = (∇θL₁ + ∇θL₂)/2`                                         | `(-0.2077391, -0.04903545, 0.1038695, 0.02451773, -0.08747298, 0.03611808, -0.1003904, 0.05019518, 0.0258471, -0.05483408, -0.3238675, 0.03541252, -0.07951528)`, `‖ḡ‖ = 0.4411865` |
| `∇J` finite difference                                        | `(-0.2077371, -0.04902482, 0.103876, 0.02451241, -0.08746981, 0.03612041, -0.1003742, 0.05018711, 0.02586841, -0.05483627, -0.3238767, 0.03539026, -0.07951259)`, max err `2.2e-5`  |
| `θ₁ = θ₀ − 0.1 ḡ`                                             | `(0.5207739, -0.9950964, 1.489613, 0.2475482, -0.4912527, 0.7463882, 0.01003904, -0.005019518, -0.00258471, 1.005483, -0.4676133, 1.996459, 0.007951528)`                           |
| Frozen hidden layer: `‖ḡ_𝒯‖`                                 | `0.3398141`                                                                                                                                                                         |
| `λ = 0.1`: `ḡ + λθ₀`                                          | `(-0.1577389, -0.1490355, 0.2538693, 0.04951775, -0.137473, 0.111118, -0.1003904, 0.05019518, 0.0258471, 0.04516602, -0.3738675, 0.2354121, -0.07951528)`                           |
| `λ = 0.1`: `∇(J + λ/2‖θ‖²)` finite difference                 | `(-0.1577139, -0.1490116, 0.253886, 0.04947185, -0.1374781, 0.1111031, -0.1003742, 0.05018711, 0.02586841, 0.04515051, -0.3738999, 0.2354085, -0.07951259)`, max err `4.6e-5`       |
| Clip `c = 0.25`: `ρ = c/‖ḡ‖`                                  | `0.5666538`; step norm `0.25`                                                                                                                                                       |
| Epoch `{x1,x2,x3}`, `B = 2`: hidden `z` of `x3` at `θ₁`, `L₃` | `(1.763071, 0.3684647, -1.367793)`, `0.01453056`                                                                                                                                    |
| 40 epochs on `{x1,x2}`: `meanLoss` at epochs 1, 20, 22, 40    | `0.3238175`, `0.0274`, `2.6e-6`, `4.1e-15`; predictions `(-0.5000001, 0.25)`                                                                                                        |

## Edge cases

- Kinks: every LeakyReLU pre-activation at `θ₀` (`0.5, 3.125, -0.625` / `-1.5, -1.25, 1.25`) is at least
  `0.5` from `0`, so the `±1e-3` finite-difference perturbations never cross a kink.
- `ηλ ≥ 1`, `η ≤ 0`, `λ < 0` or `c ≤ 0` in the configuration or `SetLearningRate`: `really_assert`.
  Not unit-tested (death tests are not used in this repo).
- An all-frozen mask, or a mask whose only trainable layers have `ParameterSize = 0`: `really_assert`
  in `SetTrainableLayers`. Not unit-tested.
- `TrainEpoch` with pending samples: `really_assert`. `TrainEpoch` with `NumberOfSamples = 0`:
  `static_assert`. Neither is a runtime test.
- `‖ḡ‖²` overflow on finite gradients (`‖ḡ‖ > ~1.8·10¹⁹`) takes the same `SkippedNonFinite` path as the
  NaN case. It is not tested separately.
- The class-index target (`Target = std::size_t`, N9 CCE) only changes the type passed through to the
  loss. It is covered by N9's tests and by compiling `MiniBatchTraining<float, 2, ModelType, std::size_t>`
  in a user project. It is not a separate runtime case here.
