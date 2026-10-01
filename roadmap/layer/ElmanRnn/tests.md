# Elman RNN (Streaming State, Truncated BPTT) — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
# anonymous namespace, next to the fixture
inline constexpr math::Vector<float, 18> flashTheta{ 0.5, -0.3, 0.2, 0.8, -0.6, 0.1,               # W_x (3×2)
                                                     0.1, -0.4, 0.3, 0.5, 0.2, -0.1, -0.3, 0.6, 0.25,   # W_h (3×3)
                                                     0.05, -0.1, 0.2 }                            # b
template<typename L> concept HasBackward = requires(L& l, const typename L::OutputVector& g) { l.Backward(g); }

class TestElmanRnn : public ::testing::Test:
    using Rnn             = neural_network::ElmanRnn<float, 2, 3, 4>     # X = 2, H = 3, K = 4 ⇒ P = 18
    using ShortWindow     = neural_network::ElmanRnn<float, 2, 3, 2>     # K = 2: truncates a 3-step sequence, wraps the ring
    using Flash           = neural_network::FlashElmanRnn<float, 2, 3>
    using InputVector     = Rnn::InputVector                             # 2
    using OutputVector    = Rnn::OutputVector                            # 3
    using StateVector     = Rnn::StateVector                             # 3 (same type as OutputVector)
    using ParameterVector = Rnn::ParameterVector                         # 18

    const Rnn::InputWeightMatrix     inputWeights{ { 0.5, -0.3 }, { 0.2, 0.8 }, { -0.6, 0.1 } }
    const Rnn::RecurrentWeightMatrix recurrentWeights{ { 0.1, -0.4, 0.3 }, { 0.5, 0.2, -0.1 }, { -0.3, 0.6, 0.25 } }
    const ParameterVector theta{ flashTheta }                            # same 18 values in RAM
    const std::array<InputVector, 3> x{ { 1.0, -0.5 }, { 0.3, 0.9 }, { -0.7, 0.4 } }
    const OutputVector g { 0.7, -1.2, 0.5 }                              # ∂L/∂h at the last step
    const std::array<OutputVector, 3> perStep{ { 0.3, 0.2, -0.4 }, { -0.6, 0.1, 0.9 }, { 0.7, -1.2, 0.5 } }
    const std::array<OutputVector, 3> lastOnly{ {}, {}, { 0.7, -1.2, 0.5 } }
    const StateVector zeroState{}
    const StateVector s0{ 0.5, -0.25, 0.8 }

    template<std::size_t N>
    float SequenceLoss(Rnn& probe, const ParameterVector& p, const StateVector& start,
                       std::span<const InputVector, N> xs, std::span<const OutputVector, N> ups):
        probe.SetParameters(p);  probe.SetState(start)
        L = 0
        for s in 0..N-1:  probe.Forward(xs[s]);  L += Σ_i ups[s][i] · probe.Output()[i]
        return L

    template<std::size_t N>
    ParameterVector NumericParameterGradient(const StateVector& start, std::span<const InputVector, N> xs,
                                             std::span<const OutputVector, N> ups):
        Rnn probe{ inputWeights, recurrentWeights }         # separate instance; N ≤ 3 < K ⇒ FD sees the untruncated sequence
        for k in 0..17:
            numeric[k] = (SequenceLoss(probe, theta + h e_k, start, xs, ups) − SequenceLoss(probe, theta − h e_k, start, xs, ups)) / (2h)
        return numeric
# each case below is a TEST_F(TestElmanRnn, <name>)
# ElmanRnn has no injected collaborator (Tanh is a by-value member, already tested) ⇒ no mock is needed
# finite difference: central, h = 1e-3 (float), tolerance math::Tolerance<float>() = 1e-3
```

## Test cases (Arrange / Act / Assert)

```
SizesAndConstructorParameterLayout:
    Arrange: const Rnn layer{ inputWeights, recurrentWeights }
    Assert:  Rnn::InputSize == 2, Rnn::OutputSize == 3, Rnn::ParameterSize == 18, Rnn::StateSize == 3, Rnn::Window == 4
             neural_network::ElmanRnn<float, 3, 16, 8>::ParameterSize == 320          # H·(X + H + 1)
             Parameters() ≈ {0.5, -0.3, 0.2, 0.8, -0.6, 0.1, 0.1, -0.4, 0.3, 0.5, 0.2, -0.1, -0.3, 0.6, 0.25, 0, 0, 0}
             State() == (0, 0, 0)                                                   # EXPECT_FLOAT_EQ

ForwardComputesElmanRecurrenceFromZeroState:
    Arrange: Rnn layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
    Act:     layer.Forward(x[0]);  h1 = Output();  Forward(x[1]);  h2 = Output();  Forward(x[2])
    Assert:  h1 ≈ (0.6043678, -0.2913126, -0.421899)          # = tanh(0.7, -0.3, -0.45): W_h h₀ = 0
             h2 ≈ (-0.01960537, 0.7469904, -0.3377696)
             Output() ≈ (-0.6762045, 0.2480861, 0.7737613)
             &layer.State() == &layer.Output()                # the state is the output object
             # a feed-forward layer (W_h ignored) would give h2 = tanh(W_x x2 + b) instead; h2 pins the W_h h₁ term

ResetStateZeroesStateAndRestartsSequence:
    Arrange: Rnn layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             Forward(x[0]); Forward(x[1]); Forward(x[2])
    Act:     layer.ResetState()
    Assert:  State() == (0, 0, 0)                             # EXPECT_FLOAT_EQ
    Act:     layer.Forward(x[0])
    Assert:  Output() ≈ (0.6043678, -0.2913126, -0.421899)    # = h1 of a fresh sequence

SetStateSeedsRecurrenceAndStartsNewTruncationWindow:
    Arrange: Rnn layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             Forward(x[0]); Forward(x[1])                     # history now holds two steps
             numeric = NumericParameterGradient(s0, {x[0]}, {g})
    Act:     layer.SetState(s0)
    Assert:  State() ≈ s0
    Act:     layer.Forward(x[0]);  layer.Backward(g)
    Assert:  Output() ≈ (0.7968782, -0.1780809, -0.5005202)   # = tanh(W_x x1 + W_h s0 + b) = tanh(1.09, -0.18, -0.55)
             ParameterGradients() ≈ (0.2554896, -0.1277448, -1.161945, 0.5809723, 0.3747398, -0.1873699,
                                     0.1277448, -0.06387239, 0.2043916, -0.5809723, 0.2904862, -0.9295557,
                                     0.1873699, -0.09368494, 0.2997918, 0.2554896, -1.161945, 0.3747398)
                                                              # one-step [δ x1ᵀ ; δ s0ᵀ ; δ], δ = g ⊙ (1 − h²)
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..17
             # if SetState kept the two old steps, G[2] would be -0.7639371 instead of -1.161945

BackwardReturnsCurrentStepInputGradient:
    Arrange: Rnn layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             numeric[j] = (SequenceLoss over (x1, x2, x3 + h e_j) − SequenceLoss over (x1, x2, x3 − h e_j)) / (2h),
                          zeroState start, ups = lastOnly,  j in 0..1
    Act:     Forward(x[0]); Forward(x[1]); Forward(x[2]);  analytic = layer.Backward(g)
    Assert:  analytic ≈ (-0.1556552, -0.9948274)              # = W_xᵀ δ₃,  δ₃ = (0.3799233, -1.126144, 0.2006467)
             EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>())

BackwardMatchesFullBpttWhenSequenceFitsWindow:
    Arrange: Rnn layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             numeric = NumericParameterGradient(zeroState, x, lastOnly)          # 3 steps ≤ K = 4 ⇒ no truncation
    Act:     Forward(x[0]); Forward(x[1]); Forward(x[2]);  layer.Backward(g)
    Assert:  ParameterGradients() ≈ (-0.5613087, -0.3146506, 1.082255, -0.7166216, -0.1514573, 0.3431987,
                                     -0.3610331, 0.4542311, 0.118505, -0.04652419, -0.8081514, 0.4282675,
                                     0.1442444, 0.07845747, -0.171213, -0.3249733, -0.9116479, 0.3612672)
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..17

BackwardTruncatesGradientAtWindowBoundary:
    Arrange: ShortWindow layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             Forward(x[0]);  h1 = Output()                    # h1 becomes the frozen state before the window
             numeric = NumericParameterGradient(h1, {x[1], x[2]}, {0, g})        # replay only the last K = 2 steps
    Act:     Forward(x[1]); Forward(x[2]);  layer.Backward(g)                    # 3 steps > K = 2: ring wrapped
    Assert:  ParameterGradients() ≈ (-0.4414609, -0.3745745, 0.7542473, -0.5526178, -0.06689905, 0.3009196,
                                     -0.3610331, 0.4542311, 0.118505, -0.04652419, -0.8081514, 0.4282675,
                                     0.1442444, 0.07845747, -0.171213, -0.2051255, -1.239655, 0.4458255)
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..17
             # differs from full BPTT on the W_x and b blocks (e.g. G[2] 0.7542 vs 1.0823, G[16] -1.2397 vs -0.9116);
             # the W_h block (k = 6..14) equals full BPTT because the dropped step-1 term is δ₁ h₀ᵀ = 0

BackwardAccumulatesPerStepLossesUntilZeroGradients:
    Arrange: Rnn layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             numeric = NumericParameterGradient(zeroState, x, perStep)           # ∇θ Σ_t perStep[t]ᵀ h_t
    Act:     for t in 0..2:  layer.Forward(x[t]);  layer.Backward(perStep[t])    # many-to-many
    Assert:  ParameterGradients() ≈ (-0.7266868, -0.8617195, 1.943975, -1.101071, -0.2287487, 1.219031,
                                     -0.7235144, 0.6289515, 0.3715471, -0.0198108, -0.8210275, 0.4096193,
                                     0.6261193, -0.1538121, -0.5076018, -0.9101899, -0.01898709, 0.8421001)
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..17
    Act:     layer.ZeroGradients()
    Assert:  ParameterGradients()[k] == 0 for k in 0..17                        # EXPECT_FLOAT_EQ

FlashElmanRnnMatchesTrainableForwardAndResets:
    Arrange: static_assert(flashTheta[0] == 0.5f)             # constant-initialised ⇒ .rodata (N10 contract)
             static_assert(!HasBackward<Flash>)               # inference only
             Flash flash{ flashTheta };  Rnn ram{ inputWeights, recurrentWeights };  ram.SetParameters(theta)
    Act:     for t in 0..2:  flash.Forward(x[t]);  ram.Forward(x[t])
    Assert:  flash.Output() ≈ (-0.6762045, 0.2480861, 0.7737613)
             |flash.Output()[i] − ram.Output()[i]| ≤ 1e-6   for i in 0..2        # same θ layout and summation order
             flash.Parameters().data() == &flashTheta[0]                         # a view, not a copy
    Act:     flash.ResetState();  flash.Forward(x[0])
    Assert:  flash.Output() ≈ (0.6043678, -0.2913126, -0.421899)
```

Integration case, deployed as `TEST_F(TestModel, ElmanRnnFeedsDenseHeadAndResetStateRestartsSequence)` in
`neural_network/model/test/TestModel.cpp` (the layer test target does not link `neural_network.model`):

```
ElmanRnnFeedsDenseHeadAndResetStateRestartsSequence:
    Arrange: Model<float, 2, 1, ElmanRnn<float, 2, 3, 4>, Dense<float, 3, 1>> model{
                 make_layer<ElmanRnn<float, 2, 3, 4>>(inputWeights, recurrentWeights),
                 make_layer<Dense<float, 3, 1>>(W, tanhActivation) }            # W = {{0.4, -0.7, 0.2}}
             model.SetParameters(theta ++ {0.4, -0.7, 0.2, 0.1})                # TotalParameters = 18 + 4 = 22
    Act:     y1 = model.Forward(x[0]);  y2 = model.Forward(x[1]);  y3 = model.Forward(x[2])
    Assert:  y1[0] ≈ 0.4311319,  y2[0] ≈ -0.4607707,  y3[0] ≈ -0.1871574       # tanh(0.1 + W h_t)
    Act:     model.ResetState();  y = model.Forward(x[0])
    Assert:  y[0] ≈ 0.4311319                                                   # state cleared through the Model
```

## Reference vectors

Computed by `scratchpad/specs/ElmanRnn/reference.py`, with float32 rounding at every operation, the same
summation order as the pseudocode, and the same central-difference step as the tests (`h = 1e-3`). A
float64 run agrees with every float32 value below to within `1e-7`.

| Quantity                                                    | Value                                                                  |
|-------------------------------------------------------------|------------------------------------------------------------------------|
| `z₁` from zero state                                        | `(0.7, -0.3, -0.45)`                                                    |
| `h₁ / h₂ / h₃` from zero state                              | `(0.6043678, -0.2913126, -0.421899)` / `(-0.01960537, 0.7469904, -0.3377696)` / `(-0.6762045, 0.2480861, 0.7737613)` |
| `SetState(s0)` then `x₁`                                    | `(0.7968782, -0.1780809, -0.5005202)`                                   |
| `G` after `SetState(s0)`, `x₁`, `Backward(g)`               | `(0.2554896, -0.1277448, -1.161945, 0.5809723, 0.3747398, -0.1873699, 0.1277448, -0.06387239, 0.2043916, -0.5809723, 0.2904862, -0.9295557, 0.1873699, -0.09368494, 0.2997918, 0.2554896, -1.161945, 0.3747398)`; FD max err `3.6e-5` |
| Same, if the two earlier steps had been kept (wrong)        | `(0.295343, -0.856547, -0.7639371, …)`                                  |
| `δ₃ = g ⊙ (1 − h₃²)`                                        | `(0.3799233, -1.126144, 0.2006467)`                                     |
| `∂L/∂x₃` analytic / FD                                      | `(-0.1556552, -0.9948274)` / `(-0.1556277, -0.9948313)` (max err `2.7e-5`) |
| `G`, full BPTT (K = 4, 3 steps, loss at step 3)             | `(-0.5613087, -0.3146506, 1.082255, -0.7166216, -0.1514573, 0.3431987, -0.3610331, 0.4542311, 0.118505, -0.04652419, -0.8081514, 0.4282675, 0.1442444, 0.07845747, -0.171213, -0.3249733, -0.9116479, 0.3612672)`; FD max err `5.7e-5` |
| `G`, truncated (K = 2, same sequence)                       | `(-0.4414609, -0.3745745, 0.7542473, -0.5526178, -0.06689905, 0.3009196, -0.3610331, 0.4542311, 0.118505, -0.04652419, -0.8081514, 0.4282675, 0.1442444, 0.07845747, -0.171213, -0.2051255, -1.239655, 0.4458255)`; FD from frozen `h₁` max err `5.7e-5` |
| Truncated − full (analytic)                                 | W_x block `(0.1198, -0.0599, -0.3280, 0.1640, 0.0846, -0.0423)`, W_h block `0`, b block `(0.1198, -0.3280, 0.0846)`; max `0.328`. This is `−δ₁ [x₁; h₀; 1]ᵀ` with `δ₁ = (-0.1198478, 0.3280077, -0.0845583)` and `h₀ = 0` |
| `G`, many-to-many (`perStep`, K = 4)                        | `(-0.7266868, -0.8617195, 1.943975, -1.101071, -0.2287487, 1.219031, -0.7235144, 0.6289515, 0.3715471, -0.0198108, -0.8210275, 0.4096193, 0.6261193, -0.1538121, -0.5076018, -0.9101899, -0.01898709, 0.8421001)`; FD max err `6.9e-5` |
| Model head `c + W h_t` / `tanh`, t = 1, 2, 3                 | `0.4612862 / 0.4311319`, `-0.4982893 / -0.4607707`, `-0.1893898 / -0.1871574` |

## Edge cases

- `K = 1`: each `Backward` uses only the current step (`δ_t [x_t; h_{t−1}; 1]ᵀ`, no recursion). Reference for the
  fixture sequence and `g`: `G = (-0.2659463, 0.1519693, 0.7883008, -0.4504576, -0.1404527, 0.08025868,
  -0.007448537, 0.283799, -0.1283265, 0.02207847, -0.8412187, 0.3803771, -0.003933753, 0.1498812,
  -0.06777235, 0.3799233, -1.126144, 0.2006467)`. This is listed for the implementer, not a separate case: the K = 2
  case already exercises the truncation boundary.
- Ring wrap-around: covered by `ShortWindow` (3 pushes into 2 slots). The newest-to-oldest walk must wrap
  `0 → K − 1`.
- Saturation (`h = ±1` exactly in float32): `δ = 0`, no NaN. Follows from the formula and is documented only.
- `Backward` right after `ResetState`/`SetState` (no `Forward`): `really_assert(count > 0)` fires. Not
  unit-tested, same as `Dense`'s `forwardDone`.
- `SetParameters` mid-window keeps the history. The resulting gradient approximation is documented, not tested.
- `InputSize`, `HiddenSize` or `BpttWindow` equal to 0: rejected by `static_assert`, not unit-tested.
