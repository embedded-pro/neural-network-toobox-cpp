# LSTM Cell (Forget Gate, Truncated BPTT) — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
# anonymous namespace, next to the fixture
inline constexpr math::Vector<float, 72> flashTheta{
    0.5, -0.3,  0.2, 0.8,  -0.6, 0.1,                                          # W_x, i rows (3×2)
    0.3, 0.4,  -0.2, 0.5,   0.7, -0.1,                                         # W_x, f rows
    0.9, -0.4, -0.5, 0.6,   0.4, 0.3,                                          # W_x, g rows
    -0.3, 0.2,  0.6, -0.5,  0.1, 0.7,                                          # W_x, o rows
    0.1, -0.4, 0.3,   0.5, 0.2, -0.1,  -0.3, 0.6, 0.25,                        # W_h, i rows (3×3)
    0.2, 0.1, -0.3,  -0.4, 0.3, 0.2,    0.1, -0.2, 0.5,                        # W_h, f rows
    0.6, -0.2, 0.1,   0.3, 0.5, -0.4,  -0.1, 0.4, 0.7,                         # W_h, g rows
    -0.2, 0.3, 0.4,   0.1, -0.5, 0.2,   0.4, 0.2, -0.3,                        # W_h, o rows
    0.05, -0.1, 0.2,  1.0, 0.8, 1.2,  0.0, 0.15, -0.05,  0.1, -0.2, 0.3 }      # b (i, f, g, o)
template<typename L> concept HasBackward = requires(L& l, const typename L::OutputVector& g) { l.Backward(g); }

class TestLstm : public ::testing::Test:
    using Cell            = neural_network::Lstm<float, 2, 3, 4>         # X = 2, H = 3, K = 4 ⇒ P = 72, S = 6
    using ShortWindow     = neural_network::Lstm<float, 2, 3, 2>         # K = 2: truncates a 3-step sequence, wraps the ring
    using Flash           = neural_network::FlashLstm<float, 2, 3>
    using InputVector     = Cell::InputVector                            # 2
    using OutputVector    = Cell::OutputVector                           # 3
    using StateVector     = Cell::StateVector                            # 6 = [h; c]
    using ParameterVector = Cell::ParameterVector                        # 72

    const Cell::InputWeightMatrix inputWeights{ { 0.5, -0.3 }, { 0.2, 0.8 }, { -0.6, 0.1 },
                                                { 0.3, 0.4 }, { -0.2, 0.5 }, { 0.7, -0.1 },
                                                { 0.9, -0.4 }, { -0.5, 0.6 }, { 0.4, 0.3 },
                                                { -0.3, 0.2 }, { 0.6, -0.5 }, { 0.1, 0.7 } }
    const Cell::RecurrentWeightMatrix recurrentWeights{ { 0.1, -0.4, 0.3 }, { 0.5, 0.2, -0.1 }, { -0.3, 0.6, 0.25 },
                                                        { 0.2, 0.1, -0.3 }, { -0.4, 0.3, 0.2 }, { 0.1, -0.2, 0.5 },
                                                        { 0.6, -0.2, 0.1 }, { 0.3, 0.5, -0.4 }, { -0.1, 0.4, 0.7 },
                                                        { -0.2, 0.3, 0.4 }, { 0.1, -0.5, 0.2 }, { 0.4, 0.2, -0.3 } }
    const ParameterVector theta{ flashTheta }                            # same 72 values in RAM; b_f = (1, 0.8, 1.2) differs from the ctor
    const std::array<InputVector, 3> x{ { 1.0, -0.5 }, { 0.3, 0.9 }, { -0.7, 0.4 } }
    const OutputVector g { 0.7, -1.2, 0.5 }                              # ∂L/∂h at the last step
    const std::array<OutputVector, 3> perStep{ { 0.3, 0.2, -0.4 }, { -0.6, 0.1, 0.9 }, { 0.7, -1.2, 0.5 } }
    const std::array<OutputVector, 3> lastOnly{ {}, {}, { 0.7, -1.2, 0.5 } }
    const StateVector zeroState{}
    const StateVector s0{ 0.5, -0.25, 0.8,  0.3, -0.6, 1.2 }            # h₀ = (0.5, -0.25, 0.8), c₀ = (0.3, -0.6, 1.2)

    template<std::size_t N>
    float SequenceLoss(Cell& probe, const ParameterVector& p, const StateVector& start,
                       std::span<const InputVector, N> xs, std::span<const OutputVector, N> ups):
        probe.SetParameters(p);  probe.SetState(start)
        L = 0
        for s in 0..N-1:  probe.Forward(xs[s]);  L += Σ_u ups[s][u] · probe.Output()[u]
        return L

    template<std::size_t N>
    ParameterVector NumericParameterGradient(const StateVector& start, std::span<const InputVector, N> xs,
                                             std::span<const OutputVector, N> ups):
        Cell probe{ inputWeights, recurrentWeights }        # separate instance; N ≤ 3 < K ⇒ FD sees the untruncated sequence
        for k in 0..71:
            numeric[k] = (SequenceLoss(probe, theta + h e_k, start, xs, ups) − SequenceLoss(probe, theta − h e_k, start, xs, ups)) / (2h)
        return numeric
# each case below is a TEST_F(TestLstm, <name>)
# Lstm has no injected collaborator (Sigmoid/Tanh are by-value final members, already tested) ⇒ no mock is needed
# finite difference: central, h = 1e-3 (float), tolerance math::Tolerance<float>() = 1e-3
```

## Test cases (Arrange / Act / Assert)

```
SizesAndConstructorParameterLayout:
    Arrange: const Cell layer{ inputWeights, recurrentWeights }
    Assert:  Cell::InputSize == 2, Cell::OutputSize == 3, Cell::ParameterSize == 72, Cell::StateSize == 6, Cell::Window == 4
             neural_network::Lstm<float, 3, 16, 8>::ParameterSize == 1280             # 4·(H·(X + H) + H)
             Parameters()[k] ≈ flashTheta[k] for k in 0..59                           # W_x rows then W_h rows, row-major
             Parameters()[60..71] == (0, 0, 0,  1, 1, 1,  0, 0, 0,  0, 0, 0)          # b_f = 1, other biases 0; EXPECT_FLOAT_EQ
             State() == (0, 0, 0, 0, 0, 0),  Output() == (0, 0, 0)                    # EXPECT_FLOAT_EQ

ForwardComputesGatedCellFromZeroState:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
    Act:     layer.Forward(x[0]);  h1 = Output();  Forward(x[1]);  h2 = Output();  Forward(x[2])
    Assert:  h1 ≈ (0.2081424, -0.1567561, 0.03930818)         # step 1: c₁ = i ⊙ g = (0.5348837, -0.2432784, 0.07685021)
             h2 ≈ (0.2318606, 0.05591217, 0.1424089)
             Output() ≈ (0.06464752, 0.1338311, 0.03905582)
             State() ≈ (0.06464752, 0.1338311, 0.03905582,  0.10765, 0.4606013, 0.06119258)     # [h₃; c₃]
             # h1 pins the gate order: TF1 LSTMCell order (i, g, f, o) gives h1 = (0.2081424, 0.09341199, 0.1831705)
             # h1 pins the cell squash: h = o ⊙ c (no tanh) gives h1 = (0.2276237, -0.1598365, 0.03938553)
             # h2 pins b_f as a plain parameter: adding forget_bias = 1 at run time (TF1) gives
             #    h2 = (0.2562376, 0.04178735, 0.1486312); h1 is unaffected because c₀ = 0

ResetStateZeroesHiddenAndCellAndRestartsSequence:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             Forward(x[0]); Forward(x[1]); Forward(x[2])
    Act:     layer.ResetState()
    Assert:  State() == (0, 0, 0, 0, 0, 0),  Output() == (0, 0, 0)                    # EXPECT_FLOAT_EQ
    Act:     layer.Forward(x[0])
    Assert:  Output() ≈ (0.2081424, -0.1567561, 0.03930818)   # = h1 of a fresh sequence (a stale c would change it)

SetStateSeedsHiddenAndCellAndStartsNewTruncationWindow:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             Forward(x[0]); Forward(x[1])                     # history now holds two steps
             numeric = NumericParameterGradient(s0, {x[0]}, {g})
    Act:     layer.SetState(s0)
    Assert:  State() ≈ s0,  Output() ≈ (0.5, -0.25, 0.8)       # Output mirrors the h half
    Act:     layer.Forward(x[0]);  layer.Backward(g)
    Assert:  Output() ≈ (0.3297209, -0.4262803, 0.4227171)
             State()[3..5] ≈ (0.8967717, -0.6707456, 1.303751)  # c₁ = f ⊙ c₀ + i ⊙ g
             ParameterGradients() ≈ (0.02708184, -0.01354092, 0.1050222, -0.05251111, 0.007910894, -0.003955447,
                                     0.009593583, -0.004796791, 0.08495615, -0.04247807, 0.005499059, -0.00274953,
                                     0.0202307, -0.01011535, -0.1191932, 0.05959658, 0.01613875, -0.008069376,
                                     0.1243282, -0.06216408, 0.1390871, -0.06954357, 0.1077926, -0.05389629,          # W_x
                                     0.01354092, -0.00677046, 0.02166547, 0.05251111, -0.02625556, 0.08401778,
                                     0.003955447, -0.001977724, 0.006328715, 0.004796791, -0.002398396, 0.007674866,
                                     0.04247807, -0.02123904, 0.06796492, 0.00274953, -0.001374765, 0.004399247,
                                     0.01011535, -0.005057674, 0.01618456, -0.05959658, 0.02979829, -0.09535453,
                                     0.008069376, -0.004034688, 0.012911, 0.06216408, -0.03108204, 0.09946253,
                                     0.06954357, -0.03477179, 0.1112697, 0.05389629, -0.02694814, 0.08623406,         # W_h
                                     0.02708184, 0.1050222, 0.007910894, 0.009593583, 0.08495615, 0.005499059,
                                     0.0202307, -0.1191932, 0.01613875, 0.1243282, 0.1390871, 0.1077926)              # b
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..71
             # h₀ = s0[0..2] ≠ 0 ⇒ the W_h block is non-zero, c₀ ≠ 0 ⇒ the f-gate rows are non-zero on the first step
             # if SetState kept the two old steps (count not reset), G[2] would be 0.1352927 instead of 0.1050222

BackwardReturnsCurrentStepInputGradient:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             numeric[j] = (SequenceLoss over (x1, x2, x3 + h e_j) − SequenceLoss over (x1, x2, x3 − h e_j)) / (2h),
                          zeroState start, ups = lastOnly,  j in 0..1
    Act:     Forward(x[0]); Forward(x[1]); Forward(x[2]);  analytic = layer.Backward(g)
    Assert:  analytic ≈ (0.1518109, 0.01044414)               # = W_xᵀ (δi; δf; δg; δo) at step 3
             EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>())

BackwardMatchesFullBpttWhenSequenceFitsWindow:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             numeric = NumericParameterGradient(zeroState, x, lastOnly)          # 3 steps ≤ K = 4 ⇒ no truncation
    Act:     Forward(x[0]); Forward(x[1]); Forward(x[2]);  layer.Backward(g)
    Assert:  ParameterGradients() ≈ (0.08660282, -0.04069027, 0.04962721, -0.04815329, 0.02852084, 0.009061829,
                                     -0.02018233, 0.03759725, 0.008190847, 0.004980799, -0.008367813, 0.009072284,
                                     0.02551855, 0.1506924, -0.007528085, -0.1093067, 0.02855564, 0.1516086,
                                     -0.008936835, 0.004442833, 0.07887217, -0.04214483, 1.202221e-05, 0.006655036,   # W_x
                                     -0.01222366, -0.004028043, -0.00796996, -0.01553419, 0.0003698098, -0.007780356,
                                     0.002866356, -0.003983472, -0.0002393189, 0.01416577, -0.001618399, 0.006546892,
                                     5.984015e-05, -0.001851667, -0.000761566, 0.004003136, 0.0001365851, 0.002104186,
                                     0.05783039, -0.01714073, 0.02222069, -0.04361267, 0.01082733, -0.01765578,
                                     0.07888515, -0.01162993, 0.03533801, 0.003983846, 0.00114284, 0.002524806,
                                     -0.02510238, -0.006610175, -0.01565612, 0.002884895, -0.0005477092, 0.00123998,  # W_h
                                     -0.007333539, -0.04783981, 0.03087654, 0.06358455, 0.001180502, 0.01767492,
                                     0.3254402, -0.2420934, 0.4845811, 0.0209998, -0.1074011, 0.01620068)             # b
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..71
             # pins the cell path: dropping the dc ⊙ f recursion (h-path only) gives G[66..68] = (0.1129978,
             # -0.07221705, 0.262667) instead of (0.3254402, -0.2420934, 0.4845811)

BackwardTruncatesGradientAtWindowBoundary:
    Arrange: ShortWindow layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             Forward(x[0]);  s1 = State()                     # (h₁; c₁) becomes the frozen state before the window
             numeric = NumericParameterGradient(s1, {x[1], x[2]}, {0, g})        # replay only the last K = 2 steps
    Act:     Forward(x[1]); Forward(x[2]);  layer.Backward(g)                    # 3 steps > K = 2: ring wrapped
    Assert:  s1 ≈ (0.2081424, -0.1567561, 0.03930818,  0.5348837, -0.2432784, 0.07685021)
             ParameterGradients() ≈ (0.04175052, -0.01826412, 0.02843461, -0.03755699, 0.01231742, 0.01716354,
                                     -0.02018233, 0.03759725, 0.008190847, 0.004980799, -0.008367813, 0.009072284,
                                     -0.03513686, 0.1810201, 0.03591614, -0.1310288, -0.1006472, 0.2162101,
                                     -0.01284461, 0.00639672, 0.07828354, -0.04185051, -0.003131685, 0.00822689,      # W_x
                                     <W_h block k = 24..59 identical to BackwardMatchesFullBpttWhenSequenceFitsWindow>,
                                     -0.05218584, -0.0690324, 0.01467311, 0.06358455, 0.001180502, 0.01767492,
                                     0.2647848, -0.1986492, 0.3553783, 0.01709203, -0.1079897, 0.01305698)            # b
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..71
             # differs from full BPTT on the i, g, o rows of W_x and b (largest: G[16] -0.1006472 vs 0.02855564);
             # the dropped step-1 terms are δ₁ [x₁; h₀; 1]ᵀ, so the W_h block is equal (h₀ = 0), and so are the
             # f rows (k = 6..11, 63..65), because δf₁ ∝ c₀ = 0

BackwardAccumulatesPerStepLossesUntilZeroGradients:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             numeric = NumericParameterGradient(zeroState, x, perStep)           # ∇θ Σ_t perStep[t]ᵀ h_t
    Act:     for t in 0..2:  layer.Forward(x[t]);  layer.Backward(perStep[t])    # many-to-many
    Assert:  ParameterGradients() ≈ (0.06051711, -0.03236509, 0.01536654, -0.02670401, 0.05976011, 0.03938815,
                                     -0.02646349, 0.01875378, 0.007631362, 0.003302344, -0.006040511, 0.01605419,
                                     -0.0467199, 0.05106547, 0.07163434, -0.1264901, 0.2590548, 0.3376352,
                                     -0.003096837, -0.06724304, 0.06042182, -0.02945734, 0.005800521, 0.03995097,     # W_x
                                     -0.01315885, -0.003323734, -0.008146572, -0.01467804, -0.0002749718, -0.00761867,
                                     0.01197427, -0.01084281, 0.001480731, 0.009807849, 0.001663632, 0.005723889,
                                     -0.0003283351, -0.001559325, -0.0008348738, 0.005617836, -0.001079477, 0.002409126,
                                     0.0309213, 0.003125021, 0.01713885, -0.03917275, 0.007483536, -0.01681729,
                                     0.1386074, -0.0566079, 0.04661668, -0.009647676, 0.011409, -4.953834e-05,
                                     -0.02441605, -0.007127068, -0.0155265, 0.01005891, -0.005950596, 0.002594809,    # W_h
                                     -0.03656436, -0.07922117, 0.09274645, 0.04264737, -0.0006844485, 0.02543259,
                                     0.1627042, -0.1479992, 0.915931, -0.01900411, -0.1235432, 0.04611596)            # b
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..71
    Act:     layer.ZeroGradients()
    Assert:  ParameterGradients()[k] == 0 for k in 0..71                        # EXPECT_FLOAT_EQ

FlashLstmMatchesTrainableForwardAndResets:
    Arrange: static_assert(flashTheta[0] == 0.5f)             # constant-initialised ⇒ .rodata (N10 contract)
             static_assert(!HasBackward<Flash>)               # inference only
             Flash flash{ flashTheta };  Cell ram{ inputWeights, recurrentWeights };  ram.SetParameters(theta)
    Act:     for t in 0..2:  flash.Forward(x[t]);  ram.Forward(x[t])
    Assert:  flash.Output() ≈ (0.06464752, 0.1338311, 0.03905582)
             |flash.State()[u] − ram.State()[u]| ≤ 1e-6   for u in 0..5          # same θ layout and summation order, h and c
             flash.Parameters().data() == &flashTheta[0]                         # a view, not a copy
    Act:     flash.ResetState();  flash.Forward(x[0])
    Assert:  flash.Output() ≈ (0.2081424, -0.1567561, 0.03930818)
```

Integration case, deployed as `TEST_F(TestModel, LstmFeedsDenseHeadAndResetStateRestartsSequence)` in
`neural_network/model/test/TestModel.cpp` (the layer test target does not link `neural_network.model`). It is
the only check that `Lstm` satisfies `detail::StatefulLayerType` with `StateSize = 2H ≠ OutputSize` and composes
through `make_layer`:

```
LstmFeedsDenseHeadAndResetStateRestartsSequence:
    Arrange: Model<float, 2, 1, Lstm<float, 2, 3, 4>, Dense<float, 3, 1>> model{
                 make_layer<Lstm<float, 2, 3, 4>>(inputWeights, recurrentWeights),
                 make_layer<Dense<float, 3, 1>>(W, tanhActivation) }            # W = {{0.4, -0.7, 0.2}}
             model.SetParameters(theta ++ {0.4, -0.7, 0.2, 0.1})                # TotalParameters = 72 + 4 = 76
    Act:     y1 = model.Forward(x[0]);  y2 = model.Forward(x[1]);  y3 = model.Forward(x[2])
    Assert:  y1[0] ≈ 0.2920884,  y2[0] ≈ 0.1801014,  y3[0] ≈ 0.03996711       # tanh(0.1 + W h_t)
    Act:     model.ResetState();  y = model.Forward(x[0])
    Assert:  y[0] ≈ 0.2920884                                                   # h and c cleared through the Model
```

## Reference vectors

Computed by `scratchpad/specs/Lstm/reference.py`, with float32 rounding at every operation, the same summation
order as the pseudocode (and the stable two-branch sigmoid of `Sigmoid.hpp`), and the same central-difference
step as the tests (`h = 1e-3`). A float64 run agrees with every float32 forward value below (states, gates, `tanh c`) to within `1e-7`. An
independent vectorised float64 implementation of the PyTorch `nn.LSTM` equations reproduces `h1..h3` and `c3`, and
its float64 central differences (`h = 1e-6`) match the float32 analytic full-BPTT gradient to `3.6e-8`, which
validates the backward formulas separately from the loop code.

| Quantity                                                    | Value                                                                  |
|-------------------------------------------------------------|------------------------------------------------------------------------|
| Step 1 from zero state `i / f / g / o / tanh c`             | `(0.6681878, 0.4255575, 0.3893608)` / `(0.7502601, 0.5866176, 0.8754467)` / `(0.800499, -0.5716699, 0.1973753)` / `(0.4255575, 0.6570104, 0.5124974)` / `(0.4891054, -0.2385899, 0.07669927)` |
| Step 2 `i / f / g / o / tanh c`                             | `(0.5063269, 0.6789148, 0.4908337)` / `(0.8121682, 0.744136, 0.8008826)` / `(0.07005261, 0.468652, 0.2766023)` / `(0.5292336, 0.4102466, 0.7310809)` / `(0.4381064, 0.1362892, 0.1947923)` |
| `h₁ / h₂ / h₃` from zero state                              | `(0.2081424, -0.1567561, 0.03930818)` / `(0.2318606, 0.05591217, 0.1424089)` / `(0.06464752, 0.1338311, 0.03905582)` |
| `c₁ / c₂ / c₃` from zero state                              | `(0.5348837, -0.2432784, 0.07685021)` / `(0.469885, 0.1371425, 0.1973137)` / `(0.10765, 0.4606013, 0.06119258)` |
| Wrong variants (float64): TF1 gate order `h₁`; `h = o ⊙ c` `h₁`; runtime `forget_bias + 1` `h₂` | `(0.2081424, 0.09341199, 0.1831705)`; `(0.2276237, -0.1598365, 0.03938553)`; `(0.2562376, 0.04178735, 0.1486312)` |
| `SetState(s0)` then `x₁`: `h / c`                           | `(0.3297209, -0.4262803, 0.4227171)` / `(0.8967717, -0.6707456, 1.303751)` |
| `G` after `SetState(s0)`, `x₁`, `Backward(g)`               | as listed in the test case; FD max err `6.0e-5`                         |
| Same, if `SetState` kept the ring (wrong)                   | `G[0..3] = (0.04331828, -0.01935135, 0.1352927, -0.1032396)`, `G[24] = 0.0139984` |
| `∂L/∂x₃` analytic / FD                                      | `(0.1518109, 0.01044414)` / `(0.1518056, 0.01042336)` (max err `2.1e-5`) |
| `G`, full BPTT (K = 4, 3 steps, loss at step 3)             | as listed; FD max err `3.4e-5`                                          |
| `G` bias block without the `dc ⊙ f` recursion (wrong)       | `(-0.05734612, -0.04865977, 0.0003882156, 0.03897631, -0.009358516, 0.01502259, 0.1129978, -0.07221705, 0.262667, 0.01734284, -0.1097729, 0.01359868)` |
| `G`, truncated (K = 2, same sequence)                       | as listed; FD from frozen `(h₁; c₁)` max err `3.0e-5`; equal to full BPTT exactly at `k ∈ 6..11, 24..59, 63..65`; max difference `0.129` at `k = 16` |
| `G`, many-to-many (`perStep`, K = 4)                        | as listed; FD max err `4.0e-5`                                          |
| Model head `c + W h_t` / `tanh`, t = 1, 2, 3                 | `0.3008479 / 0.2920884`, `0.1820875 / 0.1801014`, `0.03998841 / 0.03996711` |
| Sizes                                                       | fixture `P = 72`, 247 floats; `Lstm<float, 3, 16, 8>`: `P = 1280`, 3 531 floats, `Forward` 1 216 MACs, full `Backward` 17 088 MACs |

## Edge cases

- `K = 1`: each `Backward` uses only the current step (no recursion into `h_{t−1}` or `c_{t−1}`). Reference for the
  fixture sequence and `g`: `G[0..5] = (0.04018439, -0.02296251, 0.03440103, -0.01965773, 0.005540837, -0.003166193)`,
  `G[60..71] = (-0.05740627, -0.04914433, -0.007915482, 0.0392577, -0.007836697, 0.01367029, 0.1145723, -0.0955109,
  0.2072607, 0.01797222, -0.1106805, 0.007048778)`. Listed for the implementer, not a separate case: the K = 2 case
  already exercises the truncation boundary.
- Ring wrap-around: covered by `ShortWindow` (3 pushes into 2 slots). All eight ring arrays share the one index.
- Constructor forget-gate bias: pinned by the layout case. A forward check from the constructor parameters is not
  added: from a zero state `c₀ = 0`, so `f` cannot influence `h₁`.
- Saturation (`σ` → exactly 0/1, `tanh` → `±1` in float32): the derivative factor is `0`, no NaN. With `f = 1` and
  `i = 0` exactly, the step copies `c` and `∂L/∂c_{s−1}` gets `d̂c` unchanged. Follows from the formulas; documented only.
- Long free-running streams: `|c|` grows at most by 1 per step; no overflow test (it would need ~10³⁸ steps).
- `Backward` right after `ResetState`/`SetState` (no `Forward`): `really_assert(count > 0)` fires. Not
  unit-tested, same as `Dense`'s `forwardDone`.
- `SetParameters` mid-window keeps the history (N17 approximation). Documented, not tested.
- Keras weights with `recurrent_activation != "sigmoid"` and TF1 `LSTMCell` weights: rejected or out of scope in
  N13, not loadable here. Not tested in this item.
- `InputSize`, `HiddenSize` or `BpttWindow` equal to 0: rejected by `static_assert`, not unit-tested.
