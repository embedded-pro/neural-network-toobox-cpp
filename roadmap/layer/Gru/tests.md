# GRU Cell (reset_after, Truncated BPTT) — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
# anonymous namespace, next to the fixture
inline constexpr math::Vector<float, 63> flashTheta{
    0.5, -0.3,  0.2, 0.8,  -0.6, 0.1,                                          # W_x, r rows (3×2)
    0.3, 0.4,  -0.2, 0.5,   0.7, -0.1,                                         # W_x, z rows
    0.9, -0.4, -0.5, 0.6,   0.4, 0.3,                                          # W_x, n rows
    0.1, -0.4, 0.3,   0.5, 0.2, -0.1,  -0.3, 0.6, 0.25,                        # W_h, r rows (3×3)
    0.2, 0.1, -0.3,  -0.4, 0.3, 0.2,    0.1, -0.2, 0.5,                        # W_h, z rows
    0.6, -0.2, 0.1,   0.3, 0.5, -0.4,  -0.1, 0.4, 0.7,                         # W_h, n rows
    0.05, -0.1, 0.2,   0.1, 0.0, -0.2,   0.0, 0.15, -0.05,                     # b_x (r, z, n)
    0.02, 0.03, -0.04, -0.1, 0.2, 0.05,  0.3, -0.2, 0.1 }                      # b_h (r, z, n)
template<typename L> concept HasBackward = requires(L& l, const typename L::OutputVector& g) { l.Backward(g); }

class TestGru : public ::testing::Test:
    using Cell            = neural_network::Gru<float, 2, 3, 4>          # X = 2, H = 3, K = 4 ⇒ P = 63
    using ShortWindow     = neural_network::Gru<float, 2, 3, 2>          # K = 2: truncates a 3-step sequence, wraps the ring
    using Flash           = neural_network::FlashGru<float, 2, 3>
    using InputVector     = Cell::InputVector                            # 2
    using OutputVector    = Cell::OutputVector                           # 3
    using StateVector     = Cell::StateVector                            # 3 (same type as OutputVector)
    using ParameterVector = Cell::ParameterVector                        # 63

    const Cell::InputWeightMatrix inputWeights{ { 0.5, -0.3 }, { 0.2, 0.8 }, { -0.6, 0.1 },
                                                { 0.3, 0.4 }, { -0.2, 0.5 }, { 0.7, -0.1 },
                                                { 0.9, -0.4 }, { -0.5, 0.6 }, { 0.4, 0.3 } }
    const Cell::RecurrentWeightMatrix recurrentWeights{ { 0.1, -0.4, 0.3 }, { 0.5, 0.2, -0.1 }, { -0.3, 0.6, 0.25 },
                                                        { 0.2, 0.1, -0.3 }, { -0.4, 0.3, 0.2 }, { 0.1, -0.2, 0.5 },
                                                        { 0.6, -0.2, 0.1 }, { 0.3, 0.5, -0.4 }, { -0.1, 0.4, 0.7 } }
    const ParameterVector theta{ flashTheta }                            # same 63 values in RAM (non-zero b_x, b_h)
    const std::array<InputVector, 3> x{ { 1.0, -0.5 }, { 0.3, 0.9 }, { -0.7, 0.4 } }
    const OutputVector g { 0.7, -1.2, 0.5 }                              # ∂L/∂h at the last step
    const std::array<OutputVector, 3> perStep{ { 0.3, 0.2, -0.4 }, { -0.6, 0.1, 0.9 }, { 0.7, -1.2, 0.5 } }
    const std::array<OutputVector, 3> lastOnly{ {}, {}, { 0.7, -1.2, 0.5 } }
    const StateVector zeroState{}
    const StateVector s0{ 0.5, -0.25, 0.8 }

    template<std::size_t N>
    float SequenceLoss(Cell& probe, const ParameterVector& p, const StateVector& start,
                       std::span<const InputVector, N> xs, std::span<const OutputVector, N> ups):
        probe.SetParameters(p);  probe.SetState(start)
        L = 0
        for s in 0..N-1:  probe.Forward(xs[s]);  L += Σ_i ups[s][i] · probe.Output()[i]
        return L

    template<std::size_t N>
    ParameterVector NumericParameterGradient(const StateVector& start, std::span<const InputVector, N> xs,
                                             std::span<const OutputVector, N> ups):
        Cell probe{ inputWeights, recurrentWeights }        # separate instance; N ≤ 3 < K ⇒ FD sees the untruncated sequence
        for k in 0..62:
            numeric[k] = (SequenceLoss(probe, theta + h e_k, start, xs, ups) − SequenceLoss(probe, theta − h e_k, start, xs, ups)) / (2h)
        return numeric
# each case below is a TEST_F(TestGru, <name>)
# Gru has no injected collaborator (Sigmoid/Tanh are by-value final members, already tested) ⇒ no mock is needed
# finite difference: central, h = 1e-3 (float), tolerance math::Tolerance<float>() = 1e-3
```

## Test cases (Arrange / Act / Assert)

```
SizesAndConstructorParameterLayout:
    Arrange: const Cell layer{ inputWeights, recurrentWeights }
    Assert:  Cell::InputSize == 2, Cell::OutputSize == 3, Cell::ParameterSize == 63, Cell::StateSize == 3, Cell::Window == 4
             neural_network::Gru<float, 3, 16, 8>::ParameterSize == 1008              # 3·(H·(X + H) + 2H)
             Parameters()[k] ≈ flashTheta[k] for k in 0..44                           # W_x rows then W_h rows, row-major
             Parameters()[k] == 0 for k in 45..62                                     # b_x = b_h = 0; EXPECT_FLOAT_EQ
             State() == (0, 0, 0)                                                     # EXPECT_FLOAT_EQ

ForwardComputesResetAfterGatesFromZeroState:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
    Act:     layer.Forward(x[0]);  h1 = Output();  Forward(x[1]);  h2 = Output();  Forward(x[2])
    Assert:  h1 ≈ (0.409554, -0.352527, 0.08277316)           # step 1: u = b_h^n = (0.3, -0.2, 0.1), r = (0.672607, 0.4329071, 0.3798936)
             h2 ≈ (0.3451068, -0.06922767, 0.1931813)
             Output() ≈ (-0.09023917, 0.177579, 0.007645406)
             &layer.State() == &layer.Output()                # the state is the output object
             # h1 pins r ⊙ b_h^n: folding b_h^n into b_x^n would give h1 = (0.4205605, -0.388503, 0.1032248)
             # h2 pins reset_after: the Cho form tanh(W_x^n x + W_h^n (r ⊙ h) + b) gives h2 = (0.4068705, -0.1208201, 0.2075254)
             # h1 pins the update convention: h = z n + (1 − z) h_{t−1} (Chung) gives h1 = (0.4526272, -0.2745483, 0.1508225)

ResetStateZeroesStateAndRestartsSequence:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             Forward(x[0]); Forward(x[1]); Forward(x[2])
    Act:     layer.ResetState()
    Assert:  State() == (0, 0, 0)                             # EXPECT_FLOAT_EQ
    Act:     layer.Forward(x[0])
    Assert:  Output() ≈ (0.409554, -0.352527, 0.08277316)     # = h1 of a fresh sequence

SetStateSeedsRecurrenceAndStartsNewTruncationWindow:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             Forward(x[0]); Forward(x[1])                     # history now holds two steps
             numeric = NumericParameterGradient(s0, {x[0]}, {g})
    Act:     layer.SetState(s0)
    Assert:  State() ≈ s0
    Act:     layer.Forward(x[0]);  layer.Backward(g)
    Assert:  Output() ≈ (0.7213282, -0.5190974, 0.6911989)
             ParameterGradients() ≈ (0.006760608, -0.003380304, 0.04372862, -0.02186431, 0.01267232, -0.006336158,
                                     -0.07494815, 0.03747408, -0.1323151, 0.06615756, 0.04081457, -0.02040728,
                                     0.04967582, -0.02483791, -0.355354, 0.177677, 0.108294, -0.05414702,          # W_x
                                     0.003380304, -0.001690152, 0.005408486, 0.02186431, -0.01093215, 0.0349829,
                                     0.006336158, -0.003168079, 0.01013785, -0.03747408, 0.01873704, -0.05995853,
                                     -0.06615756, 0.03307878, -0.1058521, 0.02040728, -0.01020364, 0.03265166,
                                     0.01868132, -0.009340659, 0.02989011, -0.08218807, 0.04109403, -0.1315009,
                                     0.01931071, -0.009655356, 0.03089714,                                        # W_h
                                     0.006760608, 0.04372862, 0.01267232, -0.07494815, -0.1323151, 0.04081457,
                                     0.04967582, -0.355354, 0.108294,                                             # b_x
                                     0.006760608, 0.04372862, 0.01267232, -0.07494815, -0.1323151, 0.04081457,
                                     0.03736264, -0.1643761, 0.03862143)                                          # b_h
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..62
             # h_{s−1} = s0 ≠ 0 ⇒ the W_h block is non-zero on the first step of the window
             # if SetState kept the two old steps, G[2] would be 0.03294799 instead of 0.04372862

BackwardReturnsCurrentStepInputGradient:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             numeric[j] = (SequenceLoss over (x1, x2, x3 + h e_j) − SequenceLoss over (x1, x2, x3 − h e_j)) / (2h),
                          zeroState start, ups = lastOnly,  j in 0..1
    Act:     Forward(x[0]); Forward(x[1]); Forward(x[2]);  analytic = layer.Backward(g)
    Assert:  analytic ≈ (0.576035, -0.06134456)               # = W_x^rᵀ δr + W_x^zᵀ δz + W_x^nᵀ δn at step 3
             EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>())

BackwardMatchesFullBpttWhenSequenceFitsWindow:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             numeric = NumericParameterGradient(zeroState, x, lastOnly)          # 3 steps ≤ K = 4 ⇒ no truncation
    Act:     Forward(x[0]); Forward(x[1]); Forward(x[2]);  layer.Backward(g)
    Assert:  ParameterGradients() ≈ (-0.01854642, 0.02903655, 0.000332009, 0.01726906, -0.006218336, 0.002618558,
                                     -0.1273569, 0.08384942, -0.1588944, 0.2103905, -0.05137437, 0.00108069,
                                     -0.1381055, 0.1983663, 0.001047119, -0.291505, -0.03977643, 0.2249987,       # W_x
                                     0.01919428, -0.00848517, 0.008233333, 0.01231582, -0.006741915, 0.004579874,
                                     0.003916921, -0.0004969483, 0.002349041, 0.05634307, -0.01457947, 0.02976377,
                                     0.1092666, -0.0533351, 0.0441434, 0.001558321, 0.00646462, 0.004544119,
                                     0.0639599, -0.02893309, 0.02707865, -0.1387437, 0.07713573, -0.05095257,
                                     0.1001101, -0.04152209, 0.04442282,                                          # W_h
                                     0.05342016, 0.04018624, 0.01448362, 0.1342207, 0.2262678, -0.01067399,
                                     0.3919871, -0.7433894, 0.6171523,                                            # b_x
                                     0.05342016, 0.04018624, 0.01448362, 0.1342207, 0.2262678, -0.01067399,
                                     0.184424, -0.4336583, 0.3225946)                                             # b_h
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..62
             # b_x and b_h gradients coincide for r, z (k = 45..50 vs 54..59) and differ for n (δn vs δu = δn ⊙ r)

BackwardTruncatesGradientAtWindowBoundary:
    Arrange: ShortWindow layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             Forward(x[0]);  h1 = Output()                    # h1 becomes the frozen state before the window
             numeric = NumericParameterGradient(h1, {x[1], x[2]}, {0, g})        # replay only the last K = 2 steps
    Act:     Forward(x[1]); Forward(x[2]);  layer.Backward(g)                    # 3 steps > K = 2: ring wrapped
    Assert:  ParameterGradients() ≈ (-0.01954948, 0.02953808, -0.007117494, 0.02099381, -0.009152615, 0.004085697,
                                     -0.1005784, 0.07046017, -0.09024478, 0.1760657, -0.03150387, -0.008854558,
                                     -0.1532891, 0.2059581, 0.1527691, -0.367366, -0.1643349, 0.2872779,          # W_x
                                     <W_h block k = 18..44 identical to BackwardMatchesFullBpttWhenSequenceFitsWindow>,
                                     0.0524171, 0.03273674, 0.01154934, 0.1609991, 0.2949175, 0.009196507,
                                     0.3768035, -0.5916674, 0.4925938,                                            # b_x
                                     0.0524171, 0.03273674, 0.01154934, 0.1609991, 0.2949175, 0.009196507,
                                     0.1742114, -0.3679768, 0.2752756)                                            # b_h
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..62
             # differs from full BPTT on the W_x and bias blocks (largest: G[14] 0.1527691 vs 0.001047119);
             # the W_h block equals full BPTT because the dropped step-1 terms are δ h₀ᵀ with h₀ = 0

BackwardAccumulatesPerStepLossesUntilZeroGradients:
    Arrange: Cell layer{ inputWeights, recurrentWeights };  layer.SetParameters(theta)
             numeric = NumericParameterGradient(zeroState, x, perStep)           # ∇θ Σ_t perStep[t]ᵀ h_t
    Act:     for t in 0..2:  layer.Forward(x[t]);  layer.Backward(perStep[t])    # many-to-many
    Assert:  ParameterGradients() ≈ (-0.02990475, -0.0006458086, -0.006645132, 0.01839794, -0.00579278, 4.654017e-05,
                                     -0.1009951, 0.04566528, -0.1057695, 0.1664723, -0.07464033, -0.04265449,
                                     -0.2222133, 0.01253572, 0.1405727, -0.3222243, 0.1208042, 0.5433662,         # W_x
                                     0.005401464, 0.003387111, 0.005445727, 0.01139542, -0.005949674, 0.004393856,
                                     0.002996696, 0.0002951428, 0.002163058, 0.04659053, -0.006184889, 0.02779273,
                                     0.102497, -0.04750809, 0.04277522, -0.02003811, 0.02505393, 0.000179359,
                                     0.01603936, 0.01231491, 0.01739364, -0.1281196, 0.067991, -0.04880539,
                                     0.168432, -0.1003307, 0.05823106,                                            # W_h
                                     0.01848746, 0.03163598, 0.01333635, 0.1439137, 0.2678223, -0.07085206,
                                     0.1559564, -0.5778348, 1.043505,                                             # b_x
                                     0.01848746, 0.03163598, 0.01333635, 0.1439137, 0.2678223, -0.07085206,
                                     0.05463919, -0.3521454, 0.5071477)                                           # b_h
             EXPECT_NEAR(ParameterGradients()[k], numeric[k], math::Tolerance<float>())   for k in 0..62
    Act:     layer.ZeroGradients()
    Assert:  ParameterGradients()[k] == 0 for k in 0..62                        # EXPECT_FLOAT_EQ

FlashGruMatchesTrainableForwardAndResets:
    Arrange: static_assert(flashTheta[0] == 0.5f)             # constant-initialised ⇒ .rodata (N10 contract)
             static_assert(!HasBackward<Flash>)               # inference only
             Flash flash{ flashTheta };  Cell ram{ inputWeights, recurrentWeights };  ram.SetParameters(theta)
    Act:     for t in 0..2:  flash.Forward(x[t]);  ram.Forward(x[t])
    Assert:  flash.Output() ≈ (-0.09023917, 0.177579, 0.007645406)
             |flash.Output()[i] − ram.Output()[i]| ≤ 1e-6   for i in 0..2        # same θ layout and summation order
             flash.Parameters().data() == &flashTheta[0]                         # a view, not a copy
    Act:     flash.ResetState();  flash.Forward(x[0])
    Assert:  flash.Output() ≈ (0.409554, -0.352527, 0.08277316)
```

Integration case, deployed as `TEST_F(TestModel, GruFeedsDenseHeadAndResetStateRestartsSequence)` in
`neural_network/model/test/TestModel.cpp` (the layer test target does not link `neural_network.model`). It is
the only check that `Gru` satisfies `detail::StatefulLayerType` and composes through `make_layer`:

```
GruFeedsDenseHeadAndResetStateRestartsSequence:
    Arrange: Model<float, 2, 1, Gru<float, 2, 3, 4>, Dense<float, 3, 1>> model{
                 make_layer<Gru<float, 2, 3, 4>>(inputWeights, recurrentWeights),
                 make_layer<Dense<float, 3, 1>>(W, tanhActivation) }            # W = {{0.4, -0.7, 0.2}}
             model.SetParameters(theta ++ {0.4, -0.7, 0.2, 0.1})                # TotalParameters = 63 + 4 = 67
    Act:     y1 = model.Forward(x[0]);  y2 = model.Forward(x[1]);  y3 = model.Forward(x[2])
    Assert:  y1[0] ≈ 0.4831958,  y2[0] ≈ 0.3141457,  y3[0] ≈ -0.05880397       # tanh(0.1 + W h_t)
    Act:     model.ResetState();  y = model.Forward(x[0])
    Assert:  y[0] ≈ 0.4831958                                                   # state cleared through the Model
```

## Reference vectors

Computed by [`reference.py`](reference.py), with float32 rounding at every operation, the same summation
order as the pseudocode (and the stable two-branch sigmoid of `Sigmoid.hpp`), and the same central-difference
step as the tests (`h = 1e-3`). A float64 run agrees with every float32 value below to within `1e-7`. An
independent vectorised float64 implementation of the PyTorch `nn.GRU` equations reproduces `h1..h3`, and its
float64 central differences (`h = 1e-6`) match the float32 analytic full-BPTT gradient to `5.6e-8`, which
validates the backward formulas separately from the loop code.

| Quantity                                                    | Value                                                                  |
|-------------------------------------------------------------|------------------------------------------------------------------------|
| Step 1 from zero state `r / z / n / u`                      | `(0.672607, 0.4329071, 0.3798936)` / `(0.5249792, 0.4378235, 0.6456563)` / `(0.8621812, -0.6270753, 0.2335957)` / `(0.3, -0.2, 0.1)` |
| Step 2 `r / z / n / u`                                      | `(0.5391194, 0.6976209, 0.4393775)` / `(0.6158159, 0.5834555, 0.5306733)` / `(0.241803, 0.3275908, 0.3180211)` / `(0.6245151, -0.2865066, -0.02402499)` |
| `h₁ / h₂ / h₃` from zero state                              | `(0.409554, -0.352527, 0.08277316)` / `(0.3451068, -0.06922767, 0.1931813)` / `(-0.09023917, 0.177579, 0.007645406)` |
| Wrong variants (float64): `b_h^n` folded `h₁`; Cho reset-before `h₂`; Chung `z` swap `h₁` | `(0.4205605, -0.388503, 0.1032248)`; `(0.4068705, -0.1208201, 0.2075254)`; `(0.4526272, -0.2745483, 0.1508225)` |
| `SetState(s0)` then `x₁`                                    | `(0.7213282, -0.5190974, 0.6911989)`                                    |
| `G` after `SetState(s0)`, `x₁`, `Backward(g)`               | as listed in the test case; FD max err `8.6e-5`                         |
| Same, if the two earlier steps had been kept (wrong)        | `G[0..3] = (0.01513864, 0.01510562, 0.03294799, -0.001631974)`, `G[18] = 0.0112968` |
| `∂L/∂x₃` analytic / FD                                      | `(0.576035, -0.06134456)` / `(0.5760491, -0.06137788)` (max err `3.3e-5`) |
| `G`, full BPTT (K = 4, 3 steps, loss at step 3)             | as listed; FD max err `9.6e-5`                                          |
| `G`, truncated (K = 2, same sequence)                       | as listed; FD from frozen `h₁` max err `6.6e-5`; differs from full BPTT at every index outside `18..44` (max `0.152` at `k = 14`) |
| `G`, many-to-many (`perStep`, K = 4)                        | as listed; FD max err `1.2e-4`                                          |
| Model head `c + W h_t` / `tanh`, t = 1, 2, 3                 | `0.5271451 / 0.4831958`, `0.3251384 / 0.3141457`, `-0.05887189 / -0.05880397` |
| Sizes                                                       | fixture `P = 63`, 199 floats; `Gru<float, 3, 16, 8>`: `P = 1008`, 2 699 floats, `Forward` 912 MACs, full `Backward` 12 816 MACs |

## Edge cases

- `K = 1`: each `Backward` uses only the current step (no recursion into `h_{t−1}`). Reference for the fixture
  sequence and `g`: `G[0..5] = (-0.02469222, 0.01410984, -0.01185696, 0.006775407, -0.008832191, 0.005046966)`,
  `G[45..53] = (0.03527461, 0.01693852, 0.01261742, 0.1488781, 0.17872, 0.03426283, 0.2663302, -0.3302693, 0.312113)`,
  `G[60..62] = (0.114653, -0.18562, 0.1959764)`. Listed for the implementer, not a separate case: the K = 2 case
  already exercises the truncation boundary.
- Ring wrap-around: covered by `ShortWindow` (3 pushes into 2 slots). All six ring arrays share the one index.
- Saturation (`σ` → exactly 0/1, `tanh` → `±1` in float32): the derivative factor is `0`, no NaN. With `z = 1`
  exactly the step copies the state and `∂L/∂h_{s−1}` gets `dh` unchanged. Follows from the formulas; documented only.
- `Backward` right after `ResetState`/`SetState` (no `Forward`): `really_assert(count > 0)` fires. Not
  unit-tested, same as `Dense`'s `forwardDone`.
- `SetParameters` mid-window keeps the history (N17 approximation). Documented, not tested.
- Keras `reset_after=False` weights: rejected by the N13 exporter, not loadable here. Not tested in this item.
- `InputSize`, `HiddenSize` or `BpttWindow` equal to 0: rejected by `static_assert`, not unit-tested.
