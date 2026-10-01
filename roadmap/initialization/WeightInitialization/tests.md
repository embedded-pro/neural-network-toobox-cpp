# Weight Initialisation (Glorot, He, LeCun) + Deterministic PRNG — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```text
class TestWeightInitialization : public ::testing::Test:
    using Generator   = neural_network::Xorshift32<float>
    using Scaling     = neural_network::VarianceScaling<float>
    using Initializer = neural_network::WeightInitializer<float>
    using DenseLayer  = neural_network::Dense<float, 4, 2>          # fan_in = 4, fan_out = 2

    static constexpr std::size_t fanIn{ 4 }
    static constexpr std::size_t fanOut{ 2 }
    static constexpr std::size_t streamLength{ 65536 }
    static constexpr float       exactTolerance{ 1e-6f }            # one or two correctly rounded ops

    struct Moments { float mean; float second; float fourth; float minimum; float maximum; }
    Moments StreamMoments(auto draw):                               # streaming: no buffer, no heap
        float s1 = 0, s2 = 0, s4 = 0, lo = +inf, hi = -inf
        repeat streamLength times:
            v = draw();  v2 = v * v
            s1 += v;  s2 += v2;  s4 += v2 * v2;  lo = min(lo, v);  hi = max(hi, v)
        return { s1 / N, s2 / N, s4 / N, lo, hi }

    neural_network::ReLU<float> relu
# each case below is a TEST_F(TestWeightInitialization, <name>)
# no collaborators: Xorshift32 is a concrete value type, so there is nothing to mock (same as TestDense)
# no Backward exists (initialisation is not in the computational graph), so there is no finite-difference case
```

## Test cases (Arrange / Act / Assert)

```text
NextFollowsXorshift32RecurrenceFromScrambledSeed:
    Arrange: Generator one{ 1 };  Generator two{ 2 }
    Act:     three Next() from each
    Assert:  one: EXPECT_EQ {524866043, 2877414208, 2380002740}
             two: EXPECT_EQ {3122577100, 3576040911, 2271418240}   # not 2× seed 1: fmix32 broke the linearity

ZeroSeedFallsBackToMarsagliaDefaultState:
    Arrange: Generator zero{ 0 }
    Act:     three Next()
    Assert:  EXPECT_EQ {723471715, 2497366906, 2064144800}         # xor32 from state 2463534242, never stuck at 0

UnitConversionsStayInsideTheirHalfOpenIntervals:
    Assert:  EXPECT_EQ(Generator::UnitFromBits(0x00000000), 0.0f)
             EXPECT_EQ(Generator::UnitFromBits(0x80000000), 0.5f)
             EXPECT_EQ(Generator::UnitFromBits(0xFFFFFFFF), 0.99999994f)          # 1 − 2⁻²⁴, EXPECT_LT(…, 1.0f)
             EXPECT_EQ(Generator::PositiveUnitFromBits(0x00000000), 5.9604645e-8f) # 2⁻²⁴,     EXPECT_GT(…, 0.0f)
             EXPECT_EQ(Generator::PositiveUnitFromBits(0xFFFFFFFF), 1.0f)

UniformStreamHasUnitIntervalMoments:
    Arrange: Generator generator{ 42 }
    Act:     m = StreamMoments([&]{ return generator.Uniform(); })
    Assert:  EXPECT_NEAR(m.mean, 0.5f, 0.005f)                              # measured 0.49975556
             EXPECT_NEAR(m.second − m.mean², 1/12 = 0.083333336f, 0.0015f)  # measured 0.08276457
             EXPECT_GE(m.minimum, 0.0f);  EXPECT_LT(m.maximum, 1.0f)

NormalStreamHasStandardNormalMoments:
    Arrange: Generator generator{ 42 }
    Act:     m = StreamMoments([&]{ return generator.Normal(); })       # 131072 Next() calls
    Assert:  EXPECT_NEAR(m.mean,   0.0f, 0.02f)                         # measured −0.0010543
             EXPECT_NEAR(m.second, 1.0f, 0.03f)                         # measured  0.99526995
             EXPECT_NEAR(m.fourth, 3.0f, 0.2f)                          # measured  2.9921978; a variance-1 uniform gives 1.8
             EXPECT_LE(max(|m.minimum|, |m.maximum|), 5.7681075f)       # Box–Muller cap √(48 ln 2); measured 4.6837535

PresetsFollowVarianceScalingFormula:
    Act/Assert (fanIn = 4, fanOut = 2), EXPECT_NEAR(…, exactTolerance):
             preset                                   StandardDeviation   UniformLimit
             Scaling::GlorotUniform()                 0.57735026          1.0
             Scaling::GlorotNormal()                  0.57735026          1.0
             Scaling::HeUniform()                     0.70710677          1.2247449
             Scaling::HeNormal()                      0.70710677          1.2247449
             Scaling::HeNormal(0.1f)                  0.70359755          —            # scale 2/1.01 = 1.980198
             Scaling::LeCunUniform()                  0.5                 0.8660254
             Scaling::LeCunNormal()                   0.5                 0.8660254
             Scaling{ 1.0f, FanMode::FanOut, WeightDistribution::Normal }   0.70710677   —
             # Glorot ≠ LeCun proves fan_out enters FanAverage; FanOut row proves the third mode
    Assert:  each preset's `distribution` field is Uniform / Normal as named

HeNormalWeightsInitialiseDenseRowMajorWithZeroBiases:
    Arrange: Initializer initializer{ Scaling::HeNormal(), 1 }
    Act:     DenseLayer layer{ initializer.Weights<2, 4>(), relu }       # Weights<Out, In> == DenseLayer::WeightMatrix
    Assert:  layer.Parameters()[0..7] ≈ {−0.69886667, −0.55907273, −0.30341968,  0.38391611,
                                         −0.68643242,  0.17152607,  0.79724646, −1.43270266}
             EXPECT_NEAR(…, math::Tolerance<float>())                   # libm logf/cosf may differ by ~1 ulp
             layer.Parameters()[8..9] == {0, 0}                          # biases untouched

GlorotUniformWeightsAreLimitTimesCentredUniformDraws:
    Arrange: Initializer initializer{ Scaling::GlorotUniform(), 1 };  Generator reference{ 1 }
    Act:     w = initializer.Weights<2, 4>()                             # L = √(6/(4+2)) = 1 exactly
    Assert:  for k in 0..7:  EXPECT_EQ(w.at(k / 4, k % 4), 2.0f * reference.Uniform() − 1.0f)   # exact mapping, row-major
             w ≈ {−0.75559032, 0.33990037, 0.10827506, 0.24061728,
                   0.81145370, −0.08537471, −0.27581966, −0.62438822}   (exactTolerance)
             every |w| ≤ Scaling::GlorotUniform().UniformLimit(4, 2)

ConsecutiveCallsContinueOneStream:
    Arrange: Initializer first{ Scaling::GlorotUniform(), 7 };  Initializer second{ Scaling::GlorotUniform(), 7 }
             std::array<float, 2> tail{};  std::array<float, 10> whole{}
    Act:     first.Weights<2, 4>();  first.Fill(tail, fanIn, fanOut)
             second.Fill(whole, fanIn, fanOut)
    Assert:  EXPECT_EQ(tail[0], whole[8]);  EXPECT_EQ(tail[1], whole[9])  # {−0.07675266, 0.60809577}
             EXPECT_NE(tail[0], whole[0])                                # not a restart (whole[0] = 0.11884129)
```

## Reference vectors

Computed by [`reference.py`](reference.py). It emulates float32 rounding after
every operation (`struct` round-trip), implements `fmix32` + xorshift32 on masked Python integers, and
cross-checks both primitives against published values: raw xorshift32 from state 1 gives `270369`,
and `fmix32(1) = 0x514E28B7`.

| Quantity                                                        | Value                                                                                                                      |
|-----------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Raw xorshift32 from state 1 / state 2 (why seeds are scrambled) | `270369, 67634689, 2647435461` / `540738 = 2 × 270369`                                                                     |
| `fmix32(1)`, `fmix32(2)`, `fmix32(0)`                           | `1364076727`, `821347078`, `0`                                                                                             |
| `Next()` ×3, seed 1                                             | `524866043, 2877414208, 2380002740`                                                                                        |
| `Next()` ×3, seed 2                                             | `3122577100, 3576040911, 2271418240`                                                                                       |
| `Next()` ×3, seed 0 (state `2463534242`)                        | `723471715, 2497366906, 2064144800`                                                                                        |
| `UnitFromBits` at `0 / 0x80000000 / 0xFFFFFFFF`                 | `0 / 0.5 / 0.99999994`                                                                                                     |
| `PositiveUnitFromBits` at `0 / 0xFFFFFFFF`                      | `5.9604645e-8 / 1.0`                                                                                                       |
| Uniform stream, seed 42, 65536 draws (float sums)               | mean `0.49975556`, var `0.08276457`, min `1.26e-5`, max `0.9999964`; std errors `1.13e-3` / `2.9e-4`                       |
| Normal stream, seed 42, 65536 draws (float sums)                | mean `−0.0010543`, `E[z²] 0.99526995`, `E[z⁴] 2.9921978`, max `\|z\| 4.6837535`; std errors `3.9e-3` / `5.5e-3` / `3.8e-2` |
| Box–Muller cap `√(−2 ln 2⁻²⁴)`                                  | `5.7681075` (tail mass beyond it `8.0e-9`)                                                                                 |
| First two `Normal()`, seed 1                                    | `u1 = 0.1222049`, `u2 = 0.66995019` → `z = −0.98834670, −0.79064822`                                                       |
| σ / L, fanIn 4, fanOut 2: Glorot, He, LeCun                     | `0.57735026 / 1.0`, `0.70710677 / 1.2247449`, `0.5 / 0.8660254`                                                            |
| σ, `HeNormal(0.1)`; σ, custom `FanOut`                          | `0.70359755`; `0.70710677`                                                                                                 |
| `HeNormal` `Weights<2, 4>()`, seed 1 (= `0.70710677 · z`)       | `−0.69886667, −0.55907273, −0.30341968, 0.38391611, −0.68643242, 0.17152607, 0.79724646, −1.43270266`                      |
| `GlorotUniform` `Weights<2, 4>()`, seed 1 (= `2u − 1`, `L = 1`) | `−0.75559032, 0.33990037, 0.10827506, 0.24061728, 0.81145370, −0.08537471, −0.27581966, −0.62438822`                       |
| `GlorotUniform`, seed 7: draws 0–1 / draws 8–9                  | `0.11884129, −0.17977190` / `−0.07675266, 0.60809577`                                                                      |

Tolerances in the two stream cases are about 4–5 standard errors of the theoretical moment, so they
test the distribution, not just a regression value. The measured values sit within 2 standard errors
(the uniform variance is the closest, at 1.95σ). A variance-1 uniform source has `E[z⁴] = 1.8` and
fails the kurtosis assertion, so the normal case checks shape as well as scale.

## Edge cases

- `fanIn == 0` or `fanOut == 0`: `really_assert` fires in `Fill` (the divide would give `inf`). This
  is not unit-tested, consistent with the other size and precondition asserts.
- `u1 = 1` (`bits >> 8 = 0xFFFFFF`): `Log(1) = 0`, so `r = 0` and the draw is exactly `0`. It is
  never NaN. `u1` can never be `0`, which is covered by `UnitConversionsStayInsideTheirHalfOpenIntervals`.
- Seed 0 shares a stream with exactly one other seed (the `fmix32` preimage of `2463534242`). This is
  documented and not tested.
- `HeNormal(a)` with `a = 1` reduces to `LeCunNormal` (`scale = 1`), which is linear. This is implied by the
  formula row `HeNormal(0.1f)` and is not a separate case.
- Model-level use: `Model{ make_layer<Dense<…>>(init.Weights<…>(), act), … }` draws in brace order.
  This is a language guarantee, documented in `implementation.md` and not unit-tested.
- Parameter-gradient / `Backward` finite-difference checks: not applicable (no differentiable map).
  A freshly initialised `Dense` keeps passing `TestDense`'s existing finite-difference case, which is
  independent of the weight values.
