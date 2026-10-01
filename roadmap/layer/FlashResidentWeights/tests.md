# Flash-Resident Inference Weights — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```text
# anonymous namespace, next to the fixture
inline constexpr math::Vector<float, 8> flashTheta{ 0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.2, -0.1 }   # W row-major, b = (0.2, -0.1)
inline constexpr float flashRegion[8]{ 0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.2, -0.1 }                # raw region, same values

class ActivationFunctionMock : public neural_network::ActivationFunction<float>:
    MOCK_METHOD(float, Forward,  (float x), (const, override))
    MOCK_METHOD(float, Backward, (float x), (const, override))
    MOCK_METHOD(void,  ForwardVector,  (std::span<float> output, std::span<const float> input), (const, override))
    MOCK_METHOD(void,  BackwardVector, (std::span<float> result, std::span<const float> preActivation,
                                        std::span<const float> output, std::span<const float> outputGradient), (const, override))

template<typename L> concept HasBackward      = requires(L& l, const typename L::OutputVector& g) { l.Backward(g); }
template<typename L> concept HasSetParameters = requires(L& l, const typename L::WeightVector& p) { l.SetParameters(p); }

class TestFlashResidentWeights : public ::testing::Test:
    using FlashLayer   = neural_network::FlashDense<float, 3, 2>
    using InputVector  = FlashLayer::InputVector
    using WeightVector = FlashLayer::WeightVector

    StrictMock<ActivationFunctionMock> activationMock
    neural_network::Tanh<float>      tanhActivation
    neural_network::LeakyReLU<float> leakyRelu{ 0.1 }
    const InputVector x1{ 0.3, -0.7, 1.1 }
    const InputVector x2{ 1.0,  2.0, 3.0 }
# each case below is a TEST_F(TestFlashResidentWeights, <name>)
# the only collaborator is the ActivationFunction ⇒ StrictMock<ActivationFunctionMock> where the call contract is asserted;
# concrete Tanh / LeakyReLU / Softmax (already tested) where values are asserted
# no finite-difference case: FlashDense has no Backward (asserted at compile time below); nothing to check
```

## Test cases (Arrange / Act / Assert)

```text
ForwardPassesAffinePreActivationToActivationOnce:
    Arrange: FlashLayer layer{ flashTheta, activationMock }
             EXPECT_CALL(activationMock, ForwardVector(_, _)).WillOnce(Invoke([&](span<float> out, span<const float> in):
                 record out.data(), in.data(), in[0], in[1];  out[0] = in[0];  out[1] = in[1] ))
             # StrictMock: no scalar Forward/Backward and no BackwardVector call is allowed
    Act:     layer.Forward(x1)
    Assert:  recorded in ≈ (0.7, -0.99)                    # z = W x1 + b, EXPECT_NEAR 1e-6
             recorded out.data() == &layer.Output()[0]     # writes straight into the output buffer
             recorded in.data()  != recorded out.data()    # z is a separate buffer (no aliasing)
             layer.Output() ≈ (0.7, -0.99)

ForwardMatchesDenseWithSameParameters:
    Arrange: FlashLayer flash{ flashTheta, tanhActivation }
             neural_network::Dense<float, 3, 2> dense{ Dense<float, 3, 2>::WeightMatrix{}, tanhActivation }
             dense.SetParameters(flashTheta)                                   # RAM copy of the same θ
    Act:     flash.Forward(x1);  dense.Forward(x1)
    Assert:  flash.Output() ≈ (0.6043678, -0.7573623)                         # math::Tolerance<float>()
             |flash.Output()[i] − dense.Output()[i]| ≤ 1e-6   for i in 0..1   # same θ layout, same summation order

InferenceOnlyViewWithoutWeightCopy:
    Arrange: static_assert(flashTheta[6] == 0.2f)                              # constant-initialised ⇒ .rodata, no start-up copy
             static_assert(FlashLayer::WeightCount == neural_network::Dense<float, 3, 2>::ParameterSize)
             static_assert(std::is_base_of_v<InferenceLayer<float, 3, 2>, FlashLayer>)
             static_assert(!HasBackward<FlashLayer> && !HasSetParameters<FlashLayer>)
             static_assert(!std::is_constructible_v<FlashLayer, WeightVector&&, const ActivationFunction<float>&>)   # temporaries rejected
             static_assert(sizeof(FlashDense<float, 64, 64>) <= sizeof(math::Vector<float, 64>) + 4 · sizeof(void*))   # 280 ≤ 288 on x86-64
             FlashLayer layer{ flashTheta, tanhActivation }
    Act:     const auto view = layer.Parameters()
    Assert:  view.data() == &flashTheta[0]                                     # EXPECT_EQ: references flash, no copy
             view.size() == 8

ConcreteActivationStoredByValueAndSoftmaxSumsToOne:
    Arrange: FlashDense<float, 3, 2, Softmax<float>> layer{ flashTheta, Softmax<float>{} }   # temporary activation is safe: copied by value
    Act:     layer.Forward(x2)                                                  # z = (0.8, -0.5)
    Assert:  layer.Output() ≈ (0.785835, 0.214165)                              # = softmax(0.8, -0.5)
             layer.Output()[0] + layer.Output()[1] ≈ 1                         # EXPECT_NEAR 1e-6

SpanConstructorViewsRawFlashRegion:
    Arrange: FlashLayer layer{ FlashLayer::Weights{ std::span<const float, 8>{ flashRegion } }, leakyRelu }
    Act:     layer.Forward(x2)                                                  # z = (0.8, -0.5)
    Assert:  layer.Output() ≈ (0.8, -0.05)                                      # LeakyReLU α = 0.1
             layer.Parameters().data() == &flashRegion[0]
```

Integration cases, deployed as `TEST_F(TestModel, …)` in `neural_network/model/test/TestModel.cpp`
(the layer test target does not link `neural_network.model`). They reuse the existing `TestModel`
fixture: LeakyReLU 0.1 hidden, tanh output, `input = (2, 0.5)`, output `-0.8298019`. The fixture's
weight matrices become flat constants:

```text
inline constexpr math::Vector<float, 9> hiddenTheta{ 0.5, -1.0, 1.5, 0.25, -0.5, 0.75, 0.0, 0.0, 0.0 }   # W₁ row-major, b₁ = 0
inline constexpr math::Vector<float, 4> outputTheta{ 1.0, -0.5, 2.0, 0.0 }                              # W₂, b₂ = 0
template<typename M> concept ModelHasBackward      = requires(M& m, const typename M::OutputVector& g) { m.Backward(g); }
template<typename M> concept ModelHasGetParameters = requires(const M& m) { m.GetParameters(); }

FlashModelForwardMatchesDenseModel:
    Arrange: using FlashModel = Model<float, 2, 1, FlashDense<float, 2, 3>, FlashDense<float, 3, 1>>
             static_assert(FlashModel::TotalParameters == 0)
             static_assert(!ModelHasBackward<FlashModel> && !ModelHasGetParameters<FlashModel>)
             FlashModel flashModel{ make_layer<FlashDense<float, 2, 3>>(hiddenTheta, leakyRelu),
                                    make_layer<FlashDense<float, 3, 1>>(outputTheta, tanhActivation) }
    Act:     y = flashModel.Forward(input)[0]
    Assert:  y ≈ -0.8298019                                     # same as ForwardComposesDenseLayers
             |y − model.Forward(input)[0]| ≤ 1e-6               # the fixture's Dense model

MixedModelExposesOnlyTrainableParameters:
    Arrange: using MixedModel = Model<float, 2, 1, FlashDense<float, 2, 3>, Dense<float, 3, 1>>
             static_assert(MixedModel::TotalParameters == 4)
             static_assert(!ModelHasBackward<MixedModel> && ModelHasGetParameters<MixedModel>)
             MixedModel mixed{ make_layer<FlashDense<float, 2, 3>>(hiddenTheta, leakyRelu),
                               make_layer<Dense<float, 3, 1>>(outputWeights, tanhActivation) }
    Assert:  mixed.GetParameters() ≈ (1.0, -0.5, 2.0, 0.0)      # only the Dense slice; the flash layer adds nothing
             mixed.Forward(input)[0] ≈ -0.8298019
    Act:     mixed.SetParameters({ 0.5, 0.5, 0.5, 0.1 })
    Assert:  mixed.Forward(input)[0] ≈ 0.9546031               # tanh(0.5·(0.5 + 3.125 − 0.0625) + 0.1) = tanh(1.88125)
             mixed.GetParameters() ≈ (0.5, 0.5, 0.5, 0.1)
```

## Reference vectors

Computed by [`reference.py`](reference.py) (float32 at every operation, same
summation order as the layer). The closed forms were cross-checked in float64. The same values were
reproduced by a C++ prototype of `FlashDense` built against the current `Dense.hpp`, `Tanh`,
`LeakyReLU` and `Softmax` (`proto.cpp`), and the mock and span cases were run as gtest cases
(`mocktest.cpp`, 2/2 passed).

| Quantity                                              | Value                                                                 |
|-------------------------------------------------------|-----------------------------------------------------------------------|
| `z = W x1 + b`, `x1 = (0.3, -0.7, 1.1)`               | `(0.7, -0.99)`                                                        |
| `tanh(z)` (FlashDense and Dense, equal)               | `(0.6043678, -0.7573623)`; float64 `(0.60436778, -0.75736232)`        |
| `z = W x2 + b`, `x2 = (1, 2, 3)`                      | `(0.8, -0.5)`                                                         |
| `softmax(0.8, -0.5)`                                  | `(0.785835, 0.214165)`, sum `1.0`; float64 `(0.78583498, 0.21416502)` |
| `LeakyReLU₀.₁(0.8, -0.5)`                             | `(0.8, -0.05)`                                                        |
| Model hidden `z`, `a` (LeakyReLU 0.1)                 | `(0.5, 3.125, -0.625)`, `(0.5, 3.125, -0.0625)`                       |
| Model output `z`, `y = tanh(z)`                       | `-1.1875`, `-0.8298019` (float64 `-0.82980191`)                       |
| Mixed model after `SetParameters(0.5, 0.5, 0.5, 0.1)` | `z = 1.88125`, `y = 0.9546031` (float64 `0.95460316`)                 |
| `sizeof(FlashDense<float, 64, 64>)` on x86-64         | `280 B` (abstract reference or `Tanh` by value)                       |
| `sizeof(Dense<float, 64, 64>)` on x86-64              | `34 328 B`                                                            |

## Edge cases

- No finite-difference check exists because no `Backward` exists. `!HasBackward<FlashLayer>` and
  `!ModelHasBackward<FlashModel>` pin that at compile time. If N16 adds a frozen-layer VJP, its
  finite-difference check goes with it.
- LeakyReLU kinks: every hidden pre-activation (`0.5, 3.125, -0.625`) and `z = (0.8, -0.5)` are at least
  `0.5` from `0`, so no reference value sits on a kink.
- Dangling views: temporaries are rejected by the deleted `&&` overloads (asserted). A non-static local
  `Vector` outliving its scope is undefined behaviour and cannot be unit-tested.
- Section placement (`.rodata` → flash) is a linker property. The unit test pins only its
  precondition, constant initialisation, with a `static_assert`. An optional CI step on the
  `qemu-cortex-m4` ELF may check the exported symbols' section with `arm-none-eabi-objdump -t`.
- Explicit instantiation of the all-flash `Model` (coverage `Model.cpp`) skips `Backward`,
  `Get/SetParameters`, `Train` and `Gradients` because their constraints fail. It compiles (GCC 13
  prototype) and needs no runtime test.
- `Output()` before any `Forward` returns the value-initialised zeros. This is not unit-tested, the
  same as for `Dense`.
