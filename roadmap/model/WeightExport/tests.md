# Weight Export + Blob (De)serialisation — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
# anonymous namespace
using HiddenLayer    = neural_network::Dense<float, 2, 3>
using OutputLayer    = neural_network::Dense<float, 3, 1>
using ModelType      = neural_network::Model<float, 2, 1, HiddenLayer, OutputLayer>
using Blob           = neural_network::WeightBlob<float, HiddenLayer, OutputLayer>
using OtherShapeBlob = neural_network::WeightBlob<float, neural_network::Dense<float, 1, 2>,
                                                         neural_network::Dense<float, 2, 3>>
namespace golden = neural_network_test::weight_export_golden     # WeightExportGolden.hpp, written by the exporter:
                                                                 #   layoutHash, layer0 (9), layer1 (4), blob (76 B, alignas(4))

class LayerMock : public neural_network::Layer<float, 3, 2, 8>:
    MOCK_METHOD(void, Forward, (const InputVector&), (override))
    MOCK_METHOD(const InputVector&, Backward, (const OutputVector&), (override))
    MOCK_METHOD(const OutputVector&, Output, (), (const, override))
    MOCK_METHOD(const ParameterVector&, Parameters, (), (const, override))
    MOCK_METHOD(void, SetParameters, (const ParameterVector&), (override))

class TestWeightExport : public ::testing::Test:
    neural_network::LeakyReLU<float> leakyRelu{ 0.1 }
    neural_network::Tanh<float>      tanhActivation
    ModelType source{ make_layer<HiddenLayer>(HiddenLayer::WeightMatrix{}, leakyRelu),
                      make_layer<OutputLayer>(OutputLayer::WeightMatrix{}, tanhActivation) }
    ModelType target{ same two factories }                              # all parameters 0
    const ModelType::ParameterVector theta{ 0.5, -1.0, 1.5, 0.25, -0.5, 0.75, 0.1, -0.2, 0.05,   # layer 0: W₁ row-major, b₁
                                            1.0, -0.5, 2.0, -0.3 }                                # layer 1: W₂, b₂
    const ModelType::InputVector input{ 2.0, 0.5 }
    SetUp(): source.SetParameters(theta)

    std::array<std::uint8_t, Blob::Size> Mutated(std::size_t index, std::uint8_t value):
        copy = golden::blob;  copy[index] = value;  return copy
# each case below is a TEST_F(TestWeightExport, <name>)
# collaborators: the layer in LoadLayerParameters ⇒ StrictMock<LayerMock>; everything else uses the real
# Dense/Model (already tested), as TestModel does
# no finite-difference case: this item adds no Backward (see "Edge cases")
```

## Test cases (Arrange / Act / Assert)

```
LayoutConstantsAreFixedAtCompileTime:
    Assert: static_assert(Blob::ParameterCount == 13 && Blob::Size == 76)
            static_assert(Blob::PayloadOffset<0> == 20 && Blob::PayloadOffset<1> == 56)
            static_assert(Blob::LayoutHash == 0x81EF2FF9u)
            static_assert(OtherShapeBlob::Size == Blob::Size && OtherShapeBlob::LayoutHash == 0x4805B488u)
            static_assert(golden::layoutHash == Blob::LayoutHash)            # exporter and C++ hash agree

SaveWritesGoldenBlob:
    Arrange: std::array<std::uint8_t, Blob::Size> bytes{}
    Act:     Blob::Save(source, infra::MakeByteRange(bytes))
    Assert:  bytes == golden::blob                                           # all 76 bytes, header + payload + CRC 0x18B403EB

OpenedBlobLoadsModelLayerByLayer:
    Act:     view = Blob::Open(infra::MakeByteRange(golden::blob))
             view->LoadInto(target)
    Assert:  view.has_value()
             target.GetParameters()[k] == theta[k]   for k in 0..12         # EXPECT_EQ: bit-exact
             EXPECT_NEAR(target.Forward(input)[0], -0.8558174f, math::Tolerance<float>())

ValidateReportsFirstFailingCheck:
    # Validate(x) ≡ Blob::Validate(infra::MakeByteRange(x))
    Assert:  Validate(golden::blob)                                  == ok
             Blob::Validate(infra::Head(infra::MakeByteRange(golden::blob), 75)) == sizeMismatch   # truncated
             Validate(Mutated(0, 0x4F))                              == badMagic            # 'N' → 'O'
             Validate(Mutated(4, 0x02))                              == unsupportedVersion  # formatVersion 2
             Validate(Mutated(6, 0x02))                              == unsupportedValueType
             Validate(Mutated(20, 0x01))                             == crcMismatch         # payload bit flip
             Validate(nanBlob)                                       == nonFiniteValue
                 # nanBlob = golden::blob with bytes 20..23 = 00 00 C0 7F (quiet NaN)
                 #           and bytes 72..75 = 01 25 7E 05 (its valid CRC 0x057E2501)
             !Blob::Open(infra::MakeByteRange(Mutated(20, 0x01))).has_value()

SameSizeDifferentShapeIsLayoutMismatch:
    Act:     status = OtherShapeBlob::Validate(infra::MakeByteRange(golden::blob))
    Assert:  status == layoutMismatch                                        # same 76 B, 1→2→3 vs 2→3→1
             !OtherShapeBlob::Open(infra::MakeByteRange(golden::blob)).has_value()

DecodeIsAlignmentIndependent:
    Arrange: std::array<std::uint8_t, Blob::Size + 1> shifted{};  copy golden::blob to shifted[1..76]
             range = infra::MakeRange(shifted.data() + 1, shifted.data() + 77)     # odd address
    Act:     view = Blob::Open(range);  std::array<float, 9> decoded{};  view->Decode<0>(decoded)
    Assert:  view.has_value()
             decoded == golden::layer0                                        # EXPECT_EQ, bit-exact

WeightsViewPointsIntoFloatStorage:
    Arrange: std::array<float, Blob::Size / 4> storage{};  std::memcpy(storage.data(), golden::blob.data(), Blob::Size)
    Act:     view = Blob::Open(infra::MakeByteRange(storage));  weights = view->Weights<1>()
    Assert:  weights.data() == storage.data() + 14                            # 56 / 4: zero copy
             weights[k] == golden::layer1[k]   for k in 0..3                  # (1, -0.5, 2, -0.300000012)
             static_assert(decltype(weights)::extent == 4)

LoadLayerParametersSetsParametersExactlyOnce:
    Arrange: StrictMock<LayerMock> layer
             const std::array<float, 8> weights{ 0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.2, -0.1 }
             EXPECT_CALL(layer, SetParameters(_)).WillOnce(Invoke([&](const auto& p): captured = p))
             # StrictMock: no Forward/Backward/Output/Parameters call is allowed
    Act:     neural_network::LoadLayerParameters(layer, std::span{ weights })
    Assert:  captured[k] == weights[k]   for k in 0..7                        # EXPECT_EQ

GeneratedHeaderLoadsSameModelAsBlob:
    Arrange: static_assert(golden::layer0[6] == 0.100000001f)                # constant-initialised ⇒ .rodata
    Act:     neural_network::LoadParameters(target, golden::layer0, golden::layer1)
    Assert:  target.GetParameters()[k] == theta[k]   for k in 0..12          # EXPECT_EQ: 9-digit literals round-trip
             EXPECT_NEAR(target.Forward(input)[0], -0.8558174f, math::Tolerance<float>())
```

`LoadParameters(target, golden::layer1, golden::layer0)` (arrays swapped) must not compile. That is pinned
by the `type_identity_t` signature and is not a runtime test.

## Exporter checks (`scripts/test-export-weights.py`, Python `unittest`, NumPy only)

The C++ side cannot see framework tensors, so the conversion rules are tested in Python. Each case is one
`unittest.TestCase` method. Tolerance is `1e-6` absolute unless stated.

```
test_keras_dense_kernel_is_transposed:
    kernel (2×3) = [[0.5, 1.5, -0.5], [-1.0, 0.25, 0.75]],  bias = (0.1, -0.2, 0.05)
    ⇒ θ = (0.5, -1.0, 1.5, 0.25, -0.5, 0.75, 0.1, -0.2, 0.05)                  # float32-equal
    torch_linear(weight = kernelᵀ, bias) gives the same θ                       # no transpose on the PyTorch path

test_batchnorm_folds_before_activation:
    Dense as above → BN(γ = (1.2, 0.5, 2.0), β = (0.1, -0.3, 0.0), μ = (0.4, 2.0, -0.5),
                        σ² = (0.25, 4.0, 0.09), ε = 1e-3) → LeakyReLU 0.1
    ⇒ a = (2.395214352, 0.249968756, 6.629935441),  b = (-0.858085741, -0.799937512, 3.314967721)
    ⇒ folded θ = (1.19760716, -2.39521432, 0.374953121, 0.0624921881, -3.31496763, 4.97245169,
                  -0.618564308, -0.84993124, 3.64646459)
    at x = (2, 0.5): unfolded BN output = folded z = (0.579042874, -0.068778902, -0.497245153)
    float32 forward of the folded θ = (0.5790428, -0.0687789, -0.49724483)   # |err| ≤ 3.2e-7 vs float64
    after LeakyReLU: (0.579042874, -0.00687789, -0.049724515)

test_batchnorm_folds_into_next_dense_after_activation:
    Dense₁ (above, LeakyReLU 0.1) → BN (same statistics) → Dense₂ W₂ = (1, -0.5, 2), c₂ = -0.3
    ⇒ W₂' = (2.39521432, -0.124984376, 13.2598705),  c₂' = 5.87181854
    at x = (2, 0.5): h = (0.6, 2.925, -0.0575),  z₂ = z₂' = 6.180925191    # float32 folded: 6.1809254
    (pre-activation compared: tanh(6.18) is saturated and would hide an error)

test_batchnorm_without_linear_neighbour_emits_affine_record:
    Dense(relu) → BN (last layer)  ⇒ records [Dense(2,3,9), Affine(3,3,6)],  θ_affine = (a, b) as above
    and one warning line

test_conv1d_kernels_permute_to_output_kernel_channel:
    Keras kernel[k][c][o] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]   (K = Cin = Cout = 2)
    Torch weight[o][c][k] = [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]   (same filters)
    ⇒ both θ_W = (1, 3, 5, 7, 2, 4, 6, 8)
    at x (L = 3, channels-last) = [[1, -1], [0.5, 2], [-0.5, 0.25]]: y = [[14.5, 17.0], [5.75, 8.0]] from θ and from kernel

test_recurrent_biases_and_gate_order:
    Torch LSTM (X = H = 1): weight_ih = (1, 2, 3, 4)ᵀ, weight_hh = (5, 6, 7, 8)ᵀ,
                            b_ih = (0.1, 0.2, 0.3, 0.4), b_hh = (1, 1, 1, 1)
    ⇒ θ = (1, 2, 3, 4, 5, 6, 7, 8, 1.1, 1.2, 1.3, 1.4)                       # one fused bias
    Keras GRU (X = H = 1, reset_after): kernel = (0.1, 0.2, 0.3) [z, r, h], recurrent = (0.4, 0.5, 0.6),
                                        bias = [[1, 2, 3], [4, 5, 6]]
    ⇒ θ = (0.2, 0.1, 0.3, 0.5, 0.4, 0.6, 2, 1, 3, 5, 4, 6)                   # r, z, n; both bias rows kept
    Keras GRU reset_after=False ⇒ ValueError

test_literals_round_trip_and_reject_non_finite:
    lit(1.0) = "1.0f",  lit(-0.0) = "-0.0f",  lit(0.1) = "0.100000001f",  lit(1e-5) = "9.99999975e-06f",
    lit(3.4028235e38) = "3.40282347e+38f",  lit(1e-40) = "0.0f",  lit(-1e-40) = "-0.0f" (subnormals flushed)
    float32(float(lit(v)[:-1])) == float32(v) for every finite, normal v above
    lit(nan), lit(inf) ⇒ ValueError

test_golden_files_are_reproduced:
    export the fixture model (θ above, namespace neural_network_test::weight_export_golden, --blob-array)
    ⇒ text == neural_network/model/test/WeightExportGolden.hpp                  # byte-for-byte
    ⇒ fnv1a32(descriptor [2, 2, 3, 9, 3, 1, 4]) == 0x81EF2FF9,  fnv1a32(b"a") == 0xE40C292C (published vector)
    ⇒ zlib.crc32(blob[0:72]) == 0x18B403EB == blob[72:76] (little-endian)
```

## Reference vectors

Computed by `scratchpad/specs/WeightExport/reference.py` (blob, hash, CRC, float32 forward in the layer's
summation order) and `exporter_reference.py` (folds, permutations, literals). Folds were done in float64 and
cross-checked in float32. A C++ prototype of `WeightBlob` against the current `Dense.hpp`, `LeakyReLU`,
`Tanh` and emil `infra::Crc32` (`proto.cpp`, GCC 13 and Clang 18) reproduced the golden bytes from `Save`,
every `Validate` status below, the unaligned `Decode`, the zero-copy view and the forward value.

| Quantity                                                  | Value                                                         |
|-----------------------------------------------------------|---------------------------------------------------------------|
| Fixture `θ` (float32 literals)                            | `0.5, -1.0, 1.5, 0.25, -0.5, 0.75, 0.100000001, -0.200000003, 0.0500000007` / `1.0, -0.5, 2.0, -0.300000012` |
| Descriptor, `LayoutHash`                                  | `[2, 2, 3, 9, 3, 1, 4]` → `0x81EF2FF9`                        |
| Same-size other shape `1→2→3`                             | `[2, 1, 2, 4, 2, 3, 9]` → `0x4805B488`, `Size = 76`           |
| FNV-1a-32 check vector                                    | `"a"` → `0xE40C292C`                                          |
| `Size`, payload offsets, CRC offset                       | `76`, `20` / `56`, `72`                                       |
| Header bytes 0–19                                         | `4E 4E 57 42 01 00 01 00 F9 2F EF 81 02 00 00 00 0D 00 00 00` |
| Payload bytes 20–71                                       | `00 00 00 3F 00 00 80 BF 00 00 C0 3F 00 00 80 3E 00 00 00 BF 00 00 40 3F CD CC CC 3D CD CC 4C BE CD CC 4C 3D 00 00 80 3F 00 00 00 BF 00 00 00 40 9A 99 99 BE` |
| CRC-32 bytes 72–75                                        | `EB 03 B4 18` (`0x18B403EB`; `zlib.crc32` and `infra::Crc32` agree; `"123456789"` → `0xCBF43926`) |
| NaN variant (bytes 20–23 `00 00 C0 7F`) valid CRC         | `0x057E2501` → bytes `01 25 7E 05`                            |
| Hidden `z`, `a` (LeakyReLU 0.1) at `x = (2, 0.5)`         | `(0.6, 2.925, -0.575)`, `(0.6, 2.925, -0.0575)`              |
| Output `z`, `y = tanh(z)`                                 | `-1.2775`, `-0.8558174` (float64 `-0.855817404`)              |
| BN fold (pre-activation) `a`, `b`                         | `(2.395214352, 0.249968756, 6.629935441)`, `(-0.858085741, -0.799937512, 3.314967721)` |
| BN fold output at `x = (2, 0.5)`                          | `(0.579042874, -0.068778902, -0.497245153)`; float32 folded `(0.5790428, -0.0687789, -0.49724483)` |
| BN fold (post-activation) `W₂'`, `c₂'`, `z₂`              | `(2.39521432, -0.124984376, 13.2598705)`, `5.87181854`, `6.180925191` |
| Conv1D `θ_W` (Keras and PyTorch)                          | `(1, 3, 5, 7, 2, 4, 6, 8)`; output `[[14.5, 17], [5.75, 8]]`  |
| LSTM fused `θ`, GRU `θ`                                   | `(1..8, 1.1, 1.2, 1.3, 1.4)`, `(0.2, 0.1, 0.3, 0.5, 0.4, 0.6, 2, 1, 3, 5, 4, 6)` |

## Edge cases

- **No Backward, so no finite-difference check.** This item adds no differentiable operation.
  Serialisation is the identity on `θ` (asserted bit-exact), and a fold is an exact reparameterisation
  (asserted on the forward value in Python). The loaded `Dense`/`Model` backward passes are already covered
  by `TestDense`/`TestModel`. If N16 fine-tunes folded weights, the gradient relation stated in
  `implementation.md` belongs to N16's plan.
- LeakyReLU kinks: the hidden pre-activations `0.6, 2.925, -0.575` and the fold outputs are all at least
  `0.068` away from `0`.
- A stale blob that is valid but was trained with a different activation passes `Validate`, because
  activations are not in the layout. This is documented, not tested.
- `Weights<I>()` on a misaligned range: `really_assert` fires. It is not unit-tested, the same as the other
  precondition asserts. `Decode` covers the misaligned case.
- `Weights<I>()` on a big-endian target does not compile (`static_assert`). `Decode` stays portable.
- `Save` into a buffer whose size is not `Size`: `really_assert` fires. It is not unit-tested.
- A header whose `layerCount`/`parameterCount` fields disagree with a matching `layoutHash` can only come
  from corruption after hashing. It returns `sizeMismatch` before the CRC. It is not tested separately,
  because the check sits in the same branch as the truncation case.
- Section placement of the header arrays (`.rodata` → flash) is a linker property. The test pins its
  precondition, constant initialisation, with a `static_assert`, as N10 does.
