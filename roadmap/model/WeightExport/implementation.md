# Weight Export + Blob (De)serialisation — Implementation Pseudocode

> Roadmap ref: #N13 (Tier 2) · Target: `neural_network/model` + `scripts` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

Two paths deliver trained weights to the device. Both use the flat `θ` layout that `Parameters()` already
defines, so no layer needs to change:

- **Primary path, compile time.** `scripts/export-weights.py` reads a Keras or PyTorch model and writes a
  C++ header with one `inline constexpr std::array<float, P_ℓ>` per layer. The arrays land in flash.
- **Secondary path, run time (over-the-air).** The same exporter, or `WeightBlob::Save` on the device,
  writes a versioned little-endian blob. The device validates the blob and then loads it layer by layer.

```text
# θ layout per layer (unchanged; owned by each layer's spec)
#   Dense<T, In, Out>             θ = [W row-major (W_ij at i·In + j), b]              P = In·Out + Out
#   AffineNormalization<T, N>     θ = [a, b]                                            P = 2N          (N2)
#   Convolution1D<…, Cin, Cout, K> θ = [W[o][k][c] at o·K·Cin + k·Cin + c, b]           P = K·Cin·Cout + Cout (N11)
#   recurrent cells               θ = [W_x, W_h, b…], gate blocks in PyTorch order      (N17/N20/N21, see "Exporter")

# Blob (all words little-endian u32; payload words are IEEE-754 binary32)
#   offset  size        field
#   0       4           magic          = 0x42574E4E   (bytes "NNWB")
#   4       2           formatVersion  = 1
#   6       2           valueType      = 1            (binary32)
#   8       4           layoutHash     = FNV-1a-32 of the layout descriptor (below)
#   12      4           layerCount     = L
#   16      4           parameterCount = Σ_ℓ P_ℓ
#   20      4·P_0       layer 0 θ
#   …       4·P_ℓ       layer ℓ θ          (payload of layer ℓ starts at 20 + 4·Σ_{m<ℓ} P_m ⇒ 4-byte aligned)
#   Size−4  4           crc32          = CRC-32 (zlib / ISO-HDLC) over bytes [0, Size − 4)
#   Size = 20 + 4·Σ_ℓ P_ℓ + 4
#
# Layout descriptor: the u32 sequence [L, In_0, Out_0, P_0, In_1, Out_1, P_1, …], hashed as its
# little-endian bytes. Activations are code, not data, so they are not part of the blob.

enum class BlobStatus : std::uint8_t:
    ok, sizeMismatch, badMagic, unsupportedVersion, unsupportedValueType,
    layoutMismatch, crcMismatch, nonFiniteValue

template<typename T, typename... Layers>         # static_assert(std::is_floating_point_v<T>); instantiated for float
class WeightBlob:                                # stateless: static constants + static functions
    static_assert(std::numeric_limits<T>::is_iec559 && sizeof(T) == sizeof(std::uint32_t))   # binary32 payload
    static_assert(sizeof...(Layers) > 0)
    static_assert((std::is_same_v<typename Layers::ValueType, T> && ...))

    static constexpr std::uint32_t Magic          = 0x42574E4E
    static constexpr std::uint16_t FormatVersion  = 1
    static constexpr std::uint16_t ValueType      = 1
    static constexpr std::size_t   HeaderSize     = 20
    static constexpr std::size_t   LayerCount     = sizeof...(Layers)
    static constexpr std::array<std::size_t, LayerCount> WeightCounts{ detail::WeightCountOf<Layers>()... }
    static constexpr std::size_t   ParameterCount = (detail::WeightCountOf<Layers>() + ...)
    static constexpr std::size_t   Size           = HeaderSize + sizeof(T)·ParameterCount + sizeof(std::uint32_t)
    static constexpr std::uint32_t LayoutHash     = detail::Fnv1a32(descriptor)       # evaluated at compile time
    template<std::size_t I> using LayerType = std::tuple_element_t<I, std::tuple<Layers...>>
    template<std::size_t I> static constexpr std::size_t PayloadOffset = HeaderSize + sizeof(T)·Σ_{m<I} WeightCounts[m]

    class View:                                  # proof that a byte range passed Validate; only Open() creates it
        infra::ConstByteRange blob               # pointer + size, no copy
```

`detail::WeightCountOf<L>()` returns `L::ParameterSize` for `Layer`-derived types (a `ParameterSize = 0`
pooling layer from N12 contributes an empty record, so the topology is still hashed). For N10's
`FlashDense` it returns `L::WeightCount`. `Dense<T, In, Out>` and `FlashDense<T, In, Out>` therefore have the
same descriptor, and one blob or header feeds either layer.

## Interface

```text
# WeightBlob<T, Layers...>
static BlobStatus          Validate(infra::ConstByteRange blob)                 # full check, no side effects
static std::optional<View> Open(infra::ConstByteRange blob)                     # Validate == ok ⇒ View
template<std::size_t In, std::size_t Out>
static void                Save(const Model<T, In, Out, Layers...>& model, infra::ByteRange blob)   # blob.size() == Size

# WeightBlob<T, Layers...>::View
template<std::size_t I> void Decode(std::span<T, WeightCounts[I]> destination) const   # any alignment, any endianness
template<std::size_t I> std::span<const T, WeightCounts[I]> Weights() const            # zero-copy, for N10 FlashWeights
template<std::size_t In, std::size_t Out>
void LoadInto(Model<T, In, Out, Layers...>& model) const
    requires (detail::SettableLayer<Layers> && ...)                                    # layer by layer SetParameters

# free functions (header path)
template<typename LayerType>
void LoadLayerParameters(LayerType& layer,
                         std::span<const typename LayerType::ValueType, LayerType::ParameterSize> weights)
template<typename T, std::size_t In, std::size_t Out, typename... Layers>
void LoadParameters(Model<T, In, Out, Layers...>& model,
                    std::type_identity_t<std::span<const T, Layers::ParameterSize>>... weights)
    # type_identity_t makes the spans non-deduced, so std::array<float, P_ℓ> converts; a wrong P_ℓ is a
    # compile error (checked with GCC 13 and Clang 18)

# Model<T, In, Out, Layers...>  (addition needed by LoadInto / Save / LoadParameters)
template<std::size_t I> auto&       LayerAt()        # std::get<I>(layers)
template<std::size_t I> const auto& LayerAt() const
```

`detail::SettableLayer<L>` is `requires(L& l, const typename L::ParameterVector& p) { l.SetParameters(p); }`.
A mixed chain (N10 flash prefix + trainable head, N16) does not use `LoadInto`. Its flash layers are built
from `View::Weights<I>()`, and its trainable layers are loaded with `LoadLayerParameters(model.LayerAt<I>(), …)`.

Generated header (primary path), for `--namespace app --name keywordSpotter`:

```cpp
#pragma once
#include <array>
#include <cstdint>

namespace app::keywordSpotter
{
    inline constexpr std::uint32_t layoutHash{ 0x81EF2FF9u };
    inline constexpr std::array<float, 9> layer0{ 0.5f, -1.0f, 1.5f, 0.25f, -0.5f, 0.75f, 0.100000001f, -0.200000003f, 0.0500000007f };
    inline constexpr std::array<float, 4> layer1{ 1.0f, -0.5f, 2.0f, -0.300000012f };
}
```

The application pins the header to its model type with
`static_assert(WeightBlob<float, Hidden, Output>::LayoutHash == app::keywordSpotter::layoutHash);`, then calls
`LoadParameters(model, layer0, layer1)` (RAM `Dense`) or passes `std::span{ layer0 }` to N10's
`FlashWeights` (flash `FlashDense`). With `--blob-array` the header also holds
`alignas(4) inline constexpr std::array<std::uint8_t, Size> blob{ … }`, a factory-default image in flash.

## Algorithm (pseudocode)

```text
function detail::Fnv1a32(words):                 # constexpr loop, no recursion
    h = 0x811C9DC5
    for w in words:
        for k in 0..3:
            h = (h XOR ((w >> 8k) AND 0xFF)) · 0x01000193   mod 2³²
    return h

function ReadWord(blob, offset):                 # endianness-independent, alignment-independent
    return blob[o] | blob[o+1] << 8 | blob[o+2] << 16 | blob[o+3] << 24

function WriteWord(blob, offset, w):
    for k in 0..3: blob[o + k] = (w >> 8k) AND 0xFF

function Validate(blob):                         # OPTIMIZE_FOR_SPEED; checks in this order, first failure wins
    if blob.size() < HeaderSize + 4:                            return sizeMismatch
    if ReadWord(blob, 0) != Magic:                              return badMagic
    if (ReadWord(blob, 4) AND 0xFFFF) != FormatVersion:         return unsupportedVersion
    if (ReadWord(blob, 4) >> 16) != ValueType:                  return unsupportedValueType
    if ReadWord(blob, 8) != LayoutHash:                         return layoutMismatch
    if ReadWord(blob, 12) != LayerCount
       or ReadWord(blob, 16) != ParameterCount
       or blob.size() != Size:                                  return sizeMismatch
    crc = infra::Crc32{};  crc.Update(infra::Head(blob, Size − 4))
    nonFinite = false
    for o in HeaderSize .. Size−8 step 4:
        nonFinite |= ((ReadWord(blob, o) >> 23) AND 0xFF) == 0xFF      # exponent all ones ⇒ ±Inf or NaN
    if crc.Result() != ReadWord(blob, Size − 4):               return crcMismatch
    if nonFinite:                                               return nonFiniteValue
    return ok

function Open(blob):
    return Validate(blob) == ok ? View{ blob } : std::nullopt

function View::Decode<I>(destination):          # OPTIMIZE_FOR_SPEED
    for k in 0..WeightCounts[I]−1:
        destination[k] = std::bit_cast<T>(ReadWord(blob, PayloadOffset<I> + 4k))

function View::Weights<I>():
    static_assert(infra::isLittleEndian)
    really_assert(address(blob.begin()) mod alignof(T) == 0)
    return std::span<const T, WeightCounts[I]>{ reinterpret_cast<const T*>(blob.begin() + PayloadOffset<I>), WeightCounts[I] }

function View::LoadInto(model):                 # one layer at a time; only reachable after a successful Open
    for each I in 0..LayerCount−1 (fold over index_sequence):
        typename LayerType<I>::ParameterVector p{}             # stack, P_I floats, released after the call
        Decode<I>(std::span<T, P_I>{ p.begin(), P_I })
        model.LayerAt<I>().SetParameters(p)

function Save(model, blob):                     # e.g. persist after on-device fine-tuning (N16)
    really_assert(blob.size() == Size)
    WriteWord(blob, 0, Magic);  WriteWord(blob, 4, FormatVersion | ValueType << 16)
    WriteWord(blob, 8, LayoutHash);  WriteWord(blob, 12, LayerCount);  WriteWord(blob, 16, ParameterCount)
    o = HeaderSize
    for each I: for k in 0..P_I−1:
        WriteWord(blob, o, std::bit_cast<std::uint32_t>(model.LayerAt<I>().Parameters()[k]));  o += 4
    crc = infra::Crc32{};  crc.Update(infra::Head(blob, Size − 4));  WriteWord(blob, Size − 4, crc.Result())

function LoadLayerParameters(layer, weights):
    typename LayerType::ParameterVector p{}
    for k in 0..P−1: p[k] = weights[k]
    layer.SetParameters(p)                       # exactly one call; N8: does not touch gradients

function LoadParameters(model, weights...):
    for each I: LoadLayerParameters(model.LayerAt<I>(), weights_I)
```

### Exporter (`scripts/export-weights.py`, Python 3 + NumPy)

Each framework layer is converted to a record `(In, Out, θ)` in float64. The script folds BN, rounds θ once
to float32, and then writes the header and/or the blob. The conversion functions take NumPy arrays in the
framework's shapes, so `scripts/test-export-weights.py` tests them without TensorFlow or PyTorch installed.
Only the model walk (`model.layers`, `named_children()`) imports a framework, and it does so lazily.

```text
Dense    Keras  kernel K ∈ ℝ^{In×Out}, bias c        →  W = Kᵀ (W_ij = K_ji),  θ = [vec_rowmajor(W), c]
         Torch  weight W ∈ ℝ^{Out×In}, bias c        →  θ = [vec_rowmajor(W), c]         (no transpose)
         use_bias=False / bias=None                  →  c = 0
Conv1D   Keras  kernel ∈ ℝ^{K×Cin×Cout}  (k, c, o)   →  θ[o·K·Cin + k·Cin + c] = kernel[k][c][o]     (N11)
         Torch  weight ∈ ℝ^{Cout×Cin×K}  (o, c, k)   →  θ[o·K·Cin + k·Cin + c] = weight[o][c][k]
         Torch channels-first flatten o·Lout + t before a Linear → permute that Linear's columns to t·Cout + o
LSTM     gate order i, f, g, o (Keras i, f, c, o is the same order)
         Torch  W_x = weight_ih (4H×X), W_h = weight_hh (4H×H), b = b_ih + b_hh        (one fused bias, N21)
         Keras  W_x = kernelᵀ, W_h = recurrent_kernelᵀ, b = bias
GRU      gate order r, z, n (PyTorch); Keras stores z, r, h ⇒ permute blocks (1, 0, 2)
         reset_after (Keras default, PyTorch): n = tanh(W_xn x + b_xn + r ⊙ (W_hn h + b_hn))
         ⇒ keep both bias vectors: θ = [W_x (3H×X), W_h (3H×H), b_x (3H), b_h (3H)]  (N20; b_hn cannot be folded)
         Keras bias shape (2, 3H): row 0 = b_x, row 1 = b_h;  Torch b_ih = b_x, b_hh = b_h
         Keras reset_after=False (Cho 2014 form) is rejected: N20 does not implement it
```

The recurrent block orders are the exporter's proposal. N17/N20/N21 own the final `θ` layout, and if they
choose differently only this permutation table changes, not the blob format.

BatchNorm folding (frozen statistics, `a = γ / √(σ² + ε)`, `b = β − a μ`, with `ε` read from the framework
layer: Keras default `1e-3`, PyTorch `1e-5`). The identities are the ones stated in N2:

```text
pre-activation   Dense/Conv(linear) → BN → act     W' = diag(a) W,   c' = a ⊙ c + b         (row k scaled by a_k)
                                                   Conv: W'[o][k][ch] = a_o W[o][k][ch],  c'_o = a_o c_o + b_o
post-activation  act → BN → Dense                  W' = W diag(a),   c' = c + W b           (column j scaled by a_j)
                                                   Conv (valid padding only): W'[o][k][ch] = W[o][k][ch] a_ch,
                                                   c'_o = c_o + Σ_{k,ch} W[o][k][ch] b_ch
input BN         BN → first Dense                  same as post-activation
otherwise        (BN followed by pooling, by an activation-only layer, or last in the model)
                 → emit an AffineNormalization<N> record θ = [a, b], In = Out = N, and print a warning
```

The fold is computed in float64 and rounded to float32 once. The exporter then writes one record per library
layer, as a C++ literal `'%.9g'` with `f` appended. If the digits contain none of `.`, `e` or `n`, it inserts
`.0` first, so the output reads `1.0f`, not `1f`. It flushes float32 subnormals to a signed zero and prints
their count. It refuses NaN or ±Inf with a non-zero exit code. It prints one line per record
(`Dense<float, 2, 3>  activation=leaky_relu(0.1)`) so the user can write the matching `Model` type.

Math (forward identities the exporter relies on; there is no Backward in this item):

```text
layout:        θ_{i·In + j} = W_ij,  θ_{In·Out + i} = b_i        (Keras: W_ij = K_ji)
BN (frozen):   BN(z)_k = γ_k (z_k − μ_k)/√(σ²_k + ε) + β_k = a_k z_k + b_k                       (exact)
pre-act fold:  a ⊙ (W x + c) + b = (diag(a) W) x + (a ⊙ c + b)                                  (exact)
post-act fold: W (a ⊙ h + b) + c = (W diag(a)) h + (W b + c)                                    (exact)
LSTM bias:     W_x x + b_ih + W_h h + b_hh = W_x x + W_h h + (b_ih + b_hh)                      (exact, all gates)
GRU n-gate:    r ⊙ (W_hn h + b_hn) ≠ r ⊙ (W_hn h) + b_hn          ⇒ b_hn stays separate
```

Backward: **none is introduced.** Serialisation is the identity on `θ`, and a fold is an exact
reparameterisation of the same function, so `∂L/∂x` of a folded model equals that of the source model. If N16
fine-tunes a folded layer, its gradients are taken with respect to `W'`. They relate to the source
parameters by `∂L/∂W_kj = a_k ∂L/∂W'_kj` and `∂L/∂c_k = a_k ∂L/∂c'_k` (pre-activation fold). An SGD step on
`W'` is therefore not the same as a step on `(W, BN)`. This is harmless because the BN statistics are frozen,
and it is stated here so that N16 does not assume otherwise.

## Complexity & memory

- Compile time: `LayoutHash`, `Size`, `PayloadOffset` and `WeightCounts` are constants. At run time they
  cost nothing.
- `Validate` / `Open`: `O(Size)`. It makes `Size − 4` CRC table lookups (emil's byte-wise `Crc32`) plus
  `ΣP` integer exponent tests. RAM: `O(1)`, a 4-byte CRC state plus locals. The emil CRC table
  (`256 × u32` = 1 KB) is a `static inline constexpr` object in `.rodata`. For the 2→3→1 fixture,
  `Size = 76 B`. For `Dense<float, 64, 64>` alone, `Size = 20 + 16 640 + 4 = 16 664 B`.
- `View::Decode<I>`: `P_I` word loads and bit-casts. `View::Weights<I>`: `O(1)`, zero copies.
- `View::LoadInto` / `LoadParameters`: `ΣP` decodes (or copies) plus each layer's `SetParameters` copy.
  Transient stack is `max_ℓ P_ℓ` floats, because only one layer's `ParameterVector` is live at a time.
  `Model::SetParameters` would need `ΣP`. For a `64→64→10` model that is 4 160 floats instead of 4 810.
  The persistent RAM is the layers' own storage, which is unchanged. Only N10's flash layers avoid it.
- `Save`: `ΣP` word stores, plus one CRC pass over `Size − 4` bytes. Extra floats: 0. It reads each
  layer's `Parameters()` by reference.
- `View`: one pointer and one size (2 words). `WeightBlob` has no state.
- Header path: `ΣP` floats of flash (`.rodata`, one program-wide object per array thanks to `inline`). RAM is
  0 with N10, and `ΣP` floats (the layers' own copies) with `Dense`. `LoadParameters` adds a transient
  `max_ℓ P_ℓ` floats of stack.

## Numerical / embedded notes

- **Bit-exact transport.** Blob values are stored as their binary32 bit patterns (`std::bit_cast`), so no
  decimal conversion takes place. Header literals use 9 significant digits, which is the round-trip
  precision for binary32 (IEEE 754-2019 §5.12.2), so the compiler reproduces the exporter's float32 exactly
  (`0.1f` → `0.100000001f` → the same bits). `-0.0` is preserved as `-0.0f`.
- **`-ffast-math`.** The library compiles with `-ffast-math`, and GCC may fold `std::isfinite` to `true`.
  Validate therefore detects NaN and Inf with an integer test on the exponent bits. Subnormals are flushed
  by the exporter, so host (no FTZ) and target (possibly FTZ) compute the same function.
- **Why `std::array`, not `math::Vector`.** `Matrix`'s variadic constructor has a fold expression in its
  `requires` clause. Clang 18 rejects it for more than 256 elements ("instantiating fold expression with
  N arguments exceeded expression nesting limit of 256"), which covers every real layer: `Dense<16, 16>`
  already has 272. GCC 13 accepts 4 160 elements. `std::array` aggregate initialisation compiles in about
  0.15 s for 16 640 floats on both compilers and lands in `.rodata` (checked with `objdump -t`). N10's
  `FlashWeights(std::span<const T, Size>)` constructor consumes it directly. Its `const WeightVector&`
  overload remains for small hand-written tables.
- **Aliasing.** `View::Weights` casts a byte pointer to `const T*`. This is valid only when `T` objects (or
  memory-mapped flash that holds no C++ object) are at that address: a flash staging partition, or an
  `alignas(4)` `float` array. It is never valid for a `std::uint8_t[]` object, which would be a strict-aliasing
  violation. `Decode` is always valid, because it reads bytes and builds values with `std::bit_cast`.
- **Atomic OTA.** A blob is written to a staging area and then passed to `Open`. `LoadInto` exists only on a
  `View`, so the model is never partly updated from a corrupt or mismatched blob. A receiver that applies
  chunks directly without staging cannot commit atomically, and it is out of scope.
- **What the hash does and does not guard.** The hash covers the shape of every layer (`In`, `Out`, `P`) and
  the layer count. A blob for a `1→2→3` chain has the same `Size` (76 B) as one for a `2→3→1` chain but a
  different hash (`0x4805B488` vs `0x81EF2FF9`). Activations, layer kinds with the same shape (`Dense` vs
  `FlashDense`, which is intended) and training provenance are not covered. The application should version
  its OTA package. FNV-1a detects mismatches, not tampering, and the CRC detects transmission errors only.
  Authenticity (signatures) belongs to the OTA transport.
- **CRC choice.** `infra::Crc32` is CRC-32/ISO-HDLC (reflected `0x04C11DB7`, init and final XOR
  `0xFFFFFFFF`), the same as Python's `zlib.crc32`, so the exporter needs no custom code. Hardware CRC units
  that cannot reflect input and output (for example the STM32F4 CRC peripheral: same polynomial, no
  reflection, no final XOR, fed 32-bit words) give a different value. The software table is the reference.
- **Fold precision.** The fold is done in float64 and rounded once. The float32 forward pass of the folded
  layer matches the float64 BN reference to `3.2e-7` absolute for the test vector in `tests.md`.
- Float-only: `static_assert(std::is_floating_point_v<T>)`, plus the binary32 requirement for the blob. The
  generic `T` keeps a fixed-point specialisation cheap to add later. It would need a new `valueType`
  code, not a new format.

## Dependencies

- Builds on: `Model.hpp` (adds `LayerAt<I>()`; `Train()` keeps its signature), `Layer.hpp`
  (`ValueType`, `ParameterSize`, `const ParameterVector& Parameters() const`, `SetParameters`), `Dense.hpp`
  (the `θ` layout), emil `infra/util/Crc.hpp` (`Crc32`), `infra/util/Endian.hpp` (`isLittleEndian`),
  `infra/util/ByteRange.hpp`, `infra/util/ReallyAssert.hpp`, and `<bit>` / `<span>` / `<optional>`.
- **N10** (soft, for the zero-RAM path): the header arrays and `View::Weights<I>()` feed
  `FlashWeights(std::span<const T, P>)`. Without N10, both paths load into RAM `Dense` layers.
  `detail::WeightCountOf` needs `FlashDense::WeightCount` once N10 lands.
- **N2**: the fold identities and the `AffineNormalization` fallback record.
- **N11 / N17 / N20 / N21**: their exporter rules (permutation, fused LSTM bias, GRU `reset_after` pair)
  become active as each layer lands. Dense-only export ships first.
- **N12**: zero-parameter layers contribute `(In, Out, 0)` to the descriptor and no payload.
- **N16**: `Save` persists fine-tuned weights. N8's `ParameterGradients` are not serialised.
- Used by: N7 (on-device accuracy check of an imported model), N5/N19 (importing MobileNetV3-class
  models).

## Deployment

- Header: `neural_network/model/WeightExport.hpp`. Order: `#pragma once`, then
  `#pragma GCC optimize("O3","fast-math")`, then the includes `neural_network/model/Model.hpp`,
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ByteRange.hpp`, `infra/util/Crc.hpp`,
  `infra/util/Endian.hpp`, `infra/util/ReallyAssert.hpp`, `<array>`, `<bit>`, `<cstdint>`, `<limits>`,
  `<optional>`, `<span>` and `<type_traits>`. It holds `BlobStatus`, `detail::Fnv1a32`,
  `detail::WeightCountOf`, `detail::SettableLayer`, `WeightBlob`, `LoadLayerParameters` and
  `LoadParameters`. Put `OPTIMIZE_FOR_SPEED` on `Validate`, `View::Decode` and `Save`, and add
  `extern template class WeightBlob<float, Dense<float, 2, 3>, Dense<float, 3, 1>>;` under
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`.
- `Model.hpp`: add the two `LayerAt<I>()` accessors. The coverage `Model.cpp` instantiation is unchanged
  and now also covers them.
- Coverage: `neural_network/model/WeightExport.cpp` →
  `template class WeightBlob<float, Dense<float, 2, 3>, Dense<float, 3, 1>>;`
- Test: `neural_network/model/test/TestWeightExport.cpp`, plus the exporter-generated golden
  `neural_network/model/test/WeightExportGolden.hpp` (see `tests.md`).
- Scripts: `scripts/export-weights.py` (CLI: `--framework {keras,torch} MODEL --namespace NS --name NAME
  [--header PATH] [--blob PATH] [--blob-array]`) and `scripts/test-export-weights.py` (`unittest`, NumPy
  only). CI: add `pip install numpy` and `python scripts/test-export-weights.py` to
  `.github/workflows/validate-docs.yml`, which already sets up Python 3.12. Extend its `paths:` filter with
  `scripts/*export-weights*.py` and `neural_network/model/test/WeightExportGolden.hpp`.
- Doc: `doc/model/WeightExport.md` (per `doc/TEMPLATE.md`), with a row in `doc/model/README.md`. Add one
  sentence in `doc/model/Model.md` pointing to it for deployment.
- CMake: `WeightExport.hpp` → `target_sources(neural_network.model PRIVATE ...)`;
  `WeightExport.cpp` → `neural_network_add_coverage_sources(neural_network.model ...)`; add `infra.util`
  to `neural_network.model`'s `target_link_libraries`; `TestWeightExport.cpp` → `neural_network.model_test`
  (it already links `gmock_main`, `neural_network.activation` and `neural_network.model`, and has the QEMU
  hook).
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
