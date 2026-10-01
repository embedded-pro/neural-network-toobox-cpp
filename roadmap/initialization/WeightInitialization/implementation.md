# Weight Initialisation (Glorot, He, LeCun) + Deterministic PRNG — Implementation Pseudocode

> Roadmap ref: #N3 (Tier 1) · Target: `neural_network/initialization` (new module) · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```text
template<typename T>                           # static_assert(std::is_floating_point_v<T>); instantiated for float
class Xorshift32:
    std::uint32_t state                        # never 0 (0 is the one fixed point of the recurrence)
    static constexpr std::uint32_t defaultState = 2463534242   # Marsaglia's example seed
    # 4 bytes. A value type; copy it to fork a reproducible sub-stream.

enum class FanMode : std::uint8_t:       FanIn, FanOut, FanAverage
enum class WeightDistribution : std::uint8_t:  Uniform, Normal

template<typename T>
struct VarianceScaling:                        # Var[w] = scale / fan   (the one formula behind every preset)
    T                  scale
    FanMode            mode
    WeightDistribution distribution

template<typename T>
class WeightInitializer:
    VarianceScaling<T> configuration
    Xorshift32<T>      generator               # one stream; successive calls continue it
```

Presets (the `scale`/`mode` pairs are the standard variance-scaling table):

| Preset                    | `scale`      | `mode`       | Var[w]              | Uniform limit `L`      | Normal `σ`             |
|---------------------------|--------------|--------------|---------------------|------------------------|------------------------|
| `GlorotUniform/Normal`    | `1`          | `FanAverage` | `2/(n_in + n_out)`  | `√(6/(n_in + n_out))`  | `√(2/(n_in + n_out))`  |
| `HeUniform/Normal(a = 0)` | `2/(1 + a²)` | `FanIn`      | `2/((1 + a²) n_in)` | `√(6/((1 + a²) n_in))` | `√(2/((1 + a²) n_in))` |
| `LeCunUniform/Normal`     | `1`          | `FanIn`      | `1/n_in`            | `√(3/n_in)`            | `√(1/n_in)`            |

`a` is the LeakyReLU negative slope (`a = 0` is plain ReLU). `FanOut` is available for a custom
`VarianceScaling` (He's backward-preserving "fan_out" mode) but no preset uses it.

## Interface

```text
# Xorshift32<T>
explicit Xorshift32(std::uint32_t seed)        # state = fmix32(seed), or defaultState when that is 0
std::uint32_t Next()                           # Marsaglia xor32 (13, 17, 5); hot path
T             Uniform()                        # [0, 1),  one Next()
T             Normal()                         # N(0, 1), exactly two Next() (Box–Muller, cosine branch)
static constexpr T UnitFromBits(std::uint32_t bits)          # (bits >> 8) · 2⁻²⁴        ∈ [0, 1 − 2⁻²⁴]
static constexpr T PositiveUnitFromBits(std::uint32_t bits)  # ((bits >> 8) + 1) · 2⁻²⁴  ∈ [2⁻²⁴, 1]

# VarianceScaling<T>
static constexpr VarianceScaling GlorotUniform()
static constexpr VarianceScaling GlorotNormal()
static constexpr VarianceScaling HeUniform(T negativeSlope = T{ 0 })
static constexpr VarianceScaling HeNormal(T negativeSlope = T{ 0 })
static constexpr VarianceScaling LeCunUniform()
static constexpr VarianceScaling LeCunNormal()
T StandardDeviation(std::size_t fanIn, std::size_t fanOut) const   # √(scale / fan)
T UniformLimit(std::size_t fanIn, std::size_t fanOut) const        # √(3 · scale / fan) = √3 · σ

# WeightInitializer<T>
WeightInitializer(const VarianceScaling<T>& configuration, std::uint32_t seed)
template<std::size_t Rows, std::size_t Cols>
math::Matrix<T, Rows, Cols> Weights()          # Dense layout: Rows = OutputSize = fan_out, Cols = InputSize = fan_in
void Fill(std::span<T> values, std::size_t fanIn, std::size_t fanOut)   # generic: conv/recurrent fans; hot path
```

`Weights<Out, In>()` returns exactly `Dense<T, In, Out>::WeightMatrix`, so a layer is built with
`Dense<float, In, Out>{ initializer.Weights<Out, In>(), activation }`. Dense already zeroes its
biases, which is the bias rule of every preset here. No `Layer` interface change is needed.

## Algorithm (pseudocode)

```text
function fmix32(h):                             # MurmurHash3 finaliser: a bijection on uint32, fmix32(0) = 0
    h ^= h >> 16;  h *= 0x85EBCA6B
    h ^= h >> 13;  h *= 0xC2B2AE35
    h ^= h >> 16
    return h

function Xorshift32(seed):
    state = fmix32(seed)                        # decorrelates neighbouring seeds (raw xorshift is GF(2)-linear)
    if state == 0: state = defaultState         # only seed 0 reaches this branch

function Next():                                # OPTIMIZE_FOR_SPEED
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    return state                                # period 2³² − 1 over the non-zero states

function UnitFromBits(bits):                    # top 24 bits: exact in float, never reaches 1
    return T(bits >> 8) * (T{1} / T{16777216})

function PositiveUnitFromBits(bits):            # never 0, so Log() below is finite
    return T((bits >> 8) + 1) * (T{1} / T{16777216})

function Uniform():
    return UnitFromBits(Next())

function Normal():                              # Box–Muller; the sine branch is discarded (no cached spare)
    u1 = PositiveUnitFromBits(Next())           # (0, 1]
    u2 = UnitFromBits(Next())                   # [0, 1)
    r  = math::Sqrt(T{-2} * math::Log(u1))      # u1 = 1 ⇒ r = 0 exactly
    return r * math::Cos(T{2} * std::numbers::pi_v<T> * u2)

function Fan(mode, fanIn, fanOut):
    FanIn      → T(fanIn)
    FanOut     → T(fanOut)
    FanAverage → T(fanIn + fanOut) / T{2}

function StandardDeviation(fanIn, fanOut):
    return math::Sqrt(scale / Fan(mode, fanIn, fanOut))

function UniformLimit(fanIn, fanOut):           # U(−L, L) has variance L²/3
    return math::Sqrt(T{3} * scale / Fan(mode, fanIn, fanOut))

function HeNormal(a):                           # same body for HeUniform with distribution = Uniform
    return { T{2} / (T{1} + a * a), FanIn, Normal }

function Fill(values, fanIn, fanOut):           # OPTIMIZE_FOR_SPEED
    really_assert(fanIn > 0 and fanOut > 0)
    if distribution == Uniform:
        L = UniformLimit(fanIn, fanOut)
        for i in 0..values.size()-1:
            values[i] = L * (T{2} * generator.Uniform() - T{1})     # 2u − 1 is exact: [−1, 1 − 2⁻²³]
    else:
        σ = StandardDeviation(fanIn, fanOut)
        for i in 0..values.size()-1:
            values[i] = σ * generator.Normal()

function Weights<Rows, Cols>():
    math::Matrix<T, Rows, Cols> w{}
    Fill(span over w's Rows·Cols row-major elements, fanIn = Cols, fanOut = Rows)
    return w                                    # element (i, j) is draw i·Cols + j = Dense parameter index
```

Math. Initialisation is not part of the computational graph, so there is **no `Backward`** and no
parameter gradient. The "forward/backward" content is the variance argument that fixes `Var[w]`.
For one Dense layer `z_k = Σ_{j=1}^{n_in} w_kj x_j` (`b = 0`), with `w` i.i.d., zero mean and
independent of `x`:

```text
forward:   Var[z_k]       = n_in  · Var[w] · E[x_j²]
backward:  Var[∂L/∂x_j]   = n_out · Var[w] · Var[δ_k]          # ∂L/∂x_j = Σ_k w_kj δ_k,  δ = ∂L/∂z

linear / tanh / sigmoid-centre (f'(0) = 1, E[x²] ≈ Var[z_prev]):
    forward preserved  ⇔ n_in  Var[w] = 1           # LeCun
    backward preserved ⇔ n_out Var[w] = 1
    compromise (mean fan):  Var[w] = 2 / (n_in + n_out)             # Glorot & Bengio eq. (12)
ReLU (z_prev symmetric ⇒ E[x²] = ½ Var[z_prev]):
    ½ n_in Var[w] = 1  ⇒  Var[w] = 2 / n_in                          # He et al. eq. (10)
LeakyReLU slope a (E[x²] = ½ (1 + a²) Var[z_prev]):
    ½ (1 + a²) n_in Var[w] = 1  ⇒  Var[w] = 2 / ((1 + a²) n_in)      # He et al. eq. (15)
SELU (N6): the self-normalising fixed point assumes ω = Σ_j w_kj ≈ 0 and τ = Σ_j w_kj² ≈ 1,
    i.e. E[w] = 0 and n_in Var[w] = 1  ⇒  LeCun normal                # Klambauer et al. §"Initialization"
```

Distribution mapping for a target variance `v = scale / fan`:

```text
Uniform:  w = L (2u − 1),  u ~ U[0, 1)           ⇒  E[w] = −L·2⁻²⁴ ≈ 0,  Var[w] = L²/3 = v   ⇔  L = √(3v)
Normal:   w = σ z,  z = √(−2 ln u1) cos(2π u2)   ⇒  z ~ N(0, 1)  (Box & Muller 1958),  σ = √v
          with u1 ≥ 2⁻²⁴ the tail is capped at |z| ≤ √(48 ln 2) = 5.7681 (P(|z| > 5.77) ≈ 8·10⁻⁹)
```

## Complexity & memory

- `Next`: 3 shifts + 3 XORs, `O(1)`, no multiply, no branch.
- `Uniform`: `Next` + one int→float conversion + one multiply.
- `Normal`: two `Next` + one `Log` + one `Sqrt` + one `Cos` + ~4 multiplies. It is the costly path
  (libm `logf`/`cosf` dominate), but it runs once at set-up, never per inference.
- `Fill`: `O(n)` for `n` values — one `Uniform` or one `Normal` each, plus one `Sqrt` and one divide
  for the scale, hoisted out of the loop.
- RAM: `Xorshift32` = 1 `uint32` (4 B). `WeightInitializer` = 1 float + 2 one-byte enums + 1 `uint32`
  ≈ **12 B** with padding. `Weights<R, C>()` returns `R·C` floats by value, a **transient** stack copy
  that the `Dense` constructor copies into its own parameters. There is no heap, no table and no
  recursion.

## Numerical / embedded notes

- **Why a private PRNG.** `std::mt19937` is specified bit-exactly, but `std::normal_distribution` and
  `std::uniform_real_distribution` are implementation-defined, so the simulator's current He init
  (`std::normal_distribution<float>`) gives different weights under libstdc++, libc++ and MSVC.
  The integer stream and the `UnitFromBits` mapping here are fully specified.
- **Bit-exactness.** The integer stream is identical on every target. `(bits >> 8)` has at most 24 bits,
  so the int→float conversion is exact, the `2⁻²⁴` multiply is exact and `2u − 1` is exact. The
  uniform weights round only once (`L · (2u − 1)`), and `L` needs one divide and one `sqrt`, both
  correctly rounded under IEEE-754. `-ffast-math` may replace them with reciprocal approximations on
  some targets, so the tests compare with a tolerance. Normal weights also depend on libm
  `logf`/`cosf`, which can differ by about 1 ulp between glibc and newlib. Prefer the uniform
  presets when a bit-stable reference is needed.
- **Seed scrambling.** Raw xorshift32 is linear over GF(2). From state 1 the first output is `270369`
  (`u ≈ 6.3·10⁻⁵`, so the first weight is almost exactly `−L`), and state 2 gives exactly twice that
  (`540738`). The `fmix32` finaliser removes both effects: seed 1 gives first output `524866043`. It
  is a bijection with `fmix32(0) = 0`, so only seed 0 needs the `defaultState` fallback. Seed 0 then
  shares its stream with the one non-zero seed whose `fmix32` image is `2463534242`. That collision
  is harmless and is documented.
- **Quality.** xorshift32 has period `2³² − 1` and fails some BigCrush tests, mostly in its low bits.
  Only the top 24 bits are used. That is ample for initialisation (one stream of at most a few
  thousand draws) but not for cryptography or high-precision Monte Carlo.
- **No truncation.** Normal presets are plain Gaussians, as in He et al., Klambauer et al. and PyTorch
  `kaiming_normal_`/`xavier_normal_`. Keras instead draws a 2σ-truncated normal with σ divided by
  `0.87962566`. Neither framework's weights are reproduced bit-for-bit, since the PRNG differs, so a
  trained model is imported through N13 and never re-initialised.
- **Evaluation order.** One `WeightInitializer` shared by several layers hands out consecutive
  sub-streams in call order. Build models with brace-init (`Model{ make_layer<…>(init.Weights<…>(), …), … }`),
  whose initializer clauses are evaluated left to right, or compute the matrices into named `const`s
  first. A parenthesised call leaves the argument order unspecified, and with it which layer gets which weights.
- **Choosing a preset.** Use Glorot for tanh/sigmoid/softmax-logit layers, He for ReLU and He with
  `negativeSlope = α` for LeakyReLU(α). Use LeCun normal for SELU (N6) and for a linear/Identity output
  (N1). The simulator's tanh hidden layer currently uses He; the fix is `GlorotUniform`.
- Biases stay `0` (Dense's constructor). The layer-specific exception, the LSTM forget-gate bias `1`,
  belongs to N21 and is not set here.
- Float-only: `static_assert(std::is_floating_point_v<T>)` in all three templates; the generic `T`
  signature keeps a `Q15`/`Q31` specialisation cheap to add later. The integer generator itself does
  not depend on `T`, only its float outputs do.

## Dependencies

- Builds on: `numerical/math/Math.hpp` (`math::Sqrt`, `math::Log`, `math::Cos`),
  `numerical/math/Matrix.hpp` (`math::Matrix`, row-major `at(i, j)`),
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`, `<span>`, `<numbers>`, `<cstdint>`.
  It uses no other N-item and does not touch `Layer`/`Dense` (it fits the current
  `Dense(const WeightMatrix&, const ActivationFunction<T>&)` constructor as is).
- Used by: N6 (SELU needs `LeCunNormal`), N16 (training from scratch; its sample shuffle reuses
  `Xorshift32::Next`), N11/N18/N19 (convolution kernels via `Fill` with `fanIn = K·C_in`,
  `fanOut = K·C_out`; 2D: `K_h·K_w·C`), N17/N20/N21 (recurrent input/recurrent matrices per gate via
  `Fill`; orthogonal recurrent init is deferred), and the simulator (replaces `std::mt19937` +
  `std::normal_distribution` in `NnSimulator.cpp`).
- Not in scope: dropout masks (deferred in the roadmap), orthogonal / identity recurrent init, and a
  bounded-integer draw (N16 adds one if its shuffle needs it). The generator may move upstream to
  `numerical/math` if another numerical module needs it.

## Deployment

- Header: `neural_network/initialization/WeightInitialization.hpp`. Order: `#pragma once`, then
  `#pragma GCC optimize("O3","fast-math")`, then the includes `numerical/math/Math.hpp`,
  `numerical/math/Matrix.hpp`, `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`,
  `<cstdint>`, `<numbers>` and `<span>`. Put `OPTIMIZE_FOR_SPEED` on `Next`/`Fill`. Under
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD` declare `extern template class Xorshift32<float>;`,
  `extern template struct VarianceScaling<float>;` and `extern template class WeightInitializer<float>;`.
- Coverage: `neural_network/initialization/WeightInitialization.cpp` → `template class Xorshift32<float>;`,
  `template struct VarianceScaling<float>;`, `template class WeightInitializer<float>;` and
  `template math::Matrix<float, 2, 4> WeightInitializer<float>::Weights<2, 4>();`. The member template
  is not covered by the class instantiation.
- Test: `neural_network/initialization/test/TestWeightInitialization.cpp`.
- Doc: `doc/initialization/WeightInitialization.md` (per `doc/TEMPLATE.md`) in a new `doc/initialization/`
  folder with its `README.md`. Add the rows as `roadmap/DEPLOYMENT.md` step 6 describes. Also change
  `doc/layer/Dense.md` ("The caller supplies the initial weights") to point to it.
- CMake (new module): `neural_network/initialization/CMakeLists.txt` with
  `neural_network_add_header_library(neural_network.initialization)`, the same `target_include_directories`
  as `layer`, links `infra.util` + `numerical.math`, `WeightInitialization.hpp` →
  `target_sources(neural_network.initialization PRIVATE ...)`, `WeightInitialization.cpp` →
  `neural_network_add_coverage_sources(neural_network.initialization ...)`, `add_subdirectory(test)`.
  `test/CMakeLists.txt`: `neural_network.initialization_test` (`emil_build_for` on
  `NEURAL_NETWORK_TOOLBOX_BUILD_TESTS`, `emil_add_test`, `neural_network_link_qemu_runtime`), linking
  `gmock_main`, `neural_network.initialization`, `neural_network.layer`, `neural_network.activation`
  (for the Dense integration case). Register `add_subdirectory(initialization)` in
  `neural_network/CMakeLists.txt`, append `neural_network.initialization` to `NEURAL_NETWORK_TOOLBOX_INSTALL_TARGETS`
  in the root `CMakeLists.txt`, and append `neural_network.initialization_test` to the `targets` of both
  `qemu-cortex-m4-RelWithDebInfo` and `qemu-cortex-m7-RelWithDebInfo` in `CMakePresets.json`.
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
