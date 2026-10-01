# Flash-Resident Inference Weights — Implementation Pseudocode

> Roadmap ref: #N10 (Tier 2) · Target: `neural_network/layer` + `neural_network/model` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```
template<typename T, std::size_t Size>          # static_assert(std::is_floating_point_v<T>); instantiated for float
class FlashWeights:                              # non-owning, read-only view; reusable by N11/N17–N21
    std::span<const T, Size> values              # static extent ⇒ one pointer, no size word
    # never copies; the viewed storage must have static storage duration (a constexpr object in .rodata)

template<typename T, std::size_t InputSize, std::size_t OutputSize,
         typename Activation = ActivationFunction<T>>                 # static_assert(std::is_floating_point_v<T>)
class FlashDense final : public InferenceLayer<T, InputSize, OutputSize>:     # N8 base: Forward/Output only
    static_assert(std::is_base_of_v<ActivationFunction<T>, Activation>)
    static constexpr std::size_t WeightCount = InputSize·OutputSize + OutputSize   # == Dense<T, In, Out>::ParameterSize
    using WeightVector = math::Vector<T, WeightCount>                  # same flat layout as Dense::Parameters()
    using Weights      = FlashWeights<T, WeightCount>

    Weights      weights                         # θ = [W row-major (W_ij at i·In + j), b (at In·Out + i)], in flash
    ActivationMember activation                  # std::conditional_t<std::is_abstract_v<Activation>,
                                                 #     const Activation&,   (default: today's Dense behaviour)
                                                 #     Activation>          (concrete final type, stored by value)
    OutputVector output{}                        # a = f(z) of the last Forward — the only RAM-resident floats
    # no parameters copy, no parameterGradients, no input, no preActivation, no inputGradient, no forwardDone
```

Class diagram (N8 hierarchy, extended):

```
InferenceLayer<T, In, Out>                 Forward, Output
 ├─ FlashDense<T, In, Out, Activation>     + Parameters() → std::span<const T, In·Out+Out>   (read-only, flash)
 ├─ Layer<T, In, Out, 0>                   + Backward
 └─ Layer<T, In, Out, P>  (P > 0)          + Backward, Parameters, SetParameters, ParameterGradients, ZeroGradients
      └─ Dense<T, In, Out>                   (RAM copy of the same θ layout)
```

## Interface

```
# FlashWeights<T, Size>
constexpr explicit FlashWeights(const math::Vector<T, Size>& stored)   # views stored.begin() .. + Size
FlashWeights(const math::Vector<T, Size>&&) = delete                    # a temporary would dangle
constexpr explicit FlashWeights(std::span<const T, Size> region)       # raw flash region (N13 OTA blob)
constexpr const T&               operator[](std::size_t index) const
constexpr std::span<const T, Size> Values() const

# FlashDense<T, In, Out, Activation>
FlashDense(const WeightVector& weights, const Activation& activation)
FlashDense(const WeightVector&& weights, const Activation& activation) = delete
FlashDense(Weights weights, const Activation& activation)                # from an explicit FlashWeights view
void                          Forward(const InputVector& input) override   # hot path
const OutputVector&           Output() const override
std::span<const T, WeightCount> Parameters() const                        # the flash view itself, never a copy
# inherited: using ValueType = T; InputVector/OutputVector; InputSize/OutputSize; virtual ~InferenceLayer() = default
# deliberately absent: Backward, SetParameters, ParameterGradients, ZeroGradients, ParameterSize

# Model<T, In, Out, Layers...>  (N8 shape, relaxed so InferenceLayer types are admitted)
static constexpr std::size_t TotalParameters = (detail::ParameterSizeOf<Layers>() + ...)   # flash layers add 0
static constexpr bool        FullyTrainable  = (detail::TrainableLayerType<Layers> && ...)
OutputVector    Forward(const InputVector& input)                                  # any layer mix
void            SetParameters(const ParameterVector&)   requires(TotalParameters > 0)
ParameterVector GetParameters() const                   requires(TotalParameters > 0)
void            Train(optimizer, loss, initialParameters) requires(TotalParameters > 0)   # signature unchanged
InputVector     Backward(const OutputVector&)           requires FullyTrainable
ParameterVector Gradients() const                       requires(FullyTrainable && TotalParameters > 0)   # N8
void            ZeroGradients()                         requires(FullyTrainable && TotalParameters > 0)   # N8
```

`detail` helpers (replace `detail::is_layer_v`):

```
template<typename L> concept InferenceLayerType =
    std::is_base_of_v<InferenceLayer<typename L::ValueType, L::InputSize, L::OutputSize>, L>
template<typename L> concept TrainableLayerType =
    InferenceLayerType<L> && requires { L::ParameterSize; } &&
    std::is_base_of_v<Layer<typename L::ValueType, L::InputSize, L::OutputSize, L::ParameterSize>, L>
template<typename L> constexpr std::size_t ParameterSizeOf():
    if constexpr (TrainableLayerType<L>) return L::ParameterSize
    else                                 return 0
```

`Model` asserts `(detail::InferenceLayerType<Layers> && ...)` instead of `is_layer_v`. N8's
`static_assert(TotalParameters > 0)` becomes the `requires` clauses above, so an all-flash chain
compiles and exposes `Forward` only. `ParameterVector` stays the alias `math::Vector<T, TotalParameters>`.
With `TotalParameters == 0`, only the constrained declarations name it, so `Matrix<T, 0, 1>` is never
instantiated and its positive-dimension `static_assert` never fires (checked with GCC 13, including an
explicit instantiation of the all-flash `Model`). `make_layer<FlashDense<…>>(weights, activation)`
forwards both lvalues through `std::reference_wrapper` (the current `make_layer` contract), so the
`const WeightVector&` overload is chosen and the deleted `&&` overload is never hit.

## Algorithm (pseudocode)

```
function FlashWeights(stored):       values = std::span<const T, Size>{ stored.begin(), Size }
function FlashWeights(region):       values = region
function FlashDense(weights, act):   this.weights = Weights{ weights };  activation = act;  output = 0

function FlashDense::Forward(x):                 # OPTIMIZE_FOR_SPEED
    OutputVector z{}                             # stack, Out floats, released on return
    for i in 0..Out-1:
        s = weights[In·Out + i]                  # b_i            (flash read)
        for j in 0..In-1:
            s += weights[i·In + j] · x[j]        # W_ij x_j       (sequential flash reads, stride 1)
        z[i] = s
    activation.ForwardVector(output, z)          # distinct buffers: activations need not be alias-safe

function FlashDense::Output():       return output
function FlashDense::Parameters():   return weights.Values()

function Model::SetLayerParameters / GetLayerParameters:           # existing helpers
    if constexpr (detail::ParameterSizeOf<LayerType>() > 0): body as today
    # flash layers and zero-parameter layers own no slice of θ; the offset does not advance
```

Math (one sample):

```
forward:     z_i = b_i + Σ_{j=0}^{In−1} W_ij x_j,     a = f(z)            # identical to Dense::Forward
layout:      θ_{i·In + j} = W_ij,   θ_{In·Out + i} = b_i                   # identical to Dense::Parameters()
summation:   same order as Dense (bias first, then j ascending) ⇒ same rounding sequence;
             under -ffast-math the compiler may reassociate either loop, so agreement is ≤ 1 ulp-level, not bit-exact
```

Backward: **not provided**. The layer is inference-only, so it has no `∂L/∂θ` (the weights are
immutable) and no `∂L/∂x`. If a later item (N16 freeze mask) needs to back-propagate *through* a frozen
flash layer, for example to train an input `AffineNormalization` in front of it, the vector–Jacobian
product is

```
δ = J_f(z)ᵀ g               (element-wise: δ_i = f'(z_i) g_i; Softmax: δ = a ⊙ (g − ⟨g, a⟩))
∂L/∂x_j = Σ_i W_ij δ_i      (i.e. Wᵀ δ, W read from flash)
```

It needs `z` (Out floats, or `a` for Sigmoid/Tanh/Softmax) kept from `Forward`, plus an `In`-float
input-gradient buffer, and no parameter gradient. That is a separate `FrozenDense` decision for N16
and is not part of this item. `Model::Backward` therefore requires `FullyTrainable`.

## Complexity & memory

- `Forward`: `In·Out` MACs + the activation (`O(Out)`). It reads `In·Out + Out` floats from flash
  (sequential per row), reads `x` `Out` times from RAM, and writes `Out` floats (`z` to the stack, then
  `a` to `output`). This is the same operation count as `Dense::Forward` minus its `In`-float input
  copy and the `Out`-float `preActivation` store.
- Flash: `In·Out + Out` floats per layer, as a `.rodata` object (one per program with `inline constexpr`).
- RAM (object): `Out` floats (`output`) + one pointer (the static-extent span) + the activation (one
  reference, or the activation object by value: a vptr, plus `α` for LeakyReLU) + one vptr.
  Measured on x86-64: `sizeof(FlashDense<float, 64, 64>) = 280 B` (reference or `Tanh` by value),
  vs `sizeof(Dense<float, 64, 64>) = 34 328 B` (`2P + 2·In + 2·Out = 8576` floats + reference + `bool` + vptr).
  On a 32-bit MCU the flash layer is `256 + 3·4 = 268 B`. The 16 640 B of weights move to flash.
- RAM (transient): `Out` floats of stack for `z` during `Forward`.
- Model: `Σ_ℓ Out_ℓ` floats of RAM, because each layer keeps its `output` for the next one. For the
  2→3→1 test model: 4 floats, vs 44 floats (`28 + 16`) for the same chain built from `Dense`.
  A ping-pong activation arena (two buffers of `max Out_ℓ`, as TFLM's planner does) would go lower,
  but it needs a different `Model` data flow and is out of scope.

## Numerical / embedded notes

- **The weight object must be constant-initialised.** Declare it as
  `inline constexpr math::Vector<float, P> name{ … };`. `Matrix`'s variadic constructor is `constexpr`,
  so this is a constant expression. The object is then emitted into `.rodata`, with no start-up copy,
  and the Cortex-M linker script maps `.rodata` to flash (verified on host: `objdump -t` places such a
  vector in `.rodata`). A `const` object with a *dynamic* initialiser goes to `.bss` plus start-up code,
  which means RAM, so the tests pin constant initialisation with a `static_assert` on an element.
- **`inline constexpr`, not plain `constexpr`**, at namespace scope in a header. A non-inline `const`
  variable has internal linkage, so every translation unit that odr-uses it gets its own copy in flash.
  `inline` gives one program-wide object.
- **Lifetime.** The layer stores a view. The deleted `&&` overloads reject temporaries at compile time,
  but a non-static *local* `Vector` would still dangle once its scope ends. The contract is static
  storage duration, which N13's generated headers give by construction.
- **Flash read cost.** Flash on Cortex-M is memory-mapped, so weights are read with ordinary loads. It
  has wait states at high clock rates. The row-major, stride-1 access lets the flash accelerator's
  prefetch and line buffer (for example the STM32F4 ART accelerator) or the Cortex-M7 cache hide most of
  them. `x` (SRAM) is reused across rows. Measuring cycles is a target benchmark, not a unit test.
- **Raw regions (`std::span` constructor).** The floats must be in the target's byte order and 4-byte
  aligned (the layout `math::Vector<float, P>` has). N13's blob format guarantees both (little-endian,
  CRC-checked before the view is built).
- **Devirtualisation.** With a concrete `final` activation (for example `FlashDense<float, 64, 10, Identity<float>>`
  with N1), `activation` is a member of known dynamic type, so `ForwardVector` is a direct and inlinable
  call. This is N8's optional compile-time activation, applied to the inference layer. With the
  default `ActivationFunction<T>` the layer keeps one virtual call per `Forward`, as `Dense` does.
- **Separate `z` buffer.** `ForwardVector(output, z)` never aliases, so user activations need not be
  written for in-place use. Writing `z` straight into `output` would save `Out` stack floats but would
  impose that requirement on every activation.
- Harvard-architecture 8-bit targets (AVR `PROGMEM`) need special load instructions and are out of
  scope. The library targets Cortex-M, where `.rodata` is directly addressable.
- Float-only: `static_assert(std::is_floating_point_v<T>)` in both templates. The exported weights are
  the trained float32 values, so there is no requantisation step. The generic `T` keeps a fixed-point
  (or int8, see *Deferred* in the ROADMAP) specialisation cheap to add later.

## Dependencies

- Builds on: N8 (`InferenceLayer<T, In, Out>`; N8 explicitly hands "admitting `InferenceLayer` types
  into `Model`" to this item, and its optional compile-time activation is realised here),
  `Dense.hpp` (the `θ` layout that is reused unchanged), `ActivationFunction::ForwardVector`,
  `numerical/math/Matrix.hpp` (`constexpr` variadic constructor, `begin()` over contiguous `std::array`
  storage), `numerical/math/CompilerOptimizations.hpp`, `<span>`, and the fixed `make_layer`, which stores
  lvalue arguments as `std::reference_wrapper`.
- Used by: N13 (the exporter emits one `inline constexpr math::Vector<float, In·Out + Out>` per layer
  in the flat order, and its OTA blob path builds `FlashWeights` from a `std::span`), N11/N18/N19
  (inference-only convolutions store `FlashWeights<T, K·C_in·C_out + C_out>`), N17/N20/N21 (inference-only
  recurrent cells view `W_x`, `W_h` and `b` in flash; only the hidden state stays in RAM),
  N16 (a frozen flash prefix in front of a trainable head: `Get/SetParameters` already skip flash
  layers; propagating `Backward` only down to the first trainable layer is N16's decision).
- Pairs with: N1 (`Identity` for raw regression outputs and logits, the usual last layer), N2
  (`AffineNormalization` for input standardisation in front of a flash chain).
- Not changed: `Dense` (still the trainable, RAM-resident layer), `Model::Train`'s signature, and the
  loss contract.

## Deployment

- Header: `neural_network/layer/FlashResidentWeights.hpp`. Order: `#pragma once`, then
  `#pragma GCC optimize("O3","fast-math")`, then the includes `neural_network/layer/TrainableLayerInterface.hpp`
  (N8's `InferenceLayer`), `neural_network/activation/ActivationFunction.hpp`,
  `numerical/math/CompilerOptimizations.hpp`, `numerical/math/Matrix.hpp` and `<span>`. It holds
  `FlashWeights` and `FlashDense`. Put `OPTIMIZE_FOR_SPEED` on `FlashDense::Forward`, and add these under
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`: `extern template class FlashWeights<float, 8>;` and
  `extern template class FlashDense<float, 3, 2>;`.
- Coverage: `neural_network/layer/FlashResidentWeights.cpp` →
  `template class FlashWeights<float, 8>; template class FlashDense<float, 3, 2>;`
- `Model.hpp`: add the `detail` concepts, relax the layer check, change `TotalParameters` to
  `ParameterSizeOf`, add the `requires` clauses and the `if constexpr` guards in the parameter helpers.
  In `Model.cpp`, add `template class Model<float, 2, 1, FlashDense<float, 2, 3>, FlashDense<float, 3, 1>>;`,
  with the matching `extern template` in `Model.hpp`. Explicit instantiation skips the members whose
  constraints fail, so it builds.
- Test: `neural_network/layer/test/TestFlashResidentWeights.cpp`, plus two integration cases in
  `neural_network/model/test/TestModel.cpp` (see `tests.md`).
- Doc: `doc/layer/FlashResidentWeights.md` (per `doc/TEMPLATE.md`), with a row in `doc/layer/README.md`.
  Add one sentence in `doc/layer/Dense.md` pointing to the flash-resident variant, and one in
  `doc/model/Model.md` stating that inference-only chains expose `Forward` only.
- CMake: `FlashResidentWeights.hpp` → `target_sources(neural_network.layer PRIVATE ...)`;
  `FlashResidentWeights.cpp` → `neural_network_add_coverage_sources(neural_network.layer ...)`;
  `TestFlashResidentWeights.cpp` → `neural_network.layer_test` (already links `gmock_main` and
  `neural_network.activation`).
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
