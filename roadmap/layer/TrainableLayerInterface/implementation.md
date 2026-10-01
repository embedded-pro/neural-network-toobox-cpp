# Trainable-Layer Interface — Implementation Pseudocode

> Roadmap ref: #N8 (Tier 2) · Target: `neural_network/layer` + `neural_network/model` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```text
template<typename T, std::size_t InputSize_, std::size_t OutputSize_>
class InferenceLayer:                                   # static_assert(std::is_floating_point_v<T>); pure interface, no state
    using ValueType    = T
    using InputVector  = math::Vector<T, InputSize_>
    using OutputVector = math::Vector<T, OutputSize_>
    static constexpr std::size_t InputSize  = InputSize_
    static constexpr std::size_t OutputSize = OutputSize_

template<typename T, std::size_t InputSize_, std::size_t OutputSize_, std::size_t ParameterSize_>
class Layer : public InferenceLayer<T, InputSize_, OutputSize_>:     # primary template: ParameterSize_ > 0
    using ParameterVector = math::Vector<T, ParameterSize_>
    static constexpr std::size_t ParameterSize = ParameterSize_

template<typename T, std::size_t InputSize_, std::size_t OutputSize_>
class Layer<T, InputSize_, OutputSize_, 0> : public InferenceLayer<T, InputSize_, OutputSize_>:
    static constexpr std::size_t ParameterSize = 0      # parameter-free but differentiable (pooling, N12)
    # no ParameterVector: math::Vector<T, 0> fails Matrix's positive-dimension static_assert

template<typename T, std::size_t InputSize, std::size_t OutputSize>
class Dense : public Layer<T, InputSize, OutputSize, InputSize·OutputSize + OutputSize>:   # existing layer, migrated
    ParameterVector parameters                 # θ = [W row-major (W_ij at i·In + j), b]
    ParameterVector parameterGradients         # G = Σ ∇_θ L_s since the last ZeroGradients(); same layout as θ
    InputVector     input                      # x of the last Forward (needed for ∂L/∂W)
    OutputVector    preActivation, output      # z, a of the last Forward
    InputVector     inputGradient              # ∂L/∂x of the last Backward (per sample, never accumulated)
    bool            forwardDone = false
    # no new members: the gradient buffer already exists; only "=" becomes "+=" and it gets two accessors
```

Class diagram:

```text
InferenceLayer<T, In, Out>            Forward, Output                      ← N10 inference-only layers stop here
 ├─ Layer<T, In, Out, 0>              + Backward                           ← N12 pooling, activation-only layers
 └─ Layer<T, In, Out, P>  (P > 0)     + Backward (accumulating),
                                        Parameters, SetParameters,
                                        ParameterGradients, ZeroGradients  ← Dense, N2, N11, N17–N21
```

## Interface

```text
# InferenceLayer<T, In, Out>
virtual ~InferenceLayer() = default
virtual void                Forward(const InputVector& input) = 0                  # hot path
virtual const OutputVector& Output() const = 0
protected: defaulted ctor / copy / move (as today's Layer)

# Layer<T, In, Out, 0>  (adds only)
virtual const InputVector&  Backward(const OutputVector& outputGradient) = 0       # returns ∂L/∂x

# Layer<T, In, Out, P>, P > 0  (adds)
virtual const InputVector&     Backward(const OutputVector& outputGradient) = 0    # returns ∂L/∂x, accumulates ∂L/∂θ
virtual const ParameterVector& Parameters() const = 0
virtual void                   SetParameters(const ParameterVector& parameters) = 0 # does not touch gradients
virtual const ParameterVector& ParameterGradients() const = 0                      # G, in Parameters() order
virtual void                   ZeroGradients() = 0                                 # G = 0

# Dense<T, In, Out>  (overrides the above; constructor unchanged)

# Model<T, In, Out, Layers...>  (adds; existing members unchanged, Train() keeps its signature)
static_assert(TotalParameters > 0)                      # an all-parameter-free chain has no θ to expose
ParameterVector Gradients() const                       # [G_1; …; G_n], same order as GetParameters()
void            ZeroGradients()                         # every layer with ParameterSize > 0
```

`Model` keeps requiring `Layer`-derived types (`detail::is_layer_v`, which also matches the
`ParameterSize = 0` specialisation). Admitting pure `InferenceLayer` types into `Model` (with
`Backward`/`Gradients` constrained away) belongs to N10, which ships the first such layer.

## Algorithm (pseudocode)

```text
function Dense::Backward(g):                    # OPTIMIZE_FOR_SPEED; g = ∂L/∂a, size Out
    really_assert(forwardDone)
    δ = zeros(Out)
    activation.BackwardVector(δ, preActivation, output, g)    # δ = J_f(z)ᵀ g
    for j in 0..In-1:
        s = 0
        for i in 0..Out-1: s += W(i, j) · δ[i]
        inputGradient[j] = s                    # overwritten: per-sample ∂L/∂x
    for i in 0..Out-1:
        for j in 0..In-1:
            parameterGradients[i·In + j] += δ[i] · input[j]   # was "=" before N8
        parameterGradients[In·Out + i] += δ[i]
    return inputGradient

function Dense::ZeroGradients():
    parameterGradients = ParameterVector{}      # Matrix default ctor zero-fills

function Dense::ParameterGradients():  return parameterGradients
# construction: parameterGradients is value-initialised (all zeros), so the first Backward starts from 0

function Model::Gradients():
    G = ParameterVector{};  offset = 0
    for each layer ℓ in order (fold expression over index_sequence, no recursion):
        if constexpr (ℓ::ParameterSize > 0):
            copy ℓ.ParameterGradients() into G[offset .. offset + ℓ::ParameterSize)
            offset += ℓ::ParameterSize
    return G

function Model::ZeroGradients():
    for each layer ℓ: if constexpr (ℓ::ParameterSize > 0): ℓ.ZeroGradients()

function Model::SetLayerParameters / GetLayerParameters:     # existing helpers
    wrap the body in "if constexpr (LayerType::ParameterSize > 0)"   # zero-parameter layers own no slice of θ

function Model::Backward(g):                    # unchanged code; each layer now accumulates as a side effect
    g_n = g
    for ℓ = n..1: g_{ℓ-1} = layer_ℓ.Backward(g_ℓ)
    return g_0
```

Math (one sample `s`, upstream `g = ∂L_s/∂a`):

```text
forward:          z = W x + b,   a = f(z)                                       (unchanged)
activation VJP:   δ = ∂L_s/∂z = J_f(z)ᵀ g
                    element-wise f:  δ_i = f'(z_i) g_i
                    ReLU f' = 1 (z > 0) else 0;  LeakyReLU f' = 1 else α;
                    Sigmoid f' = σ(1 − σ);  Tanh f' = 1 − a²
                    Softmax:         δ = a ⊙ (g − ⟨g, a⟩)          # J = diag(a) − a aᵀ (symmetric)
input gradient:   ∂L_s/∂x_j = Σ_i W_ij δ_i          i.e. Wᵀ δ          # returned, per sample
parameter grads:  ∂L_s/∂W_ij = δ_i x_j,   ∂L_s/∂b_i = δ_i                # i.e. δ xᵀ and δ
accumulation:     G ← G + ∇_θ L_s                   after k samples:  G = Σ_{s=1..k} ∇_θ L_s = ∇_θ Σ_s L_s
                  (exact because θ is not changed between the Backward calls)
model:            ∇_θ L = [∇_{θ_1} L; …; ∇_{θ_n} L],  each block computed by layer ℓ with g_ℓ from layer ℓ+1
```

The mini-batch mean `Ḡ = G / B`, the optimiser step and the `ZeroGradients()` call after it are
N16's job; this item only guarantees that `G` is the exact sum.

A parameter-free layer (`Layer<T, In, Out, 0>`) implements only `Backward` returning `Jᵀ g`. For
example, max pooling (N12) routes `g` to the stored argmax, and average pooling spreads `g / K` over
the window. It has nothing to accumulate.

## Optional: compile-time activation (may be split out)

The activation is reached through `const ActivationFunction<T>&`, so every `Dense::Forward`/`Backward`
makes one virtual `ForwardVector`/`BackwardVector` call that the compiler cannot inline. A user
activation that relies on the base-class vector loops also pays one virtual `Forward`/`Backward` per
element. Both go against "no virtual calls in real-time paths". The optional extension:

```text
template<typename T, std::size_t In, std::size_t Out, typename Activation = ActivationFunction<T>>
class Dense:
    Activation-is-abstract ?  const Activation& activation    # today's behaviour (default argument)
                           :  Activation        activation    # stored by value, final type ⇒ calls devirtualise
```

All built-in activations are already `final` and override both vector methods. A by-value
`Activation` member therefore resolves `ForwardVector`/`BackwardVector` statically, and they can be
inlined into the Dense loops. The math and the gradients are identical. If deployed, the one extra test case listed in
`tests.md` applies. Otherwise N10 (inference-only Dense) takes it over.

## Complexity & memory

- `Dense::Backward`: `In·Out` MACs (input gradient) + `In·Out` MACs (weight gradient, now a
  multiply-add instead of a multiply-store) + `Out` adds (bias) + the activation VJP (`O(Out)`).
  N8 adds `In·Out + Out` additions per call and nothing else.
- `ZeroGradients`: `P = In·Out + Out` stores. `ParameterGradients`: `O(1)` (returns a reference).
- `Model::Gradients`: `TotalParameters` copies into a by-value `ParameterVector`. The caller needs
  `TotalParameters` floats of stack (N16 can instead read each layer's reference in place).
- `Model::ZeroGradients`: `TotalParameters` stores.
- RAM: **0 extra floats** per `Dense`. The gradient buffer already exists; it becomes observable.
  `Dense<T, In, Out>` holds `2P + 2·In + 2·Out` floats plus a reference, a `bool` and a vptr.
  For `Dense<float, 64, 64>` that is `2·4160 + 256 = 8576` floats (34 304 B).
- The parameter-free specialisation holds no parameter or gradient storage. The interface itself is a
  vptr only.
- Interfaces are pure virtual. `Forward`/`Backward` stay one virtual call per layer per sample, the same
  as today.

## Numerical / embedded notes

- **Accumulate, then reset explicitly.** `SetParameters` must not clear `G`. An optimiser step
  (N16) is `θ ← step(θ, G/B)`, then `SetParameters(θ)`, then `ZeroGradients()`. Clearing inside
  `SetParameters` would hide a missing reset during finite-difference checks, and would stop two
  optimisers from sharing `G`.
- **One Backward per Forward.** Each `Backward` adds the gradient of the most recent `Forward`.
  Calling `Backward` twice after one `Forward` adds that sample twice. This is the caller's contract,
  and `really_assert(forwardDone)` still guards the never-forwarded case.
- **Summation error.** Float32 accumulation of `B` terms has a worst-case relative error of about
  `B·2⁻²⁴`, which is negligible for MCU batch sizes (`B ≤ 256` ⇒ ≤ `1.5·10⁻⁵`). No compensated sum is
  needed. N16 divides by `B` once, after accumulation.
- **The input gradient is not accumulated.** It feeds the previous layer immediately. Accumulating it
  would double-count in `Model::Backward`.
- **Float-only.** Gradients span many orders of magnitude, and accumulated sums exceed `[−1, 1)`, so
  `Q15`/`Q31` cannot hold them. `static_assert(std::is_floating_point_v<T>)` on `InferenceLayer`; the
  generic `T` keeps a fixed-point specialisation cheap to add later.
- Under `-ffast-math`, non-finite upstream gradients are outside the contract. A single NaN poisons
  `G` until the next `ZeroGradients()`, which is why N16 checks before stepping.

## Dependencies

- Builds on: `Layer.hpp` (current contract: `ValueType`, `virtual ~Layer() = default`,
  `const ParameterVector& Parameters() const`), `Dense.hpp` (already computes `δ xᵀ` and `δ`
  privately), `Model.hpp` (index-sequence folds for the concatenation), `ActivationFunction::BackwardVector`
  (exact derivatives, softmax VJP), `numerical/math/Matrix.hpp` (`math::Vector`, zero-filling default
  ctor), `infra/util/ReallyAssert.hpp`. It needs no other N-item.
- Used by: N2 (AffineNormalization exposes `∂L/∂a = g ⊙ x`, `∂L/∂b = g`, which adds `N + 2N` floats),
  N5 (a trainable Swish-β slope), N9 (its loss gradient is the `g` fed to `Model::Backward`),
  N10 (inference-only layers derive from `InferenceLayer` only; N10 relaxes `Model`'s layer check),
  N11/N18/N19 (convolutions implement the accumulating `Backward`), N12 (pooling uses the
  `ParameterSize = 0` specialisation), N16 (`Gradients`/`ZeroGradients`, the per-layer freeze mask reads
  per-layer `ParameterGradients`), N17/N20/N21 (truncated BPTT sums over time steps through the same
  accumulation).
- Not changed: `Model::Train` keeps its current signature and semantics. It optimises an
  `ObjectiveFunction` over θ; N16 replaces it. Losses keep their current contract (N9).

## Deployment

- Header: `neural_network/layer/TrainableLayerInterface.hpp` (a `git mv` of `Layer.hpp`, extended).
  It holds `InferenceLayer`, the `Layer` primary template and the `Layer<…, 0>` specialisation. It is a pure interface, so it needs no `#pragma GCC optimize` and no `OPTIMIZE_FOR_SPEED`,
  the same as today's `Layer.hpp`. Add these under `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`:
  `extern template class InferenceLayer<float, 3, 2>;`, `extern template class Layer<float, 3, 2, 8>;`,
  `extern template class Layer<float, 3, 3, 0>;`.
  Update the includes in `Dense.hpp` and `Model.hpp`. Pending specs that name `Layer.hpp` (N2)
  include the new header when they deploy.
- `Dense.hpp`: change `=` to `+=` in `Backward` (keep `OPTIMIZE_FOR_SPEED`), and add the
  `ParameterGradients()`/`ZeroGradients()` overrides. `Dense.cpp` is unchanged (`Dense<float, 3, 2>`).
- `Model.hpp`: add `Gradients()`/`ZeroGradients()`, the `TotalParameters > 0` assert, and
  `if constexpr` guards in the parameter helpers. `Model.cpp` coverage is unchanged
  (`Model<float, 2, 1, Dense<float, 2, 3>, Dense<float, 3, 1>>`); its explicit instantiation now also
  covers `Gradients`/`ZeroGradients`.
- Coverage: `neural_network/layer/TrainableLayerInterface.cpp` →
  `template class InferenceLayer<float, 3, 2>; template class Layer<float, 3, 2, 8>; template class Layer<float, 3, 3, 0>;`
- Test: `neural_network/layer/test/TestTrainableLayerInterface.cpp`, plus two integration cases in
  `neural_network/model/test/TestModel.cpp` (see `tests.md`).
- Doc: `doc/layer/TrainableLayerInterface.md` (per `doc/TEMPLATE.md`), with a row in `doc/layer/README.md`.
  Drop the "overwrites / not exposed (roadmap N8)" limitation from `doc/layer/Dense.md`, and the
  "exporting them as a flat ∇θL is roadmap N8" sentence from `doc/model/Model.md`.
- CMake: `TrainableLayerInterface.hpp` replaces `Layer.hpp` in `target_sources(neural_network.layer PRIVATE ...)`;
  `TrainableLayerInterface.cpp` → `neural_network_add_coverage_sources(neural_network.layer ...)`;
  `TestTrainableLayerInterface.cpp` → `neural_network.layer_test` (it already links `gmock_main`).
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
