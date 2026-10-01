# Elman RNN (Streaming State, Truncated BPTT) — Implementation Pseudocode

> Roadmap ref: #N17 (Tier 3) · Target: `neural_network/layer` + `neural_network/model` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```text
# sizes (all compile-time):
#   X = InputSize (features per time step), H = HiddenSize, K = BpttWindow (truncation depth, K ≥ 1)
#   P = H·X + H·H + H
#
# θ layout (flat, "multiplicative blocks, then biases", as Dense / N11):
#   W_x[i, j] at i·X + j                 i ∈ [0, H), j ∈ [0, X)     (input weights, row-major H×X)
#   W_h[i, k] at H·X + i·H + k           i, k ∈ [0, H)              (recurrent weights, row-major H×H)
#   b[i]      at H·X + H·H + i                                      (one fused bias)
#   ⇒ PyTorch nn.RNN: W_x = weight_ih_l0, W_h = weight_hh_l0, b = bias_ih_l0 + bias_hh_l0 (no transpose)
#   ⇒ Keras SimpleRNN: W_x = kernelᵀ, W_h = recurrent_kernelᵀ, b = bias

template<typename T, std::size_t StateSize_>    # static_assert(std::is_floating_point_v<T>)
class StatefulLayer:                             # new pure interface (mixin), reused by N20 (GRU) and N21 (LSTM)
    using StateVector = math::Vector<T, StateSize_>
    static constexpr std::size_t StateSize = StateSize_
    # declares no ValueType/InputVector/OutputVector: a second ValueType next to Layer's would make
    # ElmanRnn::ValueType ambiguous and break Model's First::ValueType lookup

template<typename T, std::size_t InputSize, std::size_t HiddenSize, std::size_t BpttWindow>
                                                 # static_assert(std::is_floating_point_v<T>); instantiated for float
                                                 # static_assert(InputSize > 0 && HiddenSize > 0 && BpttWindow > 0)
class ElmanRnn final : public Layer<T, X, H, P>,          # N8 trainable layer (accumulating Backward)
                       public StatefulLayer<T, H>:
    Tanh<T>          activation                  # by value, final type ⇒ direct (devirtualised) call
    ParameterVector  parameters                  # θ, layout above
    ParameterVector  parameterGradients          # G = Σ ∇̃_θ L since the last ZeroGradients() (N8 contract)
    OutputVector     output                      # h_t — the output *and* the recurrent state (State() returns it)
    InputVector      inputGradient               # ∂L_t/∂x_t of the last Backward (per step, not accumulated)
    std::array<T, K·X> inputs{}                  # ring: x_s of the last K steps
    std::array<T, K·H> previousStates{}          # ring: h_{s−1} of the last K steps (the state each step read)
    std::size_t      newest = K − 1              # ring slot of the most recent step
    std::size_t      count  = 0                  # valid ring entries = min(K, steps since ResetState/SetState)
    # no preActivation history: tanh'(z_s) = 1 − h_s², and h_s is either output (s = t) or the
    # previousStates entry of step s + 1 ⇒ the ring holds exactly K·(X + H) floats

template<typename T, std::size_t InputSize, std::size_t HiddenSize>
                                                 # same static_asserts (no K: nothing to back-propagate)
class FlashElmanRnn final : public InferenceLayer<T, X, H>,  # N8 inference base, N10 storage policy
                            public StatefulLayer<T, H>:
    static constexpr std::size_t WeightCount = P                  # == ElmanRnn<T, X, H, K>::ParameterSize
    using WeightVector = math::Vector<T, WeightCount>
    using Weights      = FlashWeights<T, WeightCount>             # N10 read-only view, same θ layout
    Weights      weights                         # in flash (.rodata)
    Tanh<T>      activation
    OutputVector output{}                        # h_t — the only RAM-resident floats
```

Class diagram (N8/N10 hierarchy, extended):

```text
InferenceLayer<T, In, Out>        Forward, Output
 ├─ FlashElmanRnn<T, X, H>        (+ StatefulLayer<T, H>)                   ← inference only, weights in flash
 └─ Layer<T, In, Out, P>          + Backward, Parameters, SetParameters, ParameterGradients, ZeroGradients
      └─ ElmanRnn<T, X, H, K>     (+ StatefulLayer<T, H>)                   ← trainable, truncated BPTT
StatefulLayer<T, S>               ResetState, State, SetState               ← N20 (S = H), N21 (S = 2H: h and c)
```

## Interface

```text
# StatefulLayer<T, S>  (neural_network/layer/StatefulLayer.hpp)
virtual ~StatefulLayer() = default
virtual void               ResetState() = 0                          # state = 0; starts a new sequence
virtual const StateVector& State() const = 0
virtual void               SetState(const StateVector& state) = 0    # seeds the state; starts a new truncation window
protected: defaulted ctor / copy / move (as Layer)

# ElmanRnn<T, X, H, K>
using InputWeightMatrix     = math::Matrix<T, H, X>
using RecurrentWeightMatrix = math::Matrix<T, H, H>
static constexpr std::size_t Window = K

ElmanRnn(const InputWeightMatrix& inputWeights, const RecurrentWeightMatrix& recurrentWeights)   # b = 0, state = 0
void                   Forward(const InputVector& input) override                 # hot path; one time step
const InputVector&     Backward(const OutputVector& outputGradient) override      # hot path; truncated BPTT, accumulates G
const OutputVector&    Output() const override                                    # h_t
const ParameterVector& Parameters() const override
void                   SetParameters(const ParameterVector& parameters) override  # touches neither G nor state/history
const ParameterVector& ParameterGradients() const override                        # N8
void                   ZeroGradients() override                                   # N8
void                   ResetState() override
const StateVector&     State() const override                                     # same object as Output()
void                   SetState(const StateVector& state) override
# inherited: using ValueType = T (from Layer only); virtual ~Layer() = default

# FlashElmanRnn<T, X, H>
FlashElmanRnn(const WeightVector& weights)
FlashElmanRnn(const WeightVector&& weights) = delete                              # a temporary would dangle (N10)
explicit FlashElmanRnn(Weights weights)                                           # raw flash region (N13 OTA blob)
void                            Forward(const InputVector& input) override        # hot path
const OutputVector&             Output() const override
std::span<const T, WeightCount> Parameters() const                                # the flash view, never a copy
void ResetState() override;  const StateVector& State() const override;  void SetState(const StateVector&) override
# deliberately absent: Backward, SetParameters, ParameterGradients, ZeroGradients

# Model<T, In, Out, Layers...>  (adds; nothing else changes, Train() keeps its signature)
void ResetState()                                                                 # every StatefulLayerType layer; no-op otherwise
```

`detail` helper in `Model.hpp`:

```text
template<typename L> concept StatefulLayerType =
    requires { L::StateSize; } && std::is_base_of_v<StatefulLayer<typename L::ValueType, L::StateSize>, L>
```

## Algorithm (pseudocode)

```text
function ElmanRnn(Wx, Wh):
    for i in 0..H-1:
        for j in 0..X-1:  parameters[i·X + j]         = Wx.at(i, j)
        for k in 0..H-1:  parameters[H·X + i·H + k]   = Wh.at(i, k)
    # b, G, output (state), rings value-initialised to 0; newest = K − 1, count = 0

function Forward(x):                             # OPTIMIZE_FOR_SPEED
    newest = (newest + 1 == K) ? 0 : newest + 1  # compare-and-reset, no modulo
    inputs[newest·X .. newest·X + X)         = x
    previousStates[newest·H .. newest·H + H) = output           # h_{t−1}
    if count < K: count += 1
    z = OutputVector{}                           # stack temporary, H floats
    for i in 0..H-1:
        s = parameters[H·X + H·H + i]            # b_i
        for j in 0..X-1: s += parameters[i·X + j] · x[j]
        for k in 0..H-1: s += parameters[H·X + i·H + k] · output[k]   # output still holds h_{t−1}
        z[i] = s
    activation.ForwardVector(output, z)          # h_t = tanh(z); z and output are distinct buffers

function Backward(g):                            # OPTIMIZE_FOR_SPEED; g = ∂L_t/∂h_t, size H
    really_assert(count > 0)
    δ = OutputVector{};  next = OutputVector{}   # stack temporaries, 2H floats
    for i in 0..H-1: δ[i] = g[i] · (1 − output[i]·output[i])   # δ_t = g ⊙ tanh'(z_t)
    for j in 0..X-1:                             # ∂L_t/∂x_t = W_xᵀ δ_t — overwritten, per step
        s = 0
        for i in 0..H-1: s += parameters[i·X + j] · δ[i]
        inputGradient[j] = s
    e = newest
    for n in 0..count-1:                         # s = t − n, newest to oldest
        for i in 0..H-1:
            for j in 0..X-1: parameterGradients[i·X + j]       += δ[i] · inputs[e·X + j]
            for k in 0..H-1: parameterGradients[H·X + i·H + k] += δ[i] · previousStates[e·H + k]
            parameterGradients[H·X + H·H + i] += δ[i]
        if n + 1 == count: break                 # truncation: nothing flows into the state before the window
        for k in 0..H-1:                         # δ_{s−1} = (W_hᵀ δ_s) ⊙ (1 − h_{s−1}²)
            s = 0
            for i in 0..H-1: s += parameters[H·X + i·H + k] · δ[i]
            hp = previousStates[e·H + k]         # h_{s−1}
            next[k] = s · (1 − hp·hp)
        swap(δ, next)
        e = (e == 0) ? K − 1 : e − 1
    return inputGradient

function ResetState():       output = OutputVector{};  count = 0;  newest = K − 1
function SetState(state):    output = state;           count = 0;  newest = K − 1
function State():            return output
function Output():           return output
function SetParameters(p):   parameters = p              # history is kept (see notes)
function ZeroGradients():    parameterGradients = ParameterVector{}
function ParameterGradients(): return parameterGradients

function FlashElmanRnn::Forward(x):              # OPTIMIZE_FOR_SPEED; same summation order as ElmanRnn
    z = OutputVector{}
    for i in 0..H-1:
        s = weights[H·X + H·H + i]
        for j in 0..X-1: s += weights[i·X + j] · x[j]
        for k in 0..H-1: s += weights[H·X + i·H + k] · output[k]
        z[i] = s
    activation.ForwardVector(output, z)
function FlashElmanRnn::ResetState():   output = OutputVector{}
function FlashElmanRnn::SetState(s):    output = s

function Model::ResetState():                    # fold over index_sequence, no recursion
    for each layer ℓ: if constexpr (detail::StatefulLayerType<ℓ>): ℓ.ResetState()
```

Math (one sequence after `ResetState`/`SetState` at step `t₀`, state `h_{t₀}` given; steps `s = t₀+1, …, t`):

```text
forward:          z_s = W_x x_s + W_h h_{s−1} + b,     h_s = tanh(z_s)            (h_{t₀} = 0 after ResetState)
loss injection:   Backward(g) after Forward(x_t) receives g = ∂L_t/∂h_t (direct dependence of the step-t loss)
local error:      δ_t = g ⊙ (1 − h_t²)                                            (tanh' = 1 − tanh²)
recursion:        ∂L_t/∂h_{s−1} = W_hᵀ δ_s,   δ_{s−1} = (W_hᵀ δ_s) ⊙ (1 − h_{s−1}²)     for s = t, …, t−n+2
input gradient:   ∂L_t/∂x_t = W_xᵀ δ_t                                            (returned; current step only)
parameter grads:  ∇̃_{W_x} L_t = Σ_{s=t−n+1}^{t} δ_s x_sᵀ
                  ∇̃_{W_h} L_t = Σ_{s=t−n+1}^{t} δ_s h_{s−1}ᵀ
                  ∇̃_b     L_t = Σ_{s=t−n+1}^{t} δ_s
                  with n = count = min(K, t − t₀)
truncation:       ∇̃ treats h_{t−n} as a constant. If t − t₀ ≤ K then h_{t−n} = h_{t₀} does not depend on θ,
                  so ∇̃ L_t = ∇ L_t exactly (full BPTT); otherwise the terms through h_{t−K} are dropped.
                  Note ∂L_t/∂W_h picks up δ_{t₀+1} h_{t₀}ᵀ = 0 after ResetState.
accumulation:     G ← G + ∇̃_θ L_t per Backward (N8). Backward after every step (many-to-many):
                  G = Σ_t ∇̃_θ L_t = ∇_θ Σ_t L_t when the sequence fits the window. Backward only after the
                  last step (many-to-one, sequence classification): G = ∇̃_θ L_T.
```

This is Williams & Peng's per-step truncated BPTT, `BPTT(h)` with depth `h = K`: one depth-`K` backward
sweep at every step that injects a loss. Their cheaper `BPTT(h; h′)` variant, which runs one sweep every
`h′` steps and needs the `h′` pending upstream gradients buffered, is deferred. It does not fit the
per-sample `Backward` contract of N8.

## Complexity & memory

- `Forward`: `H·X + H²` MACs + `H` tanh + `X + H` ring copies. No loop depends on `K`.
- `Backward`: with `n = count ≤ K`: `n·(H·X + H²)` MACs (weight gradients) + `(n − 1)·H²` MACs (state
  recursion) + `H·X` MACs (input gradient) + `n·H` bias adds + `n·H` elementwise `1 − h²` products.
  `n = K`: `K·(H·X + 2H²) − H² + H·X` MACs.
- RAM, `ElmanRnn` (trainable): `2P + K·(X + H) + H + X` floats + two indices + two vptrs (`Layer` and
  `StatefulLayer`) + `Tanh`'s vptr. `Backward` adds `2H` floats of stack, `Forward` adds `H`.
  - Fixture `ElmanRnn<float, 2, 3, 4>`: `P = 18`, `36 + 20 + 3 + 2 = 61` floats.
  - Example `ElmanRnn<float, 3, 16, 8>` (3-axis IMU, 16 hidden units, 8-step window): `P = 320`,
    `640 + 152 + 16 + 3 = 811` floats (3 244 B). `Forward` costs 304 MACs; a full-window `Backward`
    costs 4 272 MACs.
  - The truncation window itself is the `K·(X + H)`-float ring (152 floats here). Doubling `K` doubles
    the ring and the `Backward` cost; `Forward` is unchanged.
- RAM, `FlashElmanRnn`: `H` floats (`output` = state) + one pointer (the static-extent span) + the vptrs;
  `H` floats of stack during `Forward`. Flash holds `P` floats (1 280 B for the example above).
- The recurrent state costs nothing extra: it is the layer's `output`, which every layer keeps anyway.

## Numerical / embedded notes

- **Tanh only.** The activation is fixed. `tanh` keeps `h ∈ (−1, 1)`, so the recurrence cannot overflow,
  and `tanh' = 1 − h²` lets the ring store `h` instead of `z`, which saves `K·H` floats. A ReLU RNN (PyTorch
  `nonlinearity="relu"`) has an unbounded state and needs identity-style initialisation to stay stable, so
  it is deferred (its derivative `[h > 0]` would also be recoverable from the stored `h`). When float32 `tanh` saturates to exactly `±1`, `1 − h² = 0` and the local
  error is exactly 0. That is the correct derivative, not a NaN.
- **Vanishing / exploding gradients.** `‖δ_{s−1}‖ ≤ ‖W_h‖₂ · ‖δ_s‖`, because `1 − h² ≤ 1`. For
  `‖W_h‖₂ < 1` contributions older than a few steps vanish, which is one reason a small `K` loses little.
  Exploding gradients (`‖W_h‖₂ > 1`) are handled by N16's global-norm clipping, not in the layer.
  Truncation bounds the growth to `‖W_h‖₂^{K−1}`.
- **Parameters change only at sequence boundaries.** The ring stores `h_s` produced by the θ in use
  at the time. If `SetParameters` runs mid-window, the next `Backward` combines stale states with the new `W_h`.
  That is the usual online-RNN approximation and has a bounded error, but it is no longer the exact truncated gradient. For an exact
  gradient, step θ between sequences and call `Model::ResetState()` afterwards. With N16, choose the batch
  size as a multiple of the sequence length. `SetParameters` deliberately does not clear the history, so a
  deployed model can be hot-swapped (N13 OTA) without losing its streaming state.
- **Input gradient is per step.** `Backward` returns `∂L_t/∂x_t` only. The earlier `∂L_t/∂x_s` (`s < t`)
  exist mathematically, but the layers below keep only their latest input and cannot use them. A
  trainable layer in front of the RNN therefore gets a truncated-to-one-step gradient. Put the RNN
  first (after an input `AffineNormalization`, N2) or freeze the layers below it. Layers above it are exact.
- **One `Backward` per `Forward`, optional.** Skipping `Backward` on steps without a loss is fine: the ring
  still records them, and a later `Backward` back-propagates through them. Calling it twice after one
  `Forward` adds the step's gradient twice (N8 caller contract).
- **Ring.** Two parallel `std::array`s. The index advances with compare-and-reset, not modulo. The window
  walk runs newest to oldest, which also reads `previousStates` in the right order for the `1 − h_{s−1}²` factor.
  `count` is the only validity marker: `ResetState`/`SetState` do not clear the arrays.
- **Summation.** Each `z_i` is an `X + H + 1`-term sum with worst-case relative error `≈ (X + H)·2⁻²⁴`.
  A weight gradient sums at most `K` terms per `Backward`. `FlashElmanRnn` and `ElmanRnn` use the same
  order (bias, then `W_x x`, then `W_h h`), so they agree to rounding. Under `-ffast-math` the compiler may
  reassociate, so the tests use `EXPECT_NEAR`.
- **State ≡ output.** `State()` and `Output()` return the same object. `SetState` writes the value the next
  `Forward` reads as `h_{t−1}`, so `Output()` shows the seeded state until then.
- Under `-ffast-math` non-finite inputs, states or weights are outside the contract.
- Float-only: `static_assert(std::is_floating_point_v<T>)` in all three templates. `h` fits `[−1, 1)`, but
  pre-activations, accumulated gradients and SGD steps do not fit `Q15`/`Q31`. The generic `T` signature
  keeps a fixed-point specialisation cheap to add later.

## Dependencies

- Requires **N8**: `Layer<T, In, Out, P>` with the accumulating `Backward`, `ParameterGradients()` and
  `ZeroGradients()`, plus `InferenceLayer` and the `TrainableLayerInterface.hpp` header.
- Also builds on: `Tanh.hpp` (`final`, `ForwardVector`; its `1 − y²` derivative is inlined on stored states),
  the Dense row loop (`z = W x + b`, same `θ` ordering idea), `numerical/math/Matrix.hpp` (`math::Vector`,
  `math::Matrix`, zero-filling default ctor, `constexpr` ctor), `numerical/math/CompilerOptimizations.hpp`,
  `infra/util/ReallyAssert.hpp`, `<array>`, `<span>`.
- **N10** for `FlashElmanRnn`: `FlashWeights<T, P>` and the relaxed `Model` that admits `InferenceLayer`
  types. `ElmanRnn` itself does not need N10. If N17 lands before N10, ship `ElmanRnn` alone and add
  `FlashElmanRnn` (and its include, coverage line and test case) together with N10.
- Soft: **N3** initialises `W_x` with `fanIn = X` and `W_h` with `fanIn = H` (Glorot/He via `Fill`;
  orthogonal recurrent init is deferred there). **N9** supplies the per-step loss gradient `g`. **N16**
  provides clipping and the optimiser loop.
- Used by: **N13** (exporter rule above: PyTorch `weight_ih_l0`/`weight_hh_l0` as-is with
  `b = b_ih + b_hh`; Keras `kernel`/`recurrent_kernel` transposed; `P = H·(X + H + 1)`), **N20/N21**
  (reuse `StatefulLayer`, `Model::ResetState`, the two-array ring and the newest-to-oldest BPTT sweep; the
  GRU stores its gate activations per step, the LSTM also stores `c`, so `StateSize = 2H`).
- Composes with `Model` through `make_layer<ElmanRnn<float, X, H, K>>(inputWeights, recurrentWeights)`.
  `OutputSize = H` chains into a `Dense<float, H, …>` head, which gives the classic Elman network
  (recurrent hidden layer plus a feed-forward output layer). One `Model::Forward` call is one time step.

## Deployment

- Header: `neural_network/layer/ElmanRnn.hpp` — `#pragma once` →
  `#pragma GCC optimize("O3","fast-math")` → includes (`neural_network/layer/TrainableLayerInterface.hpp`,
  `neural_network/layer/StatefulLayer.hpp`, `neural_network/layer/FlashResidentWeights.hpp`,
  `neural_network/activation/Tanh.hpp`, `numerical/math/Matrix.hpp`,
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`, `<array>`, `<span>`).
  `OPTIMIZE_FOR_SPEED` on `Forward`/`Backward` of both classes. Under
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`:
  `extern template class ElmanRnn<float, 2, 3, 4>;` and `extern template class FlashElmanRnn<float, 2, 3>;`
- Header: `neural_network/layer/StatefulLayer.hpp` — pure interface, like `Layer.hpp`: no
  `#pragma GCC optimize`, no `OPTIMIZE_FOR_SPEED`, no coverage source.
- Coverage: `neural_network/layer/ElmanRnn.cpp` →
  `template class ElmanRnn<float, 2, 3, 4>; template class FlashElmanRnn<float, 2, 3>;`
- `Model.hpp`: add `detail::StatefulLayerType` and `ResetState()`. The existing `Model.cpp` coverage
  instantiation then covers `ResetState` as a no-op.
- Test: `neural_network/layer/test/TestElmanRnn.cpp`, plus one integration case in
  `neural_network/model/test/TestModel.cpp` (see `tests.md`).
- Doc: `doc/layer/ElmanRnn.md` (per `doc/TEMPLATE.md`), with a row in `doc/layer/README.md`. Update
  `doc/model/Model.md` with `ResetState()` and the one-call-per-time-step usage.
- CMake: `ElmanRnn.hpp` and `StatefulLayer.hpp` → `target_sources(neural_network.layer PRIVATE ...)`;
  `ElmanRnn.cpp` → `neural_network_add_coverage_sources(neural_network.layer ...)`;
  `TestElmanRnn.cpp` → `neural_network.layer_test`.
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
