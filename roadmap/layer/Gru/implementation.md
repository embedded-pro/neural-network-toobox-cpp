# GRU Cell (reset_after, Truncated BPTT) — Implementation Pseudocode

> Roadmap ref: #N20 (Tier 4) · Target: `neural_network/layer` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```
# sizes (all compile-time):
#   X = InputSize, H = HiddenSize, K = BpttWindow (truncation depth, K ≥ 1)
#   gate order q ∈ {r = 0, z = 1, n = 2} (PyTorch order); gate row ρ = q·H + i, i ∈ [0, H)
#   P = 3H·X + 3H·H + 3H + 3H = 3·(H·(X + H) + 2H)
#
# θ layout (flat, "multiplicative blocks, then biases", as Dense / N11 / N17):
#   W_x[ρ, j]  at ρ·X + j                        ρ ∈ [0, 3H), j ∈ [0, X)    (input weights, row-major 3H×X)
#   W_h[ρ, k]  at 3H·X + ρ·H + k                 ρ ∈ [0, 3H), k ∈ [0, H)    (recurrent weights, row-major 3H×H)
#   b_x[ρ]     at 3H·X + 3H·H + ρ                                          (input-side bias)
#   b_h[ρ]     at 3H·X + 3H·H + 3H + ρ                                     (hidden-side bias)
#   ⇒ PyTorch nn.GRU: W_x = weight_ih_l0, W_h = weight_hh_l0, b_x = bias_ih_l0, b_h = bias_hh_l0 (no transpose)
#   ⇒ Keras GRU(reset_after=True): W_x = kernelᵀ, W_h = recurrent_kernelᵀ, b_x = bias[0], b_h = bias[1],
#     each with gate blocks permuted from Keras (z, r, h) to (r, z, n) — N13 exporter rule
#   Both bias vectors are kept: b_h^n sits inside r ⊙ (…) and cannot be folded into b_x^n (N13).

template<typename T, std::size_t InputSize, std::size_t HiddenSize, std::size_t BpttWindow>
                                                 # static_assert(std::is_floating_point_v<T>); instantiated for float
                                                 # static_assert(InputSize > 0 && HiddenSize > 0 && BpttWindow > 0)
class Gru final : public Layer<T, X, H, P>,              # N8 trainable layer (accumulating Backward)
                  public StatefulLayer<T, H>:            # N17 stateful mixin (StateSize = H)
    Sigmoid<T>       gate                        # by value, final type ⇒ direct (devirtualised) Forward calls
    Tanh<T>          candidate                   # by value, final type
    ParameterVector  parameters                  # θ, layout above
    ParameterVector  parameterGradients          # G = Σ ∇̃_θ L since the last ZeroGradients() (N8 contract)
    OutputVector     output                      # h_t — the output *and* the recurrent state
    InputVector      inputGradient               # ∂L_t/∂x_t of the last Backward (per step, not accumulated)
    std::array<T, K·X> inputs{}                  # ring: x_s
    std::array<T, K·H> previousStates{}          # ring: h_{s−1} (the state step s read)
    std::array<T, K·H> resetGates{}              # ring: r_s
    std::array<T, K·H> updateGates{}             # ring: z_s
    std::array<T, K·H> candidates{}              # ring: n_s
    std::array<T, K·H> hiddenCandidates{}        # ring: u_s = W_h^n h_{s−1} + b_h^n (the term r multiplies)
    std::size_t      newest = K − 1              # ring slot of the most recent step
    std::size_t      count  = 0                  # valid ring entries = min(K, steps since ResetState/SetState)
    # ring = K·(X + 5H) floats. n_s is stored because recovering it from h_s, (h_s − z h_{s−1})/(1 − z),
    # divides by ≈ 0 when z → 1; u_s is stored to save H² MACs per step in Backward

template<typename T, std::size_t InputSize, std::size_t HiddenSize>
                                                 # same static_asserts (no K: nothing to back-propagate)
class FlashGru final : public InferenceLayer<T, X, H>,       # N8 inference base, N10 storage policy
                       public StatefulLayer<T, H>:
    static constexpr std::size_t WeightCount = P                  # == Gru<T, X, H, K>::ParameterSize
    using WeightVector = math::Vector<T, WeightCount>
    using Weights      = FlashWeights<T, WeightCount>             # N10 read-only view, same θ layout
    Weights      weights                         # in flash (.rodata)
    Sigmoid<T>   gate
    Tanh<T>      candidate
    OutputVector output{}                        # h_t — the only RAM-resident floats
```

Class diagram (N8/N10/N17 hierarchy, extended):

```
InferenceLayer<T, In, Out>        Forward, Output
 ├─ FlashGru<T, X, H>             (+ StatefulLayer<T, H>)                   ← inference only, weights in flash
 └─ Layer<T, In, Out, P>          + Backward, Parameters, SetParameters, ParameterGradients, ZeroGradients
      └─ Gru<T, X, H, K>          (+ StatefulLayer<T, H>)                   ← trainable, truncated BPTT
StatefulLayer<T, S>               ResetState, State, SetState               ← from N17; S = H here
```

## Interface

```
# Gru<T, X, H, K>
using InputWeightMatrix     = math::Matrix<T, 3H, X>     # rows (r; z; n), PyTorch weight_ih_l0 shape
using RecurrentWeightMatrix = math::Matrix<T, 3H, H>     # rows (r; z; n), PyTorch weight_hh_l0 shape
static constexpr std::size_t Window = K

Gru(const InputWeightMatrix& inputWeights, const RecurrentWeightMatrix& recurrentWeights)   # b_x = b_h = 0, state = 0
void                   Forward(const InputVector& input) override                 # hot path; one time step
const InputVector&     Backward(const OutputVector& outputGradient) override      # hot path; truncated BPTT, accumulates G
const OutputVector&    Output() const override                                    # h_t
const ParameterVector& Parameters() const override
void                   SetParameters(const ParameterVector& parameters) override  # touches neither G nor state/history
const ParameterVector& ParameterGradients() const override                        # N8
void                   ZeroGradients() override                                   # N8
void                   ResetState() override                                      # N17 StatefulLayer
const StateVector&     State() const override                                     # same object as Output()
void                   SetState(const StateVector& state) override
# inherited: using ValueType = T (from Layer only); virtual ~Layer() = default; virtual ~StatefulLayer() = default

# FlashGru<T, X, H>
FlashGru(const WeightVector& weights)
FlashGru(const WeightVector&& weights) = delete                                  # a temporary would dangle (N10)
explicit FlashGru(Weights weights)                                               # raw flash region (N13 OTA blob)
void                            Forward(const InputVector& input) override       # hot path
const OutputVector&             Output() const override
std::span<const T, WeightCount> Parameters() const                               # the flash view, never a copy
void ResetState() override;  const StateVector& State() const override;  void SetState(const StateVector&) override
# deliberately absent: Backward, SetParameters, ParameterGradients, ZeroGradients
```

`Model` needs no change: `Model::ResetState()` and `detail::StatefulLayerType` (both from N17) pick up `Gru`
and `FlashGru` through `StateSize` and the `StatefulLayer` base.

## Algorithm (pseudocode)

```
# index helpers (inline): Wx(ρ, j) = parameters[ρ·X + j]          Wh(ρ, k) = parameters[3H·X + ρ·H + k]
#                         Bx(ρ)    = parameters[3H·X + 3H·H + ρ]  Bh(ρ)    = parameters[3H·X + 3H·H + 3H + ρ]
#                         GWx, GWh, GBx, GBh: the same offsets into parameterGradients

function Gru(Wx_in, Wh_in):
    for ρ in 0..3H-1:
        for j in 0..X-1:  parameters[ρ·X + j]          = Wx_in.at(ρ, j)
        for k in 0..H-1:  parameters[3H·X + ρ·H + k]   = Wh_in.at(ρ, k)
    # b_x, b_h, G, output (state), rings value-initialised to 0; newest = K − 1, count = 0

function Forward(x):                             # OPTIMIZE_FOR_SPEED
    newest = (newest + 1 == K) ? 0 : newest + 1  # compare-and-reset, no modulo
    e = newest
    inputs[e·X .. e·X + X)         = x
    previousStates[e·H .. e·H + H) = output      # h_{t−1}; read below from the ring, so output can be overwritten
    if count < K: count += 1
    hp = previousStates[e·H ..)                  # view, not a copy
    for i in 0..H-1:
        ar = Bx(i) + Bh(i)                       # r row
        az = Bx(H + i) + Bh(H + i)               # z row
        xn = Bx(2H + i)                          # n row, input side (a_n = xn + r·u)
        u  = Bh(2H + i)                          # n row, hidden side
        for j in 0..X-1:
            ar += Wx(i, j)·x[j];  az += Wx(H + i, j)·x[j];  xn += Wx(2H + i, j)·x[j]
        for k in 0..H-1:
            ar += Wh(i, k)·hp[k]; az += Wh(H + i, k)·hp[k]; u  += Wh(2H + i, k)·hp[k]
        r = gate.Forward(ar);  z = gate.Forward(az)              # stable σ (branch on sign, Sigmoid.hpp)
        n = candidate.Forward(xn + r·u)                           # reset_after: r scales (W_h^n h + b_h^n)
        resetGates[e·H + i] = r;  updateGates[e·H + i] = z
        candidates[e·H + i] = n;  hiddenCandidates[e·H + i] = u
        output[i] = n + z·(hp[i] − n)                             # = (1 − z)·n + z·h_{t−1}

function Backward(g):                            # OPTIMIZE_FOR_SPEED; g = ∂L_t/∂h_t, size H
    really_assert(count > 0)
    dh = g;  next = OutputVector{}               # stack temporaries
    δr = δz = δn = δu = OutputVector{}           # 4H floats of stack
    e = newest
    for m in 0..count-1:                         # step s = t − m, newest to oldest
        for i in 0..H-1:
            r = resetGates[e·H + i];  z = updateGates[e·H + i];  n = candidates[e·H + i]
            u = hiddenCandidates[e·H + i];  hp = previousStates[e·H + i]
            δn[i] = dh[i]·(1 − z)·(1 − n·n)      # ∂L/∂a_n
            δz[i] = dh[i]·(hp − n)·z·(1 − z)     # ∂L/∂a_z
            δr[i] = δn[i]·u·r·(1 − r)            # ∂L/∂a_r
            δu[i] = δn[i]·r                      # ∂L/∂u
        if m == 0:                               # ∂L_t/∂x_t — overwritten, per step
            for j in 0..X-1:
                s = 0
                for i in 0..H-1: s += Wx(i, j)·δr[i] + Wx(H + i, j)·δz[i] + Wx(2H + i, j)·δn[i]
                inputGradient[j] = s
        for q in {r, z, n}, i in 0..H-1:         # ρ = q·H + i
            a = (δr, δz, δn)[q][i]               # input-side error of row ρ
            c = (δr, δz, δu)[q][i]               # hidden-side error of row ρ (differs only for n)
            for j in 0..X-1: GWx(ρ, j) += a·inputs[e·X + j]
            for k in 0..H-1: GWh(ρ, k) += c·previousStates[e·H + k]
            GBx(ρ) += a;  GBh(ρ) += c
        if m + 1 == count: break                 # truncation: nothing flows into the state before the window
        for k in 0..H-1:                         # ∂L/∂h_{s−1}
            s = dh[k]·updateGates[e·H + k]       # direct path through z ⊙ h_{s−1}
            for i in 0..H-1: s += Wh(i, k)·δr[i] + Wh(H + i, k)·δz[i] + Wh(2H + i, k)·δu[i]
            next[k] = s
        swap(dh, next)
        e = (e == 0) ? K − 1 : e − 1
    return inputGradient

function ResetState():       output = OutputVector{};  count = 0;  newest = K − 1
function SetState(state):    output = state;           count = 0;  newest = K − 1
function State():            return output
function Output():           return output
function SetParameters(p):   parameters = p              # history is kept (N17 note)
function ZeroGradients():    parameterGradients = ParameterVector{}
function ParameterGradients(): return parameterGradients

function FlashGru::Forward(x):                   # OPTIMIZE_FOR_SPEED; same summation order as Gru
    hp = output                                  # H floats of stack: output is overwritten row by row
    for i in 0..H-1:
        (ar, az, xn, u) as in Gru::Forward, reading weights[...] instead of parameters[...]
        r = gate.Forward(ar);  z = gate.Forward(az);  n = candidate.Forward(xn + r·u)
        output[i] = n + z·(hp[i] − n)
function FlashGru::ResetState():   output = OutputVector{}
function FlashGru::SetState(s):    output = s
```

Math (one sequence after `ResetState`/`SetState` at step `t₀`; `W^q` = the `H` rows of gate `q`; `σ` logistic):

```
forward:     a_r = W_x^r x_s + b_x^r + W_h^r h_{s−1} + b_h^r,       r_s = σ(a_r)
             a_z = W_x^z x_s + b_x^z + W_h^z h_{s−1} + b_h^z,       z_s = σ(a_z)
             u_s = W_h^n h_{s−1} + b_h^n
             a_n = W_x^n x_s + b_x^n + r_s ⊙ u_s,                   n_s = tanh(a_n)
             h_s = (1 − z_s) ⊙ n_s + z_s ⊙ h_{s−1}                  (PyTorch/Keras; h_{t₀} = 0 after ResetState)
derivatives: σ' = σ(1 − σ),  tanh' = 1 − tanh²   (both evaluated on the stored r, z, n)
             ∂h_s/∂n_s = diag(1 − z_s),  ∂h_s/∂z_s = diag(h_{s−1} − n_s),  ∂h_s/∂h_{s−1}|direct = diag(z_s)
local error: dh = ∂L_t/∂h_s (g at s = t, else the recursion below)
             δn = dh ⊙ (1 − z_s) ⊙ (1 − n_s²)
             δz = dh ⊙ (h_{s−1} − n_s) ⊙ z_s ⊙ (1 − z_s)
             δu = δn ⊙ r_s
             δr = δn ⊙ u_s ⊙ r_s ⊙ (1 − r_s)
recursion:   ∂L_t/∂h_{s−1} = z_s ⊙ dh + W_h^rᵀ δr + W_h^zᵀ δz + W_h^nᵀ δu
input grad:  ∂L_t/∂x_t = W_x^rᵀ δr + W_x^zᵀ δz + W_x^nᵀ δn                      (returned; current step only)
param grads: ∇̃_{W_x^q} L_t = Σ_s δq_s x_sᵀ,         ∇̃_{b_x^q} L_t = Σ_s δq_s          q ∈ {r, z, n}
             ∇̃_{W_h^q} L_t = Σ_s δq̂_s h_{s−1}ᵀ,     ∇̃_{b_h^q} L_t = Σ_s δq̂_s          (r̂ = r, ẑ = z, n̂ = u)
             sums over s = t−c+1 … t, c = count = min(K, t − t₀)
truncation / accumulation: exactly as N17 — h_{t−c} is treated as a constant, so the gradient is exact
             when t − t₀ ≤ K; G ← G + ∇̃_θ L_t per Backward (N8).
```

`b_x^r`/`b_h^r` and `b_x^z`/`b_h^z` only ever appear as sums, so their gradients are identical and an SGD step
keeps their difference constant. They are stored separately anyway so imported weights load verbatim. Only
the `n` biases carry different gradients (`δn` vs `δu = δn ⊙ r`).

Truncated BPTT is the same per-step `BPTT(h)` with depth `h = K` as N17 (Williams & Peng); the deferred
`BPTT(h; h′)` variant is out of scope here too.

## Complexity & memory

- `Forward`: `3H·X + 3H²` MACs + `2H` sigmoid + `H` tanh + `O(H)` elementwise ops + `X + H` ring copies.
  No loop depends on `K`.
- `Backward`, with `c = count ≤ K`: `c·(3H·X + 3H²)` MACs (weight gradients) + `(c − 1)·3H²` MACs (state
  recursion) + `3H·X` MACs (input gradient) + `c·6H` bias adds + `O(c·H)` elementwise products. No
  transcendental calls: every derivative is a polynomial in the stored `r, z, n`.
  `c = K`: `K·(3H·X + 6H²) − 3H² + 3H·X` MACs.
- RAM, `Gru` (trainable): `2P + K·(X + 5H) + H + X` floats + two indices + vptrs (`Layer`, `StatefulLayer`,
  and one each for the by-value `Sigmoid`/`Tanh`). `Backward` adds `6H` floats of stack; `Forward` none.
  - Fixture `Gru<float, 2, 3, 4>`: `P = 63`, `126 + 68 + 3 + 2 = 199` floats.
  - Example `Gru<float, 3, 16, 8>` (3-axis IMU, 16 units, 8-step window): `P = 1008`,
    `2016 + 664 + 16 + 3 = 2699` floats (10 796 B). `Forward` costs 912 MACs; a full-window `Backward`
    costs 12 816 MACs. The same-size LSTM (N21) has `P = 1280`, so the GRU saves 21 % of the weights
    (→ 25 % as `X + H` grows); the Elman cell (N17) has `P = 320`.
  - The ring is `K·(X + 5H)` floats (664 here), versus N17's `K·(X + H)`: the three gate activations and
    `u` are the price of not re-evaluating transcendentals in `Backward`.
- RAM, `FlashGru`: `H` floats (`output` = state) + one pointer (the static-extent span) + vptrs; `H` floats
  of stack during `Forward`. Flash holds `P` floats (4 032 B for the example above).

## Numerical / embedded notes

- **State stays bounded.** `h_s` is a convex combination of `n_s ∈ (−1, 1)` and `h_{s−1}`, so with a zero
  (or `|h| ≤ 1`) start state `|h_s| < 1` forever. A seeded state outside `[−1, 1]` is pulled back towards the
  range; the recurrence cannot overflow.
- **Additive path.** `∂h_s/∂h_{s−1}` contains `diag(z_s)`: when the update gate is open (`z ≈ 1`) the error
  passes back almost unattenuated, which is why a GRU keeps long-range information better than N17. It also
  means gradients can grow; clipping is N16's job, not the layer's. Truncation still bounds the sweep to `K`
  steps.
- **Saturation.** When float32 `σ` rounds to exactly `0`/`1` or `tanh` to `±1`, the matching derivative
  factor is exactly `0` — the correct limit, not a NaN. `Sigmoid::Forward` already branches on the sign
  (`e^{−|a|}` never overflows).
- **Why store `n` and `u`.** Recovering `n = (h_s − z h_{s−1})/(1 − z)` loses all precision as `z → 1`.
  `u` could be recomputed from `h_{s−1}` at `H²` MACs per step; storing it costs `K·H` floats instead. A
  recompute-everything mode (ring `K·(X + H)`, `3H·(X + H)` MACs + `3H` transcendentals extra per step) is
  deferred.
- **reset_after only.** The Cho-2014 form `n = tanh(W_x^n x + W_h^n (r ⊙ h) + b)` is a different function;
  N13 refuses Keras `reset_after=False` weights. The gate convention `h = (1 − z) n + z h_{t−1}` matches PyTorch,
  Keras and Cho et al.; Chung et al. (2014) swap `z` and `1 − z`, which only relabels the gate but makes
  weights incompatible.
- **Same streaming contract as N17.** `Backward` returns only `∂L_t/∂x_t`; skipping `Backward` on steps
  without a loss is fine; calling it twice after one `Forward` adds the step twice; `SetParameters`
  mid-window mixes stale gates with new weights (step θ between sequences for an exact gradient);
  `ResetState`/`SetState` start a new truncation window and do not clear the ring arrays (`count` is the
  only validity marker).
- **Summation order.** `a_r`, `a_z`: `(b_x + b_h)`, then `W_x x`, then `W_h h`; `xn`: `b_x^n + W_x^n x`;
  `u`: `b_h^n + W_h^n h`; update as `n + z·(h_{t−1} − n)`. `FlashGru` uses the same order, so the two agree to
  rounding. Under `-ffast-math` the compiler may reassociate, so tests use `EXPECT_NEAR`.
- **State ≡ output.** `State()` and `Output()` return the same object; `SetState` writes the value the next
  `Forward` reads as `h_{t−1}`.
- Under `-ffast-math` non-finite inputs, states or weights are outside the contract.
- Float-only: `static_assert(std::is_floating_point_v<T>)` in both templates. `r, z ∈ (0, 1)` and
  `h ∈ (−1, 1)` would fit `Q15`, but pre-activations, `u`, accumulated gradients and SGD steps do not. The
  generic `T` signature keeps a fixed-point specialisation cheap to add later.

## Dependencies

- Requires **N8**: `Layer<T, In, Out, P>` with the accumulating `Backward`, `ParameterGradients()` and
  `ZeroGradients()`, plus `InferenceLayer`.
- Requires **N17**: `StatefulLayer<T, S>`, `Model::ResetState()` / `detail::StatefulLayerType`, the two-index
  ring and the newest-to-oldest truncated-BPTT sweep. N20 adds four parallel gate rings to the same pattern.
- Also builds on: `Sigmoid.hpp` (numerically stable, `final`), `Tanh.hpp` (`final`), the Dense row loop,
  `numerical/math/Matrix.hpp` (`math::Vector`, `math::Matrix`, zero-filling default ctor, `constexpr` ctor),
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`, `<array>`, `<span>`.
- **N10** for `FlashGru`: `FlashWeights<T, P>` and the relaxed `Model` that admits `InferenceLayer` types. If
  N20 lands before N10, ship `Gru` alone and add `FlashGru` with N10.
- Soft: **N3** initialises each gate block of `W_x` (`fanIn = X`, `fanOut = H`) and `W_h` (`fanIn = H`,
  `fanOut = H`) via `Fill`; orthogonal recurrent init is deferred there. **N9** supplies the per-step loss
  gradient. **N16** provides clipping and the optimiser loop. **N13** emits `θ` in exactly this layout
  (PyTorch as-is; Keras transposed, gate blocks permuted `(z, r, h) → (r, z, n)`, `bias[0]`/`bias[1]` kept).
- Used by: **N21** (LSTM reuses the multi-gate ring and the gate-row indexing, with `StateSize = 2H`).
- Composes with `Model` through `make_layer<Gru<float, X, H, K>>(inputWeights, recurrentWeights)`; `OutputSize = H`
  chains into a `Dense<float, H, …>` head. One `Model::Forward` call is one time step.

## Deployment

- Header: `neural_network/layer/Gru.hpp` — `#pragma once` → `#pragma GCC optimize("O3","fast-math")` →
  includes (`neural_network/layer/TrainableLayerInterface.hpp`, `neural_network/layer/StatefulLayer.hpp`,
  `neural_network/layer/FlashResidentWeights.hpp`, `neural_network/activation/Sigmoid.hpp`,
  `neural_network/activation/Tanh.hpp`, `numerical/math/Matrix.hpp`, `numerical/math/CompilerOptimizations.hpp`,
  `infra/util/ReallyAssert.hpp`, `<array>`, `<span>`). `OPTIMIZE_FOR_SPEED` on `Forward`/`Backward` of both
  classes. Under `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`:
  `extern template class Gru<float, 2, 3, 4>;` and `extern template class FlashGru<float, 2, 3>;`
- Coverage: `neural_network/layer/Gru.cpp` → `template class Gru<float, 2, 3, 4>; template class FlashGru<float, 2, 3>;`
- Test: `neural_network/layer/test/TestGru.cpp`, plus one integration case in
  `neural_network/model/test/TestModel.cpp` (see `tests.md`).
- Doc: `doc/layer/Gru.md` (per `doc/TEMPLATE.md`), with a row in `doc/layer/README.md`.
- CMake: `Gru.hpp` → `target_sources(neural_network.layer PRIVATE ...)`; `Gru.cpp` →
  `neural_network_add_coverage_sources(neural_network.layer ...)`; `TestGru.cpp` → `neural_network.layer_test`.
  `neural_network.layer` must link `neural_network.activation` (already required by N17's `Tanh`).
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
