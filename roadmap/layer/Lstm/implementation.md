# LSTM Cell (Forget Gate, Truncated BPTT) — Implementation Pseudocode

> Roadmap ref: #N21 (Tier 4) · Target: `neural_network/layer` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```text
# sizes (all compile-time):
#   X = InputSize, H = HiddenSize, K = BpttWindow (truncation depth, K ≥ 1)
#   gate order q ∈ {i = 0, f = 1, g = 2, o = 3} (PyTorch order; Keras i, f, c, o is the same); gate row ρ = q·H + u, u ∈ [0, H)
#   P = 4H·X + 4H·H + 4H = 4·(H·(X + H) + H)
#   S = StateSize = 2H, state = [h; c]  (hidden state first, cell state second)
#
# θ layout (flat, "multiplicative blocks, then biases", as Dense / N11 / N17 / N20):
#   W_x[ρ, j]  at ρ·X + j                        ρ ∈ [0, 4H), j ∈ [0, X)    (input weights, row-major 4H×X)
#   W_h[ρ, k]  at 4H·X + ρ·H + k                 ρ ∈ [0, 4H), k ∈ [0, H)    (recurrent weights, row-major 4H×H)
#   b[ρ]       at 4H·X + 4H·H + ρ                                          (one fused bias)
#   ⇒ PyTorch nn.LSTM: W_x = weight_ih_l0, W_h = weight_hh_l0, b = bias_ih_l0 + bias_hh_l0 (no transpose)
#   ⇒ Keras LSTM: W_x = kernelᵀ, W_h = recurrent_kernelᵀ, b = bias (no gate permutation)
#   The two framework biases only ever appear as a sum, so the fold is exact (N13 exporter rule).

template<typename T, std::size_t InputSize, std::size_t HiddenSize, std::size_t BpttWindow>
                                                 # static_assert(std::is_floating_point_v<T>); instantiated for float
                                                 # static_assert(InputSize > 0 && HiddenSize > 0 && BpttWindow > 0)
class Lstm final : public Layer<T, X, H, P>,             # N8 trainable layer (accumulating Backward)
                   public StatefulLayer<T, 2H>:          # N17 stateful mixin (StateSize = 2H)
    Sigmoid<T>       gate                        # by value, final type ⇒ direct (devirtualised) Forward calls
    Tanh<T>          squash                      # by value, final type; used for g and for tanh(c)
    ParameterVector  parameters                  # θ, layout above
    ParameterVector  parameterGradients          # G = Σ ∇̃_θ L since the last ZeroGradients() (N8 contract)
    StateVector      state                       # [h_t; c_t], 2H floats — what State() returns
    OutputVector     output                      # h_t, a copy of state[0..H) — what Output() returns
    InputVector      inputGradient               # ∂L_t/∂x_t of the last Backward (per step, not accumulated)
    std::array<T, K·X> inputs{}                  # ring: x_s
    std::array<T, K·H> previousOutputs{}         # ring: h_{s−1} (the hidden state step s read)
    std::array<T, K·H> previousCells{}           # ring: c_{s−1} (the cell state step s read)
    std::array<T, K·H> inputGates{}              # ring: i_s
    std::array<T, K·H> forgetGates{}             # ring: f_s
    std::array<T, K·H> candidates{}              # ring: g_s
    std::array<T, K·H> outputGates{}             # ring: o_s
    std::array<T, K·H> cellTanhs{}               # ring: tanh(c_s)
    std::size_t      newest = K − 1              # ring slot of the most recent step
    std::size_t      count  = 0                  # valid ring entries = min(K, steps since ResetState/SetState)
    # ring = K·(X + 7H) floats. tanh(c_s) is stored so Backward calls no transcendental; the gate values
    # cannot be recovered from h_s or c_s. h_{s−1} and c_{s−1} of the oldest step are the (possibly seeded)
    # window start and must be stored in any case.

template<typename T, std::size_t InputSize, std::size_t HiddenSize>
                                                 # same static_asserts (no K: nothing to back-propagate)
class FlashLstm final : public InferenceLayer<T, X, H>,      # N8 inference base, N10 storage policy
                        public StatefulLayer<T, 2H>:
    static constexpr std::size_t WeightCount = P                  # == Lstm<T, X, H, K>::ParameterSize
    using WeightVector = math::Vector<T, WeightCount>
    using Weights      = FlashWeights<T, WeightCount>             # N10 read-only view, same θ layout
    Weights      weights                         # in flash (.rodata)
    Sigmoid<T>   gate
    Tanh<T>      squash
    StateVector  state{}                         # [h_t; c_t]
    OutputVector output{}                        # h_t — state and output are the only RAM-resident floats (3H)
```

Class diagram (N8/N10/N17 hierarchy, extended):

```text
InferenceLayer<T, In, Out>        Forward, Output
 ├─ FlashLstm<T, X, H>            (+ StatefulLayer<T, 2H>)                  ← inference only, weights in flash
 └─ Layer<T, In, Out, P>          + Backward, Parameters, SetParameters, ParameterGradients, ZeroGradients
      └─ Lstm<T, X, H, K>         (+ StatefulLayer<T, 2H>)                  ← trainable, truncated BPTT
StatefulLayer<T, S>               ResetState, State, SetState               ← from N17; S = 2H here ([h; c])
```

## Interface

```text
# Lstm<T, X, H, K>
using InputWeightMatrix     = math::Matrix<T, 4H, X>     # rows (i; f; g; o), PyTorch weight_ih_l0 shape
using RecurrentWeightMatrix = math::Matrix<T, 4H, H>     # rows (i; f; g; o), PyTorch weight_hh_l0 shape
static constexpr std::size_t Window = K

Lstm(const InputWeightMatrix& inputWeights, const RecurrentWeightMatrix& recurrentWeights)
                                                 # b_i = b_g = b_o = 0, b_f = 1 (Jozefowicz et al.); state = 0
void                   Forward(const InputVector& input) override                 # hot path; one time step
const InputVector&     Backward(const OutputVector& outputGradient) override      # hot path; truncated BPTT, accumulates G
const OutputVector&    Output() const override                                    # h_t
const ParameterVector& Parameters() const override
void                   SetParameters(const ParameterVector& parameters) override  # b taken verbatim; touches neither G nor state/history
const ParameterVector& ParameterGradients() const override                        # N8
void                   ZeroGradients() override                                   # N8
void                   ResetState() override                                      # N17 StatefulLayer: h = c = 0
const StateVector&     State() const override                                     # [h_t; c_t]
void                   SetState(const StateVector& state) override                # seeds h and c; new truncation window
# inherited: using ValueType = T (from Layer only); virtual ~Layer() = default; virtual ~StatefulLayer() = default

# FlashLstm<T, X, H>
FlashLstm(const WeightVector& weights)
FlashLstm(const WeightVector&& weights) = delete                                 # a temporary would dangle (N10)
explicit FlashLstm(Weights weights)                                              # raw flash region (N13 OTA blob)
void                            Forward(const InputVector& input) override       # hot path
const OutputVector&             Output() const override
std::span<const T, WeightCount> Parameters() const                               # the flash view, never a copy
void ResetState() override;  const StateVector& State() const override;  void SetState(const StateVector&) override
# deliberately absent: Backward, SetParameters, ParameterGradients, ZeroGradients
```

`Model` needs no change: `Model::ResetState()` and `detail::StatefulLayerType` (both from N17) pick up `Lstm`
and `FlashLstm` through `StateSize = 2H` and the `StatefulLayer<T, 2H>` base.

## Algorithm (pseudocode)

```text
# index helpers (inline): Wx(ρ, j) = parameters[ρ·X + j]    Wh(ρ, k) = parameters[4H·X + ρ·H + k]
#                         B(ρ)     = parameters[4H·X + 4H·H + ρ]
#                         GWx, GWh, GB: the same offsets into parameterGradients
# u = hidden unit, q = gate; the input gate value is called ig to keep "i" free

function Lstm(Wx_in, Wh_in):
    for ρ in 0..4H-1:
        for j in 0..X-1:  parameters[ρ·X + j]          = Wx_in.at(ρ, j)
        for k in 0..H-1:  parameters[4H·X + ρ·H + k]   = Wh_in.at(ρ, k)
    for u in 0..H-1:      parameters[4H·X + 4H·H + H + u] = 1          # forget-gate bias b_f = 1
    # other biases, G, state, output, rings value-initialised to 0; newest = K − 1, count = 0

function Forward(x):                             # OPTIMIZE_FOR_SPEED
    newest = (newest + 1 == K) ? 0 : newest + 1  # compare-and-reset, no modulo
    e = newest
    inputs[e·X .. e·X + X)          = x
    previousOutputs[e·H .. e·H + H) = state[0 .. H)        # h_{t−1}
    previousCells[e·H .. e·H + H)   = state[H .. 2H)       # c_{t−1}; both read below from the ring
    if count < K: count += 1
    hp = previousOutputs[e·H ..);  cp = previousCells[e·H ..)   # views, not copies
    for u in 0..H-1:
        a_i = B(u);  a_f = B(H + u);  a_g = B(2H + u);  a_o = B(3H + u)
        for j in 0..X-1:
            a_i += Wx(u, j)·x[j];  a_f += Wx(H + u, j)·x[j];  a_g += Wx(2H + u, j)·x[j];  a_o += Wx(3H + u, j)·x[j]
        for k in 0..H-1:
            a_i += Wh(u, k)·hp[k]; a_f += Wh(H + u, k)·hp[k]; a_g += Wh(2H + u, k)·hp[k]; a_o += Wh(3H + u, k)·hp[k]
        ig = gate.Forward(a_i);  fg = gate.Forward(a_f);  og = gate.Forward(a_o)   # stable σ (Sigmoid.hpp)
        gg = squash.Forward(a_g)
        c  = fg·cp[u] + ig·gg
        tc = squash.Forward(c)
        inputGates[e·H + u] = ig;  forgetGates[e·H + u] = fg;  candidates[e·H + u] = gg
        outputGates[e·H + u] = og; cellTanhs[e·H + u] = tc
        h  = og·tc
        state[u] = h;  state[H + u] = c;  output[u] = h

function Backward(gv):                           # OPTIMIZE_FOR_SPEED; gv = ∂L_t/∂h_t, size H
    really_assert(count > 0)
    dh = gv;  next = OutputVector{};  dc = OutputVector{}  # dc = ∂L_t/∂c_s from later steps; 0 at s = t
    δ  = math::Vector<T, 4H>{}                   # (δi; δf; δg; δo); 7H floats of stack in total
    e = newest
    for m in 0..count-1:                         # step s = t − m, newest to oldest
        for u in 0..H-1:
            ig = inputGates[e·H + u];  fg = forgetGates[e·H + u];  gg = candidates[e·H + u]
            og = outputGates[e·H + u]; tc = cellTanhs[e·H + u];    cp = previousCells[e·H + u]
            dct          = dc[u] + dh[u]·og·(1 − tc·tc)          # total ∂L/∂c_s
            δ[3H + u]    = dh[u]·tc·(og·(1 − og))                 # ∂L/∂a_o
            δ[u]         = dct·gg·(ig·(1 − ig))                   # ∂L/∂a_i
            δ[H + u]     = dct·cp·(fg·(1 − fg))                   # ∂L/∂a_f
            δ[2H + u]    = dct·ig·(1 − gg·gg)                     # ∂L/∂a_g
            dc[u]        = dct·fg                                 # ∂L/∂c_{s−1}; in place, row u only
        if m == 0:                               # ∂L_t/∂x_t = W_xᵀ δ — overwritten, per step
            for j in 0..X-1:
                s = 0
                for u in 0..H-1: for q in 0..3: s += Wx(q·H + u, j)·δ[q·H + u]
                inputGradient[j] = s
        for ρ in 0..4H-1:
            for j in 0..X-1: GWx(ρ, j) += δ[ρ]·inputs[e·X + j]
            for k in 0..H-1: GWh(ρ, k) += δ[ρ]·previousOutputs[e·H + k]
            GB(ρ) += δ[ρ]
        if m + 1 == count: break                 # truncation: nothing flows into h, c before the window
        for k in 0..H-1:                         # ∂L/∂h_{s−1} = W_hᵀ δ
            s = 0
            for u in 0..H-1: for q in 0..3: s += Wh(q·H + u, k)·δ[q·H + u]
            next[k] = s
        swap(dh, next)
        e = (e == 0) ? K − 1 : e − 1
    return inputGradient

function ResetState():       state = StateVector{};  output = OutputVector{};  count = 0;  newest = K − 1
function SetState(s):        state = s;  output = s[0 .. H);                    count = 0;  newest = K − 1
function State():            return state
function Output():           return output
function SetParameters(p):   parameters = p              # b_f taken as given (no runtime +1); history kept (N17 note)
function ZeroGradients():    parameterGradients = ParameterVector{}
function ParameterGradients(): return parameterGradients

function FlashLstm::Forward(x):                  # OPTIMIZE_FOR_SPEED; same summation order as Lstm
    for u in 0..H-1:                             # reads h_{t−1} from output, c_{t−1}[u] from state[H + u]
        (a_i, a_f, a_g, a_o) as in Lstm::Forward, reading weights[...] and output[k] instead of hp[k]
        ig = gate.Forward(a_i);  fg = gate.Forward(a_f);  og = gate.Forward(a_o);  gg = squash.Forward(a_g)
        c = fg·state[H + u] + ig·gg              # row u reads and writes only c[u]
        state[H + u] = c;  state[u] = og·squash.Forward(c)       # output still holds h_{t−1} for later rows
    output = state[0 .. H)                       # no stack copy needed
function FlashLstm::ResetState():   state = StateVector{};  output = OutputVector{}
function FlashLstm::SetState(s):    state = s;  output = s[0 .. H)
```

Math (one sequence after `ResetState`/`SetState` at step `t₀`; `W^q` = the `H` rows of gate `q`; `σ` logistic):

```text
forward:     a_s = W_x x_s + W_h h_{s−1} + b                     (4H; blocks a^i, a^f, a^g, a^o)
             i_s = σ(a^i),  f_s = σ(a^f),  g_s = tanh(a^g),  o_s = σ(a^o)
             c_s = f_s ⊙ c_{s−1} + i_s ⊙ g_s
             h_s = o_s ⊙ tanh(c_s)                              (h_{t₀} = c_{t₀} = 0 after ResetState)
derivatives: σ' = σ(1 − σ),  tanh' = 1 − tanh²                  (evaluated on the stored i, f, g, o, tanh c)
             ∂h_s/∂c_s = diag(o_s ⊙ (1 − tanh² c_s)),  ∂h_s/∂o_s = diag(tanh c_s)
             ∂c_s/∂c_{s−1} = diag(f_s),  ∂c_s/∂f_s = diag(c_{s−1}),  ∂c_s/∂i_s = diag(g_s),  ∂c_s/∂g_s = diag(i_s)
local error: dh = ∂L_t/∂h_s (gv at s = t, else the recursion below); dc = ∂L_t/∂c_s through c_{s+1} (0 at s = t)
             d̂c = dc + dh ⊙ o_s ⊙ (1 − tanh² c_s)
             δo = dh ⊙ tanh(c_s) ⊙ o_s ⊙ (1 − o_s)
             δi = d̂c ⊙ g_s ⊙ i_s ⊙ (1 − i_s)
             δf = d̂c ⊙ c_{s−1} ⊙ f_s ⊙ (1 − f_s)
             δg = d̂c ⊙ i_s ⊙ (1 − g_s²)
recursion:   ∂L_t/∂c_{s−1} = d̂c ⊙ f_s                          (the additive cell path)
             ∂L_t/∂h_{s−1} = W_hᵀ δ_s,   δ_s = (δi; δf; δg; δo)
input grad:  ∂L_t/∂x_t = W_xᵀ δ_t                               (returned; current step only)
param grads: ∇̃_{W_x} L_t = Σ_s δ_s x_sᵀ,   ∇̃_{W_h} L_t = Σ_s δ_s h_{s−1}ᵀ,   ∇̃_b L_t = Σ_s δ_s
             sums over s = t−n+1 … t, n = count = min(K, t − t₀)
truncation:  h_{t−n} and c_{t−n} are treated as constants. If t − t₀ ≤ K they are the ResetState/SetState
             values and do not depend on θ, so ∇̃ L_t = ∇ L_t exactly; otherwise terms through step t − K are dropped.
accumulation: G ← G + ∇̃_θ L_t per Backward (N8); many-to-many and many-to-one exactly as N17.
```

The loss sees only `h_t` (the layer output), so each `Backward` starts with `dc = 0`. `c` reaches the loss only through
later `h` values, which is exactly what the `d̂c ⊙ f_s` recursion carries back. Truncated BPTT is the same per-step
`BPTT(h)` with depth `h = K` as N17 (Williams & Peng); `BPTT(h; h′)` stays deferred.

## Complexity & memory

- `Forward`: `4H·X + 4H²` MACs + `3H` sigmoid + `2H` tanh (`g` and `tanh c`) + `O(H)` elementwise ops +
  `X + 2H` ring copies. No loop depends on `K`.
- `Backward`, with `n = count ≤ K`: `n·(4H·X + 4H²)` MACs (weight gradients) + `(n − 1)·4H²` MACs (state
  recursion) + `4H·X` MACs (input gradient) + `n·4H` bias adds + `O(n·H)` elementwise products. No
  transcendental calls: every derivative is a polynomial in the stored `i, f, g, o, tanh c`.
  `n = K`: `K·(4H·X + 8H²) − 4H² + 4H·X` MACs.
- RAM, `Lstm` (trainable): `2P + K·(X + 7H) + 3H + X` floats (`state` 2H, `output` H, `inputGradient` X) + two
  indices + vptrs (`Layer`, `StatefulLayer`, and one each for the by-value `Sigmoid`/`Tanh`). `Backward` adds
  `7H` floats of stack (`dh`, `next`, `dc`, `δ`); `Forward` none.
  - Fixture `Lstm<float, 2, 3, 4>`: `P = 72`, `144 + 92 + 9 + 2 = 247` floats.
  - Example `Lstm<float, 3, 16, 8>` (3-axis IMU, 16 units, 8-step window): `P = 1280`,
    `2560 + 920 + 48 + 3 = 3531` floats (14 124 B). `Forward` costs 1 216 MACs; a full-window `Backward`
    costs 17 088 MACs. The same-size GRU (N20) has `P = 1008` and 2 699 floats; the Elman cell (N17) `P = 320`.
  - The ring is `K·(X + 7H)` floats (920 here): four gate activations, `tanh c`, and the two incoming states.
- RAM, `FlashLstm`: `3H` floats (`state` + `output`) + one pointer (the static-extent span) + vptrs; no stack
  buffer in `Forward`. Flash holds `P` floats (5 120 B for the example above).

## Numerical / embedded notes

- **Hidden state bounded, cell state not.** `h_s = o_s ⊙ tanh(c_s) ∈ (−1, 1)`. The cell obeys
  `|c_s| ≤ f_s|c_{s−1}| + |i_s g_s| < |c_{s−1}| + 1`, so it grows at most linearly with sequence length (and
  settles at most at `1/(1 − f_max)` when `f ≤ f_max < 1`). Float32 cannot overflow on any realistic stream, but a
  large `|c|` saturates `tanh c`, which zeroes the output-path gradient. Periodic `ResetState()` bounds it for
  free-running streams.
- **Constant error carousel.** `∂c_s/∂c_{s−1} = diag(f_s)`: with the forget gate open (`f ≈ 1`) the cell error
  passes back almost unattenuated, which is why an LSTM keeps information longer than N17. That also lets gradients
  grow; clipping is N16's job. Truncation bounds the sweep to `K` steps.
- **Forget-gate bias 1 is an initial value only.** The constructor sets `b_f = 1` so `f ≈ σ(1) ≈ 0.73` at the start
  of training when the initial weights are small (Jozefowicz et al. 2015; Keras `unit_forget_bias=True`).
  `SetParameters` and imported weights are taken verbatim. TF1 `LSTMCell` weights are outside N13's scope: that
  cell *adds* `forget_bias = 1.0` at run time and orders its gates `i, j (= g), f, o`, so loading them would need
  the `f`/`g` blocks swapped and `+1` folded into `b_f`.
- **Vanilla LSTM only.** No peephole connections, no coupled input/forget gate, no projection (`proj_size`), matching
  PyTorch `nn.LSTM` and Keras `LSTM`. Greff et al. found that dropping peepholes or coupling the input and forget
  gates does not significantly hurt, while the forget gate and the `tanh` on the cell output are critical. Gate
  activations are fixed to `σ, tanh, tanh`: N13 refuses a Keras `recurrent_activation` other than `sigmoid`
  (older standalone Keras defaulted to `hard_sigmoid`).
- **Saturation.** When float32 `σ` rounds to exactly `0`/`1` or `tanh` to `±1`, the matching derivative factor is
  exactly `0`, not a NaN. `Sigmoid::Forward` already branches on the sign (`e^{−|a|}` never overflows).
- **Why the ring stores `tanh c_s`.** `c_s` itself is only needed as `c_{s−1}` of the next step (for `δf`), which is
  already in `previousCells`; storing `tanh c_s` avoids `H` tanh calls per step in `Backward`. A recompute mode
  (recover `tanh c_s` from `previousCells` of step `s + 1`, and `h_{s−1} = o_{s−1} ⊙ tanh c_{s−1}` inside the window)
  would save about `2K·H` floats and is deferred.
- **Same streaming contract as N17/N20.** `Backward` returns only `∂L_t/∂x_t`; a trainable layer below the LSTM
  (including a second stacked `Lstm`) receives a one-step gradient. Skipping `Backward` on steps without a loss is
  fine; calling it twice after one `Forward` adds the step twice; `SetParameters` mid-window mixes stale gates with new
  weights (step θ between sequences for an exact gradient); `ResetState`/`SetState` start a new truncation window and
  do not clear the ring arrays (`count` is the only validity marker).
- **State layout.** `State()` is `[h; c]` (PyTorch returns `(h_n, c_n)`, Keras `[h, c]`); `Output()` is a separate
  `H`-float copy of `h`, because a `math::Vector<T, H>` reference cannot alias the first half of a
  `math::Vector<T, 2H>`. `SetState` updates both.
- **Summation order.** Every pre-activation: bias, then `W_x x`, then `W_h h`; the cell update as
  `f·c_{t−1} + i·g`. `FlashLstm` uses the same order, so the two agree to rounding. Under `-ffast-math` the compiler
  may reassociate, so tests use `EXPECT_NEAR`.
- Under `-ffast-math` non-finite inputs, states or weights are outside the contract.
- Float-only: `static_assert(std::is_floating_point_v<T>)` in both templates. Gates fit `(0, 1)` and `h` fits
  `(−1, 1)`, but `c`, pre-activations, accumulated gradients and SGD steps do not fit `Q15`/`Q31`. The generic `T`
  signature keeps a fixed-point specialisation cheap to add later.

## Dependencies

- Requires **N8**: `Layer<T, In, Out, P>` with the accumulating `Backward`, `ParameterGradients()` and
  `ZeroGradients()`, plus `InferenceLayer`.
- Requires **N17**: `StatefulLayer<T, S>` (here `S = 2H`), `Model::ResetState()` / `detail::StatefulLayerType`, the
  two-index ring and the newest-to-oldest truncated-BPTT sweep.
- Builds on **N20**: the multi-gate ring and the gate-row `θ` indexing (`ρ = q·H + u`); N21 adds a second recurrent
  state (`c`) and its backward recursion. No code is shared beyond the pattern, so N21 could land without N20.
- Also builds on: `Sigmoid.hpp` (numerically stable, `final`), `Tanh.hpp` (`final`), the Dense row loop,
  `numerical/math/Matrix.hpp` (`math::Vector`, `math::Matrix`, zero-filling default ctor, `constexpr` ctor),
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`, `<array>`, `<span>`.
- **N10** for `FlashLstm`: `FlashWeights<T, P>` and the relaxed `Model` that admits `InferenceLayer` types. If N21
  lands before N10, ship `Lstm` alone and add `FlashLstm` with N10.
- Soft: **N3** initialises each gate block of `W_x` (`fanIn = X`, `fanOut = H`) and `W_h` (`fanIn = H`,
  `fanOut = H`) via `Fill`; the forget-gate bias `1` is set here, not by N3. **N9** supplies the per-step loss
  gradient. **N16** provides clipping and the optimiser loop. **N13** emits `θ` in exactly this layout (PyTorch as-is
  with `b = b_ih + b_hh`; Keras transposed, no permutation).
- Composes with `Model` through `make_layer<Lstm<float, X, H, K>>(inputWeights, recurrentWeights)`; `OutputSize = H`
  chains into a `Dense<float, H, …>` head. One `Model::Forward` call is one time step.

## Deployment

- Header: `neural_network/layer/Lstm.hpp` — `#pragma once` → `#pragma GCC optimize("O3","fast-math")` →
  includes (`neural_network/layer/TrainableLayerInterface.hpp`, `neural_network/layer/StatefulLayer.hpp`,
  `neural_network/layer/FlashResidentWeights.hpp`, `neural_network/activation/Sigmoid.hpp`,
  `neural_network/activation/Tanh.hpp`, `numerical/math/Matrix.hpp`, `numerical/math/CompilerOptimizations.hpp`,
  `infra/util/ReallyAssert.hpp`, `<array>`, `<span>`). `OPTIMIZE_FOR_SPEED` on `Forward`/`Backward` of both
  classes. Under `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`:
  `extern template class Lstm<float, 2, 3, 4>;` and `extern template class FlashLstm<float, 2, 3>;`
- Coverage: `neural_network/layer/Lstm.cpp` → `template class Lstm<float, 2, 3, 4>; template class FlashLstm<float, 2, 3>;`
- Test: `neural_network/layer/test/TestLstm.cpp`, plus one integration case in
  `neural_network/model/test/TestModel.cpp` (see `tests.md`).
- Doc: `doc/layer/Lstm.md` (per `doc/TEMPLATE.md`), with a row in `doc/layer/README.md`.
- CMake: `Lstm.hpp` → `target_sources(neural_network.layer PRIVATE ...)`; `Lstm.cpp` →
  `neural_network_add_coverage_sources(neural_network.layer ...)`; `TestLstm.cpp` → `neural_network.layer_test`.
  `neural_network.layer` must link `neural_network.activation` (already required by N17's `Tanh`).
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
