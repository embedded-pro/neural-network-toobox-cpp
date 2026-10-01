# Mini-Batch SGD Training Step — Implementation Pseudocode

> Roadmap ref: #N16 (Tier 3) · Target: `neural_network/model` · Namespace `neural_network` · Type: `float` (templated on `T`, instantiated for `float` only)

## Data structures

```
enum class TrainingStatus : std::uint8_t:
    Accumulating         # sample added, batch not full yet: θ unchanged
    Stepped              # θ updated with the unclipped mean gradient
    Clipped              # θ updated, the data gradient was rescaled to ‖·‖ = maxGradientNorm
    SkippedNonFinite     # ‖ḡ‖² was NaN/±∞: θ unchanged, batch discarded
    Idle                 # Flush() with no pending sample: nothing done

template<typename T>
struct TrainingConfiguration:                    # aggregate; designated initialisers in tests
    T                learningRate{ T{ 0.01 } }   # η > 0
    T                weightDecay{ T{ 0 } }       # λ ≥ 0; R(θ) = λ/2 Σ_{trainable} θ_j²  (regularization::L2 convention)
    std::optional<T> maxGradientNorm{}           # c > 0; std::nullopt = no clipping

template<typename T>
struct StepResult:
    TrainingStatus status
    std::size_t    samples        # k: samples averaged by this step (pending count while Accumulating)
    T              meanLoss       # (1/k) Σ_s L_s at the pre-step θ, data term only; 0 while Accumulating/Idle
    T              gradientNorm   # ‖ḡ‖₂ of the masked mean data gradient before clipping; 0 while Accumulating/Idle

template<typename T>
struct EpochResult:
    T           meanLoss          # (1/N) Σ_b k_b · meanLoss_b = (1/N) Σ_s L_s
    std::size_t steps             # batches that updated θ (Stepped + Clipped)
    std::size_t skippedSteps      # batches discarded as SkippedNonFinite

template<typename T, std::size_t BatchSize, typename ModelType, typename Target = typename ModelType::OutputVector>
class MiniBatchTraining:
    static_assert(std::is_floating_point_v<T>, "MiniBatchTraining requires a floating-point type")
    static_assert(BatchSize > 0, "MiniBatchTraining requires a positive batch size")
    static_assert(std::is_same_v<typename ModelType::ValueType, T>, "Model and trainer must share T")

    using InputVector  = typename ModelType::InputVector
    using OutputVector = typename ModelType::OutputVector
    using LossType     = Loss<T, OutputVector::size, Target>          # N9 per-sample contract
    using TargetType   = Target
    using LayerMask    = std::array<bool, ModelType::NumberOfLayers>  # true = trainable

    ModelType&                model            # owns θ and the accumulated G (N8)
    const LossType&           loss             # stateless per-sample loss (N9)
    TrainingConfiguration<T>  configuration
    LayerMask                 trainable        # default: all true
    std::size_t               pending = 0      # samples accumulated since the last step, 0 ≤ pending < BatchSize
    T                         lossSum = 0      # Σ L_s of the pending samples
```

No sample buffer: each sample is consumed by one `Forward`/`Backward` and only its parameter gradient
survives, summed inside the layers (N8). The batch size is a compile-time bound on `pending`, not a
storage size.

## Interface

```
MiniBatchTraining(ModelType& model, const LossType& loss, const TrainingConfiguration<T>& configuration)
    # really_assert(η > 0, λ ≥ 0, ηλ < 1, !c || *c > 0); calls model.ZeroGradients() (discard stale G)

StepResult<T> Accumulate(const InputVector& input, const TargetType& target)   # hot path; steps on the B-th sample
StepResult<T> Flush()                                  # steps on the 0 < k < B pending samples (end of data)
template<std::size_t NumberOfSamples>
EpochResult<T> TrainEpoch(const std::array<InputVector, NumberOfSamples>& inputs,
                          const std::array<TargetType, NumberOfSamples>& targets)
                                                       # static_assert(NumberOfSamples > 0); really_assert(PendingSamples() == 0)
                                                       # in array order; ends with Flush()
void          SetTrainableLayers(const LayerMask& mask)   # really_assert(at least one trainable layer with ParameterSize > 0)
void          SetLearningRate(T learningRate)              # schedules live in application code; same asserts as the ctor
std::size_t   PendingSamples() const
private: StepResult<T> Step();  void Finish()           # shared by Accumulate (k = B) and Flush (k < B)
```

Additions to `Model<T, In, Out, Layers...>` (small, made by this item):

```
using ValueType = T
static constexpr std::size_t NumberOfLayers = sizeof...(Layers)
template<std::size_t I> auto&       LayerAt()           # std::get<I>(layers)
template<std::size_t I> const auto& LayerAt() const
# removed: Train(...) and the numerical/optimization include — this item replaces it
```

`Model::Forward`, `Model::Backward` (accumulating after N8), `Model::ZeroGradients` (N8) and
`Layer::Parameters/SetParameters/ParameterGradients` (N8) are used unchanged.

## Algorithm (pseudocode)

```
function Accumulate(x, y):                          # OPTIMIZE_FOR_SPEED
    ŷ = model.Forward(x)
    lossSum += loss.Cost(ŷ, y)
    model.Backward(loss.Gradient(ŷ, y))             # every layer adds ∇θ_ℓ L_s into its G_ℓ (N8)
    pending += 1
    if pending < BatchSize:
        return { Accumulating, pending, 0, 0 }
    return Step()

function Flush():
    if pending == 0: return { Idle, 0, 0, 0 }
    return Step()

function Step():                                    # OPTIMIZE_FOR_SPEED; pending = k ≥ 1
    k      = pending
    invK   = T{1} / T(k)
    # pass 1 — squared norm of the masked mean data gradient, read in place (no copy)
    n2 = Σ over layers ℓ with trainable[ℓ] and ParameterSize_ℓ > 0:            # fold over index_sequence
             Σ_j (G_ℓ[j] · invK)²                   # G_ℓ = LayerAt<ℓ>().ParameterGradients()
    meanLoss = lossSum · invK
    if !math::IsFinite(n2):                         # NaN/∞ in any trainable G, or ‖ḡ‖ > ~1.8e19
        Finish();  return { SkippedNonFinite, k, meanLoss, math::Sqrt(n2) }
    norm  = math::Sqrt(n2)
    scale = invK;  status = Stepped
    if configuration.maxGradientNorm and norm > *maxGradientNorm:
        scale = invK · (*maxGradientNorm / norm);  status = Clipped
    # pass 2 — per-layer update, one layer-sized scratch copy at a time
    for each layer ℓ with trainable[ℓ] and ParameterSize_ℓ > 0:                # fold over index_sequence
        θ' = LayerAt<ℓ>().Parameters()              # copy: P_ℓ floats
        G  = LayerAt<ℓ>().ParameterGradients()      # reference
        for j in 0..P_ℓ-1:
            θ'[j] -= η · (scale · G[j] + λ · θ'[j]) # θ'[j] on the right is the pre-step value
        LayerAt<ℓ>().SetParameters(θ')
    Finish()
    return { status, k, meanLoss, norm }

function Finish():
    model.ZeroGradients()                           # every layer, frozen ones included
    pending = 0;  lossSum = 0

function TrainEpoch(inputs, targets):
    really_assert(pending == 0)
    total = 0;  steps = 0;  skipped = 0
    for s in 0..N-1:
        r = Accumulate(inputs[s], targets[s]);  tally(r)
    tally(Flush())
    return { total / N, steps, skipped }
    where tally(r): if r.status ∉ { Accumulating, Idle }:
                        total += T(r.samples) · r.meanLoss
                        r.status == SkippedNonFinite ? ++skipped : ++steps
```

Frozen layers still run `Forward`/`Backward` (their input gradient feeds earlier layers, and their own
`G_ℓ` is filled). Their `G_ℓ` is simply never read and is cleared by `ZeroGradients()`, and their θ is
never written. Zero-parameter layers (N12) are skipped by `if constexpr (ParameterSize > 0)`.

### Math

Mini-batch `S` of `k` samples (`k = B`, or `1 ≤ k < B` for the flushed remainder), per-sample loss
`L_s(θ) = L(f(x_s; θ), y_s)` from N9, trainable index set `𝒯` (union of the slices of the trainable
layers), `P_𝒯 = |𝒯|`:

```
objective:    J_S(θ) = (1/k) Σ_{s∈S} L_s(θ) + (λ/2) Σ_{j∈𝒯} θ_j²

per sample:   ∇_θ L_s = J_f(x_s; θ)ᵀ · ∇_ŷ L(ŷ_s, y_s)              # reverse mode: Model::Backward(loss.Gradient(ŷ, y))
              layer ℓ receives g_ℓ = ∂L_s/∂a_ℓ and adds ∂L_s/∂θ_ℓ into G_ℓ    (Dense: δ xᵀ and δ, δ = J_f(z)ᵀ g; see N8)

batch:        G = Σ_{s∈S} ∇_θ L_s                                    # exact: θ is constant while S is accumulated
              ḡ_j = G_j / k            for j ∈ 𝒯,     ḡ_j = 0 for j ∉ 𝒯
              ∂/∂θ_j [(λ/2) θ_j²] = λ θ_j                              # = regularization::L2<T, P>::Gradient

clipping:     ρ = min(1, c / ‖ḡ‖₂)          (ρ = 1 without c)          # rescales the data term only
update:       θ_j ← θ_j − η (ρ ḡ_j + λ θ_j) = (1 − ηλ) θ_j − η ρ ḡ_j       for j ∈ 𝒯
              θ_j unchanged                                            for j ∉ 𝒯
```

Without clipping (`ρ = 1`) the update is exactly one gradient-descent step on `J_S`, i.e.
`θ ← θ − η ∇J_S(θ)` restricted to `𝒯`. This is what the finite-difference tests check. For plain SGD, coupled L2
(`+λθ` in the gradient) and decoupled weight decay (`(1 − ηλ)θ`) are the same update. They differ only for
adaptive optimisers (N14 Adam), which this item does not include. `ηλ < 1` keeps `1 − ηλ ∈ (0, 1)`, so the decay
shrinks θ and never flips its sign.

The trainer adds no backward pass of its own. The per-sample gradient is N8's back-propagation driven by
N9's `∂L/∂ŷ`. The trainer's own terms are linear (the `1/k` mean, the mask projection) or the exact
derivative of the penalty (`λθ`). Clipping is a step-size rule, not a gradient. The `ρ` factor is not
differentiated.

`meanLoss` reports `(1/k) Σ L_s` evaluated during the forward passes, at the pre-step θ. It excludes
the penalty, so computing it costs no pass over θ.

## Complexity & memory

Let `M = Σ_ℓ In_ℓ·Out_ℓ` (weights), `P = TotalParameters`, `P_𝒯` trainable parameters,
`P_max = max_ℓ P_ℓ`, `L = NumberOfLayers`.

- `Accumulate` (per sample): `Forward` `M` MACs + activations; `Backward` `2M` MACs (input gradient +
  gradient accumulate) + activation VJPs; loss `Cost` + `Gradient` `O(Out)`. Roughly `3M` MACs per sample,
  the usual back-propagation cost.
- `Step` (per batch): pass 1 `P_𝒯` multiply-adds + 1 `Sqrt`; pass 2 `P_𝒯` copies + `3·P_𝒯` flops +
  `P_𝒯` stores in `SetParameters`; `ZeroGradients` `P` stores. `O(P)` per batch, i.e. `O(P/B)` per sample
  amortised. No transcendental beyond one `Sqrt` and one division (`c/‖ḡ‖`, only when clipping).
- `TrainEpoch`: `N` × `Accumulate` + `⌈N/B⌉` × `Step`.
- Trainer RAM: two references, 3 floats of configuration (+ 1 `bool` for the optional), `L` bools of mask,
  one `std::size_t` and one float. That is about **8 words + `L` bytes**, independent of `P` and `B`.
- Stack: `Step` holds one layer's `θ'` (`P_max` floats). No `P`-sized vector is ever built: pass 1 reads
  `G_ℓ` in place, and `Model::Gradients()` (N8, by value) is not used. `Accumulate` holds `ŷ` and `∂L/∂ŷ`
  (`2·Out` floats) and the `In`-float input gradient returned by `Model::Backward`.
- Model RAM (for context, N8 `Dense`): `2P + 2Σ(In_ℓ + Out_ℓ)` floats, the same with or without training.
  Example: `Dense<64,64>` + `Dense<64,4>`: `P = 4420`, model `9232` floats (36.9 KB), `Step` scratch
  `P_max = 4160` floats (16.6 KB).
- Time and RAM per sample are independent of `B`. `B` only changes how often `Step` runs.

## Numerical / embedded notes

- **Divide once.** `G` holds the exact float sum (N8 bounds its relative error by `~B·2⁻²⁴`). `1/k` is
  applied once per step and folded into `scale`, never per sample. The flushed remainder uses its own `k`,
  so the last partial batch is still a true mean.
- **Non-finite guard.** `math::IsFinite` is bit-based, so it survives `-ffast-math`. A single NaN/∞ in any
  trainable `G_ℓ` makes `n2` non-finite. The batch is then discarded: θ is untouched, `G` is cleared,
  and `SkippedNonFinite` is returned so the application can lower η or stop. `n2` overflows for
  `‖ḡ‖ > ~1.8·10¹⁹`, which is treated as divergence (conservative). Frozen layers are not inspected.
- **Clipping order.** `ρ` rescales only the data gradient. `λθ` is added after clipping, so a clip never
  weakens the decay (the same as clipping `.grad` before an SGD step with `weight_decay` in PyTorch).
  `gradientNorm` reports the pre-clip norm, so it can be logged for monitoring.
- **Freeze mask.** A per-layer `bool`, in `Layers...` order. Typical MCU use: freeze the feature layers,
  train the head (`{false, …, false, true}`). Frozen layers are neither read in pass 1 nor written in
  pass 2. The mask may change between steps: plain SGD has no optimiser state that could carry over.
- **What the mask does not save.** Frozen layers still run `Backward` and keep their gradient buffers.
  To stop back-propagation at the first trainable layer, and to hold frozen layers as flash-resident
  inference-only layers (N10), `Model` would need a truncated `Backward` and to admit `InferenceLayer`
  types. That is a follow-up once N10 relaxes `Model`'s layer check. It is not part of this item.
- **Weight decay covers biases too.** This matches the regulariser the losses applied to their argument
  before N9, and `regularization::L2` on the flat θ. Per-parameter decay masks are out of scope.
- **Sample order.** `TrainEpoch` visits samples in array order; SGD assumes a shuffled stream. The caller
  shuffles an index array between epochs (for example Fisher–Yates driven by N3's `Xorshift32`) or passes
  pre-shuffled data. Learning-rate schedules and early stopping stay in application code
  (`SetLearningRate`, `EpochResult::meanLoss`).
- **Float-only.** Updates `η·ρ·ḡ` are routinely `10⁻⁵…10⁻⁷`, below the Q15 resolution `2⁻¹⁵ ≈ 3·10⁻⁵`,
  and accumulated gradients leave `[−1, 1)`. `static_assert(std::is_floating_point_v<T>)`; the generic `T`
  keeps a fixed-point specialisation cheap to add later.
- No heap, no recursion: the layer visits are fold expressions over `std::index_sequence`, as in `Model`.

## Dependencies

- **N8 (required)**: accumulating `Backward`, `Layer::ParameterGradients()`, `Model::ZeroGradients()`, and
  `Model::Gradients()` (tests only). Without N8, `Dense` overwrites its gradient and exposes nothing.
- **N9 (required)**: `Loss<T, N, Target>` with `const Cost/Gradient(prediction, target)`, the target
  passed per call, and no regulariser inside. `Target` may be `std::size_t` (class-index CCE), so one
  trainer type serves regression and classification.
- **N1 (recommended)**: `Identity` output for regression (MSE/Huber) and for the logit-input losses. Not
  needed by the tests (`Tanh` output + MSE).
- **N3 (optional)**: initial weights for training from scratch, and the PRNG for shuffling. Fine-tuning
  imported weights (N13) needs neither.
- **N14 (optional follow-up)**: momentum/Adam as `optimiser_ℓ.Step(θ'_ℓ, ρ ḡ_ℓ + λ θ'_ℓ)` in pass 2, one
  optimiser per trainable layer, so frozen layers carry no optimiser state (`P_ℓ` floats for momentum,
  `2P_ℓ` for Adam). With Adam it must choose coupled L2 or decoupled weight decay. Plain SGD ships
  without it.
- Upstream: `numerical/math/Math.hpp` (`Sqrt`, `IsFinite`), `numerical/math/Matrix.hpp`,
  `numerical/math/CompilerOptimizations.hpp`, `infra/util/ReallyAssert.hpp`; `numerical/regularization/L2.hpp`
  only as the test oracle for `R(θ)`.
- Replaces: `Model::Train(optimizer, objective, θ₀)` and its test `TrainAppliesOptimizerResult`.

## Deployment

- Header: `neural_network/model/MiniBatchTraining.hpp` — `#pragma once` →
  `#pragma GCC optimize("O3","fast-math")` (GCC/Clang guard) → includes `Model.hpp`,
  `neural_network/losses/LossContract.hpp` (N9), `numerical/math/Math.hpp`, `CompilerOptimizations.hpp`,
  `infra/util/ReallyAssert.hpp`, `<array>`, `<cstdint>`, `<optional>`, `<utility>`. `OPTIMIZE_FOR_SPEED` on
  `Accumulate` and `Step`. Add `extern template class MiniBatchTraining<float, 2, Model<float, 2, 1, Dense<float, 2, 3>, Dense<float, 3, 1>>>;`
  under `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`.
- Coverage: `neural_network/model/MiniBatchTraining.cpp` →
  `template class MiniBatchTraining<float, 2, Model<float, 2, 1, Dense<float, 2, 3>, Dense<float, 3, 1>>>;`
  (`TrainEpoch` is a member template; the tests instantiate it).
- `Model.hpp`: add `ValueType`, `NumberOfLayers`, `LayerAt<I>()`; delete `Train` and the
  `numerical/optimization` include. `Model.cpp` coverage is unchanged.
- Test: `neural_network/model/test/TestMiniBatchTraining.cpp` (new). In `TestModel.cpp`, delete
  `TrainAppliesOptimizerResult` with its `OptimizerMock` / objective mock.
- Doc: `doc/model/MiniBatchTraining.md` (per `doc/TEMPLATE.md`), with a row in `doc/model/README.md`.
  In `doc/model/Model.md`, replace the parameter-space training step section with a pointer to it and
  drop the "real training is roadmap N16" limitation. In `doc/model/NeuralNetwork.md`, point the
  parameter-update section at it.
- CMake: `MiniBatchTraining.hpp` → `target_sources(neural_network.model PRIVATE ...)`;
  `MiniBatchTraining.cpp` → `neural_network_add_coverage_sources(neural_network.model ...)`;
  `neural_network.model` links `neural_network.losses` (N9 dropped it) and drops `numerical.optimization`;
  `TestMiniBatchTraining.cpp` → `neural_network.model_test` (already QEMU-wired), which also links
  `numerical.regularization` for the L2 oracle.
- Generic pattern: see `roadmap/DEPLOYMENT.md`.
