# Neural Network Toolbox — Roadmap

Prioritized backlog for the neural network stack (activations, layers, losses, model, training).
Shared numerical primitives (`math`, `optimization`, `regularization`, `analysis`) are consumed from
[numerical-toolbox-cpp](https://github.com/embedded-pro/numerical-toolbox-cpp).
Every local item has a pseudocode spec under [roadmap/](roadmap/README.md); deploy one with
[roadmap/DEPLOYMENT.md](roadmap/DEPLOYMENT.md).

Difficulty legend:

| Stars | Meaning                                               | Typical effort |
|-------|-------------------------------------------------------|----------------|
| ★☆☆☆☆ | Trivial primitive — a few methods, minimal math       | Hours          |
| ★★☆☆☆ | Easy — well-defined math, small state, interface work | Half day       |
| ★★★☆☆ | Moderate — real algorithm, still finite/bounded       | 1–2 days       |
| ★★★★☆ | Advanced — gated recurrences, truncated BPTT          | Several days   |
| ★★★★★ | Hard / research-grade                                 | 1+ week        |

## Neural network algorithms

The library already has the inference building blocks: element-wise activations
([ReLU](neural_network/activation/ReLU.hpp), [LeakyReLU](neural_network/activation/LeakyReLU.hpp),
[Sigmoid](neural_network/activation/Sigmoid.hpp), [Tanh](neural_network/activation/Tanh.hpp),
[Softmax](neural_network/activation/Softmax.hpp)), a fully connected [Dense](neural_network/layer/Dense.hpp)
layer with forward/backward, four losses ([MSE](neural_network/losses/MeanSquaredError.hpp),
[MAE](neural_network/losses/MeanAbsoluteError.hpp), probability-input
[BCE](neural_network/losses/BinaryCrossEntropy.hpp), logit-input
[CCE](neural_network/losses/CategoricalCrossEntropy.hpp)) and a compile-time-checked sequential
[Model](neural_network/model/Model.hpp). What is missing:
- a **working training path**;
- a **per-sample loss contract**;
- the **MCU inference path**: linear output, flash-resident weights, convolution, pooling and weight export.

> **Top-priority open design limitation — the model is not trainable through the API.**
> - [`Model::Train`](neural_network/model/Model.hpp) minimises an `ObjectiveFunction` over the parameter
>   vector θ itself and never runs `Forward`/`Backward`.
> - Parameter gradients are not exposed: [Dense](neural_network/layer/Dense.hpp) keeps dW/db private and
>   overwrites them on every `Backward`, and [Layer.hpp](neural_network/layer/Layer.hpp) has no gradient accessor.
> - Losses are bound to one target at construction, so a dataset cannot be streamed sample by sample.
>
> **N8** (trainable-layer interface), **N9** (per-sample loss contract) and **N16** (mini-batch training
> step, replacing `Model::Train`) lift this limitation.

> **Inference-path limits.**
> - A parameter-free layer cannot be implemented, because `Matrix<T,0,1>` fails its positive-dimension
>   `static_assert` (N8 adds the zero-parameter specialisation).
> - Layers carry no sequence state (N17).
> - Dense copies its weights into RAM next to a same-size gradient buffer and parameter mirror:
>   `sizeof(Dense<float,64,64>)` = 50 960 B for 16 640 B of weights (N10).
>
> Critical path: inference N1 → N8 → N10 → N11/N12 → N13; training N8 + N9 (+ N3, N14) → N16.

All items are **float-only** ([AGENTS.md](AGENTS.md)): generic `template<typename T>` with
`static_assert(std::is_floating_point_v<T>)`, instantiated and tested on `float`. Pre-activations,
logits, losses and gradients are unbounded, while `Q15`/`Q31` span only [−1, 1), and SGD updates η·g
fall below the Q15 resolution 2⁻¹⁵ ≈ 3·10⁻⁵. Quantised inference, if ever added, would be affine int8
with per-channel scales, not `Q15`/`Q31` (see *Deferred*).

### Backlog (by priority)

| #   | Component                                                                                                                       | Target module                      | Difficulty |
|-----|---------------------------------------------------------------------------------------------------------------------------------|------------------------------------|------------|
| N1  | [Identity (linear) activation](roadmap/activation/Identity/implementation.md)                                                   | `activation`                       | ★☆☆☆☆      |
| N2  | [Per-feature affine normalisation (frozen BN / standardisation)](roadmap/layer/AffineNormalization/implementation.md)           | `layer`                            | ★☆☆☆☆      |
| N3  | [Weight initialisation (Glorot, He, LeCun) + deterministic PRNG](roadmap/initialization/WeightInitialization/implementation.md) | `initialization` (new)             | ★☆☆☆☆      |
| N4  | [Huber loss](roadmap/losses/HuberLoss/implementation.md)                                                                        | `losses`                           | ★☆☆☆☆      |
| N5  | [Swish family (SiLU / Hard-Sigmoid / Hard-Swish)](roadmap/activation/SwishActivations/implementation.md)                        | `activation`                       | ★☆☆☆☆      |
| N6  | [ELU / SELU activation](roadmap/activation/ExponentialLinearUnit/implementation.md)                                             | `activation`                       | ★☆☆☆☆      |
| N7  | [Classification metrics (argmax, accuracy, confusion matrix)](roadmap/metrics/ClassificationMetrics/implementation.md)          | `metrics` (new)                    | ★☆☆☆☆      |
| N8  | [Trainable-layer interface (inference/trainable split, gradients)](roadmap/layer/TrainableLayerInterface/implementation.md)     | `layer` + `model`                  | ★★☆☆☆      |
| N9  | [Per-sample loss contract + logit-input BCE](roadmap/losses/LossContract/implementation.md)                                     | `losses`                           | ★★☆☆☆      |
| N10 | [Flash-resident inference weights](roadmap/layer/FlashResidentWeights/implementation.md)                                        | `layer`                            | ★★☆☆☆      |
| N11 | [1D convolution](roadmap/layer/Convolution1D/implementation.md)                                                                 | `layer`                            | ★★☆☆☆      |
| N12 | [1D pooling (max / average / global-average)](roadmap/layer/Pooling1D/implementation.md)                                        | `layer`                            | ★★☆☆☆      |
| N13 | [Weight export (constexpr header) + blob (de)serialisation](roadmap/model/WeightExport/implementation.md)                       | `model` + `scripts`                | ★★☆☆☆      |
| N14 | Step optimisers (momentum, Adam) — upstream, no local spec                                                                      | `optimization` (numerical-toolbox) | ★★☆☆☆      |
| N15 | Log-mel / MFCC feature front end — upstream, no local spec                                                                      | `analysis` (numerical-toolbox)     | ★★☆☆☆      |
| N16 | [Mini-batch SGD training step (replaces `Model::Train`)](roadmap/model/MiniBatchTraining/implementation.md)                     | `model`                            | ★★★☆☆      |
| N17 | [Elman RNN (streaming state, truncated BPTT)](roadmap/layer/ElmanRnn/implementation.md)                                         | `layer`                            | ★★★☆☆      |
| N18 | [2D convolution + 2D pooling](roadmap/layer/Convolution2D/implementation.md)                                                    | `layer`                            | ★★★☆☆      |
| N19 | [Depthwise-separable convolution](roadmap/layer/DepthwiseSeparableConvolution/implementation.md)                                | `layer`                            | ★★★☆☆      |
| N20 | [GRU cell](roadmap/layer/Gru/implementation.md)                                                                                 | `layer`                            | ★★★★☆      |
| N21 | [LSTM cell](roadmap/layer/Lstm/implementation.md)                                                                               | `layer`                            | ★★★★☆      |

### Tier 1 — Trivial ★☆☆☆☆

**N1. [Identity (linear) activation](roadmap/activation/Identity/implementation.md).** `f(x) = x`, `f'(x) = 1`, so a Dense can emit raw regression outputs (MSE/Huber) and raw logits for the logit-input losses (CCE today, BCE-with-logits in N9). Today every Dense must apply a non-linearity.
- *Algorithm / paper:* I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning* (2016), §6.2.2.1 (linear output units).
- *Reuses / builds on:* [ActivationFunction.hpp](neural_network/activation/ActivationFunction.hpp); used by N4, N9, N16.

**N2. [Per-feature affine normalisation](roadmap/layer/AffineNormalization/implementation.md).** `y = a ⊙ x + b` with 2N trainable parameters, plus a static helper that builds `a = γ/√(σ² + ε)` and `b = β − a·μ` from frozen batch-norm statistics. Its main on-device use is standardising raw sensor units. Folding BN into the preceding Dense/Conv is exact, and the N13 exporter does it offline.
- *Algorithm / paper:* S. Ioffe, C. Szegedy, "Batch Normalization," *ICML*, 2015; B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," *CVPR*, 2018 (folding).
- *Reuses / builds on:* the existing [Layer](neural_network/layer/Layer.hpp) contract (no frozen parameters needed).

**N3. [Weight initialisation](roadmap/initialization/WeightInitialization/implementation.md).** Glorot/Xavier uniform `U(±√(6/(fan_in + fan_out)))`, He uniform/normal (standard deviation `√(2/fan_in)`) and LeCun normal (for SELU), drawn from a small deterministic xorshift/PCG generator so that test reference values are reproducible across standard libraries.
- *Algorithm / paper:* X. Glorot, Y. Bengio, *AISTATS*, 2010; K. He et al., "Delving Deep into Rectifiers," *ICCV*, 2015; G. Marsaglia, "Xorshift RNGs," *J. Stat. Softw.* 8(14), 2003.
- *Reuses / builds on:* Dense `initialWeights`; replaces the simulator's hand-written He init.

**N4. [Huber loss](roadmap/losses/HuberLoss/implementation.md).** Quadratic inside δ and linear outside, with a bounded gradient `clip(ŷ − y, ±δ)`. It is robust to sensor outliers. It moves to the N9 contract along with the other losses.
- *Algorithm / paper:* P. J. Huber, "Robust Estimation of a Location Parameter," *Ann. Math. Statist.* 35(1), 1964.
- *Reuses / builds on:* [MeanSquaredError.hpp](neural_network/losses/MeanSquaredError.hpp), [MeanAbsoluteError.hpp](neural_network/losses/MeanAbsoluteError.hpp) (limiting cases); N1.

**N5. [Swish family](roadmap/activation/SwishActivations/implementation.md).** SiLU `x·σ(x)`, Hard-Sigmoid `ReLU6(x + 3)/6` and Hard-Swish `x·ReLU6(x + 3)/6`. The hard variants are piecewise-linear and need no exp.
- *Algorithm / paper:* P. Ramachandran, B. Zoph, Q. Le, "Searching for Activation Functions," 2017; S. Elfwing, E. Uchibe, K. Doya, *Neural Networks* 107, 2018; A. Howard et al., "Searching for MobileNetV3," *ICCV*, 2019.
- *Reuses / builds on:* [Sigmoid.hpp](neural_network/activation/Sigmoid.hpp) (numerically stable form).

**N6. [ELU / SELU](roadmap/activation/ExponentialLinearUnit/implementation.md).** `α(eˣ − 1)` for `x ≤ 0`. SELU uses λ ≈ 1.0507 and α ≈ 1.6733 and is self-normalising only with LeCun-normal initialisation. `BackwardVector` reuses the output (`f' = f + α` for `x ≤ 0`).
- *Algorithm / paper:* D.-A. Clevert, T. Unterthiner, S. Hochreiter, *ICLR*, 2016; G. Klambauer et al., "Self-Normalizing Neural Networks," *NeurIPS*, 2017.
- *Reuses / builds on:* [ActivationFunction.hpp](neural_network/activation/ActivationFunction.hpp), `math::Exp`; N3.

**N7. [Classification metrics](roadmap/metrics/ClassificationMetrics/implementation.md).** Argmax, accuracy and a fixed-size confusion matrix for on-device evaluation.
- *Algorithm / paper:* M. Sokolova, G. Lapalme, "A systematic analysis of performance measures for classification tasks," *Inf. Process. Manage.* 45(4), 2009.
- *Reuses / builds on:* the regression counterparts in [Statistics.hpp](https://github.com/embedded-pro/numerical-toolbox-cpp/blob/main/numerical/math/Statistics.hpp).

### Tier 2 — Easy ★★☆☆☆

**N8. [Trainable-layer interface](roadmap/layer/TrainableLayerInterface/implementation.md).**
- Adds an inference base (`Forward`/`Output`) separate from a trainable one (`Backward`, `ParameterGradients()` in `Parameters()` order).
- Gradients accumulate instead of being overwritten, with `ZeroGradients()` to reset them.
- Adds a zero-parameter specialisation and `Model::Gradients()`.
- Optionally makes the activation a template parameter, so the hot path makes no per-element virtual calls.
- *Algorithm / paper:* D. Rumelhart, G. Hinton, R. Williams, "Learning representations by back-propagating errors," *Nature* 323, 1986.
- *Reuses / builds on:* [Layer.hpp](neural_network/layer/Layer.hpp) (already has `ValueType`, a virtual destructor and a const `Parameters()`), [Dense.hpp](neural_network/layer/Dense.hpp) (it already computes dW/db privately); unblocks N10, N12 and N16–N21.

**N9. [Per-sample loss contract + logit-input BCE](roadmap/losses/LossContract/implementation.md).**
- `Cost/Gradient(prediction, target)` with the target supplied at run time (integer labels allowed) and the gradient taken with respect to predictions. The loss no longer holds a regulariser; weight decay moves to θ in N16.
- Adds stable BCE-with-logits: `max(z,0) − z·y + log(1 + e^{−|z|})`, gradient `σ(z) − y`. The probability-input BCE (ε-clamped) stays for Sigmoid outputs.
- Keeps CCE's log-sum-exp (max-shift) form on logits, fused with softmax.
- Migrates MSE, MAE, BCE, CCE and Huber to the new contract.
- *Algorithm / paper:* Goodfellow et al., *Deep Learning*, §6.2.2.2; P. Blanchard, D. Higham, N. Higham, "Accurately computing the log-sum-exp and softmax functions," *IMA J. Numer. Anal.* 41(4), 2021.
- *Reuses / builds on:* [Loss.hpp](neural_network/losses/Loss.hpp); N1.

**N10. [Flash-resident inference weights](roadmap/layer/FlashResidentWeights/implementation.md).** A storage policy (or an inference-only variant) for Dense, and for every later parameterised layer, that references `const` weights placed in flash (`constexpr` arrays in `.rodata`) and drops the gradient, parameter-mirror and saved-input buffers. RAM then holds only activations.
- *Algorithm / paper:* R. David et al., "TensorFlow Lite Micro," *MLSys*, 2021; L. Lai, N. Suda, V. Chandra, "CMSIS-NN," arXiv:1801.06601, 2018.
- *Reuses / builds on:* N8; `math::Matrix` constexpr constructor.

**N11. [1D convolution](roadmap/layer/Convolution1D/implementation.md).** Multi-channel cross-correlation with valid padding and stride S. Output length is `⌊(L − K)/S⌋ + 1` and there are `K·C_in·C_out + C_out` parameters. It defines the channels-last flat `Matrix<T, L·C, 1>` layout shared by all spatial layers, so Flatten is not needed. It has an optional causal streaming mode with a `(K−1)·C_in` ring buffer.
- *Algorithm / paper:* S. Kiranyaz et al., "1D convolutional neural networks and applications: A survey," *Mech. Syst. Signal Process.* 151, 2021; Y. LeCun et al., *Proc. IEEE* 86(11), 1998.
- *Reuses / builds on:* N8, N10. It does not reuse `analysis::LinearConvolution`, which computes a full-length, single-channel convolution on BoundedVector I/O.

**N12. [1D pooling](roadmap/layer/Pooling1D/implementation.md).** Max, average and global-average downsampling with no parameters. Max-pool stores argmax indices for `Backward`. Global-average pooling replaces the flatten→Dense head.
- *Algorithm / paper:* Y.-L. Boureau, J. Ponce, Y. LeCun, *ICML*, 2010; M. Lin, Q. Chen, S. Yan, "Network In Network," *ICLR*, 2014.
- *Reuses / builds on:* N8 (`ParameterSize = 0`), N11 layout.

**N13. [Weight export + blob (de)serialisation](roadmap/model/WeightExport/implementation.md).**
- Primary path: a `scripts/` exporter for Keras/PyTorch that writes a `constexpr` C++ header in this library's flat order. It transposes Keras `[in, out]` kernels to `W[out][in]`, folds BN into the preceding layer, sums the LSTM `b_ih + b_hh` pair and keeps the GRU `reset_after` bias pairs.
- Secondary path, for over-the-air updates: a versioned little-endian blob with a magic number, a layout hash and a CRC, streamed one layer at a time into `SetParameters`.
- *Algorithm / paper:* R. David et al., "TensorFlow Lite Micro," *MLSys*, 2021 (model-import precedent).
- *Reuses / builds on:* N10, N2; emil `infra/util/Crc.hpp`, `Endian.hpp`.

**N14. Step optimisers (upstream).** A stateful `Step(θ, g)` interface, with SGD + momentum/Nesterov (P extra floats) and bias-corrected Adam (2P extra floats). This belongs in numerical-toolbox-cpp's `optimization` module, so it has no spec in this repo's `roadmap/`; it is proposed for the [numerical-toolbox roadmap](https://github.com/embedded-pro/numerical-toolbox-cpp/blob/main/ROADMAP.md).
- *Algorithm / paper:* B. Polyak, 1964; I. Sutskever et al., *ICML*, 2013; D. Kingma, J. Ba, "Adam," *ICLR*, 2015.
- *Reuses / builds on:* [Optimizer.hpp](https://github.com/embedded-pro/numerical-toolbox-cpp/blob/main/numerical/optimization/Optimizer.hpp), which today offers only full-batch `Minimize`.

**N15. Log-mel / MFCC front end (upstream).** Windowed real FFT → mel filterbank → log → DCT-II. This is the input stage for spectrogram and keyword-spotting models (N18/N19). This belongs in numerical-toolbox-cpp's `analysis` module, so it has no spec in this repo's `roadmap/`; it is proposed for the [numerical-toolbox roadmap](https://github.com/embedded-pro/numerical-toolbox-cpp/blob/main/ROADMAP.md).
- *Algorithm / paper:* S. Davis, P. Mermelstein, *IEEE Trans. ASSP* 28(4), 1980.
- *Reuses / builds on:* numerical `analysis` [RealFastFourierTransform](https://github.com/embedded-pro/numerical-toolbox-cpp/blob/main/numerical/analysis/RealFastFourierTransform.hpp), [DiscreteCosineTransform](https://github.com/embedded-pro/numerical-toolbox-cpp/blob/main/numerical/analysis/DiscreteCosineTransform.hpp), windowing.

### Tier 3 — Moderate ★★★☆☆

**N16. [Mini-batch SGD training step](roadmap/model/MiniBatchTraining/implementation.md).** Replaces `Model::Train`.
- For each sample: `Forward` → N9 loss gradient → `Backward`, which accumulates. Every B samples: `Step(θ, ḡ/B)`.
- Weight decay is applied to θ.
- Optional global-norm clipping.
- A per-layer freeze mask supports last-layer fine-tuning, the realistic MCU case.
- *Algorithm / paper:* L. Bottou, *COMPSTAT*, 2010; R. Pascanu, T. Mikolov, Y. Bengio, *ICML*, 2013 (clipping); H. Cai et al., "TinyTL," *NeurIPS*, 2020; J. Lin et al., "On-Device Training Under 256KB Memory," *NeurIPS*, 2022.
- *Reuses / builds on:* N8, N9, N3, N14 (plain SGD works without it), numerical `regularization`.

**N17. [Elman RNN](roadmap/layer/ElmanRnn/implementation.md).** Streaming `h_t = tanh(W_x x_t + W_h h_{t−1} + b)` with persisted state and `ResetState()`. `Backward` uses truncated BPTT over a compile-time window K, costing K·(X + H) floats. It introduces the stateful-layer API that N20/N21 reuse.
- *Algorithm / paper:* J. L. Elman, "Finding Structure in Time," *Cognitive Science* 14(2), 1990; R. Williams, J. Peng, *Neural Computation* 2(4), 1990.
- *Reuses / builds on:* Dense math, [Tanh.hpp](neural_network/activation/Tanh.hpp), N8, N10.

**N18. [2D convolution + 2D pooling](roadmap/layer/Convolution2D/implementation.md).** H×W×C channels-last, valid padding and stride, using direct convolution with no im2col scratch buffer. Intended for small spectrogram and low-resolution image inputs.
- *Algorithm / paper:* Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," *Proc. IEEE* 86(11), 1998.
- *Reuses / builds on:* N11, N12, N10; inputs from N15.

**N19. [Depthwise-separable convolution](roadmap/layer/DepthwiseSeparableConvolution/implementation.md).** A per-channel K×K convolution followed by a 1×1 pointwise one: `K²C + C·C'` parameters instead of `K²C·C'`. A 1D variant is included.
- *Algorithm / paper:* A. Howard et al., "MobileNets," 2017; Y. Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers," 2017.
- *Reuses / builds on:* N18 (2D), N11 (1D).

### Tier 4 — Advanced ★★★★☆

**N20. [GRU cell](roadmap/layer/Gru/implementation.md).** Update and reset gates in the PyTorch/Keras `reset_after` form, `n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))`, with `3·(H·(X + H) + 2H)` parameters. That is 25 % fewer than LSTM, making it the preferred recurrent cell on an MCU.
- *Algorithm / paper:* K. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder," *EMNLP*, 2014.
- *Reuses / builds on:* N17 stateful API and truncated BPTT, [Sigmoid.hpp](neural_network/activation/Sigmoid.hpp), [Tanh.hpp](neural_network/activation/Tanh.hpp), N3, N13 import layout.

**N21. [LSTM cell](roadmap/layer/Lstm/implementation.md).** Input, forget and output gates plus a cell state, `4·(H·(X + H) + H)` parameters with one fused bias (the exporter sums `b_ih + b_hh`), and forget-gate bias initialised to 1.
- *Algorithm / paper:* S. Hochreiter, J. Schmidhuber, *Neural Computation* 9(8), 1997; F. Gers, J. Schmidhuber, F. Cummins, *Neural Computation* 12(10), 2000; R. Jozefowicz, W. Zaremba, I. Sutskever, *ICML*, 2015.
- *Reuses / builds on:* N17, N20.

---

### Deferred / out of scope

- **Dropout**: inverted dropout does nothing at inference. It is only a training regulariser and needs a mode flag, a PRNG and parameter-free layers.
- **Training-mode batch-norm**: needs batch statistics, and the API processes one sample at a time (with a batch of 1, x − μ_B = 0).
- **Flatten / Reshape**: not needed, because layers already exchange flat `Matrix<T, N, 1>`.
- **int8 quantised inference**: the dominant MCU format (TFLM/CMSIS-NN), but it conflicts with the float-only policy. Revisit first once N10–N13 exist.
- **Attention / Transformer, Embedding**: O(T²) activations or vocabulary-sized tables.
- **GELU, LayerNorm**: only needed by those deferred blocks.
- **Hinge, KL divergence, sparse CCE, Softplus activation**: niche. N9 handles integer labels, and softplus appears inside N9's BCE-with-logits.
- **Learning-rate schedules, early stopping**: training-loop policy; N16 options or application code.
- **Residual / DAG models**: `Model` is a linear chain; revisit after N18.

> **Test-only helper (not a production component).** A finite-difference gradient check (central
> difference, relative error) goes in the `numerical.math_test_helper` INTERFACE library, as
> numerical's ROADMAP prescribes. Every `Backward` in N8–N21 is asserted against it, as
> [TESTING.md](TESTING.md) requires.
