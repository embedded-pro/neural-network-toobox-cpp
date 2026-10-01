# Flash-Resident Inference Weights — Overview

## What it is
An inference-only fully connected layer whose weights and biases stay in flash. It does not copy them
into RAM. The layer holds a read-only *view* of a constant array (`W` row by row, then `b`), the same
flat layout `Dense::Parameters()` uses. The only thing it keeps in RAM is its output vector. It has no
backward pass, no gradient buffer and no saved input. The view type itself is reusable, so later
parameterised layers (convolution, recurrent cells) can offer the same flash-backed inference variant.
`Model` also accepts these layers. A chain built only from them runs `Forward` only, and a mixed chain
exposes just the parameters of its trainable layers.

## Why it matters (embedded)
A microcontroller usually has several times more flash than SRAM (for example 512 KB flash and 128 KB
SRAM on an STM32F411). Today `Dense` copies its weights into RAM and keeps a gradient buffer of the same size, plus the
saved input, pre-activation and input gradient. `Dense<float, 64, 64>` holds 16 640 B of weights but
occupies 34 328 B of RAM (measured on host). The flash-resident variant of the same layer occupies
280 B on host (256 B of output plus three pointers), or 268 B on a 32-bit MCU. The weights move to
`.rodata`, which the linker places in flash and the core reads in place, because flash is memory-mapped
on Cortex-M. The saving decides whether a model fits the device at all. This is the standard split in
TinyML runtimes: weights in flash, activations in SRAM.

## How it works (intuition)
The forward pass is the same arithmetic as `Dense`: `z = W x + b`, then `a = f(z)`. The difference is
where `W` and `b` live. The exporter (N13) writes each layer's parameters as an `inline constexpr`
vector. Because it is constant-initialised and `const`, the compiler emits it into the read-only data
section instead of RAM, and no start-up code copies it. The layer stores only a fixed-size span
(a pointer) to that array. `z` is a short-lived stack buffer, and the activation writes the result into
the layer's output vector, which the next layer reads. Nothing a training step would need is kept. A
network deployed from this layer therefore cannot be trained on the device, and the type system says
so: `Backward` and `SetParameters` do not exist on it.

## Key parameters
- **`InputSize`, `OutputSize`** — the layer shape. The flash array holds `In·Out + Out` floats.
- **Weight source** — a `constexpr` `math::Vector<T, In·Out + Out>` (the normal path, from N13), or a
  `std::span<const T, In·Out + Out>` over a raw flash region (for example an over-the-air blob).
  Temporaries are rejected at compile time, so the view cannot dangle on a temporary.
- **`Activation`** — the interface type by default (stored by reference, one virtual call per
  `Forward`), or a concrete `final` activation stored by value, whose call the compiler can inline.

## Reference
R. David et al., "TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems," *Proceedings
of Machine Learning and Systems (MLSys)* 3, 2021 (weights read in place from the flatbuffer in flash;
activations in a RAM arena).
L. Lai, N. Suda, V. Chandra, "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs,"
arXiv:1801.06601, 2018 (fully connected kernels that take a pointer to constant weights).
J. Lin, W.-M. Chen, Y. Lin, J. Cohn, C. Gan, S. Han, "MCUNet: Tiny Deep Learning on IoT Devices,"
*NeurIPS*, 2020 (the flash budget for weights vs the SRAM budget for activations).
ISO/IEC 14882:2020, [basic.start.static] (constant initialisation: why a `constexpr` object needs no
run-time copy and can live in read-only memory).

## See also
`Dense` (the trainable RAM-resident layer with the same parameter layout), `TrainableLayerInterface`
(N8, the `InferenceLayer` base this layer derives from), `Identity` (N1, linear outputs),
`WeightExport` (N13, emits the `constexpr` arrays), `Convolution1D` (N11) and `ElmanRnn`/`Gru`/`Lstm`
(N17/N20/N21, reuse the flash view), `MiniBatchTraining` (N16, fine-tuning a trainable head behind a
frozen flash prefix).
