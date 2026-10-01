# Weight Export + Blob (De)serialisation — Overview

## What it is
This item moves a network trained in Keras or PyTorch onto the microcontroller, with exactly the same
numbers. It has two halves:
- **Exporter.** A Python script reads the trained model. It rearranges every layer's weights into this
  library's flat parameter order and folds frozen batch-norm layers into their neighbours. It writes a C++
  header with one `constexpr` array per layer.
- **Blob.** A compact binary format carries the same weights for over-the-air updates. It holds a magic
  number, a format version, a hash of the layer shapes and a CRC. The device checks the blob before it
  touches the model, then loads it one layer at a time. It can also write the blob, for example to
  persist weights after fine-tuning on the device.

## Why it matters (embedded)
Training happens on a PC, but the device needs the weights in its own layout. Keras stores a dense kernel
as `[in][out]` while this library uses `W[out][in]`. Convolution and recurrent weights differ between
frameworks in axis order, gate order and bias count. A silent layout mismatch still produces numbers, just
wrong ones, so the conversion belongs in one tested tool rather than in hand-written copy loops. The header
path puts the weights in flash as constants, at zero RAM cost when paired with the flash-resident layers
(N10). The blob path lets a fleet receive new weights without reflashing firmware. Its checks reject a
blob for a different model, an unknown format, a transmission error or a non-finite weight, and they leave
the running model untouched.

## How it works (intuition)
Every layer already exposes its parameters as one flat vector in a fixed order: weights row by row, then
biases. The exporter reproduces that order from each framework's tensors. It transposes, permutes gate
blocks, adds LSTM's two bias vectors together (they only ever appear as a sum), and keeps GRU's two bias
sets (one sits inside the reset gate, so they cannot be merged). At inference, batch-norm is just a
per-feature scale and shift. Multiplying that into the preceding layer's rows, or the following layer's
columns, removes the layer without changing the output. Weights are printed with nine significant digits,
which is enough for every float32 value to come back bit-identical.

The blob stores the raw float32 bit patterns in little-endian order behind a 20-byte header, then a
CRC-32. The layout hash is computed from the model's layer shapes at compile time. A blob for a
differently shaped network is therefore rejected even when it happens to have the same size. After a
successful check, the device can copy each layer's values into RAM, or point a flash-resident layer
straight at them without copying.

## Key parameters
- **Layer list** — the model's layer types. They determine the parameter counts, the offsets, the size and
  the layout hash, all at compile time.
- **Batch-norm `ε`** — taken from the source layer (Keras default `1e-3`, PyTorch default `1e-5`). A
  different value changes the folded weights.
- **Output form** — header only, blob only, or both. `--blob-array` also embeds the blob as a
  flash-resident default image.
- **Framework** — `keras` or `torch`. This selects the transpose and gate-order rules.

## Reference
R. David et al., "TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems," *Proceedings of
Machine Learning and Systems (MLSys)* 3, 2021 (the precedent for importing a trained model as a constant
array in flash).
S. Ioffe, C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift," *ICML*, 2015 (the inference-time transform that is folded).
B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only
Inference," *CVPR*, 2018 (batch-norm folding into the preceding layer).
P. Deutsch, "GZIP file format specification version 4.3," RFC 1952, 1996 (the CRC-32 used, identical to
`zlib.crc32`).
G. Fowler, L. C. Noll, K.-P. Vo, D. Eastlake, T. Hansen, "The FNV Non-Cryptographic Hash Algorithm," IETF
Internet-Draft draft-eastlake-fnv (FNV-1a, used for the layout hash).
IEEE Std 754-2019, *IEEE Standard for Floating-Point Arithmetic* (binary32 encoding; 9 significant digits
round-trip).

## See also
`Model` (the flat parameter vector and the `LayerAt` accessor this item adds), `Dense` (its layout),
`FlashResidentWeights` (N10, zero-copy consumer of the header arrays and blob views),
`AffineNormalization` (N2, fold identities and the fallback when a batch-norm cannot be folded),
`Convolution1D` (N11), `Gru`/`Lstm` (N20/N21, recurrent layouts), `MiniBatchTraining` (N16, saving
fine-tuned weights), `ClassificationMetrics` (N7, checking an imported model on the device).
