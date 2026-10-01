# GRU Cell (reset_after, Truncated BPTT) — Overview

## What it is
A gated recurrent layer. Like the Elman cell (N17) it keeps a hidden state `h` between calls and takes one
input frame per call, but two learned gates decide how the state changes. The **update gate** `z` chooses,
per unit, how much of the old state to keep, and the **reset gate** `r` chooses how much of the old state
feeds the new candidate `n`:
`h_t = (1 − z) ⊙ n + z ⊙ h_{t−1}`, with `n = tanh(W_x^n x + b_x^n + r ⊙ (W_h^n h_{t−1} + b_h^n))`.
With `X` inputs and `H` units it has `3·(H·(X + H) + 2H)` parameters. The new state is also the output.

## Why it matters (embedded)
A plain RNN forgets quickly: its error signal shrinks at every step back in time. The GRU's update gate
opens a near-direct path from `h_{t−1}` to `h_t`, so it can remember events tens of steps back. That is what
keyword spotting, gesture recognition and slow drift detection need. It does this with three weight blocks
instead of the LSTM's four and without a separate cell state. That makes it the recurrent cell of choice on an
MCU: a 3-input, 16-unit GRU has 1 008 parameters against the LSTM's 1 280, and 64 bytes of state. The
flash-resident variant runs inference with only those `H` floats in RAM.

## How it works (intuition)
Each step computes three blocks of weighted sums of the input and the previous state. Two are squashed by a
sigmoid into gates in `(0, 1)`, the third by `tanh` into a candidate in `(−1, 1)`. The new state blends the
candidate and the old state per unit. Because it is a blend, a state that starts at zero never leaves `(−1, 1)`. Training
uses the same truncated backpropagation through time as N17: the layer keeps a ring of the last `K` steps,
here also storing the gate values, so the backward sweep only needs multiplications, no new `exp`/`tanh`
calls. The reset gate multiplies the *already weighted* recurrent term, including its bias `b_h^n`. This
"reset_after" form is the one PyTorch and Keras (by default) use, so trained weights import directly (N13).
The paper's original form, which resets `h` before the multiplication, computes a different function and is
not supported.

## Key parameters
- **Input size `X`, hidden size `H`**: parameters and cost grow with `3H·(X + H)`.
- **Window `K`**: truncation depth of BPTT. RAM `K·(X + 5H)` floats; it does not affect inference.
- **Two bias vectors**: `b_x` and `b_h` are both kept. For the `r` and `z` gates they only add up; for `n`
  the hidden-side bias is scaled by `r` and cannot be merged.
- **Gate convention**: `z` weights the *old* state (PyTorch, Keras, Cho et al.). Some texts swap `z` and `1 − z`.

## Reference
K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, Y. Bengio, "Learning Phrase
Representations using RNN Encoder–Decoder for Statistical Machine Translation," *Proc. EMNLP*, pp. 1724–1734,
2014 (the gated recurrent unit).
J. Chung, C. Gulcehre, K. Cho, Y. Bengio, "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
Modeling," *NIPS 2014 Workshop on Deep Learning*, arXiv:1412.3555 (GRU vs LSTM; uses the swapped `z` convention).
R. Jozefowicz, W. Zaremba, I. Sutskever, "An Empirical Exploration of Recurrent Network Architectures,"
*Proc. ICML*, PMLR 37, pp. 2342–2350, 2015 (GRU competitive with LSTM).
R. J. Williams, J. Peng, "An Efficient Gradient-Based Algorithm for On-Line Training of Recurrent Network
Trajectories," *Neural Computation*, vol. 2, no. 4, pp. 490–501, 1990 (truncated BPTT).
The `reset_after` equations: PyTorch `torch.nn.GRU` and Keras `GRU(reset_after=True)` documentation
(the cuDNN-compatible form).

## See also
`ElmanRnn` (N17, the stateful API and truncated BPTT reused here), `Sigmoid`, `Tanh`, `Dense` (the usual
output head), `TrainableLayerInterface` (N8), `FlashResidentWeights` (N10), `WeightInitialization` (N3),
`WeightExport` (N13, PyTorch/Keras import), `MiniBatchTraining` (N16, clipping), `Lstm` (N21).
