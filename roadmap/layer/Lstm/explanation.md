# LSTM Cell (Forget Gate, Truncated BPTT) — Overview

## What it is
A gated recurrent layer with two pieces of state: the hidden state `h` (the output) and a separate **cell state**
`c` (the memory). Each call takes one input frame. Three sigmoid gates decide, per unit, how much new content
enters the memory (**input gate** `i`), how much old memory is kept (**forget gate** `f`) and how much of the
memory is shown as output (**output gate** `o`):
`c_t = f ⊙ c_{t−1} + i ⊙ g`, `h_t = o ⊙ tanh(c_t)`, with the candidate `g = tanh(W_x^g x + W_h^g h_{t−1} + b^g)`.
With `X` inputs and `H` units it has `4·(H·(X + H) + H)` parameters and `2H` floats of state.

## Why it matters (embedded)
The LSTM is the most widely used recurrent cell, so most published sequence models for sensor data, keyword
spotting or predictive maintenance ship LSTM weights. Supporting it lets those models be imported unchanged
from PyTorch or Keras (N13). Its memory path is additive: when `f ≈ 1` the cell carries information, and error,
across many steps almost unattenuated, which a plain RNN (N17) cannot do. It costs about 27 % more weights than a
GRU (N20): a 3-input, 16-unit LSTM has 1 280 parameters against the GRU's 1 008, plus 64 more bytes of state. So
N20 stays the default for new on-device models, and N21 is for imported ones or tasks where the LSTM trains better.
The flash-resident variant runs inference with only `3H` floats in RAM.

## How it works (intuition)
Each step computes four blocks of weighted sums of the input and the previous `h`. Three pass through a sigmoid
and become gates in `(0, 1)`; the fourth passes through `tanh` and becomes the candidate content in `(−1, 1)`.
The memory is updated by a gated sum rather than overwritten, and the output is a gated, squashed view of the
memory, so `h` always stays in `(−1, 1)` while `c` can grow slowly. Training uses the same truncated
backpropagation through time as N17: the layer keeps a ring of the last `K` steps, storing the gate values and
`tanh c`, so the backward sweep needs only multiplications. The error travels back along two paths, through `h`
and the weights, and directly through `c` scaled by `f`. The constructor sets the forget-gate bias to `1`, so the
cell starts out remembering, which makes training more reliable. The single bias vector is the sum of the two
bias vectors PyTorch stores; that sum is exact, so imported weights load without loss.

## Key parameters
- **Input size `X`, hidden size `H`**: parameters and cost grow with `4H·(X + H)`.
- **Window `K`**: truncation depth of BPTT. RAM `K·(X + 7H)` floats; it does not affect inference.
- **Forget-gate bias**: initialised to `1`, then trained or imported like any other parameter (not added at run time).
- **Gate order**: `i, f, g, o` (PyTorch; Keras `i, f, c, o` is the same). Vanilla cell: no peepholes, no projection.

## Reference
S. Hochreiter, J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780,
1997 (the cell state and the input/output gates).
F. A. Gers, J. Schmidhuber, F. Cummins, "Learning to Forget: Continual Prediction with LSTM," *Neural Computation*,
vol. 12, no. 10, pp. 2451–2471, 2000 (the forget gate).
R. Jozefowicz, W. Zaremba, I. Sutskever, "An Empirical Exploration of Recurrent Network Architectures,"
*Proc. ICML*, PMLR 37, pp. 2342–2350, 2015 (forget-gate bias 1).
K. Greff, R. K. Srivastava, J. Koutník, B. R. Steunebrink, J. Schmidhuber, "LSTM: A Search Space Odyssey,"
*IEEE Trans. Neural Networks and Learning Systems*, vol. 28, no. 10, pp. 2222–2232, 2017 (the vanilla LSTM used
here; peepholes and gate coupling are not essential).
R. J. Williams, J. Peng, "An Efficient Gradient-Based Algorithm for On-Line Training of Recurrent Network
Trajectories," *Neural Computation*, vol. 2, no. 4, pp. 490–501, 1990 (truncated BPTT).
The equations and gate order: PyTorch `torch.nn.LSTM` and Keras `LSTM` documentation.

## See also
`ElmanRnn` (N17, the stateful API and truncated BPTT reused here), `Gru` (N20, the lighter gated cell and the
multi-gate ring), `Sigmoid`, `Tanh`, `Dense` (the usual output head), `TrainableLayerInterface` (N8),
`FlashResidentWeights` (N10), `WeightInitialization` (N3), `WeightExport` (N13, PyTorch/Keras import with the fused
bias), `MiniBatchTraining` (N16, clipping).
