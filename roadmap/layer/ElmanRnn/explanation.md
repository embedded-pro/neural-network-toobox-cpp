# Elman RNN (Streaming State, Truncated BPTT) — Overview

## What it is
A recurrent layer that keeps a small hidden state `h` between calls. It takes one input frame `x_t` per call
and updates the state with `h_t = tanh(W_x x_t + W_h h_{t−1} + b)`. The new state is also the layer's
output. With `X` inputs and `H` hidden units it has `H·(X + H + 1)` parameters, however long the sequence.
Followed by a `Dense` head it forms the classic Elman network. A flash-resident variant runs inference with
only the `H` state floats in RAM.

## Why it matters (embedded)
Sensor data arrives one sample at a time. A convolution (N11) sees a fixed window, while an RNN carries a
summary of the whole past in `H` numbers and costs the same `H·X + H²` multiply-adds for every new sample.
That fits a streaming pipeline: keyword and gesture spotting, anomaly detection on vibration or current,
and filtering or prediction of IMU signals. A 3-input, 16-unit cell has 320 parameters and uses 64 bytes of
state. Training on the device stays bounded because the backward pass only looks `K` steps into the past.

## How it works (intuition)
Each step mixes the new input with the previous state, adds a bias and squashes the result with `tanh`,
which keeps the state in `(−1, 1)`. To learn, the error at step `t` has to flow back through every earlier
step, because `h_{t−1}` influenced `h_t` (backpropagation through time). Storing the whole history is
impossible on an MCU, so the layer keeps a ring of the last `K` inputs and states (`K·(X + H)` floats) and
stops the backward sweep there. This is **truncated BPTT**: influences older than `K` steps are ignored.
They usually fade anyway, because each step back multiplies the error by `W_hᵀ` and by `tanh' ≤ 1`. When a
sequence is no longer than `K`, the gradient is exact. `ResetState()` starts a new sequence at `h = 0`.
`SetState()` resumes a saved state and starts a new truncation window. `Model::ResetState()` resets every
recurrent layer at once. The parameter layout follows PyTorch `nn.RNN` (with its two bias vectors summed)
and Keras `SimpleRNN` (transposed), so trained weights import directly.

## Key parameters
- **Input size `X`, hidden size `H`**: features per step and state width. Cost and parameters grow with `H²`.
- **Window `K`**: truncation depth of BPTT. RAM `K·(X + H)` floats, backward cost roughly `K·(H·X + 2H²)`.
  It does not affect inference.
- **When to call `Backward`**: every step (many-to-many, one loss per step) or only at the end
  (many-to-one, one loss per sequence). Both accumulate into the N8 gradient.
- **Activation**: fixed to `tanh`. ReLU RNNs are deferred.

## Reference
J. L. Elman, "Finding Structure in Time," *Cognitive Science*, vol. 14, no. 2, pp. 179–211, 1990 (the
simple recurrent network).
R. J. Williams, J. Peng, "An Efficient Gradient-Based Algorithm for On-Line Training of Recurrent Network
Trajectories," *Neural Computation*, vol. 2, no. 4, pp. 490–501, 1990 (truncated BPTT).
P. J. Werbos, "Backpropagation Through Time: What It Does and How to Do It," *Proc. IEEE*, vol. 78, no. 10,
pp. 1550–1560, 1990 (BPTT).
R. Pascanu, T. Mikolov, Y. Bengio, "On the difficulty of training recurrent neural networks," *Proc. ICML*,
PMLR 28(3), pp. 1310–1318, 2013 (vanishing and exploding gradients, norm clipping).

## See also
`Dense` (the per-step arithmetic and the usual output head), `Tanh`, `TrainableLayerInterface` (N8,
gradient accumulation), `FlashResidentWeights` (N10), `Convolution1D` (N11, the fixed-window alternative),
`WeightExport` (N13, PyTorch/Keras import), `MiniBatchTraining` (N16, clipping and the optimiser loop),
`Gru` (N20) and `Lstm` (N21), which reuse the stateful API.
