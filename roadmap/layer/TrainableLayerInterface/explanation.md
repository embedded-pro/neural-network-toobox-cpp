# Trainable-Layer Interface — Overview

## What it is
A split of the layer contract into two levels. An **inference layer** can only run forward: take an
input, produce an output. A **trainable layer** can also run backward. It returns the gradient with
respect to its input, and it adds the gradient with respect to its own parameters to a buffer that the
training code can read (`ParameterGradients`) and clear (`ZeroGradients`). A parameter-free variant
(for pooling-style layers) runs backward but owns no parameters. The model gains one call that
gathers every layer's parameter gradient into one flat vector, in the same order as the parameters.

## Why it matters (embedded)
Today every layer already computes its weight and bias gradients during the backward pass, but it keeps
them private and overwrites them on the next sample. No optimiser can use them, so on-device training
through the API is impossible. Exposing and *accumulating* them costs no extra RAM, because the buffer
already exists. It turns the existing backward pass into real back-propagation, the input that
mini-batch training (N16) needs. The inference-only level lets a deployed network drop everything
training needs (gradient buffers, saved inputs), which is what flash-resident weights (N10) build on.
The parameter-free level unblocks pooling (N12), which today cannot even be declared, because a
zero-length parameter vector does not compile.

## How it works (intuition)
Back-propagation is the chain rule applied layer by layer, from the output back to the input. Each layer
receives "how much the loss changes per unit change of my output" and turns it into two things: the same
quantity for its input, handed to the layer before it, and the same quantity for its own weights. For a
dense layer `a = f(W x + b)` this means first pushing the upstream gradient through the activation's
derivative, `δ = f'(z) ⊙ g` (for softmax, the full vector–Jacobian product). Then `Wᵀδ` goes to the
previous layer, and `δ xᵀ` and `δ` are the weight and bias gradients. Because the gradient of a sum is
the sum of the gradients, adding each sample's contribution into the buffer gives exactly the gradient
of the total loss over a mini-batch. The trainer then divides by the batch size, takes a step, and
clears the buffer. The input gradient is *not* accumulated, because it is consumed immediately by the
previous layer.

## Key parameters
- **Parameter order**: gradients use exactly the `Parameters()` layout (Dense: weights row by row, then
  biases), so an optimiser can treat `θ` and `∇θ` as matching flat vectors.
- **Accumulate / reset**: `Backward` adds, `ZeroGradients` clears, and `SetParameters` leaves the
  gradients alone. The trainer owns the reset.
- **One backward per forward**: each backward uses the input and activations saved by the latest forward.
- **Optional compile-time activation**: storing the activation by type instead of by interface reference
  removes the virtual call from the hot path. The math does not change.

## Reference
D. E. Rumelhart, G. E. Hinton, R. J. Williams, "Learning representations by back-propagating errors,"
*Nature* 323(6088), pp. 533–536, 1986.
I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016, §6.5 (back-propagation and
the per-layer gradient computation).
A. G. Baydin, B. A. Pearlmutter, A. A. Radul, J. M. Siskind, "Automatic Differentiation in Machine
Learning: a Survey," *Journal of Machine Learning Research* 18(153), pp. 1–43, 2018 (reverse mode as a
chain of vector–Jacobian products).

## See also
`Dense` (the first layer migrated), `Model` (gathers the gradients), `AffineNormalization` (N2, exposes
`g ⊙ x` and `g`), `LossContract` (N9, supplies the upstream gradient), `FlashResidentWeights` (N10, the
inference-only level), `Pooling1D` (N12, the parameter-free level), `MiniBatchTraining` (N16, consumes
the accumulated gradients).
