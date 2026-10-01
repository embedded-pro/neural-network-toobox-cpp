# Identity (Linear) Activation — Overview

## What it is
The activation that does nothing: `f(z) = z`, with derivative `f'(z) = 1`. Put after a `Dense` layer,
it leaves the raw affine output `W x + b` untouched.

## Why it matters (embedded)
Today every `Dense` must apply a non-linearity, so a network cannot emit an unbounded regression value
(a temperature, a torque, a position) or raw class logits. The workaround is to squash targets into
`[0, 1]` behind a sigmoid, which distorts the loss and saturates near the ends. Identity removes that
workaround at zero cost: no arithmetic, no state, one shared instance for every output layer.

## How it works (intuition)
A final linear unit is the natural output for regression. If the network predicts the mean of a
Gaussian, minimising mean squared error is maximum likelihood, and the linear output is the matching
(canonical) link. For classification, the linear output yields logits, and the loss applies
softmax/log-sum-exp internally, which is more stable than a separate Softmax layer. The backward pass
is the identity too: the upstream gradient passes through unchanged.

## Key parameters
- **None.** Stateless; the vector length comes from the layer it is attached to.
- **Pairing** — MSE/MAE (and Huber, N4) for regression; the logit-input CCE for classification;
  keep Sigmoid in front of the probability-input BCE until BCE-with-logits (N9) lands.

## Reference
I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press (2016), §6.2.2.1 "Linear Units for
Gaussian Output Distributions"; C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer
(2006), §5.2 (identity output activation for regression).

## See also
`ActivationFunction` (the contract), `Dense` (the layer it completes), `MeanSquaredError`,
`CategoricalCrossEntropy` (logit input), `HuberLoss` (N4), `LossContract` (N9, BCE-with-logits).
