# Per-Sample Loss Contract + Logit-Input BCE — Overview

## What it is
A single interface for every loss: given one sample's prediction and its target, return the cost and
the gradient of that cost with respect to the prediction — `Cost(ŷ, y)` and `Gradient(ŷ, y)`. The
target arrives with each call (a vector, or a plain class index for classification), and the loss holds
no regulariser. On top of the contract it adds binary cross-entropy that reads raw logits,
`max(z, 0) − z·y + log(1 + e^{−|z|})` with gradient `σ(z) − y`, and a class-index form of the existing
log-sum-exp categorical cross-entropy.

## Why it matters (embedded)
Today each loss binds one fixed target in its constructor and adds a regulariser to the predictions, so
a training loop would need a new loss object per sample. With the target passed per call, one stateless
loss object serves a whole dataset, and a `K`-class label costs one index instead of `K` floats. The
gradient with respect to the prediction is exactly what `Model::Backward` consumes, which is the missing
link for on-device training (N16). Weight decay belongs to the weights, so it moves to the training step.

## How it works (intuition)
Cross-entropy on probabilities breaks in `float`: once a logit passes about 16.6, the sigmoid rounds to
exactly 1, the probability must be clamped, and the loss stops growing (a logit of 20 on the wrong
class reports 15.9 instead of 20) while the gradient explodes into the millions. Folding the sigmoid into
the loss removes the problem algebraically: the cost becomes a softplus, which only ever evaluates
`e^{−|z|}` (never overflows), and the gradient collapses to `σ(z) − y`, bounded by 1. Categorical
cross-entropy does the same with softmax: subtracting the largest logit before exponentiating keeps
every exponential in `(0, 1]` and the gradient is `softmax(z) − y`. Both losses therefore take the last
layer's raw logits (use an `Identity` output) and must not be preceded by a Sigmoid or Softmax layer.

## Key parameters
- **Target type** — a vector (hard, soft or multi-label targets) or `std::size_t` class index (CCE only).
- **Input kind** — probabilities for `BinaryCrossEntropy` (kept for Sigmoid outputs), logits for
  `BinaryCrossEntropyWithLogits` and `CategoricalCrossEntropy`.
- **Reduction** — per-sample: mean over outputs for MSE/MAE/BCE/Huber, sum over classes for CCE; the
  average over a mini-batch is the trainer's job (N16).
- **ε = 10⁻⁷** — clamp of the probability-input BCE only; the logit forms need none.

## Reference
I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press (2016), §6.2.2.2 "Sigmoid Units for
Bernoulli Output Distributions" (the softplus form of the loss) and §6.2.2.3 "Softmax Units for Multinoulli
Output Distributions"; P. Blanchard, D. J. Higham, N. J. Higham, "Accurately computing the log-sum-exp and
softmax functions," *IMA Journal of Numerical Analysis* 41(4), pp. 2311–2330, 2021; C. M. Bishop, *Pattern
Recognition and Machine Learning*, Springer (2006), §4.3.2 and §4.3.4 (the `σ − y` / `softmax − y` gradients).

## See also
`Identity` (N1, the logit output), `HuberLoss` (N4, migrated to this contract), `ClassificationMetrics`
(N7, same class-index convention), `MiniBatchTraining` (N16, the consumer that averages these gradients and
applies weight decay), `Sigmoid` / `Softmax` (the activations these losses absorb).
