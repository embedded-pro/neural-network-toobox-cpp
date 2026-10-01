# Mini-Batch SGD Training Step — Overview

## What it is
The on-device training loop for a `Model`. For every training sample it runs the network forward,
asks the loss how the output should change, and runs that gradient backward through the network. The
layers add up each sample's weight gradients. After `B` samples it averages them and moves every
trainable weight a small step downhill: `θ ← θ − η (ḡ + λθ)`. Then it clears the sums and starts the
next batch. It replaces `Model::Train`, which ran a generic optimiser on the parameter vector and never
used the network's forward or backward pass.

## Why it matters (embedded)
Devices drift: sensors age, users differ, environments change. Adapting a deployed model in the field,
without a round trip to a server, needs training that fits beside inference in a few kilobytes of RAM.
This trainer stores no samples and no gradient copies. It reads each layer's gradient where it
already lives (N8), updates one layer at a time, and keeps only a few words of its own state. A
per-layer freeze mask supports the realistic MCU pattern: keep the feature layers from the factory and
fine-tune only the last layer(s) on local data.

## How it works (intuition)
The loss over a mini-batch is the average of the per-sample losses, so its gradient is the average of the
per-sample gradients. Back-propagation gives each one, and the layers simply add them together. Averaging
over a few samples instead of one smooths the noise of pure per-sample SGD. It still costs the same
memory per sample, because the sum lives in the gradient buffers the layers already have. Three safety
rails go on top:
- **Weight decay** adds `λθ` to the gradient, which pulls unused weights toward zero (L2 regularisation).
- **Global-norm clipping** shrinks an unusually large gradient to a fixed length and keeps its direction,
  so one bad batch cannot throw the weights far away.
- **A finite-value check** discards a batch whose gradient contains NaN or infinity instead of poisoning
  every weight with it.

A final partial batch is averaged over the samples it actually holds, so it is still a true mean.

## Key parameters
- **Batch size `B`** (compile time): how many samples are summed before each step. It does not change
  RAM, only how often the weights move.
- **Learning rate `η`**: step length. It can be changed between steps for schedules.
- **Weight decay `λ`**: strength of the L2 pull toward zero (`ηλ < 1`).
- **Max gradient norm `c`** (optional): clipping threshold for the averaged data gradient. Weight decay
  is added after clipping.
- **Trainable-layer mask**: one flag per layer. Frozen layers are still used in the forward and
  backward passes but never change.

## Reference
L. Bottou, "Large-Scale Machine Learning with Stochastic Gradient Descent," in *Proc. COMPSTAT'2010*,
Physica-Verlag, pp. 177–186, 2010.
H. Robbins, S. Monro, "A Stochastic Approximation Method," *The Annals of Mathematical Statistics*
22(3), pp. 400–407, 1951.
I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016: §7.1.1 (L2 parameter
regularisation), §8.1.3 (batch and minibatch algorithms), §8.3.1 (stochastic gradient descent),
§10.11.1 (clipping gradients).
R. Pascanu, T. Mikolov, Y. Bengio, "On the difficulty of training recurrent neural networks," in
*Proc. ICML*, PMLR 28(3), pp. 1310–1318, 2013 (global-norm clipping).
I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization," *ICLR*, 2019 (why L2 and weight
decay coincide for SGD but not for Adam).
H. Cai, C. Gan, L. Zhu, S. Han, "TinyTL: Reduce Memory, Not Parameters for Efficient On-Device
Learning," *NeurIPS*, 2020; J. Lin, L. Zhu, W.-M. Chen, W.-C. Wang, C. Gan, S. Han, "On-Device Training
Under 256KB Memory," *NeurIPS*, 2022 (partial fine-tuning on microcontrollers).

## See also
`Model` (forward/backward chain this trainer drives), `TrainableLayerInterface` (N8, the accumulated
gradients), `LossContract` (N9, the per-sample loss gradient), `WeightInitialization` (N3, starting
weights and a PRNG for shuffling), `Identity` (N1, linear/logit outputs), step optimisers (N14, upstream
momentum/Adam as a follow-up), `WeightExport` (N13, imported weights to fine-tune).
