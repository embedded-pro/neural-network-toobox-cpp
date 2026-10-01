# Classification Metrics — Overview

## What it is
The small toolkit for scoring a classifier. **ArgMax** turns a network's `K` output scores into one
predicted class (the index of the largest score). A **confusion matrix** counts, for every pair
(actual class, predicted class), how often it happened: row = what the sample really was, column = what
the network said. **Accuracy**, per-class **precision**, **recall** and **F1**, **balanced accuracy**
and **macro F1** are all ratios of those counts. A binary head with one output uses a threshold
instead of ArgMax.

## Why it matters (embedded)
A model that scored well on the desktop can lose accuracy on the device: a wrong weight import, a
different feature front end, a sensor that drifts. Counting on the device is the only way to see it,
and it is cheap: a `K`-class confusion matrix is `K² + 2` 32-bit counters (408 B for 10 classes), one
increment per inference, and the ratios are computed only when someone asks. When even that is too
much RAM, a two-counter accuracy meter does the basic job.

## How it works (intuition)
Softmax and sigmoid never change which score is largest, so the prediction is taken straight from the
logits. That skips the exponentials, and it also keeps near-ties that float rounding inside softmax can
merge. Ties go to the lowest index, like `numpy.argmax`, so device and desktop agree. Accuracy alone
misleads on unbalanced data: a wake-word detector that always answers "background" can score 99 %.
Recall asks "of the samples that really were class `k`, how many did we catch?". Precision asks "of the
times we said `k`, how often were we right?". F1 is their harmonic mean. Balanced accuracy averages
recall over classes, so every class counts equally whatever its frequency. When a ratio has nothing to
divide by (a class never seen), the answer is "undefined", not a made-up zero. Counts are integers,
because a float counter stops increasing at about 16.8 million samples.

## Key parameters
- **Number of classes `K`** — compile-time; sets the matrix size (`K²` counters).
- **Threshold** (binary head) — `0.5` on a sigmoid probability, `0` on a raw logit; a sample is
  positive only when the score is strictly above it.
- **Evaluation window** — when to `Reset()`: per validation pass, per hour of field data, etc.

## Reference
M. Sokolova, G. Lapalme, "A systematic analysis of performance measures for classification tasks,"
*Information Processing & Management* 45(4), pp. 427–437, 2009; C. J. van Rijsbergen, *Information
Retrieval*, 2nd ed., Butterworths (1979), Ch. 7 (the F-measure); K. H. Brodersen, C. S. Ong,
K. E. Stephan, J. M. Buhmann, "The balanced accuracy and its posterior distribution," *ICPR*, 2010.

## See also
`Softmax` (never needed before ArgMax), `CategoricalCrossEntropy` (the differentiable training loss
behind accuracy; takes logits), `Identity` (N1, logit output), `math::MeanSquaredError` / `RSquaredScore`
in numerical `Statistics.hpp` (the regression counterparts), `MiniBatchTraining` (N16, validation
accuracy per epoch), `WeightExport` (N13, on-device accuracy check of an imported model).
