# Loss Functions

Per-sample objectives with analytic gradients, plus an optional regularization term.

## Algorithms

| Algorithm                                                      | Description                                                                                                 |
|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [Mean Squared Error](Loss.md#mean-squared-error)               | $\frac{1}{N}\sum(\hat{y} - y)^2$, gradient $2(\hat{y} - y)/N$ — regression                                  |
| [Mean Absolute Error](Loss.md#mean-absolute-error)             | $\frac{1}{N}\sum\lvert\hat{y} - y\rvert$, gradient $\operatorname{sign}(\hat{y} - y)/N$ — robust regression |
| [Binary Cross-Entropy](Loss.md#binary-cross-entropy)           | Mean BCE on probabilities clamped to $[10^{-7}, 1 - 10^{-7}]$ — binary / multi-label                        |
| [Categorical Cross-Entropy](Loss.md#categorical-cross-entropy) | Fused softmax + cross-entropy on logits via stable log-sum-exp — multi-class                                |
