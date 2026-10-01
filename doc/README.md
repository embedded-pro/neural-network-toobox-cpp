# Neural Network

Components for building and evaluating feed-forward neural networks on-device with static memory
allocation: activations, layers, losses and the sequential model.

## Categories

| Category                                     | Description                                                           |
|----------------------------------------------|-----------------------------------------------------------------------|
| [Activation Functions](activation/README.md) | Non-linear transforms: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax       |
| [Layers](layer/README.md)                    | Dense (fully connected) layer with forward and backward pass          |
| [Loss Functions](losses/README.md)           | MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy (on logits) |
| [Model](model/README.md)                     | Network overview and the compile-time checked sequential model        |

Each category page lists its algorithms with a brief description and links to the detailed
documentation. New algorithm docs follow [TEMPLATE.md](TEMPLATE.md).

> **See also:** Optimization and Regularization primitives are provided by
> [numerical-toolbox-cpp](https://github.com/embedded-pro/numerical-toolbox-cpp) and consumed via
> FetchContent.
