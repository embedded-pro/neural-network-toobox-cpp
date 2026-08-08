# Neural Network

Components for building, training, and evaluating feed-forward neural networks entirely on-device with static memory allocation.

## Components

| Component                                          | Description                                                                             |
|----------------------------------------------------|-----------------------------------------------------------------------------------------|
| [Neural Network Overview](NeuralNetwork.md)        | Architecture, training loop, and how the components fit together                        |
| [Layers](layer/Layer.md)                           | Dense (fully connected) layer with forward and backward pass                            |
| [Activation Functions](activation/Activation.md)   | Non-linear element-wise transforms: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax             |
| [Loss Functions](losses/Loss.md)                   | Objective functions: MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy          |
| [Model](model/Model.md)                            | Variadic-template model composing layers, optimizer, and loss into a trainable pipeline |

> **See also:** Optimization and Regularization primitives are provided by [numerical-toolbox-cpp](https://github.com/embedded-pro/numerical-toolbox-cpp) and consumed via FetchContent.
