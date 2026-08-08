# Neural Network Toolbox — Roadmap

Prioritized backlog for the neural network stack (activations, layers, losses, model, training).
Shared numerical primitives (`math`, `optimization`, `regularization`, `solvers`) are consumed
from [numerical-toolbox-cpp](https://github.com/embedded-pro/numerical-toolbox-cpp).

## Neural network algorithms

The library already has the core building blocks: activation functions, a dense layer, standard
loss functions, and a sequential model. What is missing is a richer layer zoo, recurrent variants,
and on-device training support.

All items are *(float-first)* — weights and activations exceed the `Q15`/`Q31` range.

### Backlog (by priority)

| #   | Component                              | Target module   | Difficulty |
|-----|----------------------------------------|-----------------|------------|
| N1  | Convolutional layer (1D)               | `layer` (new)   | ★★☆☆☆      |
| N2  | Batch normalisation layer              | `layer` (new)   | ★★☆☆☆      |
| N3  | Dropout layer (inference mask)         | `layer` (new)   | ★☆☆☆☆      |
| N4  | ELU / SELU activation                  | `activation`    | ★☆☆☆☆      |
| N5  | Recurrent layer (Elman / vanilla RNN)  | `layer` (new)   | ★★★☆☆      |
| N6  | LSTM cell                              | `layer` (new)   | ★★★☆☆      |
| N7  | On-device SGD training loop            | `model`         | ★★★☆☆      |
| N8  | Weight serialization / deserialization | `model` (new)   | ★★☆☆☆      |
