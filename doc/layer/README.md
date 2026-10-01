# Layers

Trainable transformations that make up a network, each with a forward pass, a backward pass and a
flat parameter vector.

## Algorithms

| Algorithm         | Description                                                                                              |
|-------------------|----------------------------------------------------------------------------------------------------------|
| [Dense](Dense.md) | Fully connected layer $f(W a + b)$ with back-propagation of $\delta = J_f^T g$ — $O(m \cdot n)$ per pass |
