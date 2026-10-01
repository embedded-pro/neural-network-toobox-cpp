# Activation Functions

Non-linear transforms applied after each layer's affine map, with the backward pass used by
back-propagation.

## Algorithms

| Algorithm                              | Description                                                                              |
|----------------------------------------|------------------------------------------------------------------------------------------|
| [ReLU](Activation.md#relu)             | $\max(0, z)$, derivative 1 for $z > 0$ else 0 — $O(1)$ per element                       |
| [Leaky ReLU](Activation.md#leaky-relu) | $z$ for $z > 0$ else $\alpha z$ (default $\alpha = 0.01$), derivative 1 or $\alpha$      |
| [Sigmoid](Activation.md#sigmoid)       | Numerically stable $1/(1 + e^{-z})$ in $(0, 1)$, derivative $a(1 - a)$ from the output   |
| [Tanh](Activation.md#tanh)             | $\tanh(z)$ in $(-1, 1)$, derivative $1 - a^2$ from the output                            |
| [Softmax](Activation.md#softmax)       | Max-shifted exponentials normalised to sum to 1; $O(k)$ Jacobian-vector product backward |
