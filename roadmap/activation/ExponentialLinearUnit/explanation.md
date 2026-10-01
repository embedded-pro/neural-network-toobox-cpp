# ELU / SELU Activation — Overview

## What it is
The **Exponential Linear Unit** is ReLU with a smooth negative side: `f(z) = z` for `z > 0` and
`α(eᶻ − 1)` for `z ≤ 0`. Negative inputs no longer map to a flat zero; they bend towards a floor of `−α`.
The **Scaled ELU (SELU)** multiplies the whole curve by `λ` and fixes the two constants at
`α ≈ 1.6733`, `λ ≈ 1.0507`. One class with parameters `(α, λ)` covers both.

## Why it matters (embedded)
ReLU hidden units that drift negative stop learning ("dead" neurons), and their outputs are all
non-negative, so the next layer sees inputs with a positive mean. ELU keeps a non-zero gradient on the
negative side and produces negative outputs that pull the mean activation towards zero, which speeds up
learning. SELU goes further: in a plain stack of fully connected layers it keeps activations near zero
mean and unit variance on its own, without a normalisation layer — useful here, where `Dense` is the only
trainable layer. The cost is one `exp` per negative element in the forward pass; the backward pass needs
no `exp` at all, because the derivative can be read off the stored output.

## How it works (intuition)
On the positive side the unit is linear (slope `λ`). On the negative side `eᶻ − 1` rises from `−1` to `0`,
so the output saturates softly at `−λα` for very negative inputs: large negative evidence is capped
instead of passed through, which makes the unit robust to noise. The negative-side slope is `λα eᶻ`,
which equals the output plus `λα` — the layer already keeps its outputs, so back-propagation costs an add
and a multiply. With `α = 1` (plain ELU) the slopes meet at `1` on both sides of zero, so the curve is
smooth. SELU's constants are chosen so that a standard-normal input yields an output with mean `0` and
variance `1`; with LeCun-normal weights (variance `1/fan_in`) each layer maps that distribution back to
itself, so it is stable through depth.

## Key parameters
- **α** — negative saturation level (ELU default `1`). Must be positive.
- **λ (scale)** — overall slope; `1` for ELU, `≈ 1.0507` for SELU.
- **SELU preset** — `(α₀₁, λ₀₁) ≈ (1.6733, 1.0507)`; self-normalising only with LeCun-normal
  initialisation (N3) and standardised inputs. With other initialisations it is just a scaled ELU.

## Reference
D.-A. Clevert, T. Unterthiner, S. Hochreiter, "Fast and Accurate Deep Network Learning by Exponential
Linear Units (ELUs)," *ICLR*, 2016 (arXiv:1511.07289); G. Klambauer, T. Unterthiner, A. Mayr,
S. Hochreiter, "Self-Normalizing Neural Networks," *Advances in Neural Information Processing Systems 30
(NeurIPS)*, 2017 (arXiv:1706.02515); Y. LeCun, L. Bottou, G. B. Orr, K.-R. Müller, "Efficient BackProp,"
in *Neural Networks: Tricks of the Trade*, Springer LNCS 1524, 1998 (LeCun-normal initialisation).

## See also
`ActivationFunction` (the contract), `LeakyReLU` (piecewise-linear alternative), `Sigmoid`/`Tanh` (same
output-reuse backward), `WeightInitialization` (N3, LeCun normal for SELU), `SwishActivations` (N5, the
exp-free Hard-Swish alternative).
