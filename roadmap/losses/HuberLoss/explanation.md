# Huber Loss — Overview

## What it is
A regression loss that behaves like squared error for small residuals and like absolute error for large
ones. Per output, with residual `e = ŷ − y` and threshold `δ > 0`: `½e²` when `|e| ≤ δ`, otherwise
`δ(|e| − ½δ)`. The loss is the mean over the `N` outputs, and its gradient is the residual clipped to
`[−δ, δ]`, divided by `N`.

## Why it matters (embedded)
Sensor data has outliers: a dropped I²C read, an ADC spike, a glitching encoder. Under MSE one such
sample produces a gradient proportional to the error and can wreck a training step; the squared term
can even overflow `float` for residuals above about `1.8·10¹⁹`. MAE is robust but has a constant-size
gradient that never shrinks near the optimum, so it converges poorly. Huber keeps the smooth, shrinking
MSE gradient for inliers and caps the gradient at `δ/N` for outliers, at the cost of two compares per
output — no exp, log, or division.

## How it works (intuition)
The two pieces are glued at `|e| = δ` so that both the value (`½δ²`) and the slope (`±δ`) match; the
loss is therefore smooth enough for gradient descent everywhere. Inside the band it is a parabola, so
small errors are corrected in proportion to their size; outside it is a straight line, so an error ten
times larger pulls no harder than one just past the threshold. Setting `δ` larger than every residual
gives exactly half of MSE; setting it smaller than every residual gives `δ` times MAE minus a constant.
`δ` is in the units of the target and should sit just above the typical inlier noise.

## Key parameters
- **δ (delta)** — the quadratic/linear switch point, `> 0`, default `1`. Around `1.345σ` of the inlier
  noise gives 95 % of MSE's statistical efficiency on clean Gaussian data while staying robust.
- **Target** — bound when the loss is created, like the other losses in this library.
- **Output activation** — pair with an unbounded output (`Identity`, N1); a squashing output already
  limits the residual and hides what Huber is for.

## Reference
P. J. Huber, "Robust Estimation of a Location Parameter," *Annals of Mathematical Statistics* 35(1),
pp. 73–101, 1964; T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2nd ed.,
Springer (2009), §10.6 "Loss Functions and Robustness".

## See also
`MeanSquaredError` and `MeanAbsoluteError` (its two limiting cases), `Identity` (N1, the matching
output activation), `LossContract` (N9, which moves Huber to the per-sample `Cost/Gradient(prediction,
target)` contract), `MiniBatchTraining` (N16).
