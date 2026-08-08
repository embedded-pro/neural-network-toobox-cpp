# Testing Strategy — Algorithm Metrics

The library's test strategy: every algorithm is validated against the **mathematical invariants of
its family**, not golden output. This is the reference for the `unit-tester` agent (and humans) when
writing **unit tests** for `neural_network/`. It answers one question per algorithm: *which
mathematical properties must a correct implementation satisfy, and how do we assert them?*

## Rationale

A numerical algorithm is not validated by "it compiles and doesn't crash." Every family has a small
set of **characteristic invariants** — properties that hold for any correct implementation
regardless of parameters. A unit test earns its place by pinning one such invariant against a
**known ground truth**, not by re-running the implementation and trusting its own output.

Canonical rules still apply ([AGENTS.md](AGENTS.md), [testing.instructions.md](.github/instructions/testing.instructions.md)):
`TEST_F` on `float`, one behaviour per test, no redundant cases, **no heap in tests**, assert with
`EXPECT_NEAR` + `math::Tolerance<float>()` against reference values.

## How the agent uses this

1. Identify the target algorithm's **family** (section below).
2. From that family's row, take the **applicable metric types** and author **one `TEST_F` per
   distinct property** — not per parameter permutation.
3. Prefer **analytic ground truth** (closed-form response, known transform pair, hand-solved system)
   over self-consistency. Fall back to a cross-method check only when no closed form exists.
4. Always include the cross-cutting metrics (accuracy, boundary, determinism/reset) plus the
   family-specific ones. Keep signal/data generation on the stack (`std::array`, bounded buffers).

## Metric types (the vocabulary)

| #  | Metric type                   | What it asserts                                              | Typical assertion                                                       |
|----|-------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| M1 | **Numerical accuracy**        | output matches a closed-form / reference value               | `EXPECT_NEAR(out, ref, tol)`; ULP error for math funcs                  |
| M5 | **Convergence**               | iterative process reaches the answer                         | iterations-to-tolerance; monotonic objective/residual; contraction rate |
| M6 | **Boundary / edge**           | zero, saturation, extreme magnitude, min sizes               | clamp limits; zero-in→zero-out; no NaN/Inf at extremes                  |
| M7 | **Invariants & conservation** | energy/Parseval, norm, probability mass, orthogonality       | Parseval residual; ‖q‖=1; Σsoftmax=1; energy drift bound                |

## Family → metric-type matrix

| Family          | M1 | M5 | M6 | M7 |
|-----------------|:--:|:--:|:--:|:--:|
| Neural network  | ●  | ○  | ●  | ●  |

● primary   ○ situational

## Per-component implementation checklist

### Neural network — `neural_network/`

`activation/*`, `layer/Dense`, `losses/*`, `model/Model`.

- **M1 activation values** — reference points: `sigmoid(0)=0.5`, `tanh(0)=0`, `relu(−x)=0`,
  `leaky_relu` slope; output range bounds; monotonicity where expected.
- **M7 softmax** — outputs sum to 1 and are non-negative; shift-invariance.
- **M1 loss values & gradients** — MSE of identical vectors = 0; loss non-negative; analytic gradient
  matches finite-difference (M1 gradient check).
- **M1 dense layer** — `output = W·x + b`; back-prop gradient check.
- **Model (M6)** — forward pass is deterministic and equals the manual layer composition.

---

## Anti-patterns

- Golden-output snapshots with no independent reference ("the output is whatever it printed").
- One test per parameter value instead of one per property (violates *no redundant tests*).
- Asserting only "no NaN / no crash" without a numerical reference.
- Heap-allocated signal buffers in tests — use `std::array` / bounded buffers.
- Re-deriving the algorithm inside the test as the "reference" — the reference must be independent
  (closed form, hand computation, or a distinct method).
