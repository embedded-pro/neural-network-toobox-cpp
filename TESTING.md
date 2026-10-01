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
4. Always include the cross-cutting metrics — accuracy (M1), boundary (M6; determinism/reset
   checks are filed here too, as in the sibling repos) — plus the family-specific ones. Keep signal/data generation on the stack (`std::array`, bounded buffers).

## Metric types (the vocabulary)

| #  | Metric type                   | What it asserts                                        | Typical assertion                                                       |
|----|-------------------------------|--------------------------------------------------------|-------------------------------------------------------------------------|
| M1 | **Numerical accuracy**        | output matches a closed-form / reference value         | `EXPECT_NEAR(out, ref, tol)`; ULP error for math funcs                  |
| M5 | **Convergence**               | iterative process reaches the answer                   | iterations-to-tolerance; monotonic objective/residual; contraction rate |
| M6 | **Boundary / edge**           | zero, saturation, extreme magnitude, min sizes         | clamp limits; zero-in→zero-out; no NaN/Inf at extremes                  |
| M7 | **Invariants & conservation** | energy/Parseval, norm, probability mass, orthogonality | Parseval residual; ‖q‖=1; Σsoftmax=1; energy drift bound                |

## Family → metric-type matrix

| Family         | M1 | M5 | M6 | M7 |
|----------------|:--:|:--:|:--:|:--:|
| Neural network | ●  | ○  | ●  | ●  |

● primary   ○ situational

The IDs follow the shared M1–M9 vocabulary of robotics-toolbox-cpp and numerical-toolbox-cpp so
they mean the same thing across the three repositories. Only the families that apply to neural
networks are listed; M2 (frequency response), M3 (transient response), M4 (stability),
M8 (statistical consistency) and M9 (conditioning) are intentionally absent — do not cite them here.

## Per-component implementation checklist

### Neural network — `neural_network/`

`activation/*`, `layer/Dense`, `losses/*`, `model/Model`.

- **M1 activation values** — reference points: `sigmoid(0)=0.5`, `tanh(0)=0`, `relu(−x)=0`,
  `leaky_relu` slope; exact derivatives (`relu'(x>0)=1`, `sigmoid'(0)=0.25`, `tanh'(0)=1`);
  backward (scalar and vector) matches a central finite difference.
- **M6 activation extremes** — sigmoid stays in `[0, 1]` with no NaN for `|x|` ≫ 88 (stable
  branch); softmax of large logits (e.g. `[1000, 0, 0]`) is finite.
- **M7 softmax** — outputs sum to 1 (no clamping) and are non-negative; shift-invariance; backward
  (Jacobian-vector product) matches finite difference and its components sum to 0.
- **M1 loss values & gradients** — mean normalisation (`1/N`) of MSE/MAE/BCE; CCE equals
  `logsumexp(z)·Σy − Σ yᵢzᵢ`; loss of a perfect prediction is 0 (MSE/MAE); analytic gradient matches
  finite difference; regularization term and gradient are added (StrictMock).
- **M6 loss boundaries** — MAE gradient is 0 at equality; BCE is finite at `p = 0` and `p = 1`
  (clamp `ε = 1e-7`); CCE cost and gradient are finite for logits of magnitude `1e30`.
- **M1 dense layer** — `output = f(W·x + b)` against a hand computation; input and parameter
  gradients match finite difference; parameters flatten row-major (`W` rows, then `b`).
- **Model (M1 + M6)** — forward pass equals the manual layer composition (M1); repeated calls are
  deterministic and parameter export/import round-trips in the documented order (M6).

---

## Anti-patterns

- Golden-output snapshots with no independent reference ("the output is whatever it printed").
- One test per parameter value instead of one per property (violates *no redundant tests*).
- Asserting only "no NaN / no crash" without a numerical reference.
- Heap-allocated signal buffers in tests — use `std::array` / bounded buffers.
- Re-deriving the algorithm inside the test as the "reference" — the reference must be independent
  (closed form, hand computation, or a distinct method).
