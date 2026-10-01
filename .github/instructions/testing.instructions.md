---
description: "Neural Network Toolbox testing (float-only): TEST_F on float, StrictMock only, anonymous-namespace fixtures, no plain TEST(), no redundant cases, Arrange-Act-Assert. Canonical: AGENTS.md."
applyTo: "**/test/**"
---

# Neural Network Toolbox Testing Guidelines

## File Structure

- Test files: `neural_network/{domain}/test/Test{ComponentName}.cpp`
  (`{domain}` is `activation`, `layer`, `losses` or `model`)
- CMake: tests added via `add_subdirectory(test)` with standard test target patterns
  (`neural_network.{domain}_test`, wired for the QEMU presets like the existing test targets)

## Framework

- GoogleTest for assertions — **`TEST_F` on `float`** (no `TYPED_TEST`, no multi-type)
- GoogleMock (`testing::StrictMock<>`) only when needed
- No heap allocation in tests — same rules as production code
- **NEVER use plain `TEST()` macro** — cppcheck reports `syntaxError`

## Fixture Test Pattern (float)

```cpp
#include "neural_network/activation/Sigmoid.hpp"
#include "numerical/math/Tolerance.hpp"
#include <gtest/gtest.h>

namespace
{
    class TestSigmoid
        : public ::testing::Test
    {
    protected:
        neural_network::Sigmoid<float> activation;
    };
}

TEST_F(TestSigmoid, ForwardAtZeroIsOneHalf)
{
    EXPECT_NEAR(activation.Forward(0.0f), 0.5f, math::Tolerance<float>());
}
```

## Rules

- Fixture class and type aliases go inside anonymous `namespace {}`
- `TEST_F` macros go **outside** the anonymous namespace
- Include `<gtest/gtest.h>` (not `<gmock/gmock.h>`) unless gmock matchers are needed
- Use `testing::StrictMock<MockType>` for strict mock expectations (e.g. a mocked regularization term in loss tests)
- **Math in tests**: use `std::exp`, `std::log`, `std::tanh`, `std::abs`, etc. directly — never `math::Exp`, `math::Log`, etc. in test code
- **ONLY `StrictMock`**: Never use `testing::NiceMock<>` or bare mock instantiation — `NiceMock` silences unexpected-call warnings, masking test gaps; `StrictMock` enforces all interactions explicitly
- Test `float` only (single type) — no multi-type tests
- **No redundant tests** — implement exactly the spec's enumerated cases; no overlapping/extra cases
- Test numerical accuracy against known reference values (not just "doesn't crash")
- Check every analytic gradient (activation backward, loss gradient, layer parameter/input gradient) against a central finite difference
- Pick the properties to assert from [`TESTING.md`](../../TESTING.md) — the per-family metric reference (accuracy, convergence, boundaries, invariants)
- Test genuine edge cases (zero input, saturation, extreme logits) without duplicating coverage
- One behavior per test — keep tests focused
- Use descriptive test names that explain the scenario
- Allman brace style and PascalCase naming apply to test code too

## TDD Approach

- **Clarify requirements first**: Before writing any code, define and document all use cases, inputs, outputs, and edge cases as test cases
- **Write tests before implementation**: Tests define the expected behavior; implementation exists only to satisfy the tests
- **Red-Green-Refactor cycle**: Write a failing test, make it pass with minimal code, then refactor while keeping tests green

## Coverage for Template Code

When `EMIL_ENABLE_COVERAGE` is set, template code needs explicit instantiation in a `.cpp` file that is compiled with coverage flags. Add to the header (guarded):

```cpp
#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
extern template class Dense<float, 3, 2>;
#endif
```

And in the matching `.cpp` file, registered with `neural_network_add_coverage_sources()`:

```cpp
#include "neural_network/layer/Dense.hpp"

namespace neural_network
{
    template class Dense<float, 3, 2>;
}
```
