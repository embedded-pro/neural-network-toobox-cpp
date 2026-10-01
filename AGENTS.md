# Neural Network Toolbox — Agent Rules (canonical)

Single source of truth for **Claude and Copilot**. `CLAUDE.md` and
`.github/copilot-instructions.md` point here. Code-file specifics load on demand from
`.github/instructions/`. Deployment recipe: `roadmap/DEPLOYMENT.md`.

Neural network algorithms library (activations, layers, losses, model) for
resource-constrained embedded systems. Real-time, deterministic, no heap. The shared numerical
primitives (`math`, `optimization`, `regularization`, …) are consumed from
[numerical-toolbox-cpp](https://github.com/embedded-pro/numerical-toolbox-cpp) via `FetchContent`
and included as `numerical/<domain>/…`.

## Scope

- The rules below apply to the library (`neural_network/**`, including its tests) and to
  `platform/**`.
- `simulator/**` is host-only Qt tooling and is **exempt from the no-heap and float-only rules**:
  `std::vector`, `std::string`, `QString` and other Qt/heap types are allowed there, and exceptions
  may be used if they are caught at the UI boundary. Style, no-comments, `TEST_F` and `StrictMock`
  rules still apply to simulator code and tests. Library code must never depend on the simulator.

## Numeric policy — FLOAT-ONLY (current)

- New algorithms are **float-only**. Write generic `template<typename T>`, add
  `static_assert(std::is_floating_point_v<T>, "...")`, and **instantiate/test `float` only**.
- Do **NOT** implement `Q15`/`Q31` now. The generic `T` keeps a fixed-point specialisation cheap later.
- `std::numbers::pi_v<float>` — never hardcode constants.

## Memory — no heap

- Forbidden: `new`/`delete`/`malloc`/`free`, `make_unique`/`make_shared`,
  `std::vector`/`string`/`deque`/`list`/`map`/`set`.
- Use: `infra::BoundedVector<T>::WithMaxSize<N>`, `infra::BoundedString::WithStorage<N>`,
  `infra::BoundedDeque<T>::WithMaxSize<N>`, `infra::BoundedList<T>::WithMaxSize<N>`,
  `std::array<T,N>`, `std::optional<T>`. Stack/static only. No recursion. **Tests too.**

## Embedded optimizations (algorithm headers)

```cpp
#pragma once
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif
#include "numerical/math/CompilerOptimizations.hpp"
```

`OPTIMIZE_FOR_SPEED` on hot paths (`Forward/Backward/ForwardVector/BackwardVector/Cost/Gradient`).
Pure interfaces exempt. It expands only when `NEURAL_NETWORK_TOOLBOX_ENABLE_OPTIMIZATIONS` is on: the option
is forwarded to `NUMERICAL_TOOLBOX_ENABLE_OPTIMIZATIONS`, and `NumericalToolbox_ENABLE_OPTIMIZATIONS` reaches
every target as a usage requirement of `numerical.math`. Never re-add it with `add_definitions`.

## Style

- Allman braces, 4-space indent, `.clang-format` authoritative.
- Brace-init `{}` everywhere. PascalCase types/methods, camelCase members, lowercase namespaces.
- Functions ≤ ~30 lines. `const`/`constexpr`-correct. Fixed-width ints. Full descriptive names.
- **No comments** except license headers, `NOLINT`, and a brief note on genuinely non-obvious math.

## Interfaces & errors

- Interfaces = pure virtual classes; `virtual ~I() = default` — **never** `= 0` destructors.
- No exceptions. Use `std::optional<T>` or status enums. `assert`/`really_assert` for preconditions.
- SOLID + DIP: constructor injection; depend on abstractions. RAII. No virtual calls in real-time paths.

## Namespaces

`neural_network`. Shared primitives keep their upstream namespaces (`math`, `optimization`,
`regularization`, …) from numerical-toolbox.

## Math functions

- **Production code** calls `math::Exp`, `math::Log`, `math::Tanh`, etc. from
  `numerical/math/Math.hpp`. Never call `std::` cmath functions directly in production.
- If a needed `<cmath>` function is missing from `Math.hpp`, add the `constexpr` wrapper upstream in
  numerical-toolbox-cpp first, bump the pinned revision, then use `math::FunctionName`.
- Each function is individually overridable at compile time via `#ifndef MATH_<NAME>_OVERRIDE`
  (e.g. `MATH_EXP_OVERRIDE`). Define the macro in CMake and supply your own template definition.
- **Unit tests** call `std::exp`, `std::log`, `std::tanh`, etc. directly. Never use `math::` in tests.

## Testing

- GoogleTest. **`TEST_F` on `float`** — no `TYPED_TEST`, no multi-type. **Never plain `TEST()`**.
- **`StrictMock` only** (no `NiceMock`/bare). Fixture + aliases in an anonymous namespace; macros outside it.
- **No redundant tests** — implement exactly the spec's enumerated cases, one behavior per test,
  Arrange/Act/Assert, `EXPECT_NEAR` + `math::Tolerance<float>()`. No heap.
- **Coverage ≥ 90 %** on new code — target for every algorithm before merge. CI (`static-analysis.yml`)
  builds the `coverage` preset, runs the tests, exports a gcovr report and uploads it to SonarCloud;
  the SonarCloud quality gate on the pull request reports the result. The workflow step itself does
  not fail on low coverage, so check the SonarCloud result before merging.

## CMake (neural_network/ targets)

- `neural_network_add_header_library(<target>)`, `neural_network_add_coverage_sources(<target> <Name>.cpp)`,
  `${NEURAL_NETWORK_VISIBILITY}`.
- Coverage `.cpp`: `template class <Name><float, ...>;`. `extern template` guarded by
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`.
- Test executables are also wired for the QEMU presets exactly like the existing
  `neural_network/<domain>/test/CMakeLists.txt` files.

## Docs

Design-first `doc/<domain>/<Name>.md` per `doc/TEMPLATE.md`. Math/theory/complexity/pitfalls —
no code, no class names, no usage examples. Update `doc/<domain>/README.md` when adding a new algorithm.
`doc/README.md` is the category index linking each `doc/<domain>/README.md`. A family of closely
related algorithms may share one doc (e.g. `doc/activation/Activation.md`); each algorithm still gets
its own row in the domain README, linking its section anchor. `python3 scripts/validate-docs.py` must
pass.

## Assistant behavior — be terse

- Minimal prose. No preamble/postamble, no restating the plan, no summaries unless asked.
- Report results as file paths + pass/fail. Don't narrate routine tool calls.
- Don't re-read files already read; batch reads; prefer targeted edits.
- Build: `cmake --preset host && cmake --build --preset host` · Test: `ctest --preset host`
  (needs Qt6 for the simulator; without Qt6 use `host-single-Debug` for all three steps).
