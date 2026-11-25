# neural-network-toobox-cpp — Agent Rules

<!-- shared:begin 0.6.0 sha256:f57badde9903d1ff -->
Canonical rules for Claude, Copilot and sub-agents. Sections marked *(shared)* come from embedded-pro/embedded-agent-kit:
edit them there, not here. *(repo)* sections belong to this repository. *Exceptions* overrides a shared rule.

## Behaviour *(shared)*

- **Be as terse as possible.** No preamble, summary or narration. Report file paths and pass/fail.
- Everything you write — code, docs, agent files, PR text — uses the fewest words that stay unambiguous.
- Prose (docs, comments, agent files, commit and PR text) follows ASD-STE100 Simplified Technical English:
  one word, one meaning; active voice; one instruction per sentence; sentences ≤ 20 words (procedures) or ≤ 25 (descriptions).
- Ambiguous request: ask. Clear request: act.
- Do only what was asked. No unrelated refactors, no extra features.
- Batch reads, don't re-read, prefer targeted edits.
- Never change a shared section. Add a deviation to *Exceptions* only when the user approves it.

## Workflow *(shared)*

1. Multi-file or architectural change: update the design document first. Get approval before you write code.
2. Read *Domain* and *Build*. Read each `.github/instructions/*.instructions.md` whose `applyTo` matches a file you change.
3. Find the nearest existing pattern. Follow it.
4. Per behaviour: write a failing test, write the minimum code that passes it, refactor with all tests green.
5. Add each new file to its `CMakeLists.txt`.
6. Update the documentation for each behavioural change.
7. Run the *Build* commands until build and tests pass. Never disable, skip or weaken a test to get green.
8. Non-trivial change: run the `reviewer` agent, or ask the user to run it. Fix each CRITICAL and WARNING finding.
9. Report changed files, build pass/fail, test pass/fail.

## Memory *(shared)*

Scope: production code. Tests and host tools may use the heap.

- Forbidden: `new`/`delete`/`malloc`/`free`, `make_unique`/`make_shared`,
  `std::vector`/`string`/`deque`/`list`/`map`/`set`, recursion.
- Use: `infra::BoundedVector<T>::WithMaxSize<N>`, `infra::BoundedString::WithStorage<N>`,
  `infra::BoundedDeque`, `infra::BoundedList`/`IntrusiveList`, `std::array`, `std::optional`, `std::variant`.

## Interfaces & errors *(shared)*

- No virtual destructors: each adds a deleting destructor and thunks to flash.
  Interface = pure virtual methods + `protected: ~I() = default;` (non-virtual).
- Public `virtual ~Base() = default;` only on a base destroyed through a base pointer:
  `delete`/`std::unique_ptr<Base>`, `infra::PolymorphicVariant<Base, ...>`, `infra::OptionalForPolymorphicObjects<Base, ...>`.
  Put it on that base, not on every interface.
- Not needed for `std::variant`, `std::optional<Concrete>`, `infra::SharedPtr`: they destroy the concrete type.
- No exceptions. `std::optional`, `bool` or status enum; `really_assert` for preconditions. Never swallow an error.
- Constructor injection; no global state.

## Optimisation *(shared)*

- No `#pragma GCC optimize`, no `optimize` attribute: GCC won't inline across differing options.
  The consumer sets flags per translation unit.
- `OPTIMIZE_FOR_SPEED` only on hot functions defined in a header.
- Hot paths: no heap, no virtual dispatch, no blocking.

## Style *(shared)*

- `.clang-format` is authoritative. Allman, 4 spaces, `{}` init.
- PascalCase types/methods; camelCase members and enum values; lowercase namespaces.
  No `I` prefix; implementations end in `Impl`.
- `#pragma once` only.
- Fixed-width integers. `const`/`constexpr`-correct. Functions ≤ 30 lines (hard limit 50).
- Comments: non-obvious *why* only, one line. No `TODO`/`FIXME`, commented-out code, or restating the code.
  Fix or delete a comment the change makes stale.

## Testing *(shared)*

- `TEST_F` for fixtures, `TYPED_TEST` for types, plain `TEST()` for stateless checks.
- `StrictMock` only. `EXPECT_NEAR` with explicit tolerance. Fixture in an anonymous namespace, macros outside.
- Heap allowed in tests.

## Documentation *(shared)*

- Update docs before or with any behavioural change.
- One fact, one place: link, never restate.
- Visuals: Mermaid or ASCII.

## Desktop tools *(shared)*

Scope: host tools that people install and run, GUI or console.

- Open the log file first in `main`, before the application object.
  A GUI-subsystem build (`WIN32_EXECUTABLE`) has no stderr: the log file is its only output.
- Write the log to a per-user writable folder (`AppLocalDataLocation`) and rotate it.
  Capture the toolkit's messages (`qInstallMessageHandler`) from debug level.
- Install a crash handler after the log. It writes a `FATAL` line on a crash signal, `std::terminate`
  or an unhandled SEH exception, and a minidump on Windows. Qt tools use ui-cpp's `IssueLog` and `CrashReporter`.
- Log each start-up step: version, OS, toolkit version, arguments, restored settings, first frame.
- Provide `--safe-start`: it skips the saved state, so a bad saved state cannot block start-up.
- Give the log folder a menu item (Help ▸ Open Log Folder).
- Install test: start the tool, close it normally, then start it again with saved state.
  Print the log. On a failure, upload the log folder and the dumps.
<!-- shared:end -->

## Domain *(repo)*

- Neural-network algorithms (activations, layers, losses, model) for embedded targets. Namespace `neural_network`.
- Source: `neural_network/<domain>/` with `<domain>` = `activation`, `layer`, `losses`, `model`.
- `simulator/` is host-only Qt tooling. Heap, `std::string`, `QString` and exceptions caught at the UI boundary are
  allowed there; float-only does not apply. Style and test rules apply. Library code never depends on the simulator.
- Shared primitives come from numerical-toolbox-cpp through `FetchContent`: `numerical/<domain>/...`, namespaces
  `math`, `optimization`, `regularization`. A missing `<cmath>` wrapper goes into numerical-toolbox `Math.hpp`
  first; then bump the pinned revision.
- Float only: `template<typename T>` with `static_assert(std::is_floating_point_v<T>)`. Instantiate and test
  `float`. No `Q15`/`Q31`. Use `std::numbers::pi_v<float>`, never a literal constant.
- Production code calls `math::Exp`, `math::Log`, `math::Tanh`, ... from `numerical/math/Math.hpp`, never `std::`
  cmath. `MATH_<NAME>_OVERRIDE` replaces one function at compile time. Tests call `std::` math, never `math::`.
- `OPTIMIZE_FOR_SPEED` on `Forward`, `Backward`, `ForwardVector`, `BackwardVector`, `Cost`, `Gradient`. It expands
  only with `NEURAL_NETWORK_TOOLBOX_ENABLE_OPTIMIZATIONS`; never add it again with `add_definitions`.
- No virtual calls in real-time paths.
- CMake: `neural_network_add_header_library(<target>)` with `${NEURAL_NETWORK_VISIBILITY}`. Wire each test target
  for the QEMU presets like the existing `neural_network/<domain>/test/CMakeLists.txt`.
- Coverage: at the header bottom, `extern template class <Name><float, ...>;` inside
  `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD`. The matching `.cpp` holds `template class <Name><float, ...>;`,
  added with `neural_network_add_coverage_sources(<target> <Name>.cpp)`.
- Tests: `neural_network/<domain>/test/Test<Name>.cpp`, `TEST_F` on `float` only, no `TYPED_TEST`. Check each
  analytic gradient against a central finite difference (`activation/test/ActivationFiniteDifference.hpp`,
  `losses/test/LossTestSupport.hpp`). Select the properties from `TESTING.md`.
- Coverage ≥ 90 % on new code. CI does not fail on it: check the SonarCloud result before merge.
- Docs: design-first `doc/<domain>/<Name>.md` per `doc/TEMPLATE.md`: math, theory, complexity, pitfalls; no code,
  class names or usage examples. A family may share one doc; each algorithm gets its own row, linking its anchor,
  in `doc/<domain>/README.md`. `doc/README.md` indexes the domain READMEs.
- Roadmap deployment: `roadmap/DEPLOYMENT.md` and the `algo-implementer` skill. Tests: the `unit-tester` skill.
  Pre-roadmap refactors: the `modernizer` agent.

## Build *(repo)*

```bash
cmake --preset host && cmake --build --preset host-Debug
ctest --preset host
python3 scripts/validate-docs.py
```

The `host` preset builds the simulator and needs Qt6. Without Qt6, use `host-single-Debug` for configure, build and
test. Other presets: `coverage`, `qemu-cortex-m4`, `qemu-cortex-m7`.

## Exceptions *(repo)*

- No plain `TEST()`: cppcheck reports `syntaxError` on it. Use `TEST_F`.
