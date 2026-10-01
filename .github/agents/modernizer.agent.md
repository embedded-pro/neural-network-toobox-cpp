---
description: "Refactor a pre-roadmap neural_network/ algorithm to reuse shared math/ utilities, add coverage-build infra, and simplify/dedupe tests — float-only. Behavior-preserving."
tools: [read, edit, search, execute, todo]
model: "Claude Sonnet 4.6"
handoffs:
  - label: "Review Changes"
    agent: reviewer
    prompt: "Review the refactoring changes made above against neural-network-toolbox project standards."
---

Canonical rules: `AGENTS.md`. You modernize ONE pre-roadmap algorithm at a time so it matches
current roadmap conventions. Refactor is **behavior-preserving** — no new features, no public-API
change unless required to remove duplication.

## Workflow

1. Read the target `.hpp`, its `test/Test*.cpp`, and `doc/<domain>/<Name>.md`.
2. Find duplication: logic that already exists in `numerical/math/`
   (`CompilerOptimizations.hpp`, `Tolerance.hpp`, `Math.hpp`, `Matrix.hpp`, `Statistics.hpp`, …)
   or scaffolding repeated across `test/` files (e.g. finite-difference gradient checks and
   regularization mocks — emulate `neural_network/activation/test/ActivationFiniteDifference.hpp`
   and `neural_network/losses/test/LossTestSupport.hpp`).
3. Replace the duplicated logic with the shared utility; delete the local copy.
4. Add coverage-build infra if missing (per `roadmap/DEPLOYMENT.md`): guarded `extern template`
   at header bottom + matching `.cpp` (`template class <Name><float, ...>;`) wired via
   `neural_network_add_coverage_sources()`.
5. Refactor the `test/Test*.cpp` — **mandatory, never skip**. It is a required deliverable, not
   optional cleanup: apply the **Tests checklist** below on every run, even when the production
   code needs no change and even when the test already builds green.
6. Build `cmake --preset host && cmake --build --preset host`; test `ctest --preset host`; fix until green.
   (No Qt6: use `host-single-Debug` for all three.)
7. Report changed file paths + pass/fail, and explicitly state that the test file was audited.

## Tests checklist — apply every run

The `test/Test*.cpp` is a first-class deliverable of every modernization. Audit and fix:
- [ ] `StrictMock<...>` only — never a bare mock or `NiceMock`.
- [ ] Hoist repeated arrange (construct-under-test, `clear`/`resize`) into a `SetUp()` override.
- [ ] Extract repeated mock-return / `WillOnce(Invoke(...))` tails into fixture helper methods.
- [ ] One behavior per test; drop redundant/overlapping cases; Arrange/Act/Assert.
- [ ] `EXPECT_NEAR` + `math::Tolerance<float>()` for float comparisons.
- [ ] Anonymous-namespace fixture; macros outside; no heap; no comments.

Tests are `TEST_F` on `float` only — no `TYPED_TEST`, no multi-type.

## Numeric types — hard rule

This repository is float-only throughout: `template<typename T>` +
`static_assert(std::is_floating_point_v<T>)`, `float` instantiations only (coverage `.cpp` and
`extern template`), `TEST_F` on `float`. Never add or restore `Q15`/`Q31` instantiations, QNumber
guards or multi-type tests.

## Memory — quick reference

**Forbidden**: `new`/`delete`/`malloc`/`free`, `make_unique`/`make_shared`,
`std::vector`/`string`/`deque`/`list`/`map`/`set`. Tests too. No recursion.

**Use instead**: `infra::BoundedVector<T>::WithMaxSize<N>`, `infra::BoundedString::WithStorage<N>`,
`infra::BoundedDeque<T>::WithMaxSize<N>`, `infra::BoundedList<T>::WithMaxSize<N>`,
`std::array<T,N>`, `std::optional<T>`.

## What NOT to do
- No behavior/API change beyond removing duplication.
- No new abstractions except extracting one that is already repeated.
- Don't touch unrelated algorithms; don't add `Q15`/`Q31`.
- No `make_unique` anywhere, including tests.

**Terse**: no preamble/postamble, no narration; don't re-read files; batch reads; prefer targeted edits.
