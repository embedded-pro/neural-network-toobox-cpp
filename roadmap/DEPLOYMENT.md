# Deploying a Roadmap Algorithm (float-only recipe)

Turn one `roadmap/<domain>/<Name>/` spec into shipped code. Rules: `../AGENTS.md`.
Read the spec's three files first (`implementation.md`, `tests.md`, `explanation.md`), then:

1. **Header** `neural_network/<domain>/<Name>.hpp`
   - `#pragma once` → `#pragma GCC optimize("O3","fast-math")` (GCC/Clang guard) →
     `#include "numerical/math/CompilerOptimizations.hpp"`.
   - `template<typename T[, std::size_t sizes...]>` with
     `static_assert(std::is_floating_point_v<T>, "<Name> supports floating-point types");`.
   - Implement per `implementation.md`; `OPTIMIZE_FOR_SPEED` on the hot path(s)
     (`Forward/Backward/ForwardVector/BackwardVector/Cost/Gradient`). Transcendentals via
     `math::` from `numerical/math/Math.hpp`.
   - Bottom: `#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD` / `extern template class <Name><float, ...>;` / `#endif`.

2. **Coverage** `neural_network/<domain>/<Name>.cpp`
   - Include the header; `namespace neural_network { template class <Name><float, ...>; }`.
   - Float only — the same `<float, ...>` arguments as the `extern template` in step 1.

3. **Test** `neural_network/<domain>/test/Test<Name>.cpp`
   - Per `tests.md`. `TEST_F` on `float`; `StrictMock` only; fixture in an anonymous namespace.
   - Implement **exactly** the enumerated cases — no extra/redundant tests. `EXPECT_NEAR` +
     `math::Tolerance<float>()` (or explicit tol) against known reference values.

4. **CMake**
   - Add `.hpp` to `target_sources(...)`, `.cpp` to `neural_network_add_coverage_sources(...)`,
     `Test<Name>.cpp` to the `_test` target's `target_sources`.
   - New module: create `neural_network/<module>/CMakeLists.txt` via
     `neural_network_add_header_library(...)` and `neural_network_add_coverage_sources(...)`, add a
     `test/` subdir, register it in the parent `CMakeLists.txt`, append `neural_network.<module>` to
     `NEURAL_NETWORK_TOOLBOX_INSTALL_TARGETS` in the root `CMakeLists.txt` (otherwise it is left out of
     the installed export set), and add a `doc/<module>/` folder.

5. **QEMU wiring** — a new `neural_network.<module>_test` executable gets the same QEMU hook as the
   existing test targets (the helper from `cmake/NeuralNetworkQemuHelpers.cmake`, called right after
   `target_link_libraries` — copy it from a sibling `neural_network/<domain>/test/CMakeLists.txt`),
   and append `neural_network.<module>_test` to the `targets` list of both the
   `qemu-cortex-m4-RelWithDebInfo` and `qemu-cortex-m7-RelWithDebInfo` build presets in
   `CMakePresets.json` (those presets build only the listed targets, so an unlisted test is never built
   and its QEMU ctest entry fails). Tests added to an existing target need nothing extra.

6. **Doc** `doc/<domain>/<Name>.md` per `doc/TEMPLATE.md` (design-first; no code/class names/usage)
   — or a new section in the domain's shared family doc (e.g. `doc/activation/Activation.md`).
   Add its row to `doc/<domain>/README.md`; a new domain also gets a row in the `doc/README.md`
   category index (which `README.md`'s Documentation table links). These README tables are the
   booklet's ordering source, so the booklet (`scripts/build-booklet.py`) regenerates automatically
   in CI; never edit the booklet by hand. `python3 scripts/validate-docs.py` must pass.

7. **Build & test**, fix until green:
   `cmake --preset host && cmake --build --preset host && ctest --preset host`
   (needs Qt6 for the simulator; without it use `host-single-Debug` for all three steps; scope to the
   target/test where possible). Optionally `cmake --preset coverage && cmake --build --preset coverage
   && ctest --preset coverage` to confirm the coverage TU builds.

8. **Remove roadmap spec** — delete the entire `roadmap/<domain>/<Name>/` directory (including its
   `reference.py` scripts) once all tests are green.

**Report**: the file paths created/edited/deleted + the test result. Nothing else.

Recap: float-only (generic `T`, `float` instantiation), no heap, no comments, terse.
