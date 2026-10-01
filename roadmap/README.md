# Roadmap — Pseudocode Specifications

Design-level pseudocode for every proposed component in [../ROADMAP.md](../ROADMAP.md).
These are **specifications, not compilable code** — they describe *what* to build and *how*
the algorithm works, so an implementer can produce the real templated C++ afterwards.
Deploy one with [DEPLOYMENT.md](DEPLOYMENT.md).

## Convention

The tree mirrors the `neural_network/` layout. Each algorithm gets its own folder with **three files**:

```text
roadmap/<domain>/<AlgorithmName>/
├── implementation.md   # data structures, interface, algorithm pseudocode, complexity, float notes, deployment
├── tests.md            # GoogleTest test plan in pseudocode (TEST_F on float, StrictMock, no heap)
└── explanation.md      # short plain-language overview + the reference paper
```

A folder whose `tests.md` quotes computed reference values also holds the script that produced them
(`reference.py`, numpy float32; run it from that folder with `python3 reference.py`).

All pseudocode respects the library constraints: **no heap**, no recursion, bounded containers /
`std::array`, `OPTIMIZE_FOR_SPEED` on hot paths, and the **float-only** policy — a generic
`template<typename T>` interface guarded by `static_assert(std::is_floating_point_v<T>)` and
instantiated and tested for **`float`** only (no `Q15`/`Q31`).

The canonical worked example is
[activation/Identity](activation/Identity/implementation.md).

## Deployment shape (float-only)

Each spec maps to this concrete artifact set when deployed into `neural_network/`:

| Spec file           | Deploys to                                                                   |
|---------------------|------------------------------------------------------------------------------|
| `implementation.md` | `neural_network/<domain>/<Name>.hpp` + `<Name>.cpp` (coverage instantiation) |
| `tests.md`          | `neural_network/<domain>/test/Test<Name>.cpp`                                |
| `explanation.md`    | `doc/<domain>/<Name>.md` (expanded to follow `doc/TEMPLATE.md`)              |

Header shape (mirroring existing components such as `Dense.hpp`):

```cpp
#pragma once
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif
#include "numerical/math/CompilerOptimizations.hpp"

namespace neural_network
{
    template<typename T /*, std::size_t sizes... */>
    class <Name>
    {
        static_assert(std::is_floating_point_v<T>, "<Name> requires a floating-point type");
    public:
        // OPTIMIZE_FOR_SPEED on hot paths (Forward/Backward/ForwardVector/BackwardVector/Cost/Gradient)
    };

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class <Name><float /*, sizes... */>;
#endif
}
```

Coverage `.cpp`:

```cpp
#include "neural_network/<domain>/<Name>.hpp"
namespace neural_network { template class <Name><float /*, sizes... */>; }
```

CMake wiring:
- add `<Name>.hpp` to `target_sources(neural_network.<domain> PRIVATE ...)`
- add `<Name>.cpp` to `neural_network_add_coverage_sources(neural_network.<domain> ...)`
- add `Test<Name>.cpp` to the `neural_network.<domain>_test` target's `target_sources`
- new module (`initialization`, `metrics`): create it with `neural_network_add_header_library(...)`
  and register it in [neural_network/CMakeLists.txt](../neural_network/CMakeLists.txt)

Tests: single-type **`TEST_F` on `float`**, `StrictMock` only, no heap, no redundant cases; validate
reference vectors with `EXPECT_NEAR` and `math::Tolerance<float>()` (or an explicit tolerance), and
assert every `Backward` against a finite-difference gradient check.

## Index

| Domain           | Algorithm                                                                                    | Status  |
|------------------|----------------------------------------------------------------------------------------------|---------|
| `activation`     | [Identity](activation/Identity/implementation.md) (N1)                                       | pending |
| `activation`     | [SwishActivations](activation/SwishActivations/implementation.md) (N5)                       | pending |
| `activation`     | [ExponentialLinearUnit](activation/ExponentialLinearUnit/implementation.md) (N6)             | pending |
| `layer`          | [AffineNormalization](layer/AffineNormalization/implementation.md) (N2)                      | pending |
| `layer`          | [TrainableLayerInterface](layer/TrainableLayerInterface/implementation.md) (N8)              | pending |
| `layer`          | [FlashResidentWeights](layer/FlashResidentWeights/implementation.md) (N10)                   | pending |
| `layer`          | [Convolution1D](layer/Convolution1D/implementation.md) (N11)                                 | pending |
| `layer`          | [Pooling1D](layer/Pooling1D/implementation.md) (N12)                                         | pending |
| `layer`          | [ElmanRnn](layer/ElmanRnn/implementation.md) (N17)                                           | pending |
| `layer`          | [Convolution2D](layer/Convolution2D/implementation.md) (N18)                                 | pending |
| `layer`          | [DepthwiseSeparableConvolution](layer/DepthwiseSeparableConvolution/implementation.md) (N19) | pending |
| `layer`          | [Gru](layer/Gru/implementation.md) (N20)                                                     | pending |
| `layer`          | [Lstm](layer/Lstm/implementation.md) (N21)                                                   | pending |
| `initialization` | [WeightInitialization](initialization/WeightInitialization/implementation.md) (N3)           | pending |
| `losses`         | [HuberLoss](losses/HuberLoss/implementation.md) (N4)                                         | pending |
| `losses`         | [LossContract](losses/LossContract/implementation.md) (N9)                                   | pending |
| `metrics`        | [ClassificationMetrics](metrics/ClassificationMetrics/implementation.md) (N7)                | pending |
| `model`          | [WeightExport](model/WeightExport/implementation.md) (N13)                                   | pending |
| `model`          | [MiniBatchTraining](model/MiniBatchTraining/implementation.md) (N16)                         | pending |

## Upstream (numerical-toolbox-cpp)

These backlog items are shared numerical primitives. They belong in
[numerical-toolbox-cpp](https://github.com/embedded-pro/numerical-toolbox-cpp), are consumed here via
FetchContent (`numerical/…` includes), and have **no spec folder in this repo**:

| #   | Component                                       | Upstream module          |
|-----|-------------------------------------------------|--------------------------|
| N14 | Step optimisers (SGD + momentum/Nesterov, Adam) | `numerical/optimization` |
| N15 | Log-mel / MFCC feature front end                | `numerical/analysis`     |

The finite-difference gradient-check test helper also lives upstream, in the
`numerical.math_test_helper` INTERFACE library.
