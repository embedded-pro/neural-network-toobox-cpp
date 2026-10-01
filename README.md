# Neural Network Algorithms Library

## Overview

This library provides neural network algorithms — activations, layers, losses, and model —
designed for robust and efficient real-time use on resource-constrained embedded systems (no heap,
deterministic, float-only). It builds on the shared numerical primitives (linear algebra, solvers,
optimization, regularization) provided by
[numerical-toolbox-cpp](https://github.com/embedded-pro/numerical-toolbox-cpp), which it consumes
via CMake `FetchContent`.

## Getting Started

The library is a set of CMake targets. Consume it from your own project with `FetchContent`.

Like its siblings, neural-network-toolbox only fetches its own dependencies when it is built
standalone. As a subproject it expects the consuming project to provide
[embedded-infra-lib](https://github.com/embedded-pro/embedded-infra-lib) (`emil`) and
[numerical-toolbox-cpp](https://github.com/embedded-pro/numerical-toolbox-cpp) first, so declare and
make them available before neural-network-toolbox:

```cmake
include(FetchContent)

FetchContent_Declare(
    emil
    GIT_REPOSITORY https://github.com/embedded-pro/embedded-infra-lib.git
    GIT_TAG        2848dbf163a6ebf159e29f7f4f73b96f7773d20d
)

set(EMIL_INCLUDE_MBEDTLS Off CACHE BOOL "" FORCE)
set(EMIL_INCLUDE_ECHO Off CACHE BOOL "" FORCE)
set(EMIL_FETCH_ECHO_COMPILERS Off CACHE BOOL "" FORCE)
set(EMIL_BUILD_ECHO_COMPILERS Off CACHE BOOL "" FORCE)
set(EMIL_ENABLE_DOCKER_TOOLS Off CACHE BOOL "" FORCE)

FetchContent_Declare(
    numerical_toolbox
    GIT_REPOSITORY https://github.com/embedded-pro/numerical-toolbox-cpp.git
    GIT_TAG        1225b3c422beab93e0ff8fd7231f8d7e087f6da4
)

FetchContent_Declare(
    neural_network_toolbox
    GIT_REPOSITORY https://github.com/embedded-pro/neural-network-toobox-cpp.git
    GIT_TAG        main
)

FetchContent_MakeAvailable(emil numerical_toolbox neural_network_toolbox)

target_link_libraries(my_app PRIVATE neural_network.activation neural_network.layer neural_network.losses neural_network.model)
```

Use the same `emil` and `numerical_toolbox` revisions that this repository pins in its root
`CMakeLists.txt`. Includes are namespaced by domain, e.g.
`#include "neural_network/activation/ReLU.hpp"` and — for shared primitives —
`#include "numerical/math/Matrix.hpp"`.

The CPack package produced by the `host-Debug-WithPackage` build preset also installs the headers
and an exported CMake package, usable with `find_package(NeuralNetworkToolbox)` and the
`neural-network-toolbox::` target namespace.

## Documentation

| Category                                                     | Description                                       |
|--------------------------------------------------------------|---------------------------------------------------|
| [Neural Network](doc/README.md)                              | Activations, Layers, Losses, Model                |

Each category page lists its algorithms with a brief description and links to the detailed
documentation.

### Booklet

The entire documentation set is also published as a single book — read it online as a
[GitHub Pages site](https://embedded-pro.github.io/neural-network-toobox-cpp/) or download the latest
PDF from the [Releases page](https://github.com/embedded-pro/neural-network-toobox-cpp/releases/latest). Both are generated automatically from `doc/`
(cover, Summary/table of contents, one chapter per category, consolidated references, back cover).

Build it locally with [Pandoc](https://pandoc.org) + XeLaTeX installed:

```bash
python scripts/build-booklet.py --format all   # writes build/booklet/{NeuralNetworkToolbox.pdf,index.html}
```

## Simulator

The `simulator/` directory contains an interactive Qt-based GUI application (Neural Network) that
trains a small multilayer perceptron on an XOR or sine-approximation demo and plots the loss
history and predictions. It is a standalone development demo with its own independent MLP
implementation; it does not exercise the library's layers, activations or losses, and it is
separate from the core embedded-targeted library.

### Building the Simulator

The simulator requires Qt6 and is disabled by default. Enable it with:

```bash
cmake --preset host  # host preset enables it automatically
# or manually:
cmake -DNEURAL_NETWORK_TOOLBOX_BUILD_SIMULATOR=ON ...
```

The executable target is `neural_network.simulator.model.neural_network`. Prerequisites:
`qt6-base-dev` and `libgl1-mesa-dev` (Ubuntu/Debian). Set `QT_QPA_PLATFORM=offscreen` to run the
simulator tests on a headless machine.

## Roadmap

Planned components are tracked in [ROADMAP.md](ROADMAP.md) — a prioritized backlog of neural
network layers, activations, losses and training components ordered by implementation difficulty.
Design-level pseudocode specifications live under [roadmap/](roadmap/README.md).

## Testing

Every algorithm is validated against the **mathematical invariants of its family** — not golden
output. Tests are `TEST_F` on `float`, no heap, one behaviour per test, asserted against independent
reference values. The per-family metric strategy is documented in **[TESTING.md](TESTING.md)**;
framework rules live in
[.github/instructions/testing.instructions.md](.github/instructions/testing.instructions.md).

Build & test locally (the `host` preset also builds the simulator and therefore needs Qt6):

```bash
cmake --preset host && cmake --build --preset host-Debug
ctest --preset host
```

Without Qt6, use the single-configuration preset:

```bash
cmake --preset host-single-Debug && cmake --build --preset host-single-Debug
ctest --preset host-single-Debug
```

## Contributing

Contributions, issues, and feature requests are welcome. Please check the contributing guidelines
before submitting pull requests.

## License

See [LICENSE](LICENSE).
