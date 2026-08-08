# Neural Network Algorithms Library

## Overview

This library provides neural network algorithms — activations, layers, losses, and model —
designed for robust and efficient real-time use on resource-constrained embedded systems (no heap,
deterministic, float-only). It builds on the shared numerical primitives (linear algebra, solvers,
optimization, regularization) provided by
[numerical-toolbox-cpp](https://github.com/embedded-pro/numerical-toolbox-cpp), which it consumes
via CMake `FetchContent`.

## Getting Started

The library is a set of CMake targets. Consume it from your own project with `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(
    neural_network_toolbox
    GIT_REPOSITORY https://github.com/embedded-pro/neural-network-toobox-cpp.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(neural_network_toolbox)

target_link_libraries(my_app PRIVATE neural_network.activation neural_network.layer neural_network.losses neural_network.model)
```

The numerical-toolbox dependency is fetched automatically. Includes are namespaced by domain, e.g.
`#include "neural_network/activation/ReLU.hpp"` and — for shared primitives —
`#include "numerical/math/Matrix.hpp"`.

## Documentation

| Category                                                     | Description                                       |
|--------------------------------------------------------------|---------------------------------------------------|
| [Neural Network](doc/neural_network/README.md)               | Activations, Layers, Losses, Model                |

Each category page lists its algorithms with a brief description and links to the detailed
documentation.

### Booklet

The entire documentation set is also published as a single book — read it online as a
[GitHub Pages site](https://embedded-pro.github.io/neural-network-toobox-cpp/) or download the latest
PDF from the [Releases page](../../releases/latest). Both are generated automatically from `doc/`
(cover, Summary/table of contents, one chapter per category, consolidated references, back cover).

Build it locally with [Pandoc](https://pandoc.org) + XeLaTeX installed:

```bash
python scripts/build-booklet.py --format all   # writes build/booklet/{NeuralNetworkToolbox.pdf,index.html}
```

## Simulator

The `simulator/` directory contains an interactive Qt-based GUI application (Neural Network) for
visualizing and experimenting with the library's algorithms. It is a desktop tool for development
and exploration, separate from the core embedded-targeted library.
