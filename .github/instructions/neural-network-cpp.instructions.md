---
description: "Neural Network C++ rules (float-only): no heap, bounded containers, generic template<typename T> instantiated for float, embedded pragmas, Allman/brace-init, SOLID, const-correct. Canonical: AGENTS.md."
applyTo: "**/*.{hpp,cpp,h}"
---

# Neural Network Toolbox C++ Rules

This project is a neural network algorithms library (activations, layers, losses, model) targeting
resource-constrained embedded systems. It consumes the shared numerical primitives (`math`,
`optimization`, `regularization`, …) from numerical-toolbox-cpp via FetchContent. Follow these
rules strictly.

## Memory — No Heap Allocation

Never use `new`, `delete`, `malloc`, `free`, `std::make_unique`, or `std::make_shared`.

Replace standard containers:
- `std::vector<T>` → `infra::BoundedVector<T>::WithMaxSize<N>`
- `std::string` → `infra::BoundedString::WithStorage<N>`
- `std::deque<T>` → `infra::BoundedDeque<T>::WithMaxSize<N>`
- `std::list<T>` → `infra::BoundedList<T>::WithMaxSize<N>`
- Use `std::array<T, N>` for fixed-size arrays
- Use `std::optional<T>` for optional values
- No recursion — stack usage must be predictable

## Numeric Types — Float-Only

Write generic `template<typename T[, std::size_t N]>` with
`static_assert(std::is_floating_point_v<T>, "...")`, and **instantiate/test `float` only**.
Do not implement `Q15`/`Q31` — the generic `T` keeps that a cheap future add.
Use `std::numbers::pi_v<float>` — never hardcode `3.14159265f`.

## Embedded Optimizations

Every algorithm header MUST include:

```cpp
#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif
```

Apply `OPTIMIZE_FOR_SPEED` (from `numerical/math/CompilerOptimizations.hpp`) on hot-path methods:
`Forward()`, `Backward()`, `Compute()`, `Update()`.

## Naming

- Classes/Methods: `PascalCase` (e.g., `Dense`, `Forward()`)
- Member variables: `camelCase` (e.g., `weights`, `biases`)
- Namespaces: lowercase (`neural_network`)
- Template parameters: descriptive (`typename T`, `std::size_t InputSize`)

## Style

- Allman braces (opening brace on new line), 4-space indent
- Functions ~30 lines max (hard limit ~50)
- Self-documenting code — avoid unnecessary comments
- `const` on all non-mutating methods, `constexpr` where possible
- Fixed-size types: `uint8_t`, `int32_t`, etc.
- **Brace initialization**: Use `{}` for all variable and object initialization.

## Design

- SOLID principles — constructor injection, depend on abstractions
- DRY — extract shared logic into helpers or templates
- RAII for resource management
- **No pure virtual destructors**: Use `virtual ~Interface() = default`.
- No virtual calls in ISR-callable or real-time critical paths

## Documentation — MANDATORY

For every algorithm added or modified, update the corresponding `doc/{domain}/{AlgorithmName}.md`
file. Follow `doc/TEMPLATE.md` exactly. Documentation is **design-first**: cover mathematical
background, algorithm behaviour, complexity, pitfalls, and connections. Do **not** include
implementation details, class names, template parameters, or usage code examples.

Canonical rules: [AGENTS.md](../../AGENTS.md).
