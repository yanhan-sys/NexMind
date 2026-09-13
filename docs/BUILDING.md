# Building NexMind

## Supported baseline

NexMind targets modern C++ with C++17 as the initial language baseline.

Required tools:

- CMake 3.20 or newer
- C++17-capable compiler
- Git
- CTest (provided by CMake)

Supported compiler families:

- Microsoft Visual C++ (MSVC)
- GCC
- Clang

The first implementation is CPU-only. CUDA is a later acceleration layer and is not required for the reference implementation.

## Windows

Recommended toolchain:

- Visual Studio with Desktop development with C++
- Windows 10/11 SDK
- CMake

Configure:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026"
```

Build:

```powershell
cmake --build build --config Release
```

Run tests:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

If the installed Visual Studio generator differs, use the generator reported by `cmake --help`.

## Linux

Typical configuration:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Configuration policy

Build configuration must not depend on credentials or machine-specific secrets. Local paths, API keys, cloud credentials, model access tokens, and private datasets must stay outside the repository.

## Development progression

1. CPU reference implementation.
2. Correctness and deterministic tests.
3. Profiling and optimization.
4. SIMD acceleration.
5. CUDA acceleration.
6. FP16/BF16 and memory optimization.
7. Efficient attention and larger models.
8. Distributed training and production inference.

The optimized implementations must remain consistent with the tested reference behavior.
