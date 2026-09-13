# NexMind Contribution Guide

## Development branch

Use `main` as the primary development branch for this project.

## Build requirements

- C++17 or newer where explicitly documented
- CMake
- A supported C++ compiler: MSVC, GCC, or Clang
- CTest for automated tests

## Coding rules

- Use `.h` and `.cpp` files.
- Keep the core implementation independent of GUI frameworks.
- Keep platform-specific code behind explicit abstraction boundaries.
- Use UTF-8 internally.
- Prefer the C++ standard library unless a dependency provides a clear technical benefit.
- Keep ownership and lifetime explicit; avoid unnecessary global state.
- Add tests for behavior that can regress.

## Definition of done

A change is considered complete only when the relevant code builds successfully and the applicable tests pass. Claims about benchmarks, training, or generation must be based on an actual reproducible run.

## Dependencies and licenses

Before adding a third-party dependency, document why it is needed and record its license in the project documentation when appropriate.

## Security

Never commit credentials, API keys, access tokens, private keys, or other secrets. See `SECURITY.md` for reporting security issues.
