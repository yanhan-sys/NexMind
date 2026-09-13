# NexMind Roadmap

## Phase 0 — Project foundation

- Repository documentation
- Apache-2.0 license
- CMake build foundation
- Cross-platform development rules
- Security and contribution rules

## Phase 1 — CPU Mini Transformer

- Tensor abstraction
- Character tokenizer
- Embedding
- Causal self-attention
- Feed-forward network
- Transformer block
- Language-model head
- Cross-entropy loss
- Backpropagation
- Adam/AdamW
- Training loop
- Checkpoint save/load
- Text generation
- CTest coverage

## Phase 2 — Engineering hardening

- Deterministic tests
- Numerical gradient checks
- Profiling
- Memory accounting
- Reproducible training configuration
- Dataset tooling
- Benchmark suite

## Phase 3 — Acceleration

- SIMD kernels
- CUDA backend
- FP16/BF16
- Memory-efficient attention
- Faster inference

## Phase 4 — Scale

- Larger model configurations
- Dataset pipelines
- Distributed training
- Checkpoint sharding
- Evaluation harness
- SFT and other training workflows

## Phase 5 — Production direction

- High-throughput inference
- Quantization
- Serving interfaces
- Model/version management
- Hardware-aware optimization

The roadmap is intentionally incremental. No later phase should hide correctness problems in an earlier phase.
