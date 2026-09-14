# NexMind

NexMind is an open-source project for building a trainable Mini Transformer and evolving LLM infrastructure from scratch in modern C++.

## What works now

- C++20 CPU tensor and autograd engine
- Linear, ReLU, MLP, LayerNorm and Adam optimizer
- Embedding and sinusoidal positional encoding
- Multi-head self-attention with causal masking
- Pre-Norm Transformer blocks and stacked Transformer encoder
- Autoregressive Transformer language model
- Next-token language-model training
- UTF-8 byte tokenizer with a fixed 256-token vocabulary
- Sequential text dataset for fixed-length next-token samples
- Binary model checkpoint save/load
- Greedy and temperature text generation
- `train` and `generate` command-line entry points
- Automated CTest coverage for the core training and inference path

## Quick start

Build with any C++20 compiler and CMake 3.20+.

Train a small model directly from a UTF-8 text file:

```text
nexmind train data.txt model.nxm 1000 64 64 2 4 128 0.001
```

Arguments after the checkpoint are optional:

```text
steps sequence_length embed_dim layers heads feed_forward_dim learning_rate
```

Generate text from a saved checkpoint:

```text
nexmind generate model.nxm "Once upon a time" 128 0.8
```

Use temperature `0` or omit it for deterministic greedy generation.

## Architecture

```text
UTF-8 text
   ↓
ByteTokenizer
   ↓
Token IDs
   ↓
Embedding + Sinusoidal Position Encoding
   ↓
Causal Transformer
   ├─ LayerNorm
   ├─ Multi-Head Causal Self-Attention
   ├─ Residual
   ├─ LayerNorm
   ├─ MLP
   └─ Residual
   ↓
LM Head
   ↓
Cross Entropy
   ↓
Autograd → Adam
   ↓
Checkpoint
   ↓
Text Generation
```

## Engineering principles

1. Correctness before performance.
2. A feature is not complete until it builds, runs, and has tests.
3. Prefer explicit C++ implementations over opaque framework magic during the reference stage.
4. Keep dependencies small and document every external dependency and license.
5. Keep model, data, training, checkpointing, and inference responsibilities separated.
6. Do not commit credentials, API keys, private datasets, model secrets, or other sensitive material.

## Current limitations

This is a CPU reference implementation intended for learning, validation, and small experiments. It does not yet provide GPU acceleration, packed tensor kernels, mixed precision, distributed training, batched data loading, AdamW, or a subword tokenizer.

The byte tokenizer is deliberately simple: UTF-8 is preserved as raw bytes, so the vocabulary is fixed at 256 entries. This makes the end-to-end training path dependency-free and easy to inspect.

## Development

The default development branch is `main`.

Run the test suite with CTest after configuring the build:

```text
ctest --test-dir build -C Release --output-on-failure
```

## License

NexMind is released under the Apache License 2.0. See [LICENSE](LICENSE).
