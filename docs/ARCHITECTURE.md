# NexMind Architecture

## Layers

```text
Data / Tokenizer
       |
       v
Tensor / Math primitives
       |
       v
Transformer model
       |
       +----> Language-model loss
       |
       v
Autograd / Backward
       |
       v
Optimizer / Training
       |
       +----> Checkpoint
       |
       v
Inference / Generation
```

## Reference implementation

The CPU implementation is the correctness reference. It should favor simple, inspectable data structures and explicit operations over premature optimization.

## Model boundary

The Transformer layer should not depend on the command-line application, test framework, or GUI. Training and inference orchestration should consume model interfaces rather than embed model internals.

## Acceleration boundary

SIMD and CUDA implementations are acceleration backends. They must not redefine model semantics. Platform-specific code belongs behind explicit interfaces or implementation boundaries.

## Data boundary

Tokenization and dataset loading are separate from model computation. This allows the same model to be trained from different datasets and makes data preprocessing independently testable.

## Checkpoint boundary

Checkpoint files must contain enough information to reproduce model parameters and the metadata required by the corresponding model configuration. Serialization formats will be documented when introduced.

## Testing strategy

Tests should cover small deterministic examples first: tensor operations, tokenizer behavior, attention shapes and masks, forward/backward consistency, optimizer updates, checkpoint round trips, and generation behavior.
