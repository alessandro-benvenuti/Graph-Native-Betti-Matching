# 3D RelationFormer model contract

## Supported architecture

The active model is the 3D RelationFormer path used by the established Plants
and syntheticMRI experiments:

```text
[B,1,D,H,W]
    -> 3D SE-ResNet encoder
    -> 1x1x1 input projection
    -> 3D sine positional encoding
    -> single-level deformable transformer encoder/decoder
    -> object and relation tokens
    -> node-class and coordinate heads
```

The implementation lives in:

- `models/seresnet.py`;
- `models/position_encoding.py`;
- `models/deformable_transformer.py`;
- `models/relationformer.py`;
- `models/ops/` for the 3D deformable-attention operator.

`models.build_model(config)` is the supported construction entry point.

## Forward interface

The input is a float tensor `[B,C,D,H,W]`. The model returns:

```python
tokens, predictions, projected_features = model(samples)
```

- `tokens`: `[B,Q,H]`, containing all object, relation, and dummy tokens;
- `predictions["pred_logits"]`: `[B,Q_object,num_classes]`;
- `predictions["pred_nodes"]`: `[B,Q_object,6]`, passed through sigmoid;
- `projected_features`: `[B,H,D',H',W']`, retained for diagnostics and future
  trainer parity.

The first three coordinate channels are consumed as normalized node positions
by the legacy losses. The remaining three channels are retained for checkpoint
and bounding-box-loss compatibility.

The relation classifier remains a model submodule named `relation_embed`. It is
not called by `RelationFormer.forward`; the relation loss will consume it when
that loss is migrated. Keeping it here is necessary for checkpoint compatibility.

## Baseline architecture corrections

Two legacy configuration fields did not describe the model that was actually
constructed:

1. `ENC_LAYERS` was configured as 4 but was never passed by the builder. The
   transformer constructor defaulted to 6 encoder layers. The new configuration
   explicitly says `encoder_layers: 6`, and the builder now consumes it.
2. The first value of encoder `STRIDES` was ignored because layer 1 always used
   its function default of 1. The new configuration explicitly records the
   effective stage strides `[1,2,2,2]`.

These are schema corrections, not architecture changes. They preserve the
effective model and expected checkpoint shapes while making future YAML edits
real.

Legacy `DROP_RATE`, `DROP_PATH_RATE`, `LABEL_SMOOTHING`, encoder `CELL_SIZE`,
and decoder `TWO_STAGE` fields were removed because this 3D path never read
them. Transformer dropout remains configurable through `model.decoder.dropout`.

The legacy relation-attention branch is also preserved exactly: when enabled,
deformable cross-attention is applied to every query except the final token.
With two configured relation tokens, this means only the final relation token
skips cross-attention. This behavior is unusual, but changing it before a
checkpoint-backed equivalence test would alter the baseline.

## Intentional exclusions

The port does not include:

- image or instance domain discriminators;
- gradient reversal;
- 2D/pre-2D backbones and padding paths;
- general-transformer adapters;
- MoCo;
- diameter smoothing;
- model-embedded debug printing and random visualization;
- compiled objects, shared libraries, or previous build directories.

Legacy discriminator keys are the only removed model keys accepted by
`models.checkpoint.checkpoint_compatibility`. Missing active keys, unexpected
non-domain keys, and shape mismatches remain hard failures.

## Deformable-attention implementations

The model has two execution paths with the same trainable state:

- `model.decoder.use_cuda_extension: true` uses the compiled CUDA extension and
  is the training baseline;
- `false` uses PyTorch `grid_sample`, intended for CPU tests and debugging.

The fallback is not intended for full training. It exists so tensor contracts,
gradients, and small model forwards can be tested without compiling CUDA.

Build the extension on the cluster from the repository root:

```bash
bash scripts/build_deformable_attention.sh
```

The build requires the CUDA toolkit used by the active PyTorch installation.
Build products are intentionally ignored by Git.

## Tests

CPU-safe component and configuration checks:

```bash
python -m unittest -v tests.test_model_components
```

All repository tests:

```bash
python -m unittest discover -v
```

After building the CUDA extension on the cluster:

```bash
python -m unittest -v tests.test_model_cluster.CudaModelTests
```

This compares the CUDA operator with the PyTorch fallback and runs a small
end-to-end RelationFormer forward on CUDA.

## MRI checkpoint verification

Set the trusted finetuning checkpoint path and run:

```bash
export GNBM_MRI_CHECKPOINT=/absolute/path/to/checkpoint.pt
python -m unittest -v \
  tests.test_model_cluster.RealCheckpointTests.test_mri_checkpoint_matches_all_active_model_tensors
```

The test accepts legacy `module.` wrappers and reports old discriminator tensors
as explicitly ignored. Every active tensor must otherwise exist with the exact
expected shape. Passing this test is required before declaring the model port
checkpoint-compatible.

After schema compatibility passes, the next checkpoint test should load the
weights into both repositories and compare evaluation outputs for the same
stored input volume. That numerical comparison requires the original and new
CUDA extensions to be built in compatible environments.
