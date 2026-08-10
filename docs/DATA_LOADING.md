# Data loading contract

`load_config` and `build_data_loaders` are the only configuration-driven data
entry points. The removed uppercase compatibility builders are not part of the
new repository API.

```python
from configs import load_config
from data.loaders import build_data_loaders

config = load_config("configs/pretrain_mixed.yaml")
train_loader, validation_loader = build_data_loaders(config)
```

Configuration-driven loading requires separate `root/train` and `root/val`
directories, each containing `raw`, `seg`, and `vtp`. Direct leaf roots remain
available to the low-level discovery functions for inspection and integration
tests, but they cannot silently serve as both training and validation data.

## Model-facing batch

The six-part interface consumed by the established model and trainer is kept:

1. images: float tensor `[B,C,D,H,W]`;
2. segmentations: float tensor `[B,C,D,H,W]`, foreground positive;
3. nodes: list of float tensors `[N,>=3]` in normalized `(D,H,W)` order;
4. edges: list of long tensors `[E,2]`;
5. projection positions: one value per sample (`None` for native 3D);
6. domain labels: long tensor, source `0` and target `1`.

Coordinates use `voxel_index / axis_size`. Image, segmentation, and graph
nodes always receive the same sampled geometric transformation.

## SyntheticMRI

Discovery accepts complete `*_data.nii` or `*_data.nii.gz`, matching
`*_seg` and `*_graph.vtp` files, in sorted order. Missing partners fail during
discovery rather than later in a worker.

The loader applies foreground mean-centering, shifts segmentation by `-0.5`,
and resizes only when necessary. Coordinate storage must be declared as
`normalized` or `voxel`; range-based guessing was removed. Training applies
the configured quarter rotations, zoom, MONAI Gaussian noise, and final clamp.
Validation performs deterministic preprocessing only.

## Plants

Plants PNGs are decoded to grayscale and divided by 255. Images use MONAI's
area-style resize. Segmentation is thresholded at `0.3` at native resolution,
then downsampled as a categorical foreground-occupancy mask so thin branches
cannot be averaged below a second threshold. The result is projected into the
configured odd number of central slices. Stored graph points use generator
order `(y,x)` and are swapped once to match the loaded array axes before z is
appended.

Training applies the configured quarter rotations and D/H/W flips, followed by
MONAI intensity scaling and symmetric padding. Validation uses the same
deterministic projection, scaling, and padding but samples no random geometry.
Inactive elastic deformation, real-set growth, Gaussian smoothing, fake
continuous rotation, and unused transform arguments were removed.

Plants file order deliberately follows `os.listdir`, preserving the subset
selected by the established sample cap. SyntheticMRI discovery is sorted.

## Mixed-domain sampling

Source and target datasets are concatenated. When target upsampling is enabled,
every source item has weight `1`, every target item has weight
`source_size / target_size`, and an epoch contains exactly `2 * source_size`
draws. Validation is concatenated without sampling or random augmentation.

## Cluster verification

The optional end-to-end tests automatically use the real datasets when their
environment variables are present:

```bash
export SYNTHETIC_MRI_DATASET=/data/scavone/syntheticMRI/patches/syntheticMRI
export PLANTS_DATASET=/data/scavone/plants_3d2cut/patches_3d
export AUGMENTATION_DATASET_SAMPLES=8

python3 -m unittest -v tests.test_data_loader_datasets
python3 -m unittest discover -v
```

The Plants integration test compares preprocessing-only alignment with active
geometry and prints native/resized coordinate-order diagnostics on failure.
The SyntheticMRI check permits a one-voxel neighbourhood because nearest
shrinking can remove a one-voxel-wide segmentation branch even when the graph
transform is correct.
