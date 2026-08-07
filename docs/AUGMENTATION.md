# Data augmentation contract

Image, segmentation, and graph geometry must never be transformed independently.

## Coordinate convention

All graph nodes use spatial order `[D, H, W]` and are normalized as:

```text
normalized_coordinate = voxel_index / axis_size
```

The maximum valid value on an axis of size `S` is `(S - 1) / S`. Values equal to `1.0` are outside the voxel-centre grid.

Dataset loaders must declare whether stored graph points are already normalized or are voxel coordinates. They must not infer the convention by looking at the coordinate range.

## Preserved training policies

The policies reproduce every augmentation active in the trusted pipeline.

### syntheticMRI training

1. Independent 90-degree rotations in the `(D,H)`, `(D,W)`, and `(H,W)` planes.
2. Isotropic zoom sampled uniformly from `[0.6, 1.0]`.
3. Gaussian noise with probability `0.35`.
4. As in MONAI `RandGaussianNoise`, applied standard deviation is sampled uniformly from `[0, 0.015]`.
5. Image is clamped to `[-0.5, 0.5]` after the optional noise step. The clamp
   still runs on samples for which Gaussian noise was not selected, matching
   the legacy transform chain.

Image zoom uses trilinear interpolation (`grid_sample` calls this mode `bilinear`). Segmentation zoom always uses nearest-neighbour interpolation.

### Plants training

1. Project the resized 2D sample into the central five slices of a 3D volume.
2. Independent 90-degree rotations in all three spatial planes.
3. Independent flips on D, H, and W with probability `0.5` per axis.
4. Min-max intensity scaling to `[-0.5, 0.5]`.
5. Symmetric padding from the inner volume to the configured model input size, with graph coordinates remapped to the padded grid.

The old elastic-deformation branch, `real_set_augment`, Gaussian smoothing objects, and the misleading continuous-rotation option were not active and are not part of the compatibility policy.

## Evaluation policy

Validation and test policies are identities. Deterministic preprocessing such as normalization, 2D-to-3D projection, and padding still applies, but no random rotation, zoom, flip, or noise is sampled.

## Tests

Run the dependency-light exact geometry suite:

```bash
python3 -m unittest -v tests.test_augmentation_geometry
```

Run it through test discovery:

```bash
python3 -m unittest discover -v
```

To check existing datasets on Magnolia (or another machine with the optional
PNG/NIfTI/VTP readers installed):

```bash
export SYNTHETIC_MRI_DATASET=/path/to/syntheticMRI/train
export PLANTS_DATASET=/path/to/plants_3d2cut/patches_3d
python3 -m unittest -v tests.test_augmentation_dataset
python3 -m unittest -v tests.test_augmentation_plants_dataset
```

Both tests select eight evenly spaced filenames by default, covering the
beginning, middle, and end of each sorted dataset without depending on random
sampling. Change the count explicitly when needed:

```bash
export AUGMENTATION_DATASET_SAMPLES=16
```

The existing generator stores syntheticMRI graph points in normalized
coordinates, so that is the test default. For another explicitly voxel-space
dataset, set `SYNTHETIC_MRI_COORDINATES=voxel`. The test never guesses a
coordinate convention from value ranges.

The syntheticMRI integration test measures graph/foreground alignment on every
selected patch. Rotations and flips must preserve exact foreground hits. The
zoom check additionally reports exact hits, but allows a one-voxel
neighbourhood because nearest-neighbour shrinking can remove one-voxel-wide
segmentation branches even when the continuous graph transform is correct.

The Plants integration test uses the generated `*_data.png`, `*_seg.png`, and
`*_graph.vtp` triplets. It follows the stored `coordinate / patch_size`
convention and the generator's stored `(y,x)` order, grayscale handling,
`0.3` segmentation threshold, central
five-slice projection, 90-degree rotations, D/H/W flips, and symmetric padding.
The test uses PIL only as a test-time PNG reader; no production loader module
has been replaced. MONAI resize/scaling remain part of the loader-porting step,
while their geometry-independent contracts are covered by unit tests.

`PLANTS_COORDINATE_ORDER` defaults to `yx`, as declared by
`generate_plants_data.py`. It can be set to `xy` only when checking a dataset
created by a different generator with that explicit on-disk order.

Plants exact baseline hits are reported but are not required to be perfect:
an endpoint may lie next to a one-pixel-wide rasterized skeleton, and a
continuous node exactly between pixels has two equally valid nearest voxels.
Rounding such a tie does not necessarily commute with a rotation, reflection,
or renormalization after padding. The stored dataset must instead meet the
configured baseline within a one-voxel neighbourhood, and rotations, flips,
and padding must preserve that neighbourhood rate. Exact rates are still
printed as diagnostics. Separate integer-landmark unit tests require exact
voxel agreement and remain the strict guard against off-by-one errors.

Override an unaugmented baseline threshold only when investigating a known
dataset:

```bash
export AUGMENTATION_MIN_NODE_HIT_RATE=0.70
export PLANTS_MIN_NODE_HIT_RATE=0.70
```
