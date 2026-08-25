# SyntheticMRI dataset split and patch-selection report

**Status:** discussion document for the next supervision meeting  
**Audit date:** 25 August 2026  
**Dataset:** `/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI`

## Executive summary

The current SyntheticMRI dataset contains 136 volumes. Its existing split is
patient-level, with 55 training, 14 validation, and 67 test volumes
(approximately 40.4%/10.3%/49.3%). Consequently, patches from the same volume
are not mixed across splits, and the current split does not have the suspected
patch-level leakage.

The inherited patch-selection code uses ground-truth information. This is an
oracle procedure in principle, especially if it is used to decide which test
patches are evaluated. However, the complete audit shows that it has virtually
no effect on this particular synthetic dataset: it rejects only 2 of 130,560
full-grid candidates (0.0015%), both because the cropped graph has two nodes.
There are no segmentation-empty or graph-empty candidates. An additional
MRI-only empty-patch filter is therefore not justified by the observed data.

The material loss of data instead comes from the incomplete legacy grid and
the historical quota of 4,000/1,000/5,000 saved train/validation/test patches.
The audited boundary-complete grid contains 130,560 candidates, about 13.1
times the 10,000 patches in the current saved dataset. Before regeneration,
its final-boundary placement should be improved by distributing starts evenly
between zero and the last valid start rather than appending one unusually close
final patch. This retains 960 candidates per volume while avoiding extreme
boundary overlap. The proposed new dataset should retain every candidate in
its catalogue. Training cost should be controlled through an explicit sampler
or coverage schedule, not by permanently deleting valid data.

## Questions for Maria

1. Is strict comparability with the historical 40/10/50 patient split the
   primary objective, or may the main experiment use a larger training split
   such as 70/15/15?
2. Should the project report both:
   - a reproducibility result on the unchanged legacy dataset and split; and
   - a primary result on a new, versioned dataset and split?
3. For a new split, should volumes be balanced using foreground fraction,
   graph nodes, graph edges, bifurcations, and graph Betti numbers rather than
   assigned with an unconstrained random shuffle?
4. Should validation and test inference cover the complete predetermined grid,
   with metrics aggregated per volume as well as per patch?
5. What training compute budget should define an epoch when the complete
   training catalogue is much larger than the historical 4,000 patches?

## Provenance and audit scope

The read-only audit used:

- `raw/<patient_id>.nii.gz` for MRI volumes;
- `seg/<patient_id>.nii.gz` for ground-truth vessel segmentations;
- `graphs/<patient_id>/` for ground-truth graphs; and
- `splits.csv` for the existing patient assignment.

It did not rename or modify source data, existing patches, or the split file.
It enumerated both the proposed full grid and the inherited randomized grid.
Ground truth was recorded only after a candidate coordinate had been selected;
it did not control inclusion in the audit.

The legacy audit reproduces the inherited candidate traversal with seed 42,
but it deliberately does not reproduce quota-based early stopping because the
exact historical command-line invocation has not been verified. Accordingly,
the legacy counts describe available candidates, not necessarily the exact
sequence of patches previously saved.

## Source-volume inventory

| Property | Result |
|---|---:|
| Volumes | 136 |
| Scan shape | 325 x 304 x 600 for all volumes |
| Voxel spacing | 1.0 x 1.0 x 1.0 for all volumes |
| Raw/segmentation affine mismatches | 0 |
| Segmentation foreground fraction | 1.6668% to 2.3300% |
| Full-graph nodes per volume | 3,858 to 4,271 |
| Full-graph edges per volume | 3,924 to 4,363 |

The volumes are geometrically uniform, but their foreground and graph
statistics still vary enough to consider distribution-aware stratification.

## Existing patient split

| Split | Volumes | Percentage |
|---|---:|---:|
| Train | 55 | 40.4% |
| Validation | 14 | 10.3% |
| Test | 67 | 49.3% |
| **Total** | **136** | **100%** |

The split is made by volume/patient, not by patch. This is correct and avoids
leakage between spatially correlated patches from the same scan. The unusual
feature is the allocation of almost half of all volumes to the test set, which
limits the diversity available for learning.

An approximate 70/15/15 allocation would contain 95/20/21 volumes (the two
smaller assignments can be exchanged). Because all volumes have the same
shape, this would yield 91,200 training candidates, 19,200 candidates in the
20-volume split, and 20,160 in the 21-volume split.

## Patch geometry

Both audits retain the inherited geometry:

- stored patch size: 64 x 64 x 64;
- padding: 5 voxels on every side;
- effective source crop: 54 x 54 x 54;
- overlap parameter: 0.25; and
- resulting stride: 40 voxels.

The audited full grid is deterministic and includes the final boundary crop on
every axis. Every volume therefore produces exactly 960 candidates. The legacy
grid uses randomized offsets and cyclic starting positions, so its candidate
count varies by volume. A full-grid coordinate is not necessarily an exact
superset of the legacy coordinates because the coordinate origins differ.

### Boundary-placement issue in the audited full grid

The audit constructs regular stride-40 starts from zero and appends the last
valid start when it is not already present. For the axis of length 304, the
effective crop has length 54 and the last valid start is `304 - 54 = 250`:

```text
audited starts: 0, 40, 80, 120, 160, 200, 240, 250
```

The final step is only 10 voxels, so the crops starting at 240 and 250 overlap
by 44/54 voxels (81.5%). This does provide full coverage, but it is inconsistent
with the approximately 25.9% overlap between ordinary stride-40 neighbours.
Changing the nominal stride to 50 does not generally fix this problem: 250 is
divisible by 50, but the other final valid starts, 271 and 546, are not. No
practical single integer stride aligns all three axes exactly.

### Preferred endpoint-distributed grid

The preferred policy treats 40 as a maximum target step and distributes the
starts approximately evenly from zero through the last valid start. For each
axis:

```text
maximum_start = axis_size - crop_size
number_of_intervals = ceil(maximum_start / target_stride)
starts = round(linspace(0, maximum_start, number_of_intervals + 1))
```

For the 304-voxel axis this gives approximately:

```text
preferred starts: 0, 36, 71, 107, 143, 179, 214, 250
```

All gaps are now 35 or 36 voxels rather than ending with a 10-voxel gap. Across
the three dataset axes, the preferred grid has:

| Axis size | Last valid start | Intervals | Starts | Actual step sizes | Approximate overlap |
|---:|---:|---:|---:|---:|---:|
| 325 | 271 | 7 | 8 | 38-39 | 27.8%-29.6% |
| 304 | 250 | 7 | 8 | 35-36 | 33.3%-35.2% |
| 600 | 546 | 14 | 15 | 39 | 27.8% |

It therefore retains `8 x 8 x 15 = 960` candidates per volume and full boundary
coverage, but eliminates the isolated 81.5% boundary overlap. The effective
overlap is slightly higher than the nominal 25% because the scan dimensions do
not divide exactly into 40-voxel steps.

The completed audit describes the append-boundary coordinates, not these newly
proposed coordinates. The total candidate count remains 130,560 because the
number of starts is unchanged, but the candidate contents differ slightly.
After implementing the endpoint-distributed grid, the lightweight candidate
summary should be rerun before patch materialization to verify the GT statistics.

## Full-grid audit

| Split | Candidates | Segmentation empty | Graph empty | Fewer than 3 nodes | Accepted by old rules |
|---|---:|---:|---:|---:|---:|
| Train | 52,800 | 0 | 0 | 0 | 52,800 |
| Validation | 13,440 | 0 | 0 | 1 | 13,439 |
| Test | 64,320 | 0 | 0 | 1 | 64,319 |
| **Total** | **130,560** | **0** | **0** | **2** | **130,558** |

The inherited GT rules accept 99.9985% of the complete grid. No candidate is
rejected for segmentation presence, foreground ratio, GT-derived SNR,
graph-coordinate validity, or an empty graph.

### Two sparse graph crops

| Patient | Split | Start (d, h, w) | Foreground voxels | Nodes | Edges | Old decision |
|---|---|---|---:|---:|---:|---|
| 66 | Validation | (160, 0, 360) | 906 | 2 | 1 | Reject: fewer than 3 nodes |
| 130 | Test | (80, 0, 40) | 1,576 | 2 | 1 | Reject: fewer than 3 nodes |

Neither patch is empty: each contains a one-edge, two-node graph and substantial
segmentation foreground. Both lie on the `h = 0` scan boundary. Unless visual
quality assurance identifies an annotation or graph-cropping defect, they
should be retained. Sparse valid targets are part of the inference distribution
and should not be removed merely because they are difficult.

## Legacy-grid audit

| Split | Candidates | Segmentation empty | Graph empty | Fewer than 3 nodes | Accepted by old rules |
|---|---:|---:|---:|---:|---:|
| Train | 32,891 | 0 | 0 | 0 | 32,891 |
| Validation | 7,705 | 0 | 0 | 0 | 7,705 |
| Test | 38,951 | 0 | 0 | 0 | 38,951 |
| **Total** | **79,547** | **0** | **0** | **0** | **79,547** |

The full grid contains 51,013 more candidates than the legacy traversal, an
increase of 64.1%. The increase differs by split because the randomized legacy
grid produces a variable number of candidates per scan.

## Effect of the historical patch quota

The currently reported saved dataset contains 4,000 training, 1,000 validation,
and 5,000 test patches. Comparing those totals with the audit gives:

| Split | Existing saved patches | Legacy candidates | Full-grid candidates | Full/existing factor |
|---|---:|---:|---:|---:|
| Train | 4,000 | 32,891 | 52,800 | 13.2x |
| Validation | 1,000 | 7,705 | 13,440 | 13.4x |
| Test | 5,000 | 38,951 | 64,320 | 12.9x |
| **Total** | **10,000** | **79,547** | **130,560** | **13.1x** |

Assuming these stated saved totals correspond to the audited sources, the
historical dataset retains approximately 12.6% of the acceptable legacy-grid
candidates. Thus the quota, rather than GT rejection, is the dominant selection
operation. The exact historical selection sequence remains unverified until
the original generator invocation is recovered.

## Updated interpretation of the oracle-filter concern

The original concern was methodologically correct: choosing evaluated test
patches using segmentation or graph ground truth would not be available during
deployment and could hide failure on empty inputs. The proposed workaround was
an MRI-only filter followed by evaluation of every patch, including patches
classified as empty.

The audit changes the practical conclusion. There are no empty candidates, and
the inherited GT logic rejects only two non-empty sparse graphs. An MRI-only
filter would add complexity and a new source of error without addressing an
observed dataset property. The cleaner pipeline is therefore:

```text
MRI volume
    -> fixed, deterministic, boundary-complete grid
    -> retain every candidate (no GT or MRI selection filter)
    -> run the graph model on every candidate
    -> evaluate every candidate against ground truth
    -> aggregate results per patch and per volume
```

Ground truth may still be used to describe or stratify the training data after
the split has been fixed, but it should not determine which validation or test
examples enter the reported evaluation.

## Proposed dataset and training policy

1. Preserve the current dataset and `splits.csv` as a versioned legacy
   benchmark. Do not silently overwrite them.
2. Create a new patient-level split with a fixed seed and a saved manifest.
3. Consider approximately 70/15/15, subject to supervisor approval.
4. Balance the split distributions using foreground fraction, node count, edge
   count, bifurcation count, and graph Betti numbers. These are acceptable for
   constructing a fixed research split; they must not be used as an inference
   gate.
5. Catalogue every candidate from the deterministic endpoint-distributed grid,
   using 40 voxels as the maximum target step. Do not permanently discard valid
   candidates.
6. Run validation and test exhaustively on their complete grids.
7. Report per-volume aggregates in addition to pooled per-patch metrics so that
   correlated patches and scan size do not distort the result.
8. Retain evaluation on the unchanged legacy test set as a secondary
   comparability experiment. Do not use that test result for model selection.

### Controlling training cost without discarding data

A 70/15/15 split would expose approximately 91,200 training candidates, 22.8
times the historical 4,000-patch training set. Retaining all candidates does
not require processing all of them in every epoch. Suitable options include:

- a patient-balanced sampler, preventing individual volumes from dominating;
- shuffled sampling without replacement across a multi-epoch coverage cycle;
- a fixed optimizer-step definition of an epoch for controlled comparisons;
- periodic full validation rather than full validation after every short
  training epoch; and
- an ablation comparing the historical sample budget with increasing coverage
  budgets from the complete catalogue.

This preserves access to all data while keeping experiments computationally
tractable and making the effective training budget explicit. Before physically
materializing 130,560 patches, storage requirements and the alternative of
on-demand cropping from a coordinate manifest should be measured.

## Decisions not yet made

- Final patient split and random seed.
- Exact stratification algorithm and relative importance of its statistics.
- Exact implementation and verification of the endpoint-distributed grid.
- Whether patches are materialized or cropped on demand.
- Training samples/optimizer steps per epoch and coverage schedule.
- Frequency of exhaustive validation.
- Per-volume prediction merging and metric aggregation protocol.
- Whether the two sparse boundary graphs need visual or graph-construction QA.

No existing dataset paths should be renamed while current experiments depend
on them. Any replacement should be written under explicit versioned names and
activated only after the legacy run is reproducibly preserved.
