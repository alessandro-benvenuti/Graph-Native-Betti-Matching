# Full-data node-focal pipeline

This experiment trains from scratch on the original limited-target mixed regime
for 50 epochs: 25,900 Plants and 4,000 synthetic-MRI training samples with
balanced source/target sampling. It then initializes a 100-epoch MRI-only
specialization on the complete new MRI split from the pretraining checkpoint
selected by validation edge mAP.

Both stages use unweighted node focal loss from epoch zero and the baseline
matched-node edge cross-entropy objective. Dataset caps are disabled only for
MRI specialization; pretraining deliberately retains the historical caps of
480 Plants and 200 MRI validation samples. MRI cap membership is selected by a
stable SHA-256 ranking seeded with `364505`, shared by every loss recipe. Each
run writes the resulting IDs to `dataset-manifest.json`.

The launcher defaults to one H100 for pretraining and four H100s for MRI
specialization. It automatically selects `32 / stage GPUs` per process, keeping
the global batch fixed at 32 in both stages.

`SYNTHETIC_MRI_DATASET` must point directly to the generated `new_patches`
directory. Its sibling `new_split.csv` records the patient-level assignment;
the loader consumes the already materialized `new_patches/{train,val,test}`
directories and never reconstructs a split from that CSV.
