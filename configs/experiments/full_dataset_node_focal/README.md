# Full-data node-focal pipeline

This experiment trains from scratch on the complete mixed Plants + synthetic-MRI
training split for 50 epochs, then initializes a 100-epoch MRI-only specialization
from the pretraining checkpoint selected by validation edge mAP.

Both stages use unweighted node focal loss from epoch zero and the baseline
matched-node edge cross-entropy objective. Dataset caps are disabled. Mixed
pretraining disables weighted replacement sampling so every available sample is
visited once per epoch.

The configured batch size is per process. The production launcher defaults to
four H100s and automatically selects `32 / GNBM_GPUS` per process, preserving
a global batch of 32 for supported one-, two-, and four-GPU launches.

`SYNTHETIC_MRI_DATASET` must point directly to the generated `new_patches`
directory. Its sibling `new_split.csv` records the patient-level assignment;
the loader consumes the already materialized `new_patches/{train,val,test}`
directories and never reconstructs a split from that CSV.
