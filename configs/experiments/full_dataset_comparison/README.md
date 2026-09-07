# Full-data comparison pipelines

These recipes add two controlled comparisons to the full-data node-focal run:

1. baseline node CE plus baseline matched-node edge CE;
2. immediate unweighted node focal plus immediate unweighted edge focal on
   Hungarian matched--matched pairs only.

Every recipe uses the original limited-target mixed pretraining regime (25,900
Plants and 4,000 MRI training samples with balanced domain sampling) for 50
epochs, followed by up to 250 epochs on the complete new MRI split (50-epoch
patience, no minimum). The default
launcher uses one H100 for pretraining and four H100s for specialization while
preserving a global batch of 32. The MRI stage starts from the pretraining
checkpoint selected by validation edge mAP.

`finetune_nodefocal_edgefocal_mm_fgw_controlled.yaml` is the matcher-only FGW
counterpart to the saved node-focal/edge-focal matched-only checkpoint recipe.
Its dataset split, augmentation, losses, initialization contract, seed,
optimizer, schedule, batch size, and evaluation settings are inherited without
change. Only the experiment identity and matcher block differ. The separately
named smoke derivative deliberately bounds samples, epochs, workers, and
checkpoint/evaluation work and must not be used as the full comparison.

The 4,000/200 MRI subset is selected reproducibly by a SHA-256 ranking seeded
with `364505`. All recipes therefore use identical subset membership, recorded
in each run's `dataset-manifest.json`.
