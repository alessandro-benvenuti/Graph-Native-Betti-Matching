# Full-data comparison pipelines

These recipes add two controlled comparisons to the full-data node-focal run:

1. baseline node CE plus baseline matched-node edge CE;
2. immediate unweighted node focal plus immediate unweighted edge focal on
   Hungarian matched--matched pairs only.

Every recipe uses the original limited-target mixed pretraining regime (25,900
Plants and 4,000 MRI training samples with balanced domain sampling) for 50
epochs, followed by 100 epochs on the complete new MRI split. The default
launcher uses one H100 for pretraining and four H100s for specialization while
preserving a global batch of 32. The MRI stage starts from the pretraining
checkpoint selected by validation edge mAP.

The 4,000/200 MRI subset is selected reproducibly by a SHA-256 ranking seeded
with `364505`. All recipes therefore use identical subset membership, recorded
in each run's `dataset-manifest.json`.
