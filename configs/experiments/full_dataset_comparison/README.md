# Full-data comparison pipelines

These recipes add two controlled comparisons to the full-data node-focal run:

1. baseline node CE plus baseline matched-node edge CE;
2. immediate unweighted node focal plus immediate unweighted edge focal on
   Hungarian matched--matched pairs only.

Every recipe uses all available Plants and new synthetic-MRI patches, 50 epochs
of mixed pretraining, 100 epochs of MRI specialization, and a global batch of
32 when launched on four H100s. The MRI stage starts from the pretraining
checkpoint selected by validation edge mAP.

