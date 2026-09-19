# Node-aware Betti pilot on 4,000 MRI patches

This is a paired 100-epoch continuation from one pretrained node+edge-focal
checkpoint. `control.yaml` retains the original objective;
`node_aware_betti.yaml` changes only topology supervision. Both configurations
use the same deterministic 4,000-patch training subset, 500-patch validation
subset, seed, augmentation, optimizer, and validation schedule.

The topology arm keeps unmatched-edge focal supervision disabled. Edges
incident to unmatched topology vertices participate in the forward filtration,
but their relation probabilities are detached; topology corrects those
structures through node existence instead of assigning relation labels to
nonexistent target vertices.

The launcher defaults to `WANDB_MODE=offline` so unavailable external network
connectivity cannot abort an A100 allocation before training starts. Set
`GNBM_PILOT_WANDB_MODE=online` only when online initialization has been verified;
offline runs can be synchronized after training.
