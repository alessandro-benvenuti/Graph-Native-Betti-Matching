# Edge candidate/loss ablation

These four new recipes complete the missing cells around the original focal
matrix while preserving its seed, data, scheduler, validation protocol and
100-epoch mixed-pretraining to 600-epoch MRI-specialization schedule.

| Node loss | Edge recipe | Candidate pool |
|---|---|---|
| weighted CE | unweighted immediate focal, no ratio upsampling | M-M only |
| unweighted immediate focal | unweighted immediate focal, no ratio upsampling | M-M only |
| weighted CE | CE with legacy M-M ratio upsampling | M-M plus top-256 hard U-M/U-U negatives |
| unweighted immediate focal | CE with legacy M-M ratio upsampling | M-M plus top-256 hard U-M/U-U negatives |

For CE-all, ratio upsampling is applied only to matched candidates. The hard
unmatched candidates are all no-edge targets, use ordinary CE, and enter with
per-candidate weight one. No Betti loss is active.

Together with the four existing immediate recipes (baseline, node focal, edge
focal and combined focal), these runs separate candidate-pool effects within
each edge objective. They do not completely separate CE/focal from edge ratio
upsampling because the legacy CE recipe retains upsampling while focal does
not.

