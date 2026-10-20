"""Small real CPU collective to test the study's torchrun launcher, not training."""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time


def progress(stage):
    print(f"probe rank={os.environ.get('RANK', '?')} stage={stage} time={time.time():.3f}", flush=True)


if __name__ == "__main__":
    progress("import-torch")
    import torch
    import torch.distributed as dist

    progress("init-process-group")
    dist.init_process_group("gloo", init_method="env://", timeout=timedelta(seconds=30))
    try:
        progress("all-reduce")
        value = torch.tensor(float(dist.get_rank() + 1))
        dist.all_reduce(value)
        expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
        assert value.item() == expected
        assert os.environ["TORCHELASTIC_USE_AGENT_STORE"] == "True"
        if dist.get_rank() == 0:
            Path(sys.argv[1]).write_text(json.dumps({
                "world_size": dist.get_world_size(), "sum": value.item(),
                "agent_store": True, "master_port": int(os.environ["MASTER_PORT"]),
            }))
    finally:
        progress("destroy-process-group")
        dist.destroy_process_group()
    progress("complete")
