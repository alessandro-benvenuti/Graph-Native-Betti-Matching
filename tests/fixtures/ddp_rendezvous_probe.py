"""Small real CPU collective to test the study's torchrun launcher, not training."""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist


if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://", timeout=timedelta(seconds=30))
    try:
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
        dist.destroy_process_group()
