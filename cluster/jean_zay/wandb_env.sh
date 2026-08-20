#!/usr/bin/env bash
# Non-secret W&B destination shared by Jean-Zay login and batch environments.
# Authentication remains in W&B's user credential store after `wandb login`;
# never add WANDB_API_KEY to this file.

export WANDB_ENTITY="${WANDB_ENTITY:-alessandrobenvenuti2002-politecnico-di-torino}"
export WANDB_PROJECT="${WANDB_PROJECT:-focal-loss}"
