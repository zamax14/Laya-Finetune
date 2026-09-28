#!/bin/bash
# Queues the whole flow for one task, one job per step. Run it on the login node from the repo root:
#
#     TASK=invoices bash slurm/pipeline.sh
#
# Each job waits for the one it needs (--dependency), so the queue runs them in order, and the independent ones
# (the base model's evaluation, the 8k and 32k stages) run at the same time on other GPUs. Every step keeps its own
# log in logs/.

export TASK=${TASK:?pass the task: TASK=<name in tasks/> bash slurm/pipeline.sh}
mkdir -p logs

generate=$(sbatch --parsable --export=ALL slurm/generate.sh)
verify=$(sbatch --parsable --export=ALL --dependency=afterok:$generate slurm/verify.sh)
train_1k=$(sbatch --parsable --export=ALL --dependency=afterok:$verify slurm/train_1k.sh)
train_8k=$(sbatch --parsable --export=ALL --dependency=afterok:$train_1k slurm/train_8k.sh)
train_32k=$(sbatch --parsable --export=ALL --dependency=afterok:$train_1k slurm/train_32k.sh)  # 8k and 32k both start from 1k, in parallel
val_base=$(sbatch --parsable --export=ALL slurm/val.sh)

echo "$TASK: generate $generate · verify $verify · train 1k $train_1k · 8k $train_8k · 32k $train_32k · base $val_base"
