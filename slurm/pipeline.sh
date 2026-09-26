#!/bin/bash
# Queues the whole tool-routing flow, one job per step: run it on the login node with `bash slurm/pipeline.sh`.
# Each job waits for the one it needs (--dependency), so the queue runs them in order, and the independent ones
# (the base model's evaluation) run at the same time on another GPU. Every step keeps its own log in ~/logs.

cd $HOME/Laya-Finetune

generate=$(sbatch --parsable slurm/generate.sh)
verify=$(sbatch --parsable --dependency=afterok:$generate slurm/verify.sh)
train_1k=$(sbatch --parsable --dependency=afterok:$verify slurm/train_1k.sh)
train_8k=$(sbatch --parsable --dependency=afterok:$train_1k slurm/train_8k.sh)
train_32k=$(sbatch --parsable --dependency=afterok:$train_8k slurm/train_32k.sh)
val_base=$(sbatch --parsable slurm/val.sh)

echo "generate $generate · verify $verify · train 1k $train_1k · 8k $train_8k · 32k $train_32k · base $val_base"
