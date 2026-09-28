#!/bin/bash
# Queues the whole context-prefilter flow, one job per step: run it on the login node with
# `bash slurm/prefilter/pipeline.sh`. Each job waits for the one it needs (--dependency=afterok).

cd $HOME/Laya-Finetune

synthesize=$(sbatch --parsable slurm/prefilter/synthesize.sh)
generate=$(sbatch --parsable --dependency=afterok:$synthesize slurm/prefilter/generate.sh)
pair=$(sbatch --parsable --dependency=afterok:$generate slurm/prefilter/pair.sh)
verify=$(sbatch --parsable --dependency=afterok:$pair slurm/prefilter/verify.sh)
train_1k=$(sbatch --parsable --dependency=afterok:$verify slurm/prefilter/train_1k.sh)
train_8k=$(sbatch --parsable --dependency=afterok:$train_1k slurm/prefilter/train_8k.sh)

echo "synthesize $synthesize · generate $generate · pair $pair · verify $verify · 1k $train_1k · 8k $train_8k"
