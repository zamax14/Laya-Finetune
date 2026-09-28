#!/bin/bash
#SBATCH --job-name=layaft_train_8k
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j_%x.out
##SBATCH --partition=<your partition>

set -e  # A failed training stops here: no evaluation, and the next stage (afterok) never starts.

TASK=${TASK:?pass the task: sbatch --export=ALL,TASK=<name in tasks/> slurm/train_8k.sh}

pwd; hostname; date
cd "$SLURM_SUBMIT_DIR"

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
# 8k from the 1k stage: positions mmBERT already has, now trained on long states too
layaft train task=$TASK model=runs/$TASK-1k profile=full teacher=none data=data/${TASK}_train.jsonl ctx=8k long=2000 out=runs/$TASK-8k
layaft val task=$TASK model=runs/$TASK-8k ctx=8k

# Option 2: the same from Python
# python - <<PY
# from layaft import LayaFT
# model = LayaFT("runs/$TASK-1k")
# model.train(task="$TASK", profile="full", teacher="none", data="data/${TASK}_train.jsonl", ctx="8k", long=2000, out="runs/$TASK-8k")
# model.val(task="$TASK", ctx="8k")
# PY

pwd; hostname; date
