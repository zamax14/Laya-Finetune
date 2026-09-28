#!/bin/bash
#SBATCH --job-name=layaft_train_1k
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j_%x.out
##SBATCH --partition=<your partition>

set -e  # A failed training stops here: no evaluation, and the next stage (afterok) never starts.

TASK=${TASK:?pass the task: sbatch --export=ALL,TASK=<name in tasks/> slurm/train_1k.sh}

pwd; hostname; date
cd "$SLURM_SUBMIT_DIR"

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
# 1,024 tokens, the checkpoint's own context (teacher=none: the judges already dropped wrong labels)
layaft train task=$TASK model=multilingual profile=full teacher=none data=data/${TASK}_train.jsonl out=runs/$TASK-1k
layaft val task=$TASK model=runs/$TASK-1k

# Option 2: the same from Python
# python - <<PY
# from layaft import LayaFT
# model = LayaFT("multilingual")
# model.train(task="$TASK", profile="full", teacher="none", data="data/${TASK}_train.jsonl", out="runs/$TASK-1k")
# model.val(task="$TASK")
# PY

pwd; hostname; date
