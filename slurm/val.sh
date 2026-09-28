#!/bin/bash
#SBATCH --job-name=layaft_val
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j_%x.out
##SBATCH --partition=<your partition>

TASK=${TASK:?pass the task: sbatch --export=ALL,TASK=<name in tasks/> slurm/val.sh}

pwd; hostname; date
cd "$SLURM_SUBMIT_DIR"

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
# Laya as shipped, the reference every stage is compared with (each train_*.sh evaluates its own checkpoint).
layaft val task=$TASK model=multilingual

# Option 2: the same from Python
# python -c "from layaft import LayaFT; LayaFT('multilingual').val(task='$TASK')"

pwd; hostname; date
