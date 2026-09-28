#!/bin/bash
#SBATCH --job-name=prefilter_train_1k
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=logs/%j_prefilter_train_1k.out
##SBATCH --nodelist=<your node>
#SBATCH --gres=gpu:1
#SBATCH --partition=<your-partition>

set -e  # A failed step stops here, and the jobs that wait for this one (afterok) never start.

pwd; hostname; date

cd $HOME/Laya-Finetune

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
layaft train task=context_prefilter model=multilingual profile=full teacher=none data=data/context_prefilter_train.jsonl out=runs/context_prefilter-1k
layaft val task=context_prefilter model=runs/context_prefilter-1k

# Option 2: the same from Python
# python examples/prefilter.py train_1k

pwd; hostname; date
