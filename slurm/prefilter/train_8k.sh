#!/bin/bash
#SBATCH --job-name=prefilter_train_8k
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=logs/%j_prefilter_train_8k.out
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
# 8k from the 1k stage: the request wrapped in earlier session transcripts, for callers that send recent turns too.
layaft train task=context_prefilter model=runs/context_prefilter-1k profile=full teacher=none data=data/context_prefilter_train.jsonl ctx=8k long=2000 out=runs/context_prefilter-8k
layaft val task=context_prefilter model=runs/context_prefilter-8k ctx=8k

# Option 2: the same from Python
# python examples/prefilter.py train_8k

pwd; hostname; date
