#!/bin/bash
#SBATCH --job-name=layaft_train
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=logs/%j_layaft_train.out
##SBATCH --nodelist=<your node>
#SBATCH --gres=gpu:1
#SBATCH --partition=<your-partition>

pwd; hostname; date

cd $HOME/Laya-Finetune

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
layaft train task=helpdesk model=multilingual profile=full ctx=32k

# Option 2: Python script (same result)
# python examples/train.py

pwd; hostname; date
