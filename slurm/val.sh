#!/bin/bash
#SBATCH --job-name=layaft_val
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=logs/%j_layaft_val.out
##SBATCH --nodelist=<your node>
#SBATCH --gres=gpu:1
#SBATCH --partition=<your-partition>

pwd; hostname; date

cd $HOME/Laya-Finetune

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export CPATH=$PWD/.venv/python-headers/usr/include/python3.12:$PWD/.venv/python-headers/usr/include  # From train.sh.

# Option 1: command line
layaft val task=tool_routing model=multilingual
layaft val task=tool_routing model=runs/tool_routing-1k
layaft val task=tool_routing model=runs/tool_routing-8k ctx=8k
layaft val task=tool_routing model=runs/tool_routing-32k ctx=32k

# Option 2: Python script (same result)
# python examples/val.py

pwd; hostname; date
