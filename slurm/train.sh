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

# Option 1: command line (teacher=none: without a Jev key the constructed label is smoothed instead)
# 1. 1,024 tokens, the checkpoint's own context → runs/tool_routing-1k
layaft train task=tool_routing model=multilingual profile=full teacher=none
# 2. 8k: positions mmBERT already has, now trained on long states → runs/tool_routing-8k
layaft train task=tool_routing model=runs/tool_routing-1k profile=full ctx=8k teacher=none
# 3. 32k: YaRN ×4 on the global-attention layers → runs/tool_routing-32k
layaft train task=tool_routing model=runs/tool_routing-8k profile=full ctx=32k teacher=none

# Option 2: Python script (same result)
# python examples/train.py

pwd; hostname; date
