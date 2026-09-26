#!/bin/bash
#SBATCH --job-name=layaft_train_8k
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=logs/%j_layaft_train_8k.out
##SBATCH --nodelist=<your node>
#SBATCH --gres=gpu:1
#SBATCH --partition=<your-partition>

set -e  # A failed training stops here: no evaluation, and the next stage (afterok) never starts.

pwd; hostname; date

cd $HOME/Laya-Finetune

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
# 8k from the 1k stage: positions mmBERT already has, now trained on long states too
layaft train task=tool_routing model=runs/tool_routing-1k profile=full teacher=none data=data/tool_routing_train.jsonl ctx=8k long=2000 out=runs/tool_routing-8k
layaft val task=tool_routing model=runs/tool_routing-8k ctx=8k

# Option 2: the same from Python
# python - <<'EOF'
# from layaft import LayaFT
# model = LayaFT("runs/tool_routing-1k")
# model.train(task="tool_routing", profile="full", teacher="none", data="data/tool_routing_train.jsonl", ctx="8k", long=2000, out="runs/tool_routing-8k")
# model.val(task="tool_routing", ctx="8k")
# EOF

pwd; hostname; date
