#!/bin/bash
#SBATCH --job-name=layaft_train_1k
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=logs/%j_layaft_train_1k.out
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
# 1,024 tokens, the checkpoint's own context (teacher=none: the judges already dropped wrong labels)
layaft train task=tool_routing model=multilingual profile=full teacher=none data=data/tool_routing_train.jsonl out=runs/tool_routing-1k
layaft val task=tool_routing model=runs/tool_routing-1k

# Option 2: the same from Python
# python - <<'EOF'
# from layaft import LayaFT
# model = LayaFT("multilingual")
# model.train(task="tool_routing", profile="full", teacher="none", data="data/tool_routing_train.jsonl", out="runs/tool_routing-1k")
# model.val(task="tool_routing")
# EOF

pwd; hostname; date
