#!/bin/bash
#SBATCH --job-name=layaft_verify
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=logs/%j_layaft_verify.out
##SBATCH --nodelist=<your node>
#SBATCH --partition=<your-partition>

# No --gres=gpu on purpose: the judge talks to the `ollama serve` daemon already
# running on this node, which manages its own GPU outside Slurm.

pwd; hostname; date

cd $HOME/Laya-Finetune

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
# A judge of another family (gemma4, the cases come from qwen) answers every case blind:
# data/tool_routing.jsonl → data/tool_routing_verified.jsonl (kept) and data/tool_routing_rejected.jsonl (to audit)
layaft verify task=tool_routing backend=ollama llm=gemma4:31b parallel=8

# Option 2: Python script (same result)
# python examples/verify.py

pwd; hostname; date
