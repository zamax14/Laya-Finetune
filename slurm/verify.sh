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
# gemma sees an internal database in almost any business request. A second, larger judge re-reads only the rejected
# cases: those whose label it confirms come back (2 of 3 agree: the generator and this judge).
layaft verify task=tool_routing backend=ollama llm=qwen3.5:122b parallel=8 data=data/tool_routing_rejected.jsonl
# Training data: what the first judge kept plus what the second recovered.
cat data/tool_routing_verified.jsonl data/tool_routing_rejected_verified.jsonl > data/tool_routing_train.jsonl

# Option 2: Python script (same result)
# python examples/verify.py

pwd; hostname; date
