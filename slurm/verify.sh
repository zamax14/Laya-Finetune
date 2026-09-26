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
# A second, larger judge re-reads only the rejected cases. Where it confirms the label, the case comes back; where it
# answers exactly like gemma, the two judges' answer becomes the label (_relabelled): the message was written for one
# answer and reads as another, the hard cases.
layaft verify task=tool_routing backend=ollama llm=qwen3.5:122b parallel=8 data=data/tool_routing_rejected.jsonl
# Training data: what the first judge kept, plus what the second confirmed or relabelled with the first.
cat data/tool_routing_verified.jsonl data/tool_routing_rejected_verified.jsonl data/tool_routing_rejected_relabelled.jsonl > data/tool_routing_train.jsonl

# Option 2: Python script (same result)
# python examples/verify.py

pwd; hostname; date
