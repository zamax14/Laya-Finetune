#!/bin/bash
#SBATCH --job-name=layaft_generate
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=logs/%j_layaft_generate.out
##SBATCH --nodelist=<your node>
#SBATCH --partition=<your-partition>

# No --gres=gpu on purpose: generation talks to the `ollama serve` daemon already
# running on this node, which manages its own GPU outside Slurm.

pwd; hostname; date

cd $HOME/Laya-Finetune

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
# 1. Two cases per answer combination (33) in each of the task's 16 contexts: about 1,000 cases.
layaft generate task=tool_routing backend=ollama llm=qwen3.8:latest n=66 context=all parallel=8
# 2. "Ask the user" and "answer directly" are only 2 of the 33 combinations: add more of them.
layaft generate task=tool_routing backend=ollama llm=qwen3.8:latest n=20 context=all parallel=8 only='{"action": ["ask_user", "answer_directly"]}'
# 3. Neutral documents that long-context training wraps around the cases.
layaft generate task=tool_routing backend=ollama llm=qwen3.8:latest n=0 filler=200 context=all parallel=8

# Option 2: Python script (same result)
# python examples/generate.py

pwd; hostname; date
