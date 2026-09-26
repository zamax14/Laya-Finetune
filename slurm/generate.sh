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
# 150 cases in each of the task's 40 contexts, about 6,000, in the real mix of answers (`weights` in the task),
# plus 400 neutral documents that long-context training wraps around the cases.
layaft generate task=tool_routing backend=ollama llm=qwen3.6:35b n=150 filler=400 context=all parallel=8
# The judge keeps about 1 in 5 "ask the user" cases and fewer multi-tool ones than single-tool: write more of both.
layaft generate task=tool_routing backend=ollama llm=qwen3.6:35b n=50 context=all parallel=8 only='{"action": ["ask_user"]}'
layaft generate task=tool_routing backend=ollama llm=qwen3.6:35b n=50 context=all parallel=8 only='{"action": ["use_tools"]}'

# Option 2: Python script (same result)
# python examples/generate.py

pwd; hostname; date
