#!/bin/bash
#SBATCH --job-name=layaft_generate
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=logs/%j_%x.out
##SBATCH --partition=<your partition>

# No --gres=gpu on purpose: generation talks to an `ollama serve` daemon already running on the node,
# which manages its own GPU outside Slurm. Run from the repo root: sbatch --export=ALL,TASK=invoices slurm/generate.sh

TASK=${TASK:?pass the task: sbatch --export=ALL,TASK=<name in tasks/> slurm/generate.sh}
LLM=${LLM:-qwen3.6:35b}  # any model the daemon already has: generation never pulls one

pwd; hostname; date
cd "$SLURM_SUBMIT_DIR"

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
# 150 cases in each of the task's contexts, in the task's mix of answers (`weights`),
# plus 400 neutral documents that long-context training wraps around the cases.
layaft generate task=$TASK backend=ollama llm=$LLM n=150 filler=400 context=all parallel=8

# Option 2: Python script (same result)
# python examples/generate.py

pwd; hostname; date
