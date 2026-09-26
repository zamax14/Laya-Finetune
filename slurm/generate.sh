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
layaft generate task=helpdesk backend=ollama llm=qwen3.8:latest n=216 context=all parallel=8

# Option 2: Python script (same result)
# python examples/generate.py

pwd; hostname; date
