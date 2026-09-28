#!/bin/bash
#SBATCH --job-name=prefilter_synthesize
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=logs/%j_prefilter_synthesize.out
##SBATCH --nodelist=<your node>
#SBATCH --partition=<your-partition>

# No --gres=gpu on purpose: it talks to the `ollama serve` daemon of this node, which manages its own GPU.

set -e  # A failed step stops here, and the jobs that wait for this one (afterok) never start.

pwd; hostname; date

cd $HOME/Laya-Finetune

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# MCP tools for the registry's servers that list none, and rules by theme, written by the node's LLM; then the
# pools of the task: tasks/prefilter/pool_train.jsonl and pool_heldout.jsonl (real catalog + these items).
python tasks/prefilter/synthesize.py qwen3.6:35b

pwd; hostname; date
