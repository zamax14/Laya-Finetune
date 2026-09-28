#!/bin/bash
#SBATCH --job-name=prefilter_pair
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=logs/%j_prefilter_pair.out
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

# Option 1: command line
# Every useful request with the 2 most similar items (hard negatives, judged next) and 5 random ones (kept as no).
layaft pair task=context_prefilter near=2 random=5 embed=bge-m3:latest
layaft pair task=context_prefilter near=2 random=5 embed=bge-m3:latest data=data/prefilter_heldout.jsonl pool=tasks/prefilter/pool_heldout.jsonl

# Option 2: the same from Python
# python examples/prefilter.py pair

pwd; hostname; date
