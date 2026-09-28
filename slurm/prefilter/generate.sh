#!/bin/bash
#SBATCH --job-name=prefilter_generate
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=logs/%j_prefilter_generate.out
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
# Requests for items of the training pool: 300 per project context (30), useful 60 % and near misses 40 %.
layaft generate task=context_prefilter backend=ollama llm=qwen3.6:35b n=300 context=all parallel=8
# The held-out test: the same on the held-out pool (sources never trained on), 20 per context.
layaft generate task=context_prefilter backend=ollama llm=qwen3.6:35b n=20 context=all parallel=8 pool=tasks/prefilter/pool_heldout.jsonl out=data/prefilter_heldout.jsonl
# Earlier Claude Code sessions for the 8k stage to wrap the requests in.
layaft generate task=context_prefilter backend=ollama llm=qwen3.6:35b n=0 filler=400 context=all parallel=8

# Option 2: the same from Python
# python examples/prefilter.py generate

pwd; hostname; date
