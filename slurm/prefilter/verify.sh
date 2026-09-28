#!/bin/bash
#SBATCH --job-name=prefilter_verify
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=logs/%j_prefilter_verify.out
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
# gemma answers every generated case and every near pair blind; qwen3.5:122b re-reads what it rejected: a confirmed
# label comes back, and where both judges agree against the label, their answer becomes it (_relabelled).
layaft verify task=context_prefilter backend=ollama llm=gemma4:31b parallel=8 data=data/context_prefilter.jsonl
layaft verify task=context_prefilter backend=ollama llm=qwen3.5:122b parallel=8 data=data/context_prefilter_rejected.jsonl
layaft verify task=context_prefilter backend=ollama llm=gemma4:31b parallel=8 data=data/context_prefilter_near.jsonl
layaft verify task=context_prefilter backend=ollama llm=qwen3.5:122b parallel=8 data=data/context_prefilter_near_rejected.jsonl
# The same for the held-out test.
layaft verify task=context_prefilter backend=ollama llm=gemma4:31b parallel=8 data=data/prefilter_heldout.jsonl
layaft verify task=context_prefilter backend=ollama llm=qwen3.5:122b parallel=8 data=data/prefilter_heldout_rejected.jsonl
layaft verify task=context_prefilter backend=ollama llm=gemma4:31b parallel=8 data=data/prefilter_heldout_near.jsonl
layaft verify task=context_prefilter backend=ollama llm=qwen3.5:122b parallel=8 data=data/prefilter_heldout_near_rejected.jsonl

# Training data: what the judges kept or relabelled, plus the random pairs.
cat data/context_prefilter_verified.jsonl data/context_prefilter_rejected_verified.jsonl data/context_prefilter_rejected_relabelled.jsonl \
    data/context_prefilter_near_verified.jsonl data/context_prefilter_near_rejected_verified.jsonl data/context_prefilter_near_rejected_relabelled.jsonl \
    data/context_prefilter_random.jsonl > data/context_prefilter_train.jsonl
# The held-out test, the same way, on items never trained on.
cat data/prefilter_heldout_verified.jsonl data/prefilter_heldout_rejected_verified.jsonl data/prefilter_heldout_rejected_relabelled.jsonl \
    data/prefilter_heldout_near_verified.jsonl data/prefilter_heldout_near_rejected_verified.jsonl data/prefilter_heldout_near_rejected_relabelled.jsonl \
    data/prefilter_heldout_random.jsonl > tasks/prefilter/test_heldout.jsonl

# Option 2: the same from Python
# python examples/prefilter.py verify

pwd; hostname; date
