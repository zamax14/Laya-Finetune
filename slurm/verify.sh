#!/bin/bash
#SBATCH --job-name=layaft_verify
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=logs/%j_%x.out
##SBATCH --partition=<your partition>

# No --gres=gpu on purpose: the judges talk to an `ollama serve` daemon already running on the node,
# which manages its own GPU outside Slurm.

TASK=${TASK:?pass the task: sbatch --export=ALL,TASK=<name in tasks/> slurm/verify.sh}
JUDGE=${JUDGE:-gemma4:31b}          # of another family than the generator
SECOND_JUDGE=${SECOND_JUDGE:-qwen3.5:122b}

pwd; hostname; date
cd "$SLURM_SUBMIT_DIR"

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
# A judge answers every case blind: data/$TASK.jsonl → data/${TASK}_verified.jsonl (kept) and _rejected.jsonl (to audit)
layaft verify task=$TASK backend=ollama llm=$JUDGE parallel=8
# A second, larger judge re-reads only the rejected cases. Where it confirms the label, the case comes back; where it
# answers exactly like the first judge, the two judges' answer becomes the label (_relabelled): the text was written
# for one answer and reads as another, the hard cases.
layaft verify task=$TASK backend=ollama llm=$SECOND_JUDGE parallel=8 data=data/${TASK}_rejected.jsonl
# Training data: what the first judge kept, plus what the second confirmed or relabelled with the first.
cat data/${TASK}_verified.jsonl data/${TASK}_rejected_verified.jsonl data/${TASK}_rejected_relabelled.jsonl > data/${TASK}_train.jsonl

# Option 2: Python script (same result)
# python examples/verify.py

pwd; hostname; date
