#!/bin/bash
#SBATCH --job-name=layaft_train
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=logs/%j_layaft_train.out
##SBATCH --nodelist=<your node>
#SBATCH --gres=gpu:1
#SBATCH --partition=<your-partition>

pwd; hostname; date

cd $HOME/Laya-Finetune

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Long context uses flex_attention, whose kernels Triton compiles against the Python headers this node lacks:
# unpack them inside the venv (nothing is installed on the system) and point the compiler at them.
if [ ! -d .venv/python-headers ]; then
    apt-get download libpython3.12-dev
    dpkg-deb -x libpython3.12-dev_*.deb .venv/python-headers
    rm libpython3.12-dev_*.deb
fi
export CPATH=$PWD/.venv/python-headers/usr/include/python3.12:$PWD/.venv/python-headers/usr/include

# Option 1: command line (teacher=none: without a Jev key the label is smoothed; the judges already dropped wrong ones)
# 1. 1,024 tokens, the checkpoint's own context → runs/tool_routing-1k
layaft train task=tool_routing model=multilingual profile=full teacher=none data=data/tool_routing_train.jsonl
# 2. 8k: positions mmBERT already has, now trained on long states too → runs/tool_routing-8k
layaft train task=tool_routing model=runs/tool_routing-1k profile=full teacher=none data=data/tool_routing_train.jsonl ctx=8k long=2000
# 3. 32k: YaRN ×4 on the global-attention layers → runs/tool_routing-32k
layaft train task=tool_routing model=runs/tool_routing-8k profile=full teacher=none data=data/tool_routing_train.jsonl ctx=32k long=2000

# Option 2: Python script (same result)
# python examples/train.py

pwd; hostname; date
