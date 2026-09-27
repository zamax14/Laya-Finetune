#!/bin/bash
#SBATCH --job-name=layaft_train_32k
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=logs/%j_layaft_train_32k.out
##SBATCH --nodelist=<your node>
#SBATCH --gres=gpu:1
#SBATCH --partition=<your-partition>

set -e  # A failed training stops here: no evaluation, and the next stage (afterok) never starts.

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

# Option 1: command line
# 32k from the 1k stage, at the same time as 8k: YaRN ×4 on the global-attention layers, trained on short
# cases and long copies of every length up to 32k
layaft train task=tool_routing model=runs/tool_routing-1k profile=full teacher=none data=data/tool_routing_train.jsonl ctx=32k long=2000 out=runs/tool_routing-32k
layaft val task=tool_routing model=runs/tool_routing-32k ctx=32k

# Option 2: the same from Python
# python - <<'EOF'
# from layaft import LayaFT
# model = LayaFT("runs/tool_routing-1k")
# model.train(task="tool_routing", profile="full", teacher="none", data="data/tool_routing_train.jsonl", ctx="32k", long=2000, out="runs/tool_routing-32k")
# model.val(task="tool_routing", ctx="32k")
# EOF

pwd; hostname; date
