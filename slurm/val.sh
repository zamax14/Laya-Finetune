#!/bin/bash
#SBATCH --job-name=layaft_val
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=logs/%j_layaft_val.out
##SBATCH --nodelist=<your node>
#SBATCH --gres=gpu:1
#SBATCH --partition=<your-partition>

pwd; hostname; date

cd $HOME/Laya-Finetune

# Everything is installed inside the project's virtual environment, never in the system Python.
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option 1: command line
# Laya as shipped, the reference every stage is compared with (each train_*.sh evaluates its own checkpoint).
layaft val task=tool_routing model=multilingual

# Option 2: the same from Python
# python -c 'from layaft import LayaFT; LayaFT("multilingual").val(task="tool_routing")'


pwd; hostname; date
