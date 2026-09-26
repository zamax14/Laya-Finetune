#!/bin/bash
#SBATCH --job-name=layaft
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=logs/%j_layaft.out
##SBATCH --nodelist=<your node>
#SBATCH --gres=gpu:1
#SBATCH --partition=<your-partition>
#
# Every layaft mode as a Slurm job, from the repo folder (on the DGX, python only runs inside jobs):
#
#   sbatch slurm.sh test                                                     # the unit tests
#   sbatch --gres=none slurm.sh generate task=helpdesk llm=qwen3.8:latest n=216 context=all parallel=8
#   sbatch --gres=none slurm.sh generate task=helpdesk llm=qwen3.8:latest n=0 filler=300 context=all parallel=8
#   sbatch slurm.sh train task=helpdesk profile=full ctx=32k
#   sbatch slurm.sh val task=helpdesk model=runs/helpdesk-32k ctx=32k
#
# The first job creates .venv and installs the package; later jobs reinstall only if pyproject.toml changed.
# With backend=ollama (the default of generate) and no ollama_url=, the job uses the node's Ollama daemon, or starts
# its own server if there is none; llm= is pulled if missing. API keys are read from files in this folder (openrouter, OPENAI,
# typesafe, HF_TOKEN) or from the environment (sbatch --export=ALL,OPENROUTER_API_KEY=...).
# torch comes from TORCH_INDEX (CUDA 13 by default, driver 580 or newer); for an older driver:
#   sbatch --export=ALL,TORCH_INDEX=https://download.pytorch.org/whl/cu128 slurm.sh train ...
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "$(date) · $(hostname) · $(pwd) · layaft $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ---------------------------------------------------------------- environment
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
STAMP=".venv/.installed-$(sha1sum pyproject.toml | cut -c1-12)"
if [ ! -f "$STAMP" ]; then
    pip install --quiet --upgrade pip
    pip install --quiet --extra-index-url "$TORCH_INDEX" -e ".[notebooks]"
    touch "$STAMP"
fi

MODE="${1:-}"
if [ "$MODE" = "test" ]; then
    exec python -m unittest discover -s tests -t .
fi

# ---------------------------------------------------------------- Ollama, when generating with it
# The node's own `ollama serve` daemon (localhost:11434) manages its GPU outside Slurm: submit generation with
# --gres=none and it is used as is. Without a daemon, the job starts its own server on the GPU it was given.
ARGS=("$@")
BACKEND=ollama; LLM=gemma3:12b; HAS_URL=0
for arg in "${@:2}"; do
    case "$arg" in
        backend=*) BACKEND="${arg#backend=}" ;;
        llm=*) LLM="${arg#llm=}" ;;
        ollama_url=*) HAS_URL=1 ;;
    esac
done
if [ "$MODE" = "generate" ] && [ "$BACKEND" = "ollama" ] && [ "$HAS_URL" = 0 ]; then
    if curl -sf --max-time 10 http://localhost:11434/api/tags >/dev/null; then
        echo "Using the node's Ollama daemon"
    else
        command -v ollama >/dev/null || { echo "No Ollama daemon and ollama is not in PATH on $(hostname)"; exit 1; }
        PORT=$((11500 + ${SLURM_JOB_ID:-0} % 400))  # One port per job: two jobs never share a server.
        export OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-8}"
        ollama serve > "ollama-${SLURM_JOB_ID:-local}.log" 2>&1 &
        OLLAMA_PID=$!
        trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
        for _ in $(seq 60); do curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 1; done
        ARGS+=("ollama_url=http://$OLLAMA_HOST")
    fi
    ollama list | grep -q "^${LLM}[ :]" || ollama pull "$LLM"
fi

# ---------------------------------------------------------------- run
layaft "${ARGS[@]}"
echo "$(date) · done"
