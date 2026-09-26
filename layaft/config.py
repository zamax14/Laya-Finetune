"""Paths, API keys and checkpoint aliases.

Imported before torch (`layaft/__init__.py` does it first): it keeps the Hugging Face cache inside the project unless
HF_HOME is already set, reads HF_TOKEN from a local file and sets TORCH_DISABLE_NATIVE_JIT, which torch reads on import.
"""
import os
from pathlib import Path

WORKDIR = Path.cwd()
DATA = WORKDIR / "data"
RUNS = WORKDIR / "runs"
TASKS = WORKDIR / "tasks"

# Pinned revisions: a new upload on the Hub never changes a run silently.
MODELS = {
    "multilingual": "convaiinnovations/laya-multilingual@82d57fc4f2d1be3d2caac494045f2ec51d0842f3",
    "english": "convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
}
DEFAULT_MODEL = "multilingual"


def secret(env, filename=None):
    """Key from the environment or, if missing, from a file in the working directory (git-ignored)."""
    value = os.environ.get(env)
    path = WORKDIR / (filename or env)
    if not value and path.is_file():
        value = path.read_text(encoding="utf-8").strip()
    return value or None


if secret("HF_TOKEN"):  # Authenticated Hugging Face downloads.
    os.environ["HF_TOKEN"] = secret("HF_TOKEN")
os.environ.setdefault("HF_HOME", str(WORKDIR / ".model-cache" / "huggingface"))
# torch 2.14 sends some GPU ops to Triton kernels that compile C and need Python.h; this official switch uses plain
# torch ops instead.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("USE_TF", "0")
