"""Carga de Laya Multilingual: el checkpoint de Hugging Face o uno local que deja entrenar.py.

Este módulo se importa antes que torch: fija la caché de Hugging Face dentro del proyecto, lee las llaves locales y
activa TORCH_DISABLE_NATIVE_JIT, que torch lee al importarse.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT = "convaiinnovations/laya-multilingual@82d57fc4f2d1be3d2caac494045f2ec51d0842f3"
# 8192 es el máximo del encoder (mmBERT, max_position_embeddings); el checkpoint base se entrenó hasta 1024.
MAX_LEN, HEAD_MAX_LEN = 8192, 256


def secret(env, filename):
    """Llave desde la variable de entorno o, si falta, desde un archivo de la raíz (ignorado en git)."""
    value = os.environ.get(env)
    if not value and (ROOT / filename).is_file():
        value = (ROOT / filename).read_text(encoding="utf-8").strip()
    return value or None


if secret("HF_TOKEN", "HF_TOKEN"):  # Descargas autenticadas de Hugging Face.
    os.environ["HF_TOKEN"] = secret("HF_TOKEN", "HF_TOKEN")
HF_HOME = ROOT / ".model-cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
# torch 2.14 manda algunas operaciones de GPU a kernels de Triton que compilan C y necesitan Python.h. Con este
# interruptor oficial usa las operaciones normales de torch.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("USE_TF", "0")


def base_dir(patterns=("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")):
    """Carpeta local del checkpoint base; se descarga la primera vez (~650 MB)."""
    from huggingface_hub import snapshot_download
    repo, revision = CHECKPOINT.split("@")
    return Path(snapshot_download(repo, revision=revision, allow_patterns=list(patterns)))


def load(path=None, device="cuda"):
    """Un agente de Laya listo para predecir, con los topes de contexto de este proyecto.

    `build_model` crea el encoder con pesos aleatorios justo antes de sobrescribirlos con el checkpoint;
    no_init_weights se salta ese relleno (en CPU, de ~14 s a ~1,4 s) con los mismos pesos.
    """
    import laya
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers < 5 lo exponía en modeling_utils.
        from transformers.modeling_utils import no_init_weights
    with no_init_weights():
        agent = laya.load(str(path or base_dir()), device=device)
    agent.cfg["max_len"], agent.cfg["head_max_len"] = MAX_LEN, HEAD_MAX_LEN
    return agent
