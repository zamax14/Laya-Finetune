"""Longer context for a Laya checkpoint: YaRN on the encoder's global-attention layers.

mmBERT and ModernBERT alternate one global-attention layer with two local ones (a 128-token window) and were trained
up to 8192 positions. Local layers never see more than 128 positions, so only the global layers' RoPE is scaled.
Laya rebuilds the encoder from the checkpoint's `encoder/config.json`, so the extended checkpoint loads with the
official `laya.load`. The weights do not change here: `layaft train ctx=...` then teaches it long states.
"""
import json
import shutil
import warnings
from pathlib import Path

MAX_CTX, TESTED_CTX = 65536, 32768


def parse_ctx(value):
    """8192, "8192", "8k" or "8K" → 8192."""
    text = str(value).strip().lower()
    return int(float(text[:-1]) * 1024) if text.endswith("k") else int(text)


def check_ctx(ctx):
    if ctx > MAX_CTX:
        raise ValueError(f"ctx={ctx}: the limit is {MAX_CTX} tokens")
    if ctx > TESTED_CTX:
        warnings.warn(f"ctx={ctx} is experimental: stages up to {TESTED_CTX} were measured, beyond it quality and "
                      "memory are not", stacklevel=2)
    import transformers
    if ctx > 8192 and int(transformers.__version__.split(".")[0]) < 5:
        raise RuntimeError("Context above 8192 needs transformers>=5: 4.x reads only RoPE's theta and ignores YaRN")


def extend(src, ctx, out):
    """Copies the checkpoint in `src` to `out` with room for `ctx` tokens. Returns `out`."""
    check_ctx(ctx)
    src, out = Path(src), Path(out)
    if src.resolve() != out.resolve():
        shutil.copytree(src, out, dirs_exist_ok=True)  # Follows the Hub cache's symlinks: real files.
    encoder_path, agent_path = out / "encoder" / "config.json", out / "rl_agent_config.json"
    encoder = json.loads(encoder_path.read_text())
    full = encoder["rope_parameters"]["full_attention"]
    native = full.get("original_max_position_embeddings", encoder["max_position_embeddings"])
    if ctx > native:
        full.update(rope_type="yarn", factor=ctx / native, original_max_position_embeddings=native)
        encoder["max_position_embeddings"] = ctx
    encoder_path.write_text(json.dumps(encoder, indent=2))
    agent = json.loads(agent_path.read_text())
    agent["max_len"] = ctx
    agent_path.write_text(json.dumps(agent, indent=2))
    return out


def attention_for(ctx):
    """flash_attention_2 if installed; flex_attention above 8192, since sdpa builds a dense ctx² mask for the
    sliding-window layers (~2 GB per layer at 32k); sdpa otherwise, as Laya ships."""
    import importlib.util
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "flex_attention" if ctx > 8192 else "sdpa"
