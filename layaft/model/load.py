"""Laya checkpoints: an alias (multilingual, english), a Hub repo[@revision] or a local folder."""
import json
from pathlib import Path

from layaft import config
from layaft.model.context import attention_for

FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


def resolve(model=config.DEFAULT_MODEL):
    """Local folder of a checkpoint; a Hub one is downloaded the first time (~650 MB multilingual, ~1.7 GB english)."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download
    repo, _, revision = config.MODELS.get(model, model).partition("@")
    return Path(snapshot_download(repo, revision=revision or None, allow_patterns=list(FILES)))


def position_limit(path):
    """The most tokens the checkpoint's encoder has positions for."""
    return json.loads((Path(path) / "encoder" / "config.json").read_text())["max_position_embeddings"]


def load(model=config.DEFAULT_MODEL, device="cuda", ctx=None):
    """A Laya agent ready to predict; `ctx` raises its token budget up to what the encoder supports.

    `build_model` creates the encoder with random weights right before overwriting them with the checkpoint;
    no_init_weights skips that fill (on CPU, from ~14 s to ~1.4 s) with the same weights.
    """
    import laya
    from transformers.initialization import no_init_weights

    path = resolve(model)
    if ctx and ctx > position_limit(path):
        raise ValueError(f"{model} has positions for {position_limit(path)} tokens: extend it first "
                         f"(`layaft extend model={model} ctx={ctx}`) or train with ctx={ctx}, which extends it")
    with no_init_weights():
        agent = laya.load(str(path), device=device)
    if ctx:
        agent.cfg["max_len"] = ctx
    # Above 8192 tokens sdpa builds a dense mask for the sliding-window layers (on a 6 GB GPU 16k ran out of memory);
    # flex_attention keeps it small (16k in 2.7 GB) where Triton can compile it.
    agent.model.encoder.set_attn_implementation(attention_for(agent.cfg["max_len"]))
    return agent
