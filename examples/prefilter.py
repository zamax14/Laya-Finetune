"""The context-prefilter flow from Python, one step per argument, the same as slurm/prefilter/<step>.sh:

    python examples/prefilter.py synthesize generate pair verify train_1k train_8k

A Laya that tells Claude Code which installed skills, rules, agents and MCP tools a request needs (claude-decide).
"""
import subprocess
import sys
from pathlib import Path

from layaft import LayaFT

TASK, HELD_OUT_POOL, HELD_OUT = "context_prefilter", "tasks/prefilter/pool_heldout.jsonl", "data/prefilter_heldout.jsonl"
JUDGES = ("gemma4:31b", "qwen3.5:122b")


def synthesize():
    subprocess.run([sys.executable, "tasks/prefilter/synthesize.py", "qwen3.6:35b"], check=True)


def generate():
    ft = LayaFT()
    ft.generate(task=TASK, llm="qwen3.6:35b", n=300, context="all", parallel=8)
    ft.generate(task=TASK, llm="qwen3.6:35b", n=20, context="all", parallel=8, pool=HELD_OUT_POOL, out=HELD_OUT)
    ft.generate(task=TASK, llm="qwen3.6:35b", n=0, filler=400, context="all", parallel=8)


def pair():
    LayaFT().pair(task=TASK, near=2, random=5)
    LayaFT().pair(task=TASK, near=2, random=5, data=HELD_OUT, pool=HELD_OUT_POOL)


def verify():
    for data in ("data/context_prefilter", "data/context_prefilter_near", "data/prefilter_heldout", "data/prefilter_heldout_near"):
        LayaFT().verify(task=TASK, llm=JUDGES[0], parallel=8, data=f"{data}.jsonl")
        LayaFT().verify(task=TASK, llm=JUDGES[1], parallel=8, data=f"{data}_rejected.jsonl")
    for stem, out in (("data/context_prefilter", "data/context_prefilter_train.jsonl"),
                      ("data/prefilter_heldout", "tasks/prefilter/test_heldout.jsonl")):
        parts = [f"{stem}{near}{kind}.jsonl" for near in ("", "_near")
                 for kind in ("_verified", "_rejected_verified", "_rejected_relabelled")] + [f"{stem}_random.jsonl"]
        Path(out).write_text("".join(Path(p).read_text(encoding="utf-8") for p in parts), encoding="utf-8")


def train_1k():
    model = LayaFT("multilingual")
    model.train(task=TASK, profile="full", teacher="none", data="data/context_prefilter_train.jsonl",
                out="runs/context_prefilter-1k")
    model.val(task=TASK)


def train_8k():
    model = LayaFT("runs/context_prefilter-1k")
    model.train(task=TASK, profile="full", teacher="none", data="data/context_prefilter_train.jsonl", ctx="8k",
                long=2000, out="runs/context_prefilter-8k")
    model.val(task=TASK, ctx="8k")


if __name__ == "__main__":
    steps = {f.__name__: f for f in (synthesize, generate, pair, verify, train_1k, train_8k)}
    for step in sys.argv[1:]:
        steps[step]()
