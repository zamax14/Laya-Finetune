"""A judge of another family answers every generated case blind; train only on the cases where it agrees.

Same as the CLI:
    layaft verify task=tool_routing backend=ollama llm=gemma4:31b parallel=8
    layaft verify task=tool_routing backend=ollama llm=qwen3.5:122b parallel=8 data=data/tool_routing_rejected.jsonl
"""
from pathlib import Path

from layaft import LayaFT

# data/tool_routing.jsonl → data/tool_routing_verified.jsonl (kept) and data/tool_routing_rejected.jsonl (to audit)
LayaFT().verify(task="tool_routing", backend="ollama", llm="gemma4:31b", parallel=8)

# gemma sees an internal database in almost any business request. A second, larger judge re-reads only the rejected
# cases: those whose label it confirms come back (2 of 3 agree: the generator and this judge).
LayaFT().verify(task="tool_routing", backend="ollama", llm="qwen3.5:122b", parallel=8,
                data="data/tool_routing_rejected.jsonl")

# Training data: what the first judge kept plus what the second recovered.
parts = ["data/tool_routing_verified.jsonl", "data/tool_routing_rejected_verified.jsonl"]
Path("data/tool_routing_train.jsonl").write_text("".join(Path(p).read_text(encoding="utf-8") for p in parts), encoding="utf-8")
