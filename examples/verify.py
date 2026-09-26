"""A judge of another family answers every generated case blind; train only on the cases where it agrees.

Same as the CLI:
    layaft verify task=tool_routing backend=ollama llm=gemma4:31b parallel=8
"""
from layaft import LayaFT

# data/tool_routing.jsonl → data/tool_routing_verified.jsonl (kept) and data/tool_routing_rejected.jsonl (to audit)
LayaFT().verify(task="tool_routing", backend="ollama", llm="gemma4:31b", parallel=8)
