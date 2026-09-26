"""Fine-tune Laya multilingual on the verified tool-routing cases, growing the context stage by stage.

Same as the CLI:
    layaft train task=tool_routing model=multilingual profile=full teacher=none data=data/tool_routing_verified.jsonl
    layaft train task=tool_routing model=runs/tool_routing-1k profile=full teacher=none data=data/tool_routing_verified.jsonl ctx=8k long=2000
    layaft train task=tool_routing model=runs/tool_routing-8k profile=full teacher=none data=data/tool_routing_verified.jsonl ctx=32k long=2000

teacher="none": without a Jev key the constructed label is smoothed instead (the judge already dropped the wrong ones).
long=2000: long copies of training cases wrapped in filler, on top of every short case.
"""
from layaft import LayaFT

DATA = "data/tool_routing_verified.jsonl"

model = LayaFT("multilingual")
model.train(task="tool_routing", data=DATA, profile="full", teacher="none")                          # 1,024 tokens → runs/tool_routing-1k
model.train(task="tool_routing", data=DATA, profile="full", teacher="none", ctx="8k", long=2000)     # mmBERT's own positions → runs/tool_routing-8k
model.train(task="tool_routing", data=DATA, profile="full", teacher="none", ctx="32k", long=2000)    # YaRN ×4 → runs/tool_routing-32k
print("Last checkpoint:", model.model)
