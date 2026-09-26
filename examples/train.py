"""Fine-tune Laya multilingual on the tool-routing task, growing the context stage by stage.

Same as the CLI:
    layaft train task=tool_routing model=multilingual profile=full teacher=none
    layaft train task=tool_routing model=runs/tool_routing-1k profile=full ctx=8k teacher=none
    layaft train task=tool_routing model=runs/tool_routing-8k profile=full ctx=32k teacher=none

teacher="none": without a Jev key the constructed label is smoothed instead.
"""
from layaft import LayaFT

model = LayaFT("multilingual")
model.train(task="tool_routing", profile="full", teacher="none")              # 1,024 tokens → runs/tool_routing-1k
model.train(task="tool_routing", profile="full", teacher="none", ctx="8k")    # mmBERT's own positions → runs/tool_routing-8k
model.train(task="tool_routing", profile="full", teacher="none", ctx="32k")   # YaRN ×4 → runs/tool_routing-32k
print("Last checkpoint:", model.model)
