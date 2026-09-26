"""Fine-tune Laya multilingual on the tool-routing cases, growing the context stage by stage, and evaluate each stage.

On a cluster each stage is its own job (slurm/train_1k.sh, train_8k.sh, train_32k.sh); here, one after the other.
Same as the CLI:
    layaft train task=tool_routing model=multilingual profile=full teacher=none data=data/tool_routing_train.jsonl out=runs/tool_routing-1k
    layaft train task=tool_routing model=runs/tool_routing-1k profile=full teacher=none data=data/tool_routing_train.jsonl ctx=8k long=2000 out=runs/tool_routing-8k
    layaft train task=tool_routing model=runs/tool_routing-8k profile=full teacher=none data=data/tool_routing_train.jsonl ctx=32k long=2000 out=runs/tool_routing-32k

teacher="none": without a Jev key the label is smoothed (the judges already dropped wrong labels).
long=2000: long copies of training cases wrapped in filler, on top of every short case.
"""
from layaft import LayaFT

DATA = "data/tool_routing_train.jsonl"

model = LayaFT("multilingual")
model.train(task="tool_routing", data=DATA, profile="full", teacher="none", out="runs/tool_routing-1k")
model.val(task="tool_routing")
model.train(task="tool_routing", data=DATA, profile="full", teacher="none", ctx="8k", long=2000, out="runs/tool_routing-8k")
model.val(task="tool_routing", ctx="8k")
model.train(task="tool_routing", data=DATA, profile="full", teacher="none", ctx="32k", long=2000, out="runs/tool_routing-32k")
model.val(task="tool_routing", ctx="32k")
