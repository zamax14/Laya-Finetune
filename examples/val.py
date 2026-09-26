"""Evaluate base Laya and every stage on the 120 hand-written tool-routing cases; the long stages also at long lengths.

Same as the CLI:
    layaft val task=tool_routing model=multilingual
    layaft val task=tool_routing model=runs/tool_routing-1k
    layaft val task=tool_routing model=runs/tool_routing-8k ctx=8k
    layaft val task=tool_routing model=runs/tool_routing-32k ctx=32k
"""
from layaft import LayaFT

LayaFT("multilingual").val(task="tool_routing")
LayaFT("runs/tool_routing-1k").val(task="tool_routing")
LayaFT("runs/tool_routing-8k").val(task="tool_routing", ctx="8k")
LayaFT("runs/tool_routing-32k").val(task="tool_routing", ctx="32k")
