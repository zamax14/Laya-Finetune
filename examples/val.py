"""Evaluate base Laya and the 32k checkpoint on the tool-routing test set; the 32k one also at long lengths.

Same as the CLI:
    layaft val task=tool_routing model=multilingual
    layaft val task=tool_routing model=runs/tool_routing-32k ctx=32k
"""
from layaft import LayaFT

LayaFT("multilingual").val(task="tool_routing")
LayaFT("runs/tool_routing-32k").val(task="tool_routing", ctx="32k")
