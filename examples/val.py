"""Evaluate Laya as shipped on the 120 hand-written tool-routing cases: the reference every stage is compared with.

Same as the CLI:
    layaft val task=tool_routing model=multilingual
"""
from layaft import LayaFT

LayaFT("multilingual").val(task="tool_routing")
