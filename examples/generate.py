"""Generate the tool-routing dataset with an Ollama model of the node.

Same as the CLI:
    layaft generate task=tool_routing backend=ollama llm=qwen3.6:35b n=150 filler=400 context=all parallel=8
"""
from layaft import LayaFT

model = LayaFT()

# 150 cases in each of the task's 40 contexts, about 6,000, in the real mix of answers (`weights` in the task),
# plus 400 neutral documents that long-context training wraps around the cases.
model.generate(task="tool_routing", backend="ollama", llm="qwen3.6:35b", n=150, filler=400, context="all", parallel=8)

# The judge keeps about 1 in 5 "ask the user" cases and fewer multi-tool ones than single-tool: write more of both.
model.generate(task="tool_routing", backend="ollama", llm="qwen3.6:35b", n=50, context="all", parallel=8,
               only={"action": ["ask_user"]})
model.generate(task="tool_routing", backend="ollama", llm="qwen3.6:35b", n=50, context="all", parallel=8,
               only={"action": ["use_tools"]})
