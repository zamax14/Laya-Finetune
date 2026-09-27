"""Generate the tool-routing dataset with an Ollama model of the node.

Same as the CLI:
    layaft generate task=tool_routing backend=ollama llm=qwen3.6:35b n=150 filler=400 context=all parallel=8
"""
from layaft import LayaFT

model = LayaFT()

# 150 cases in each of the task's 40 contexts, about 6,000, in the real mix of answers (`weights` in the task),
# plus 400 neutral documents that long-context training wraps around the cases.
model.generate(task="tool_routing", backend="ollama", llm="qwen3.6:35b", n=150, filler=400, context="all", parallel=8)

# The judges keep few "ask the user" and "answer directly" cases, and fewer multi-tool than single-tool: write more.
for action, n in (("ask_user", 110), ("use_tools", 50), ("answer_directly", 30)):
    model.generate(task="tool_routing", backend="ollama", llm="qwen3.6:35b", n=n, context="all", parallel=8,
                   only={"action": [action]})
