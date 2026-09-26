"""Generate synthetic cases for the tool-routing task with the Ollama model of the node.

Same as the CLI:
    layaft generate task=tool_routing backend=ollama llm=qwen3.8:latest n=66 context=all parallel=8
"""
from layaft import LayaFT

model = LayaFT()

# 1. Two cases per answer combination (33) in each of the task's 16 contexts: about 1,000 cases.
model.generate(task="tool_routing", backend="ollama", llm="qwen3.8:latest", n=66, context="all", parallel=8)

# 2. "Ask the user" and "answer directly" are only 2 of the 33 combinations: add more of them.
model.generate(task="tool_routing", backend="ollama", llm="qwen3.8:latest", n=20, context="all", parallel=8,
               only={"action": ["ask_user", "answer_directly"]})

# 3. Neutral documents that long-context training wraps around the cases.
model.generate(task="tool_routing", backend="ollama", llm="qwen3.8:latest", n=0, filler=200, context="all", parallel=8)
