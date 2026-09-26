"""Generate synthetic cases for the helpdesk task with Ollama.

Same as the CLI:
    layaft generate task=helpdesk backend=ollama llm=qwen3.8:latest n=216 context=all parallel=8
"""
from layaft import LayaFT

model = LayaFT()
model.generate(task="helpdesk", backend="ollama", llm="qwen3.8:latest", n=216, context="all", parallel=8)
