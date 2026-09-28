"""Generate a task's dataset with an Ollama model.

Same as the CLI:
    layaft generate task=invoices backend=ollama llm=qwen3.6:35b n=150 filler=400 context=all parallel=8
"""
from layaft import LayaFT

TASK = "invoices"  # tasks/invoices.yaml: the example task of the README, or any path to a YAML file

# 150 cases in each of the task's contexts, in the task's mix of answers (`weights`),
# plus 400 neutral documents that long-context training wraps around the cases.
LayaFT().generate(task=TASK, backend="ollama", llm="qwen3.6:35b", n=150, filler=400, context="all", parallel=8)
