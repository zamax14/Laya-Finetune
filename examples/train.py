"""Fine-tune Laya multilingual on the helpdesk task with a 32k-token context.

Same as the CLI:
    layaft train task=helpdesk model=multilingual profile=full ctx=32k
"""
from layaft import LayaFT

model = LayaFT("multilingual")
model.train(task="helpdesk", profile="full", ctx="32k")
print("Checkpoint saved in", model.model)
