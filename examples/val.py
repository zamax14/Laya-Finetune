"""Evaluate a fine-tuned checkpoint on the helpdesk test set, also at long lengths.

Same as the CLI:
    layaft val task=helpdesk model=runs/helpdesk-32k ctx=32k
"""
from layaft import LayaFT

model = LayaFT("runs/helpdesk-32k")
model.val(task="helpdesk", ctx="32k")
