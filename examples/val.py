"""Evaluate Laya as shipped on a task's hand-written test set: the reference every stage is compared with.

Same as the CLI:
    layaft val task=invoices model=multilingual
"""
from layaft import LayaFT

TASK = "invoices"  # tasks/invoices.yaml: the example task of the README, or any path to a YAML file

LayaFT("multilingual").val(task=TASK)
