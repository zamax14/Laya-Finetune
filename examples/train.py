"""Fine-tune Laya multilingual on a task's verified cases, then grow its context to 8k and to 32k, and evaluate each.

8k and 32k both start from the 1k checkpoint, so with two GPUs they can run in parallel; here, one after the other.
Same as the CLI:
    layaft train task=invoices model=multilingual profile=full teacher=none data=data/invoices_train.jsonl out=runs/invoices-1k
    layaft train task=invoices model=runs/invoices-1k profile=full teacher=none data=data/invoices_train.jsonl ctx=8k long=2000 out=runs/invoices-8k
    layaft train task=invoices model=runs/invoices-1k profile=full teacher=none data=data/invoices_train.jsonl ctx=32k long=2000 out=runs/invoices-32k

teacher="none": the label is smoothed instead of graded by Jev (the judges already dropped wrong labels).
long=2000: long copies of training cases wrapped in filler, on top of every short case.
"""
from layaft import LayaFT

TASK = "invoices"  # tasks/invoices.yaml: the example task of the README, or any path to a YAML file
DATA = f"data/{TASK}_train.jsonl"

model = LayaFT("multilingual")
model.train(task=TASK, data=DATA, profile="full", teacher="none", out=f"runs/{TASK}-1k")
model.val(task=TASK)
for ctx in ("8k", "32k"):
    model = LayaFT(f"runs/{TASK}-1k")
    model.train(task=TASK, data=DATA, profile="full", teacher="none", ctx=ctx, long=2000, out=f"runs/{TASK}-{ctx}")
    model.val(task=TASK, ctx=ctx)
