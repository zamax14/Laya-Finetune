"""A judge of another family answers every generated case blind; train only on the cases where it agrees.

Same as the CLI:
    layaft verify task=invoices backend=ollama llm=gemma4:31b parallel=8
    layaft verify task=invoices backend=ollama llm=qwen3.5:122b parallel=8 data=data/invoices_rejected.jsonl
"""
from pathlib import Path

from layaft import LayaFT

TASK = "invoices"  # tasks/invoices.yaml: the example task of the README, or any path to a YAML file

# data/invoices.jsonl → data/invoices_verified.jsonl (kept) and data/invoices_rejected.jsonl (to audit)
LayaFT().verify(task=TASK, backend="ollama", llm="gemma4:31b", parallel=8)

# A second, larger judge re-reads only the rejected cases. Where it confirms the label, the case comes back; where it
# answers exactly like the first judge, the two judges' answer becomes the label (_relabelled): the text was written
# for one answer and reads as another, the hard cases.
LayaFT().verify(task=TASK, backend="ollama", llm="qwen3.5:122b", parallel=8, data=f"data/{TASK}_rejected.jsonl")

# Training data: what the first judge kept, plus what the second confirmed or relabelled with the first.
parts = [f"data/{TASK}_verified.jsonl", f"data/{TASK}_rejected_verified.jsonl", f"data/{TASK}_rejected_relabelled.jsonl"]
Path(f"data/{TASK}_train.jsonl").write_text("".join(Path(p).read_text(encoding="utf-8") for p in parts), encoding="utf-8")
