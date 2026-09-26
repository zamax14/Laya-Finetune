"""Where a task's data lives and how it is read and appended: one JSON object per line."""
import json
import threading

from layaft import config

_lock = threading.Lock()


def cases_path(task):
    return config.DATA / f"{task.name}.jsonl"


def teacher_path(task):
    return config.DATA / f"{task.name}_teacher.jsonl"


def filler_path(task):
    return config.DATA / f"{task.name}_filler.jsonl"


def fillers(task, held_out=False):
    """Filler texts: a tenth is held out for evaluation, so test states never reuse training filler."""
    texts = [r["text"] for r in read(filler_path(task))]
    return texts[::10] if held_out else [t for i, t in enumerate(texts) if i % 10]


def read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path, rows):
    """Thread-safe append: the generator writes each answer combination as soon as it is done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
