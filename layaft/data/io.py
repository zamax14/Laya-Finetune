"""JSON Lines: read and append (thread-safe). Where each file lives is the task's `data:` section."""
import json
import threading

_lock = threading.Lock()


def fillers(task, held_out=False):
    """Filler texts: a tenth is held out for evaluation, so test states never reuse training filler."""
    texts = [r["text"] for r in read(task.filler_path)]
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
