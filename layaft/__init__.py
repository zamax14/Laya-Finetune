"""Laya Finetune: fine-tune Laya, the open System One decision model, for any typed-decision task.

    from layaft import LayaFT
    m = LayaFT("multilingual")                                  # or "english", a Hub repo, a local checkpoint
    m.generate(task="helpdesk", backend="ollama", n=216)        # synthetic cases labelled by construction
    m.verify(task="helpdesk", llm="gemma4:31b")                 # a second LLM drops the mislabelled ones
    m.train(task="helpdesk", ctx="16k", profile="full")          # RLCD + calibration → runs/helpdesk-16k
    m.val(task="helpdesk")
    m.predict({"ticket": "The VPN drops every hour"}, task="helpdesk")

The same modes from the shell: `layaft train task=helpdesk ctx=16k profile=full`.
"""
from layaft import config  # First: sets HF_HOME and TORCH_DISABLE_NATIVE_JIT before torch is imported.
from layaft.model.context import parse_ctx
from layaft.task import Task

__all__ = ["LayaFT", "Task"]


def _task(task):
    return task if isinstance(task, Task) else Task.load(task)


def _ctx(ctx):
    return parse_ctx(ctx) if ctx else None


class LayaFT:
    """Facade: every mode delegates to its module. After `train` or `extend`, `self.model` is the new checkpoint."""

    def __init__(self, model=config.DEFAULT_MODEL):
        self.model = str(model)

    def generate(self, task, backend="ollama", n=72, context=None, filler=0, llm=None, api_key=None, base_url=None,
                 ollama_url=None, parallel=None, only=None, seed=None):
        """About `n` cases per context; `context="all"` walks the task's contexts file. `filler=N` also writes N
        neutral documents (spread over the contexts) for long-context training."""
        from layaft.backends import create_backend
        from layaft.data.generate import generate, generate_filler
        task = _task(task)
        llm_ = create_backend(backend, llm, api_key, base_url, ollama_url, parallel)
        contexts = task.contexts() if context == "all" else [context or task.default_context]
        total, cost = 0, 0.0
        for i, ctx in enumerate(contexts, 1):
            rows, spent = generate(task, llm_, n, ctx, seed=seed, only=only) if n else ([], 0.0)
            docs, spent_filler = generate_filler(task, llm_, -(-filler // len(contexts)), ctx) if filler else ([], 0.0)
            total, cost = total + len(rows), cost + spent + spent_filler
            print(f"[{i}/{len(contexts)}] {len(rows)} cases and {len(docs)} filler documents · {ctx}", flush=True)
        print(f"Total: {total} cases" + (f" for US${cost:.2f}" if cost else ""))
        return total

    def verify(self, task, llm, backend="ollama", data=None, api_key=None, base_url=None, ollama_url=None, parallel=None):
        """A second LLM (another family than the generator) answers every case blind; the cases where it agrees with
        the label go to <data>_verified.jsonl, the rest to <data>_rejected.jsonl. Run on a rejected file, the cases
        where it answers like the first judge go to <data>_relabelled.jsonl. Returns how many were kept or relabelled."""
        from pathlib import Path
        from layaft.backends import create_backend
        from layaft.data.verify import verify
        judge = create_backend(backend, llm, api_key, base_url, ollama_url, parallel)
        kept, relabelled, _ = verify(_task(task), judge, data and Path(data))
        return len(kept) + len(relabelled)

    def train(self, task, ctx=None, profile="test", epochs=None, teacher="jev", data=None, out=None, gpu_limit=None,
              long=600):
        """RLCD fine-tune; ctx above the checkpoint's positions extends it with YaRN first. Returns the checkpoint."""
        from pathlib import Path
        from layaft.train.pipeline import TrainPipeline
        self.model = str(TrainPipeline(_task(task), self.model, _ctx(ctx), profile, epochs, teacher,
                                       data and Path(data), out and Path(out), gpu_limit, long).run())
        return self.model

    def val(self, task, ctx=None, device="cuda", save=True):
        """The hand-written test set, saved in runs/val/<task>-<model>/; with ctx above 2048 (compose.LONG), also wrapped in filler at 8k/16k/32k/64k up to ctx."""
        import json
        from layaft.data.compose import LONG
        from layaft.evaluate import report, run
        from layaft.model.load import load
        task, ctx = _task(task), _ctx(ctx)
        agent = load(self.model, device, ctx)
        cases = task.test_cases()
        summary, rows = run.evaluate(agent, cases, task)
        out = {"model": self.model, "fingerprint": task.fingerprint(cases), "summary": summary, "rows": rows}
        if ctx and ctx > LONG:
            out["by_length"] = run.by_ctx(agent, cases, task)
            print(report.table(task, out["by_length"]))
        else:
            print(report.table(task, {self.model: summary}))
        if save:
            from pathlib import Path
            path = config.RUNS / "val" / f"{task.name}-{Path(self.model).name.split('@')[0]}"
            path.mkdir(parents=True, exist_ok=True)
            (path / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
            report.chart(task, out.get("by_length") or {self.model: summary}, path / "chart.svg",
                         subtitle=f"{len(cases)} hand-written test cases, never trained on", footer=f"fingerprint {out['fingerprint']}")
            print("Saved in", path)
        return out

    def predict(self, state, task, ctx=None, device="cuda", fields=None):
        """Typed answers for one state (a string or a dict), with the task's questions; `fields` fills the templated
        ones, e.g. {"name": "git-commits", ...} for a question on one candidate."""
        from layaft.model.load import load
        if getattr(self, "_loaded", (None,))[0] != self.model:  # Reloaded after train/extend change the model.
            self._loaded = (self.model, load(self.model, device, _ctx(ctx)))
        task = _task(task)
        return self._loaded[1].predict(state, task.laya_for({"fields": fields or {}}))["answers"]

    def extend(self, ctx, out=None):
        """A copy of the checkpoint with room for ctx tokens (YaRN); train it with `train(ctx=...)`."""
        from pathlib import Path
        from layaft.model.context import extend
        from layaft.model.load import resolve
        ctx = _ctx(ctx)
        name = Path(self.model).name if Path(self.model).is_dir() else self.model.split("/")[-1].split("@")[0]
        self.model = str(extend(resolve(self.model), ctx, Path(out) if out else config.RUNS / f"{name}-{ctx // 1024}k"))
        print("Extended checkpoint in", self.model)
        return self.model
