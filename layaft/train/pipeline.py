"""The whole fine-tune as a pipeline of short steps: each one a method, `run()` chains them.

    cases → teacher → split → model (extended if ctx needs it) → long copies → baseline → RLCD → calibration
          → comparison → checkpoint in Laya's format, checked by reloading it
"""
import json
import random
import shutil

import torch

from layaft import config
from layaft.data import io
from layaft.data.compose import LONG, LongStateBuilder, state_budget, token_counter
from layaft.evaluate import report, run
from layaft.model.context import extend
from layaft.model.load import load, position_limit, resolve
from layaft.teachers import create_teacher
from layaft.train.calibrate import calibrate
from layaft.train.rlcd import Trainer

SEED = 20260923
SPLIT = (0.8, 0.1, 0.1)  # Train, calibration, validation.
# test: 2 cases per combination and only the top 6 encoder layers (of 22) plus the head, ~45 M trainable parameters.
# full: every case and the whole model.
PROFILES = {"test": {"cases_per_combo": 2, "epochs": 1, "tokens_per_batch": 2048, "grad_accum": 4, "freeze_layers": 16,
                     "gpu_limit_gb": 3.0},
            "full": {"cases_per_combo": None, "epochs": 4, "tokens_per_batch": 4096, "grad_accum": 8, "freeze_layers": 0,
                     "gpu_limit_gb": None}}


class TrainPipeline:
    def __init__(self, task, model=config.DEFAULT_MODEL, ctx=None, profile="test", epochs=None, teacher="jev",
                 data=None, out=None, gpu_limit=None, long=600, seed=SEED):
        self.task, self.model_name, self.seed = task, model, seed
        self.cfg = {**PROFILES[profile], "profile": profile}
        self.cfg["epochs"] = epochs or self.cfg["epochs"]
        if gpu_limit is not None:
            self.cfg["gpu_limit_gb"] = gpu_limit or None
        self.source = resolve(model)
        self.ctx = ctx or json.loads((self.source / "rl_agent_config.json").read_text())["max_len"]
        self.teacher, self.long = create_teacher(teacher), long
        self.data = data or task.train_path
        self.out = out or _next_run(f"{task.name}-{self.ctx // 1024}k" + ("-test" if profile == "test" else ""))

    def run(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.check_gpu()
        cases = self.load_cases()
        teacher = self.teacher.label(cases, self.task, self.task.teacher_path)
        train, calib, val = self.split(cases, teacher)
        agent = self.load_model()
        train, calib, val, teacher = self.lengthen(agent, train, calib, val, teacher)
        trainer = Trainer(agent, self.cfg, self.seed)
        base = self.measure(agent, val)
        items = [trainer.items(d, self.task, teacher) for d in (train, calib, val)]
        lengths = [len(it["ids"]) for it in items[0]]
        print(f"{len(items[0])} sequences · tokens: median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}", flush=True)
        best_loss, improved = trainer.train(items[0], items[2])
        if not improved:
            raise SystemExit("No epoch improved validation: nothing is saved.")
        temperatures = calibrate(trainer, items[1])
        tuned = self.measure(agent, val)
        print(report.table(self.task, {"validation: base": base["validation"], "validation: tuned": tuned["validation"],
                                       "test: base": base["test"], "test: tuned": tuned["test"]}))
        report_data = {"task": self.task.name, "model": self.model_name, "ctx": self.ctx, "config": self.cfg,
                       "data": str(self.data), "teacher_answers": len(teacher),
                       "cases": {"train": len(train), "calibration": len(calib), "validation": len(val)},
                       "best_validation_ce": best_loss, "max_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       "base": base, "tuned": tuned}
        return self.save(trainer, temperatures, report_data)

    # ------------------------------------------------------------------ steps

    def check_gpu(self):
        if not torch.cuda.is_available():
            raise SystemExit("Training needs a CUDA GPU.")
        limit = self.cfg["gpu_limit_gb"]
        if limit:
            free, total = torch.cuda.mem_get_info()
            if free < limit * 2**30:
                raise SystemExit(f"Only {free / 2**30:.1f} GB free on the GPU: close other processes using it "
                                 "(Laya takes 1.6 GB; Ollama with gemma3:12b, about 9 GB).")
            # Hard cap: past it training fails with OutOfMemoryError instead of filling the GPU. The CUDA context
            # adds ~0.4 GB outside it.
            torch.cuda.set_per_process_memory_fraction((limit - 0.4) * 2**30 / total)
            print(f"GPU cap: {limit} GB ({limit - 0.4:.1f} GB for tensors) of {total / 2**30:.1f} GB")

    def load_cases(self):
        cases = self.task.read_cases(self.data)
        if not cases:
            raise SystemExit(f"No cases in {self.data}: generate them with `layaft generate task={self.task.name}`.")
        per_combo = self.cfg["cases_per_combo"]
        if per_combo:  # At most per_combo cases per combination, always the same for a seed.
            groups = {}
            for case in cases:
                groups.setdefault(json.dumps(case["answers"], sort_keys=True), []).append(case)
            rng = random.Random(self.seed)
            cases = [c for key in sorted(groups) for c in rng.sample(groups[key], min(per_combo, len(groups[key])))]
        print(f"{len(cases)} cases from {self.data}")
        return cases

    def split(self, cases, teacher):
        """Drops the cases whose choice answers the teacher does not see, and splits the rest."""
        kept = [c for c in cases if self.agrees(c, teacher)]
        graded = [c for c in cases if c["id"] in teacher]
        if graded:
            print(f"The teacher agrees with the constructed label in {sum(c in kept for c in graded)}/{len(graded)} cases; "
                  "the rest are dropped.")
        random.Random(self.seed).shuffle(kept)
        n_train, n_calib = int(SPLIT[0] * len(kept)), int(SPLIT[1] * len(kept))
        return kept[:n_train], kept[n_train:n_train + n_calib], kept[n_train + n_calib:]

    def agrees(self, case, teacher):
        t = teacher.get(case["id"])
        return not t or all(t[qid]["choice"] == case["answers"][qid]
                            for qid, q in self.task.questions.items() if q.type == "choice")

    def load_model(self):
        if self.ctx > position_limit(self.source):
            self.source = extend(self.source, self.ctx, self.out / "extended")
            print(f"Extended {self.model_name} to {self.ctx} tokens with YaRN")
        agent = load(self.source, device="cuda", ctx=self.ctx)
        print(f"{self.model_name} · ctx {self.ctx} · attention {agent.model.encoder.config._attn_implementation}")
        return agent

    def lengthen(self, agent, train, calib, val, teacher):
        """Above LONG: `long` long copies of train cases (a tenth for calibration and validation), graded by the
        teacher too. The short cases stay, so short states are not forgotten."""
        if self.ctx <= LONG:
            return train, calib, val, teacher
        builder = LongStateBuilder(self.task, io.fillers(self.task), token_counter(agent.tok))
        rng, budget = random.Random(self.seed), state_budget(agent)
        splits = [builder.spread(rng.sample(cases, min(n, len(cases))), budget, rng.randrange(2**32))
                  for cases, n in ((train, self.long), (calib, self.long // 10), (val, self.long // 10))]
        teacher = {**teacher, **self.teacher.label([c for s in splits for c in s], self.task, self.task.teacher_path)}
        long_train, long_calib, long_val = ([c for c in s if self.agrees(c, teacher)] for s in splits)
        print(f"Long copies up to {budget} tokens: train {len(long_train)} · calibration {len(long_calib)} · validation {len(long_val)}")
        return train + long_train, calib + long_calib, val + long_val, teacher

    def measure(self, agent, val):
        """Validation, the hand-written test set, and above LONG the test set at 8k/16k/32k/64k up to ctx."""
        out = {"validation": run.evaluate(agent, val, self.task)[0], "test": run.evaluate(agent, self.task.test_cases(), self.task)[0]}
        if self.ctx > LONG:
            out["test_by_length"] = run.by_ctx(agent, self.task.test_cases(), self.task)
        return out

    def save(self, trainer, temperatures, report_data):
        from safetensors.torch import save_file
        out = self.out
        # Copies files, not the Hub cache's links, except the ones written below.
        shutil.copytree(self.source, out, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("model.safetensors", "rl_agent_config.json", "extended"))
        save_file({k: v.contiguous() for k, v in trainer.snapshot().items()}, out / "model.safetensors")
        agent = trainer.agent
        cfg = {**agent.cfg, "fine_tuned": True, "model_name": f"laya-{self.task.name}", "temperature": temperatures,
               "fine_tuned_from": config.MODELS.get(self.model_name, str(self.model_name))}
        cfg.pop("temperature_by_options", None)
        (out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
        (out / "results.json").write_text(json.dumps(report_data, ensure_ascii=False, indent=2))
        shutil.rmtree(out / "extended", ignore_errors=True)
        # Check: the saved checkpoint loads with Laya and decides like the model in memory.
        state = self.task.state(self.task.test_cases()[0])
        qid = next(iter(self.task.laya))
        a = agent.predict(state, self.task.laya)["answers"][qid]
        agent.model.to("cpu")  # Two copies on the GPU would pass the test profile's cap.
        torch.cuda.empty_cache()
        b = load(out, device="cuda").predict(state, self.task.laya)["answers"][qid]
        assert a == b, f"The reloaded checkpoint decides differently: {a} against {b}"
        print("Saved in", out, "and checked by reloading it")
        return out


def _next_run(name):
    """runs/<name>, runs/<name>-2, …: a new run never overwrites an old one."""
    path, i = config.RUNS / name, 2
    while path.exists():
        path, i = config.RUNS / f"{name}-{i}", i + 1
    return path
