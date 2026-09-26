"""Targets, batching and the pipeline's data steps, without GPU or downloads."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from layaft.task import Task
from layaft.train import pipeline
from layaft.train.rlcd import batches, targets

TASK = Task.load("helpdesk")


def case(i, categoria="redes", prioridad="alta", bloqueo=True):
    fields = {"titulo": f"caso {i}", "solicitante": "Ana", "area": "Ventas", "descripcion": "texto"}
    return {"id": str(i), "fields": fields, "answers": {"categoria": categoria, "prioridad": prioridad, "bloqueo": bloqueo}}


def fake_pipeline(tmp, **kw):
    ckpt = Path(tmp) / "ckpt"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "rl_agent_config.json").write_text(json.dumps({"max_len": 1024}))
    with mock.patch.object(pipeline.config, "RUNS", Path(tmp) / "runs"):
        return pipeline.TrainPipeline(TASK, model=str(ckpt), teacher="none", **kw)


class TrainChecks(unittest.TestCase):
    def test_targets_blend_the_teacher_or_smooth_without_it(self):
        teacher = {"1": {"categoria": {"probabilities": {k: 1 / 6 for k in TASK.questions["categoria"].keys}},
                         "prioridad": {"probabilities": {str(i): .25 for i in range(4)}}, "bloqueo": {"noul": .2}}}
        t = targets(TASK, case(1), teacher)
        self.assertAlmostEqual(t["categoria"][TASK.questions["categoria"].index("redes")], .7 + .3 / 6)
        self.assertAlmostEqual(t["bloqueo"][1], .7 + .3 * .2)
        self.assertAlmostEqual(targets(TASK, case(1), {})["categoria"][2], .9)

    def test_batches_respect_the_token_budget_and_keep_every_item(self):
        items = [{"ids": [0] * n} for n in (100, 120, 500, 30000, 90)]
        out = batches(items, 1000, random.Random(0))
        self.assertEqual(sorted(len(it["ids"]) for b in out for it in b), [90, 100, 120, 500, 30000])
        for b in out:
            self.assertTrue(len(b) == 1 or len(b) * max(len(it["ids"]) for it in b) <= 1000)

    def test_split_drops_teacher_disagreements(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = fake_pipeline(tmp)
            cases = [case(i) for i in range(10)]
            teacher = {"0": {"categoria": {"choice": "software"}}, "1": {"categoria": {"choice": "redes"}}}
            train, calib, val = p.split(cases, teacher)
            self.assertEqual(len(train) + len(calib) + len(val), 9)
            self.assertEqual((len(train), len(calib)), (7, 0))

    def test_test_profile_caps_cases_per_combination_and_runs_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = fake_pipeline(tmp)
            self.assertEqual((p.ctx, p.out.name), (1024, "helpdesk-1k-test"))
            p.data = Path(tmp) / "cases.jsonl"
            p.data.write_text("".join(json.dumps(case(i, prioridad=("alta", "baja")[i % 2], bloqueo=False)) + "\n" for i in range(10)))
            self.assertEqual(len(p.load_cases()), 4)  # 2 combinations × 2.
            p.out.mkdir(parents=True)
            self.assertEqual(fake_pipeline(tmp, ctx=1024).out.name, "helpdesk-1k-test-2")


if __name__ == "__main__":
    unittest.main()
