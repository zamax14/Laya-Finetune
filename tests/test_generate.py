"""The generator with a fake LLM: spread, filters and JSONL."""
import json
import random
import tempfile
import unittest
from pathlib import Path

from layaft.data import generate as gen
from layaft.data import io
from layaft.task import Task
from tests import HELPDESK

TASK = Task.load(HELPDESK)
WORDS = " ".join(["palabra"] * 30)


class FakeLLM:
    model, parallel = "fake", 1

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, schema):
        self.calls += 1
        good = [{"titulo": f"Caso {self.calls}-{i}", "solicitante": "Ana Ruiz", "area": "Ventas",
                 "descripcion": f"Desde ayer falla algo en mi puesto ({self.calls}-{i}). {WORDS}"} for i in range(3)]
        bad = [{"titulo": "El portátil no enciende", "solicitante": "x", "area": "x", "descripcion": WORDS},  # A test title.
               {"titulo": "Pista", "solicitante": "x", "area": "x", "descripcion": "Esto no es un fallo de red. " + WORDS}]
        return json.dumps({"items": good + bad}), 0.001


class GenerateChecks(unittest.TestCase):
    def test_generates_balanced_rows_without_leaks_and_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            rows, cost = gen.generate(TASK, FakeLLM(), 72, "tickets de un hospital", path, seed=1)
            self.assertEqual(len(rows), 72)
            self.assertEqual({r["answers"]["categoria"] for r in rows}, set(TASK.questions["categoria"].keys))
            self.assertFalse({r["fields"]["titulo"] for r in rows} & {"El portátil no enciende", "Pista"})
            self.assertGreater(cost, 0)
            cases = io.read(path)
            self.assertEqual(len(cases), 72)
            self.assertEqual(cases[0]["id"], TASK.case_id(cases[0]["fields"]))
            self.assertFalse(any(c["answers"]["bloqueo"] for c in cases if c["answers"]["prioridad"] in ("baja", "media")))
            gen.generate(TASK, FakeLLM(), 36, "otro contexto", path, seed=2)
            self.assertEqual(len(io.read(path)), 108)

    def test_only_limits_the_combinations_and_prompt_demands_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = gen.generate(TASK, FakeLLM(), 12, "banco", Path(tmp) / "c.jsonl", seed=3,
                                   only={"categoria": ["seguridad"]})
        self.assertEqual({r["answers"]["categoria"] for r in rows}, {"seguridad"})
        self.assertEqual(len(rows), 12)
        combo = {"categoria": "seguridad", "prioridad": "alta", "bloqueo": False}
        prompt = gen.prompt_for(TASK, combo, "banco", [], random.Random(0), 5)
        self.assertIn(TASK.questions["categoria"].options["seguridad"].signals, prompt)

    def test_weights_follow_the_rules_and_vary_picks_an_option(self):
        spec = {**TASK.spec, "generation": {**TASK.spec["generation"], "prompt": "{n} {tono}",
                                            "weights": [{"categoria": "seguridad", "weight": 4},
                                                        {"bloqueo": True, "weight": 0.5}],
                                            "vary": {"tono": ["seco", "amable"]}}}
        task = Task(spec, TASK.root)
        self.assertEqual(task.weight({"categoria": "seguridad", "prioridad": "alta", "bloqueo": True}), 2.0)
        self.assertEqual(task.weight({"categoria": "redes", "prioridad": "alta", "bloqueo": False}), 1.0)
        prompts = {gen.prompt_for(task, {}, "banco", [], random.Random(i), 5) for i in range(20)}
        self.assertEqual(prompts, {"5 seco", "5 amable"})
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = gen.generate(task, FakeLLM(), 200, "banco", Path(tmp) / "c.jsonl", seed=4)
        share = sum(r["answers"]["categoria"] == "seguridad" for r in rows) / len(rows)
        self.assertAlmostEqual(share, 4 / 9, delta=0.05)  # Weight 4 against 1 for each of the other 5 categories.

    def test_pool_fields_come_from_the_file_and_the_llm_writes_the_rest(self):
        class RequestLLM(FakeLLM):
            prompts = []

            def __call__(self, prompt, schema):
                self.prompts.append(prompt)
                assert list(schema["$defs"]["Item"]["properties"]) == ["request"]
                return json.dumps({"items": [{"request": f"pedido {len(self.prompts)}-{i}"} for i in range(2)]}), 0.0

        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog.jsonl"
            io.append(catalog, [{"name": "git-commits", "description": "commit rules"},
                                {"name": "find-docs", "description": "library docs"}])
            (Path(tmp) / "t.jsonl").write_text("")
            task = Task({"name": "t", "fields": {"request": {}, "name": {}, "description": {}},
                         "state": {"request": "{request}"},
                         "questions": {"needed": {"type": "noul", "instructions": "Is «{name}» useful?"}},
                         "generation": {"text_field": "request", "id_fields": ["request", "name"], "per_call": 2,
                                        "pool": {"file": "catalog.jsonl", "fields": ["name", "description"]},
                                        "prompt": "{n} requests where «{name}» ({description}) → {needed.label}"},
                         "data": {"test": "t.jsonl"}}, tmp)
            llm = RequestLLM()
            rows, _ = gen.generate(task, llm, 8, "dev", Path(tmp) / "c.jsonl", seed=5)
        self.assertEqual(len(rows), 8)
        for r in rows:  # Each row keeps the item of the prompt that wrote it ("pedido <call>-<i>").
            prompt = llm.prompts[int(r["fields"]["request"].split()[1].split("-")[0]) - 1]
            self.assertIn(f"«{r['fields']['name']}» ({r['fields']['description']})", prompt)
            self.assertEqual(set(r["fields"]), {"request", "name", "description"})

    def test_default_prompt_for_a_task_without_one(self):
        task = Task({**TASK.spec, "generation": {**TASK.spec["generation"], "prompt": None}})
        prompt = gen.prompt_for(task, {"categoria": "redes", "prioridad": "alta", "bloqueo": True}, "banco", [],
                                random.Random(0), 3)
        self.assertIn("Write 3 distinct texts", prompt)
        self.assertIn("→ Redes (network:", prompt)
        self.assertIn("For area, one of:", prompt)

    def test_leaks_detects_hints_and_answer_names(self):
        combo = {"categoria": "seguridad", "prioridad": "alta", "bloqueo": False}
        self.assertTrue(gen.leaks(TASK, "Parece un problema de Seguridad", combo))
        self.assertTrue(gen.leaks(TASK, "no hay indicios de ataque", {**combo, "categoria": "correo"}))
        self.assertFalse(gen.leaks(TASK, "La VPN se desconecta cada hora", {**combo, "categoria": "redes"}))

    def test_filler_keeps_long_documents_only(self):
        class FillerLLM(FakeLLM):
            def __call__(self, prompt, schema):
                return json.dumps({"items": [{"text": " ".join(["log"] * 150)}, {"text": "corto"}]}), 0.0

        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = gen.generate_filler(TASK, FillerLLM(), 6, path=Path(tmp) / "f.jsonl")
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
