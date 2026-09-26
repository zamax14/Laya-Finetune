"""The helpdesk task and its 20 test tickets."""
import hashlib
import re
import unittest
from pathlib import Path

import yaml

from pydantic import ValidationError

from layaft.task import Task

TASK = Task.load("helpdesk")


class TaskChecks(unittest.TestCase):
    def test_twenty_tickets_with_valid_references(self):
        cases = TASK.test_cases()  # Also validates every answer against its question.
        self.assertEqual(len({c["id"] for c in cases}), 20)
        for c in cases:
            self.assertIsInstance(c["answers"]["bloqueo"], bool)
            self.assertNotIn("answers", TASK.state(c))  # The reference never reaches the model.

    def test_descriptions_do_not_leak_the_answer(self):
        # A test set that says the answer in the text measures nothing.
        category = TASK.questions["categoria"]
        for c in TASK.test_cases():
            text = c["fields"]["descripcion"].lower()
            self.assertFalse([h for h in TASK.leak_phrases if h in text], c["id"])
            if c["answers"]["categoria"]:
                self.assertFalse([w for w in category.leak_terms(c["answers"]["categoria"]) if w in text], c["id"])

    def test_combos_respect_exclude_rules(self):
        combos = TASK.combos()
        self.assertEqual(len(combos), 36)  # 6 categories × (2 levels × 1 + 2 levels × 2).
        self.assertFalse([c for c in combos if c["prioridad"] in ("baja", "media") and c["bloqueo"]])
        self.assertEqual([q.type for q in TASK.questions.values()], ["choice", "score", "noul"])

    def test_state_adds_padding_around_the_case(self):
        case = TASK.test_cases()[0]
        long = TASK.state({**case, "padding": {"before": "old thread", "after": "logs"}})
        self.assertEqual(list(long), ["history", "ticket", "attachments"])
        self.assertEqual(long["ticket"], TASK.state(case)["ticket"])

    def test_schema_rejects_thin_or_extra_fields(self):
        words = " ".join(["palabra"] * 30)
        item = {"titulo": " VPN caída ", "solicitante": "Ana", "area": "Ventas", "descripcion": words}
        self.assertEqual(TASK.schema.model_validate({"items": [item]}).items[0].titulo, "VPN caída")
        for bad in ({**item, "descripcion": "muy corta"}, {**item, "solicitante": ""}, {**item, "extra": "x"}):
            with self.assertRaises(ValidationError):
                TASK.schema.model_validate({"items": [bad]})

    def test_case_id_is_stable(self):
        # The same hash as the first generator (title + description), so the teacher's cache keeps working.
        self.assertEqual(TASK.case_id({"titulo": "VPN", "descripcion": "caída"}),
                         hashlib.sha1("VPNcaída".encode()).hexdigest()[:12])

    def test_readme_example_task_loads(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
        task = Task(yaml.safe_load(re.search(r"```yaml\n(name: invoices.*?)```", readme, re.S).group(1)))
        self.assertEqual(len(task.combos()), 9)  # 3 × 2 × 2, minus the 3 with high urgency and duplicate: true.
        self.assertEqual(task.questions["expense_type"].options["travel"].signals, "a booking, a route or a stay")


if __name__ == "__main__":
    unittest.main()
