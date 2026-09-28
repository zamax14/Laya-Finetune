"""Evaluation and report with a fake agent: no model, no GPU."""
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from layaft.data.compose import LongStateBuilder
from layaft.evaluate import report, run
from layaft.task import Task
from tests import HELPDESK

TASK = Task.load(HELPDESK)
LEVELS = TASK.questions["prioridad"].keys


class FakeAgent:
    """Answers every case with its own reference (or fixed wrong answers), at a fixed confidence."""
    model = SimpleNamespace(eval=lambda: None)

    def __init__(self, cases, right=True, confidence=.95):
        self.by_state = {str(TASK.state(c)["ticket"]): c for c in cases}
        self.right, self.confidence = right, confidence

    def predict(self, state, questions):
        ref = self.by_state[state["ticket"]]["answers"]
        category = ref["categoria"] or "hardware"
        if not self.right:
            category = "seguridad" if category != "seguridad" else "redes"
        return {"answers": {"categoria": {"choice": category, "confidence": self.confidence},
                            "prioridad": {"score": LEVELS.index(ref["prioridad"]) if self.right else 0, "confidence": .5},
                            "bloqueo": {"noul": float(ref["bloqueo"]) if self.right else .5, "confidence": .5}}}


class EvaluateChecks(unittest.TestCase):
    def test_summary_counts_hits(self):
        cases = TASK.test_cases()
        s, rows = run.evaluate(FakeAgent(cases), cases, TASK)
        q = s["questions"]
        self.assertEqual((q["categoria"]["correct"], q["categoria"]["total"]), (19, 19))
        self.assertEqual((q["prioridad"]["correct"], q["bloqueo"]["correct"], q["bloqueo"]["brier"]), (20, 20, 0))
        self.assertEqual(q["categoria"]["lights"]["green"]["correct"], 19)
        self.assertAlmostEqual(q["categoria"]["ece"], .05)  # Right every time with 95 % confidence.
        self.assertEqual(len(rows), 20)
        self.assertEqual((s["all_correct"], s["total"]), (20, 20))  # The ticket without a category counts on the rest.

    def test_wrong_and_overconfident_scores_worse(self):
        cases = TASK.test_cases()
        s, _ = run.evaluate(FakeAgent(cases, right=False, confidence=.99), cases, TASK)
        self.assertEqual(s["questions"]["categoria"]["correct"], 0)
        self.assertEqual(s["all_correct"], 0)
        self.assertGreater(s["questions"]["categoria"]["ece"], .9)

    def test_by_length_keeps_the_answers(self):
        cases = TASK.test_cases()[:6]
        builder = LongStateBuilder(TASK, [" ".join(["relleno"] * 300)] * 3, lambda t: len(t.split()))
        out = run.by_length(FakeAgent(cases), cases, TASK, builder, [1500])
        self.assertEqual(list(out), ["short", "1500"])
        self.assertEqual(out["1500"]["questions"]["prioridad"]["correct"], 6)

    def test_table_and_chart(self):
        cases = TASK.test_cases()
        s, _ = run.evaluate(FakeAgent(cases), cases, TASK)
        text = report.table(TASK, {"a": s, "b": s})
        self.assertIn("| Category | 19/19 | 19/19 |", text)
        self.assertIn("| Priority ±1 |", text)
        self.assertIn("| All answers right | 20/20 | 20/20 |", text)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.svg"
            report.chart(TASK, {"a": s, "b": s}, path)
            self.assertTrue(path.read_text().startswith("<svg"))


if __name__ == "__main__":
    unittest.main()
