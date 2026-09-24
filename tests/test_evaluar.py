"""Métricas y gráfica de evaluar.py con respuestas inventadas, sin modelo."""
import tempfile
import unittest
from pathlib import Path

import evaluar
from tarea import CATEGORIES, TICKETS


def answers(category, confidence, score, blocking):
    return {"categoria": {"choice": category, "confidence": confidence},
            "prioridad": {"score": score}, "bloqueo": {"noul": blocking}}


class EvaluateChecks(unittest.TestCase):
    def test_row_and_summary_count_hits(self):
        rows = []
        for t in TICKETS:
            category, priority, _ = t["referencia"]
            level = list(evaluar.LEVELS).index(priority)
            rows.append(evaluar.row(t, answers(category or "hardware", .95, level, float(t["bloquea"])), 10))
        s = evaluar.summarize(rows)
        self.assertEqual((s["category_correct"], s["category_total"]), (19, 19))
        self.assertEqual((s["priority_correct"], s["blocking_correct"], s["blocking_brier"]), (20, 20, 0))
        self.assertEqual(s["lights"]["verde"]["correct"], 19)
        self.assertAlmostEqual(s["category_ece"], .05)  # Acierta todo con 95 % de confianza.

    def test_wrong_and_overconfident_scores_worse(self):
        wrong = [evaluar.row(t, answers("seguridad" if t["referencia"][0] != "seguridad" else "redes", .99, 0, .5), 10)
                 for t in TICKETS]
        s = evaluar.summarize(wrong)
        self.assertEqual(s["category_correct"], 0)
        self.assertGreater(s["category_ece"], .9)

    def test_table_and_chart(self):
        rows = [evaluar.row(t, answers(next(iter(CATEGORIES)), .7, 1, .4), 5) for t in TICKETS]
        s = evaluar.summarize(rows)
        self.assertIn("| Categoría |", evaluar.table({"a": s, "b": s}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.svg"
            evaluar.chart({"a": s, "b": s}, path)
            self.assertTrue(path.read_text().startswith("<svg"))


if __name__ == "__main__":
    unittest.main()
