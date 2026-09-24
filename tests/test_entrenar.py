"""Objetivos, profesor y reparto de entrenar.py, sin GPU."""
import unittest

import entrenar
from tarea import CATEGORIES, TICKETS


def case(category, priority, blocking, id_):
    return {"id": id_, "titulo": id_, "descripcion": "x", "solicitante": "a", "area": "b",
            "referencia": [category, priority, CATEGORIES[category][1]], "bloquea": blocking}


class TrainChecks(unittest.TestCase):
    def test_targets_blend_label_and_teacher_or_smooth_without_it(self):
        c = case("redes", "alta", True, "1")
        teacher = {"1": {"categoria": {"probabilities": {k: 1 / 6 for k in CATEGORIES}},
                         "prioridad": {"probabilities": {str(i): .25 for i in range(4)}}, "bloqueo": {"noul": .2}}}
        t = entrenar.targets(c, teacher)
        self.assertAlmostEqual(t["categoria"][list(CATEGORIES).index("redes")], .7 + .3 / 6)
        self.assertAlmostEqual(t["bloqueo"][1], .7 + .3 * .2)
        smooth = entrenar.targets(c, {})
        self.assertAlmostEqual(smooth["categoria"][list(CATEGORIES).index("redes")], .9)
        for dist in (*t.values(), *smooth.values()):
            self.assertAlmostEqual(sum(dist), 1)

    def test_split_drops_teacher_disagreements_and_pick_caps_per_combination(self):
        cases = [case("redes", "alta", False, str(i)) for i in range(10)]
        teacher = {"0": {"categoria": {"choice": "software"}}, "1": {"categoria": {"choice": "redes"}}}
        train, calib, val = entrenar.split(cases, teacher)
        self.assertEqual(len(train) + len(calib) + len(val), 9)
        self.assertEqual(len(entrenar.pick(cases, 2)), 2)
        self.assertNotIn(TICKETS[0]["id"], {c["id"] for c in train})


if __name__ == "__main__":
    unittest.main()
