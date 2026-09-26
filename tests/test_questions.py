"""The three question types: targets, normalization of other models' answers and metrics."""
import unittest

from layaft.questions import Question, ece

CHOICE = Question.create("c", {"type": "choice", "instructions": "i", "review": {"green": 80, "yellow": 60},
                               "options": {"a": {"criteria": "A", "label": "Alpha"}, "b": "B", "c": "C"}})
SCORE = Question.create("s", {"type": "score", "instructions": "i", "levels": {"low": "L", "mid": "M", "high": "H"}})
NOUL = Question.create("n", {"type": "noul", "instructions": "i", "words": {True: "sí", False: "no"}})


class QuestionChecks(unittest.TestCase):
    def test_factory_builds_laya_questions_and_rejects_unknown_types(self):
        self.assertEqual(CHOICE.laya(), {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B", "c": "C"}})
        self.assertEqual(SCORE.laya()["criteria"], ["L", "M", "H"])
        self.assertEqual(NOUL.laya(), {"type": "noul", "instructions": "i"})  # `words` only feeds the generator.
        self.assertEqual(NOUL.keys, [False, True])
        with self.assertRaises(ValueError):
            Question.create("x", {"type": "rank", "instructions": "i"})

    def test_target_blends_teacher_or_smooths_without_it(self):
        self.assertAlmostEqual(CHOICE.target("b", [1 / 3] * 3)[1], .7 + .3 / 3)
        self.assertAlmostEqual(NOUL.target(True, [.8, .2])[1], .7 + .3 * .2)
        self.assertEqual(SCORE.target("low"), [.9, .05, .05])
        for q, v in ((CHOICE, "c"), (SCORE, "high"), (NOUL, False)):
            self.assertAlmostEqual(sum(q.target(v)), 1)

    def test_normalize_fills_distribution_from_the_stated_answer(self):
        a = CHOICE.normalize({"choice": "b"})
        self.assertEqual((a["choice"], a["probabilities"]), ("b", {"a": 0, "b": 1, "c": 0}))
        s = SCORE.normalize({"probabilities": {"0": 1, "2": 3}})
        self.assertEqual(s["score"], 1.5)
        self.assertEqual(NOUL.normalize({"probability": .25})["confidence"], .75)
        with self.assertRaises(ValueError):
            NOUL.normalize({"noul": 1.5})
        self.assertEqual(CHOICE.teacher_dist(a), [0, 1, 0])

    def test_records_and_metrics(self):
        records = [CHOICE.record({"choice": "a", "confidence": .95}, "a"),
                   CHOICE.record({"choice": "b", "confidence": .7}, "a"),
                   CHOICE.record({"choice": "c", "confidence": .5}, None)]  # Ambiguous: not graded.
        m = CHOICE.metrics(records)
        self.assertEqual((m["correct"], m["total"]), (1, 2))
        self.assertEqual(m["lights"]["green"], {"correct": 1, "total": 1})
        self.assertEqual(m["lights"]["yellow"], {"correct": 0, "total": 1})
        s = SCORE.metrics([SCORE.record({"score": 1.4, "confidence": .5}, "high"),
                           SCORE.record({"score": 0, "confidence": .5}, "high")])
        self.assertEqual((s["correct"], s["near"], s["total"]), (0, 1, 2))
        n = NOUL.metrics([NOUL.record({"noul": 1.0, "confidence": 1}, True), NOUL.record({"noul": .0, "confidence": 1}, False)])
        self.assertEqual((n["correct"], n["brier"]), (2, 0))

    def test_light_thresholds_and_describe(self):
        self.assertEqual([CHOICE.light(c) for c in (100, 80.1, 80, 60, 59.9)], ["green", "green", "yellow", "yellow", "red"])
        self.assertEqual(CHOICE.describe("a").label, "Alpha")
        self.assertEqual(NOUL.describe(True).label, "sí")
        self.assertEqual(CHOICE.leak_terms("a"), ["alpha"])

    def test_ece(self):
        self.assertAlmostEqual(ece([.95] * 4, [True] * 4), .05)
        self.assertGreater(ece([.99] * 4, [False] * 4), .9)


if __name__ == "__main__":
    unittest.main()
