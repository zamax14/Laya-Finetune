"""Long states: length, position and the case kept intact."""
import json
import random
import unittest

from layaft.data.compose import LongStateBuilder
from layaft.task import Task

TASK = Task.load("helpdesk")
count = lambda text: len(text.split())  # Words stand in for tokens.
FILLERS = [" ".join(f"doc{i}w{j}" for j in range(200)) for i in range(10)]


class ComposeChecks(unittest.TestCase):
    def test_reaches_the_length_without_passing_it_and_keeps_the_case(self):
        builder, case = LongStateBuilder(TASK, FILLERS, count), TASK.test_cases()[0]
        for position in ("start", "middle", "end"):
            long = builder.build(case, 3000, position, random.Random(0))
            size = count(json.dumps(TASK.state(long), ensure_ascii=False))
            self.assertLessEqual(size, 3000)
            self.assertGreater(size, 2800)
            self.assertEqual(TASK.state(long)["ticket"], TASK.state(case)["ticket"])
            self.assertEqual(long["answers"], case["answers"])
            self.assertNotEqual(long["id"], case["id"])
        self.assertEqual(builder.build(case, 3000, "start", random.Random(0))["padding"]["before"], "")
        self.assertEqual(builder.build(case, 3000, "end", random.Random(0))["padding"]["after"], "")

    def test_shorter_target_than_the_case_adds_nothing(self):
        long = LongStateBuilder(TASK, FILLERS, count).build(TASK.test_cases()[0], 10, "middle", random.Random(0))
        self.assertFalse(any(long["padding"].values()))

    def test_spread_is_deterministic_and_needs_filler(self):
        builder, cases = LongStateBuilder(TASK, FILLERS, count), TASK.test_cases()[:5]
        self.assertEqual(builder.spread(cases, 2000, seed=1, min_tokens=500), builder.spread(cases, 2000, seed=1, min_tokens=500))
        with self.assertRaises(ValueError):
            LongStateBuilder(TASK, [], count)


if __name__ == "__main__":
    unittest.main()
