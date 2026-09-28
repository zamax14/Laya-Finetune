"""The judge with a fake LLM: agreement splits the cases, and judged cases are not asked again."""
import json
import tempfile
import unittest
from pathlib import Path

from layaft.data import io
from layaft.data.verify import judge_schema, outputs, verify
from layaft.task import Task
from tests import HELPDESK

TASK = Task.load(HELPDESK)
CASES = [c for c in TASK.test_cases() if None not in c["answers"].values()]


class FakeJudge:
    """Agrees with the label except when the ticket says 'MARK', where it answers bloqueo=false."""
    model, parallel, calls = "fake", 1, 0

    def __init__(self, labels):
        self.labels = labels

    def __call__(self, prompt, schema):
        self.calls += 1
        ticket = json.loads(prompt.split("State:\n")[1].split("\n\nQuestions")[0])["ticket"]
        answers = dict(self.labels[ticket])
        if "MARK" in ticket:
            answers["bloqueo"] = False
        return json.dumps(answers), 0.0


def labels(cases):
    return {TASK.state(c)["ticket"]: c["answers"] for c in cases}


class VerifyChecks(unittest.TestCase):
    def test_splits_by_agreement_and_skips_judged_cases(self):
        cases = CASES[:4]
        cases[0] = {**cases[0], "fields": {**cases[0]["fields"], "titulo": "MARK"}}
        cases[0]["id"] = TASK.case_id(cases[0]["fields"])
        cases[0]["answers"] = {**cases[0]["answers"], "prioridad": "alta", "bloqueo": True}
        judge = FakeJudge(labels(cases))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "helpdesk.jsonl"
            io.append(path, cases)
            kept, relabelled, rejected = verify(TASK, judge, path)
            self.assertEqual((len(kept), len(relabelled), len(rejected)), (3, 0, 1))
            self.assertFalse(rejected[0]["judge"]["bloqueo"])
            kept_path, relabelled_path, rejected_path = outputs(path)
            self.assertEqual(len(io.read(kept_path)), 3)
            self.assertEqual(verify(TASK, judge, path), ([], [], []))  # Nothing new to judge.
            self.assertEqual(judge.calls, 4)

            # A second judge on the rejected file that answers like the first one relabels the case.
            _, relabelled, _ = verify(TASK, judge, rejected_path)
            self.assertEqual(len(relabelled), 1)
            self.assertEqual(relabelled[0]["answers"], rejected[0]["judge"])
            self.assertTrue(relabelled[0]["generated"]["bloqueo"])
            self.assertEqual(len(io.read(outputs(rejected_path)[1])), 1)

    def test_relabel_only_to_a_valid_combination(self):
        case = CASES[0]
        invalid = {**case["answers"], "prioridad": "baja", "bloqueo": True}  # Low priority never blocks: excluded.
        judge = FakeJudge({TASK.state(case)["ticket"]: invalid})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            io.append(path, [{**case, "judge": invalid}])
            self.assertEqual([len(x) for x in verify(TASK, judge, path)], [0, 0, 1])
            self.assertTrue(all(p.exists() for p in outputs(path)))  # All three, even the empty ones.

    def test_schema_only_accepts_valid_keys(self):
        schema = judge_schema(TASK).model_json_schema()
        self.assertEqual(schema["properties"]["categoria"]["enum"], list(TASK.questions["categoria"].keys))
        self.assertEqual(schema["properties"]["bloqueo"]["enum"], [False, True])


if __name__ == "__main__":
    unittest.main()
