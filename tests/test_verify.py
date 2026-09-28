"""The judge with a fake LLM: agreement splits the cases, and judged cases are not asked again."""
import json
import tempfile
import unittest
from pathlib import Path

from layaft.data import io
from layaft.data.verify import judge_schema, outputs, verify
from layaft.task import Task

TASK = Task.load("tool_routing")


class FakeJudge:
    """Agrees with the label except when the message says 'web', where it answers web_search=false."""
    model, parallel, calls = "fake", 1, 0

    def __init__(self, labels):
        self.labels = labels

    def __call__(self, prompt, schema):
        self.calls += 1
        message = json.loads(prompt.split("State:\n")[1].split("\n\nQuestions")[0])["message"]
        answers = dict(self.labels[message])
        if "web" in message:
            answers["web_search"] = False
        return json.dumps(answers), 0.0


class VerifyChecks(unittest.TestCase):
    def test_splits_by_agreement_and_skips_judged_cases(self):
        cases = TASK.test_cases()[:4]
        cases[0] = {**cases[0], "fields": {**cases[0]["fields"], "message": "busca en la web el clima"}}
        cases[0]["id"] = TASK.case_id(cases[0]["fields"])
        cases[0]["answers"] = {**cases[0]["answers"], "action": "use_tools", "web_search": True}
        judge = FakeJudge({c["fields"]["message"]: c["answers"] for c in cases})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tool_routing.jsonl"
            io.append(path, cases)
            kept, relabelled, rejected = verify(TASK, judge, path)
            self.assertEqual((len(kept), len(relabelled), len(rejected)), (3, 0, 1))
            self.assertFalse(rejected[0]["judge"]["web_search"])
            kept_path, relabelled_path, rejected_path = outputs(path)
            self.assertEqual(len(io.read(kept_path)), 3)
            self.assertEqual(verify(TASK, judge, path), ([], [], []))  # Nothing new to judge.
            self.assertEqual(judge.calls, 4)

            # A second judge on the rejected file that answers like the first one relabels the case.
            _, relabelled, _ = verify(TASK, judge, rejected_path)
            self.assertEqual(len(relabelled), 1)
            self.assertEqual(relabelled[0]["answers"], rejected[0]["judge"])
            self.assertTrue(relabelled[0]["generated"]["web_search"])
            self.assertEqual(len(io.read(outputs(rejected_path)[1])), 1)

    def test_relabel_only_to_a_valid_combination(self):
        case = TASK.test_cases()[0]
        invalid = {**case["answers"], "action": "answer_directly", "email": True}  # Answering directly uses no tool.
        judge = FakeJudge({case["fields"]["message"]: invalid})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            io.append(path, [{**case, "judge": invalid}])
            self.assertEqual([len(x) for x in verify(TASK, judge, path)], [0, 0, 1])
            self.assertTrue(all(p.exists() for p in outputs(path)))  # All three, even the empty ones.

    def test_schema_only_accepts_valid_keys(self):
        schema = judge_schema(TASK).model_json_schema()
        self.assertEqual(schema["properties"]["action"]["enum"], ["use_tools", "ask_user", "answer_directly"])
        self.assertEqual(schema["properties"]["email"]["enum"], [False, True])


if __name__ == "__main__":
    unittest.main()
