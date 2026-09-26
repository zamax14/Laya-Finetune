"""The teacher without network: Jev's reply is replaced and inspected."""
import json
import unittest
from unittest import mock

from layaft import teachers
from layaft.task import Task


class TeacherChecks(unittest.TestCase):
    def test_jev_normalizes_and_prices_typesafe_answers(self):
        task = Task.load("helpdesk")
        reply = {"answers": {"categoria": {"choice": "redes", "confidence": .8,
                                           "probabilities": {"redes": .8, "software": .2}},
                             "prioridad": {"score": 2.0, "probabilities": {"2": 1}}, "bloqueo": {"noul": 1.0}},
                 "usage": {"input_tokens": 1_000_000}}
        with mock.patch.object(teachers.config, "secret", side_effect=lambda env, f=None: "k" if env == "TYPESAFE_API_KEY" else None), \
                mock.patch("layaft.teachers.post", return_value=reply) as post:
            jev = teachers.create_teacher("jev")
            answers = jev.ask({"ticket": "x"}, task)
        self.assertEqual(post.call_args.args[0], "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(post.call_args.args[1]["questions"], task.laya)
        self.assertEqual(answers["categoria"]["probabilities"]["hardware"], 0)
        self.assertEqual(answers["bloqueo"]["confidence"], 1.0)
        self.assertAlmostEqual(jev.cost, 0.042)
        self.assertEqual(teachers.create_teacher("none").label([{"id": "1"}], task, None), {})
        self.assertIsInstance(teachers.create_teacher(None), teachers.NoTeacher)
        json.dumps(answers)  # Cached as JSONL.


if __name__ == "__main__":
    unittest.main()
