"""The key=value command line."""
import unittest
from unittest import mock

from layaft import cli
from layaft.model.context import parse_ctx


class CliChecks(unittest.TestCase):
    def test_values(self):
        self.assertEqual([cli.parse_value(v) for v in ("true", "False", "none", "216", "0.5", "8k", '["a"]', "gemma3:12b")],
                         [True, False, None, 216, 0.5, "8k", ["a"], "gemma3:12b"])
        self.assertEqual(parse_ctx(cli.parse_value("16k")), 16384)

    def test_parse_splits_model_from_mode_arguments(self):
        mode, model, kwargs = cli.parse(["train", "task=helpdesk", "model=english", "ctx=32k", "gpu-limit=0"])
        self.assertEqual((mode, model, kwargs), ("train", "english", {"task": "helpdesk", "ctx": "32k", "gpu_limit": 0}))
        with self.assertRaises(SystemExit):
            cli.parse(["fit", "task=x"])
        with self.assertRaises(SystemExit):
            cli.parse(["train", "helpdesk"])

    def test_main_calls_the_facade_mode(self):
        with mock.patch("layaft.LayaFT.generate", return_value=0) as generate:
            cli.main(["generate", "task=helpdesk", "backend=openrouter", "n=10", 'only={"categoria": ["redes"]}'])
        generate.assert_called_once_with(task="helpdesk", backend="openrouter", n=10, only={"categoria": ["redes"]})


if __name__ == "__main__":
    unittest.main()
