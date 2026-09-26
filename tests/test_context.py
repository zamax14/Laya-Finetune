"""Context extension on a fake checkpoint: only configs change."""
import json
import tempfile
import unittest
import warnings
from unittest import mock
from pathlib import Path

from layaft.model.context import attention_for, extend, parse_ctx

ENCODER = {"max_position_embeddings": 8192, "rope_parameters": {
    "full_attention": {"rope_theta": 160000, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 160000, "rope_type": "default"}}}


def checkpoint(root):
    (root / "encoder").mkdir(parents=True)
    (root / "encoder" / "config.json").write_text(json.dumps(ENCODER))
    (root / "rl_agent_config.json").write_text(json.dumps({"max_len": 1024, "head_max_len": 256}))
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def read(root):
    return json.loads((root / "encoder" / "config.json").read_text()), json.loads((root / "rl_agent_config.json").read_text())


class ContextChecks(unittest.TestCase):
    def test_parse_ctx(self):
        self.assertEqual([parse_ctx(v) for v in ("8k", "32K", 16384, "1024")], [8192, 32768, 16384, 1024])

    def test_yarn_only_on_global_layers_and_from_the_native_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = checkpoint(Path(tmp) / "base")
            out = extend(src, 32768, Path(tmp) / "32k")
            encoder, agent = read(out)
            self.assertEqual(encoder["max_position_embeddings"], 32768)
            self.assertEqual(encoder["rope_parameters"]["full_attention"],
                             {"rope_theta": 160000, "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192})
            self.assertEqual(encoder["rope_parameters"]["sliding_attention"]["rope_type"], "default")
            self.assertEqual(agent, {"max_len": 32768, "head_max_len": 256})
            self.assertEqual(read(src)[0], ENCODER)  # The source is untouched.
            self.assertEqual((out / "model.safetensors").read_bytes(), b"weights")
            again = extend(out, 16384, Path(tmp) / "16k")  # 8k → 32k → 16k: the factor stays relative to 8192.
            self.assertEqual(read(again)[0]["rope_parameters"]["full_attention"]["factor"], 2.0)

    def test_up_to_the_native_length_only_max_len_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = extend(checkpoint(Path(tmp) / "base"), 8192, Path(tmp) / "8k")
            self.assertEqual(read(out), (ENCODER, {"max_len": 8192, "head_max_len": 256}))

    def test_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = checkpoint(Path(tmp) / "base")
            with self.assertRaises(ValueError):
                extend(src, 131072, Path(tmp) / "x")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                extend(src, 65536, Path(tmp) / "64k")
            self.assertIn("experimental", str(caught[0].message))

    def test_attention_choice(self):
        self.assertEqual(attention_for(1024), "sdpa")
        self.assertIn(attention_for(32768), ("flex_attention", "flash_attention_2", "sdpa"))  # sdpa without Python.h.
        with tempfile.TemporaryDirectory() as tmp, mock.patch("sysconfig.get_paths", return_value={"include": "/none"}), \
                mock.patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(attention_for(32768), "sdpa")
            (Path(tmp) / "Python.h").touch()
            with mock.patch.dict("os.environ", {"CPATH": tmp}):
                self.assertEqual(attention_for(32768), "flex_attention")  # Headers unpacked next to the venv.


if __name__ == "__main__":
    unittest.main()
