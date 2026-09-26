"""Backends and teacher without network: the HTTP call is replaced and inspected."""
import unittest
from unittest import mock

from layaft import backends
from layaft.backends import FallbackChain, OpenAICompatible, create_backend


class Fake:
    model, parallel, provider = "fake", 1, "fake"

    def __init__(self, error=None):
        self.error = error

    def __call__(self, prompt, schema):
        if self.error:
            raise RuntimeError(self.error)
        return '{"items": []}', 0.0

    def unload(self):
        pass


class BackendChecks(unittest.TestCase):
    def test_custom_endpoint_gets_base_url_key_and_schema(self):
        reply = {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}
        with mock.patch("layaft.backends.openai_compat.post", return_value=reply) as post:
            llm = create_backend("custom", base_url="http://localhost:1234/v1/", model="qwen3", api_key="k")
            self.assertEqual(llm("hola", {"type": "object"}), ("{}", 0.0))
        url, body, key = post.call_args.args
        self.assertEqual((url, key, body["model"]), ("http://localhost:1234/v1/chat/completions", "k", "qwen3"))
        self.assertEqual(body["response_format"]["json_schema"]["schema"], {"type": "object"})

    def test_presets_read_the_explicit_key_and_price_the_call(self):
        llm = create_backend("openai", api_key="sk-test")
        self.assertEqual((llm.key, llm.model, llm.url), ("sk-test", "gpt-6-luna", "https://api.openai.com/v1"))
        self.assertAlmostEqual(llm.cost({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}), 0.6)
        self.assertEqual(llm.cost({"cost": 0.02}), 0.02)  # OpenRouter reports its own cost.
        with mock.patch.object(backends.config, "secret", return_value=None), self.assertRaises(ValueError):
            create_backend("openrouter")
        with self.assertRaises(ValueError):
            create_backend("custom", model="x")  # No base_url.

    def test_openrouter_falls_back_to_openai_only_with_its_key(self):
        with mock.patch.object(backends.config, "secret", return_value="key"):
            self.assertIsInstance(create_backend("openrouter"), FallbackChain)
            self.assertIsInstance(create_backend("openrouter", api_key="forced"), OpenAICompatible)

    def test_fallback_switches_only_when_out_of_credit(self):
        chain = FallbackChain([Fake("HTTP 402: Insufficient credits"), Fake()])
        self.assertEqual(chain("x", {}), ('{"items": []}', 0.0))
        self.assertEqual(len(chain.llms), 1)
        with self.assertRaises(RuntimeError):  # Other errors do not change provider.
            FallbackChain([Fake("HTTP 500: down"), Fake()])("x", {})


if __name__ == "__main__":
    unittest.main()
