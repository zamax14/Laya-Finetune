"""Any chat-completions API with structured output: OpenAI, OpenRouter, vLLM, LM Studio, Groq…"""
from layaft.backends.base import LLM
from layaft.http import post


class OpenAICompatible(LLM):
    def __init__(self, base_url, model, api_key=None, extra=None, prices=None, provider="custom"):
        """`extra` goes into every request body; `prices` = (input, output) US$ per million tokens, for APIs that
        do not return the cost themselves (OpenRouter does)."""
        self.url, self.model, self.key = base_url.rstrip("/"), model, api_key
        self.extra, self.prices, self.provider = extra or {}, prices, provider

    def __call__(self, prompt, schema):
        body = post(f"{self.url}/chat/completions", {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "batch", "strict": True, "schema": schema}},
            **self.extra}, self.key)
        return body["choices"][0]["message"]["content"], self.cost(body.get("usage") or {})

    def cost(self, usage):
        if usage.get("cost") is not None:
            return float(usage["cost"])
        if not self.prices:
            return 0.0
        price_in, price_out = self.prices
        return (usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out) / 1e6
