"""LLM backends for the data generator, created by name.

    create_backend("ollama")                                    # local, gemma3:12b
    create_backend("openrouter", api_key="sk-or-...")           # falls back to OpenAI if it runs out of credit
    create_backend("openai", model="gpt-6-luna")
    create_backend("custom", base_url="http://localhost:1234/v1", model="qwen3")   # vLLM, LM Studio…
"""
from layaft import config
from layaft.backends.base import LLM
from layaft.backends.fallback import FallbackChain
from layaft.backends.ollama import Ollama
from layaft.backends.openai_compat import OpenAICompatible

OPENROUTER = "https://openrouter.ai/api/v1"
# Each preset: where it lives, which key it reads (env var, local file), its default model and request extras.
PRESETS = {
    "openrouter": {"base_url": OPENROUTER, "key": ("OPENROUTER_API_KEY", "openrouter"), "model": "openai/gpt-5.6-luna",
                   "extra": {"reasoning": {"effort": "low"}, "usage": {"include": True}}},
    "openai": {"base_url": "https://api.openai.com/v1", "key": ("OPENAI_API_KEY", "OPENAI"), "model": "gpt-6-luna",
               "extra": {"reasoning_effort": "low"}, "prices": {"gpt-6-luna": (0.1, 0.5), "gpt-5.6-luna": (0.2, 1.2)}},
}
# qwen3.5:9b does not work: without reasoning it ignores the schema, reasoning it took 148 s to return nothing.
OLLAMA_MODEL = "gemma3:12b"


def create_backend(name="ollama", model=None, api_key=None, base_url=None, ollama_url="http://localhost:11434",
                   parallel=None):
    if name == "ollama":
        llm = Ollama(model or OLLAMA_MODEL, ollama_url)
    elif name == "custom":
        if not base_url or not model:
            raise ValueError("backend=custom needs base_url= and model=")
        llm = OpenAICompatible(base_url, model, api_key, provider="custom")
    elif name in PRESETS:
        llm = _preset(name, model, api_key, base_url)
        # OpenRouter out of credit → OpenAI directly, when there is a key for it and none was forced for OpenRouter.
        if name == "openrouter" and not api_key and config.secret(*PRESETS["openai"]["key"]):
            llm = FallbackChain([llm, _preset("openai")])
    else:
        raise ValueError(f"Unknown backend {name!r}: use ollama, openai, openrouter or custom")
    llm.parallel = parallel or llm.parallel
    return llm


def _preset(name, model=None, api_key=None, base_url=None):
    p = PRESETS[name]
    key = api_key or config.secret(*p["key"])
    if not key:
        raise ValueError(f"Missing the {name} key: pass api_key=, set {p['key'][0]} or create the file «{p['key'][1]}»")
    model = model or p["model"]
    return OpenAICompatible(base_url or p["base_url"], model, key, p["extra"], p.get("prices", {}).get(model), name)


__all__ = ["LLM", "Ollama", "OpenAICompatible", "FallbackChain", "create_backend", "PRESETS", "OPENROUTER"]
