"""Generation with Ollama through its official client: a local server, or the shared daemon of a GPU node."""
from ollama import Client

from layaft.backends.base import LLM


class Ollama(LLM):
    parallel = 2  # One GPU: more threads only queue.
    temperature = 0.9  # Varied texts; the judge of `verify` sets 0.

    def __init__(self, model="gemma3:12b", url=None):
        """`url=None` reads OLLAMA_HOST, else localhost:11434. Models are never pulled here: the server is shared."""
        self.model, self.client = model, Client(host=url, timeout=600)
        available = sorted(m.model for m in self.client.list().models)
        if model not in available:
            raise ValueError(f"Ollama has no model {model!r}; available: {', '.join(available)}")

    def __call__(self, prompt, schema):
        reply = self.client.chat(model=self.model, messages=[{"role": "user", "content": prompt}], format=schema,
                                 think=False, options={"temperature": self.temperature})
        return reply.message.content, 0.0
