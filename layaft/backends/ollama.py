"""Local generation with Ollama: free, one GPU."""
from layaft.backends.base import LLM
from layaft.http import post


class Ollama(LLM):
    parallel = 2  # One GPU: more threads only queue.

    def __init__(self, model="gemma3:12b", url="http://localhost:11434"):
        self.model, self.url = model, url.rstrip("/")

    def __call__(self, prompt, schema):
        body = post(f"{self.url}/api/chat", {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                                              "format": schema, "stream": False, "think": False,
                                              "options": {"temperature": 0.9}}, timeout=600)
        return body["message"]["content"], 0.0

    def unload(self):
        """Frees the VRAM: training usually comes next on the same GPU."""
        post(f"{self.url}/api/generate", {"model": self.model, "keep_alive": 0})
