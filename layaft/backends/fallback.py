"""A chain of backends that is itself a backend (composite)."""
import threading

from layaft.backends.base import LLM


class FallbackChain(LLM):
    """Uses the first backend until it runs out of credit (HTTP 402), then the next, without stopping the run."""

    def __init__(self, llms):
        self.llms, self.parallel, self.lock = list(llms), llms[0].parallel, threading.Lock()

    @property
    def model(self):
        return self.llms[0].model  # Rows record the model that actually wrote them.

    def __call__(self, prompt, schema):
        llm = self.llms[0]
        try:
            return llm(prompt, schema)
        except RuntimeError as exc:
            if "HTTP 402" not in str(exc) or len(self.llms) == 1:
                raise
            with self.lock:
                if self.llms[0] is llm:
                    self.llms.pop(0)
                    print(f"{getattr(llm, 'provider', llm.model)} out of credit: continuing with {self.llms[0].model}", flush=True)
            return self.llms[0](prompt, schema)

    def unload(self):
        for llm in self.llms:
            llm.unload()
