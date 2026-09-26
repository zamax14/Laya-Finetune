"""The interface every LLM backend of the data generator implements."""


class LLM:
    """`llm(prompt, schema)` → (the reply's JSON text, cost in US$). `schema` is the JSON Schema the reply must follow."""
    model: str
    parallel = 8  # Simultaneous calls the generator makes.

    def __call__(self, prompt, schema):
        raise NotImplementedError
