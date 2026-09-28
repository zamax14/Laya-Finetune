"""Long states from short cases: the case untouched, wrapped in neutral filler up to a token length.

Writing thousands of 32k-token documents with an LLM does not scale, and the label of a long document would be as
hard to trust as a short one's. The builder reuses what already works: a short case labelled by construction goes
at the start, the middle or the end of a pile of documents that need no action (`generate_filler`). The label
still holds, and the teacher (Jev reads up to ~32k tokens) drops the long cases where it does not.
"""
import json
import random

POSITIONS = ("start", "middle", "end")
LONG = 2048  # Above this context, training adds long copies and evaluation measures by length.


def token_counter(tokenizer):
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


def state_budget(agent):
    """Tokens the state can take: the context minus the question and options (`head_max_len`) and a margin."""
    return agent.cfg["max_len"] - agent.cfg["head_max_len"] - 16


class LongStateBuilder:
    def __init__(self, task, fillers, count_tokens):
        """`fillers`: texts of neutral documents; `count_tokens(text)` → int, the model's tokenizer."""
        if not fillers:
            raise ValueError(f"No filler documents for {task.name}: run `layaft generate task={task.name} filler=200`")
        self.task, self.count = task, count_tokens
        self.docs = [(text, count_tokens(text)) for text in fillers]

    def tokens(self, case):
        return self.count(json.dumps(self.task.state(case), ensure_ascii=False))

    def build(self, case, tokens, position, rng):
        """A copy of `case` whose state has about `tokens` tokens (never more), with its own id."""
        need = tokens - self.tokens(case)
        docs, total = [], 0
        while total < need:
            text, size = rng.choice(self.docs)
            docs.append(text)
            total += size + 2
        before, after = _split(docs, position)
        long = {**case, "id": f"{case['id']}~{tokens}{position[0]}", "padding": {"before": before, "after": after}}
        while self.tokens(long) > tokens and any(long["padding"].values()):  # Serialization adds a little: trim the far end.
            over = self.tokens(long) - tokens
            side = "before" if len(long["padding"]["before"]) > len(long["padding"]["after"]) else "after"
            words = long["padding"][side].split(" ")
            cut = max(1, over * len(words) // max(1, self.count(long["padding"][side])) + 1)
            long["padding"][side] = " ".join(words[cut:] if side == "before" else words[:-cut])
        return long

    def spread(self, cases, max_tokens, seed, min_tokens=1024):
        """One long copy of each case, with lengths uniform in [min_tokens, max_tokens] and random positions."""
        rng = random.Random(seed)
        return [self.build(c, rng.randint(min_tokens, max_tokens), rng.choice(POSITIONS), rng) for c in cases]


def _split(docs, position):
    text = "\n\n".join(docs)
    if position == "start":  # The case first, everything after it.
        return "", text
    if position == "end":
        return text, ""
    half = len(docs) // 2
    return "\n\n".join(docs[:half]), "\n\n".join(docs[half:])
