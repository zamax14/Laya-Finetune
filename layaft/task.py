"""A task: the typed questions, how a case becomes Laya's state, what to generate and the hand-written test set.

    task = Task.load("helpdesk")          # tasks/helpdesk.yaml, or any path to a YAML file
    task.questions["categoria"]           # a questions.Question
    task.combos()                         # every answer combination the generator writes texts for
    task.state(case)                      # what Laya reads for a case

A case is `{"id", "fields": {...}, "answers": {question id: value}}`; generated cases add context, model and date.
"""
import hashlib
import itertools
import json
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, create_model, model_validator

from layaft import config
from layaft.questions import Question


class _Item(BaseModel):
    """One generated text. The checks stay out of the JSON Schema (OpenRouter's strict mode rejects them)."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    min_words: ClassVar[dict] = {}

    @model_validator(mode="after")
    def complete(self):
        for name, value in self:
            if not value:
                raise ValueError(f"{name} is empty")
            if len(value.split()) < self.min_words.get(name, 0):
                raise ValueError(f"{name} has {len(value.split())} words")
        return self


class Task:
    def __init__(self, spec, root=config.TASKS):
        self.spec, self.root, self.name = spec, Path(root), spec["name"]
        self.fields = spec["fields"]
        self.questions = {qid: Question.create(qid, q) for qid, q in spec["questions"].items()}
        self.template = spec["state"]
        gen = spec.get("generation", {})
        self.title_field = gen.get("title_field")  # Titles must not repeat; with the text they make the case id.
        self.text_field = gen.get("text_field", next(iter(self.fields)))  # Checked for leaks.
        self.leak_phrases = [p.lower() for p in gen.get("leak_phrases", [])]
        self.prompt = gen.get("prompt")
        self.default_context = gen.get("default_context", f"texts for the task {self.name}")
        self.per_call = gen.get("per_call", 5)
        clash = set(self.fields) & set(self.questions)
        if clash:
            raise ValueError(f"task {self.name}: {sorted(clash)} is both a field and a question")

    @classmethod
    def load(cls, name_or_path):
        path = Path(name_or_path)
        if not path.is_file():
            path = config.TASKS / f"{name_or_path}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"No task {name_or_path!r}: pass a YAML path or create {path}")
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")), path.parent)

    # ------------------------------------------------------------------ questions

    @cached_property
    def laya(self):
        """The questions in Laya's format, ready for `agent.predict(state, task.laya)`."""
        return {qid: q.laya() for qid, q in self.questions.items()}

    def combos(self):
        """Every combination of answers, minus the `exclude` rules (e.g. a low priority never blocks)."""
        rules = self.spec.get("exclude", [])
        all_ = [dict(zip(self.questions, values)) for values in itertools.product(*(q.keys for q in self.questions.values()))]
        return [c for c in all_ if not any(_matches(c, rule) for rule in rules)]

    # ------------------------------------------------------------------ cases

    def state(self, case):
        """The state template filled with the case's fields; a long case adds its padding around it."""
        state = {key: value.format(**case["fields"]) for key, value in self.template.items()}
        padding = case.get("padding") or {}
        return {**({"history": padding["before"]} if padding.get("before") else {}), **state,
                **({"attachments": padding["after"]} if padding.get("after") else {})}

    def case_id(self, fields):
        text = (fields.get(self.title_field, "") if self.title_field else "") + fields[self.text_field]
        return hashlib.sha1(text.encode()).hexdigest()[:12]

    def title(self, case):
        return case["fields"].get(self.title_field) or case["fields"][self.text_field][:60]

    @cached_property
    def schema(self):
        """Pydantic model of a generated batch: its JSON Schema constrains the LLM and it validates the reply."""
        item = create_model("Item", __base__=_Item, **{name: (str, ...) for name in self.fields})
        item.min_words = {name: f["min_words"] for name, f in self.fields.items() if f and f.get("min_words")}
        return create_model("Batch", __config__=ConfigDict(extra="forbid"), items=(list[item], ...))

    def test_cases(self):
        """The hand-written test set: never trained on, only measured."""
        path = self.root / self.spec["test"]
        cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for case in cases:
            for qid, value in case["answers"].items():
                if value is not None and value not in self.questions[qid].keys:
                    raise ValueError(f"{path.name} {case['id']}: {qid}={value!r} is not one of {self.questions[qid].keys}")
        return cases

    def contexts(self):
        path = self.spec.get("generation", {}).get("contexts")
        if not path:
            return [self.default_context]
        return [line.strip() for line in (self.root / path).read_text(encoding="utf-8").splitlines() if line.strip()]

    def fingerprint(self, cases):
        """Two evaluations with the same fingerprint measured the same questions on the same cases."""
        data = json.dumps([cases, self.laya], ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:12]


def _matches(combo, rule):
    return all(combo[qid] in (value if isinstance(value, list) else [value]) for qid, value in rule.items())
