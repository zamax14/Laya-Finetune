"""A task: the typed questions, how a case becomes Laya's state, what to generate and the hand-written test set.

    task = Task.load("invoices")          # tasks/invoices.yaml, or any path to a YAML file
    task.questions["expense_type"]        # a questions.Question
    task.combos()                         # every answer combination the generator writes texts for
    task.state(case)                      # what Laya reads for a case

Dataset format: JSON Lines, one case per line, the same for generated data, your own labelled data and the test set:

    {"fields": {"message": "..."}, "answers": {"action": "use_tools", "web_search": true}}

`fields` has every field of the task; `answers` has every question with one of its keys (a `noul` is true/false).
`id` is optional (a hash of the text by default); any other key (context, model…) is kept and ignored. The task's
`data:` section says where the files are, relative to the YAML file:

    data:
      train: ../data/my_task.jsonl      # default: data/<name>.jsonl in the working directory
      test: my_task_test.jsonl          # required: hand-written, never trained on; a null answer is not graded
      filler: ../data/my_filler.jsonl   # default: data/<name>_filler.jsonl; {"text": ...} per line, for long context
"""
import hashlib
import itertools
import json
from functools import cached_property
from string import Formatter
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
    optional: ClassVar[set] = set()
    single_line: ClassVar[set] = set()

    @model_validator(mode="after")
    def complete(self):
        for name, value in self:
            if not value and name in self.optional:
                continue
            if not value:
                raise ValueError(f"{name} is empty")
            if len(value.split()) < self.min_words.get(name, 0):
                raise ValueError(f"{name} has {len(value.split())} words")
            if "\n" in value and name in self.single_line:  # A line break here is the LLM echoing the dialogue or the prompt.
                raise ValueError(f"{name} has several lines")
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
        # `pool`: fields drawn from a JSONL file (one row per call) instead of written by the LLM, e.g. a catalog item.
        pool = gen.get("pool") or {}
        self.pool_fields = list(pool.get("fields", []))
        self.pool_path = self.root / pool["file"] if pool else None
        # The fields hashed into a case's id: one text paired with several candidates needs the candidate's too.
        self.id_fields = gen.get("id_fields") or [f for f in (self.title_field, self.text_field) if f]
        self.leak_phrases = [p.lower() for p in gen.get("leak_phrases", [])]
        self.prompt = gen.get("prompt")
        self.default_context = gen.get("default_context", f"texts for the task {self.name}")
        self.per_call = gen.get("per_call", 5)
        data = spec.get("data", {})
        self.train_path = self.root / data["train"] if "train" in data else config.DATA / f"{self.name}.jsonl"
        self.filler_path = self.root / data["filler"] if "filler" in data else config.DATA / f"{self.name}_filler.jsonl"
        self.test_path = self.root / data["test"]
        self.teacher_path = config.DATA / f"{self.name}_teacher.jsonl"  # The teacher's answers, a cache.
        # Questions whose instructions name a field, e.g. "Is `{name}` useful?": filled per case by `laya_for`.
        self.templated = {qid for qid, q in self.questions.items()
                          if any(name in self.fields for _, name, _, _ in Formatter().parse(q.instructions) if name)}
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
        if self.templated:
            raise ValueError(f"task {self.name}: {sorted(self.templated)} take fields of each case, use laya_for(case)")
        return {qid: q.laya() for qid, q in self.questions.items()}

    def laya_for(self, case):
        """The questions for one case: a templated question gets the case's fields in its instructions."""
        if not self.templated:
            return self.laya
        return {qid: {**q.laya(), "instructions": q.instructions.format(**case["fields"])} if qid in self.templated
                else q.laya() for qid, q in self.questions.items()}

    def combos(self):
        """Every combination of answers, minus the `exclude` rules (e.g. a low priority never blocks)."""
        rules = self.spec.get("exclude", [])
        all_ = [dict(zip(self.questions, values)) for values in itertools.product(*(q.keys for q in self.questions.values()))]
        return [c for c in all_ if not any(_matches(c, rule) for rule in rules)]

    def weight(self, combo):
        """How often the generator writes this combination, relative to the others: the product of the `weight` of
        every `generation.weights` rule it matches (1 without rules), to follow the real mix of answers."""
        weight = 1.0
        for rule in self.spec.get("generation", {}).get("weights", []):
            rule = dict(rule)
            factor = rule.pop("weight")
            weight *= factor if _matches(combo, rule) else 1.0
        return weight

    # ------------------------------------------------------------------ cases

    def state(self, case):
        """The state template filled with the case's fields; a long case adds its padding around it."""
        state = {key: value.format(**case["fields"]) for key, value in self.template.items()}
        padding = case.get("padding") or {}
        return {**({"history": padding["before"]} if padding.get("before") else {}), **state,
                **({"attachments": padding["after"]} if padding.get("after") else {})}

    def case_id(self, fields):
        text = "".join(fields.get(f, "") for f in self.id_fields)
        return hashlib.sha1(text.encode()).hexdigest()[:12]

    def title(self, case):
        return case["fields"].get(self.title_field) or case["fields"][self.text_field][:60]

    @cached_property
    def pool_rows(self):
        return [json.loads(line) for line in self.pool_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @property
    def written_fields(self):
        """The fields the LLM writes: all but the ones drawn from the pool."""
        return [name for name in self.fields if name not in self.pool_fields]

    @cached_property
    def schema(self):
        """Pydantic model of a generated batch: its JSON Schema constrains the LLM and it validates the reply."""
        item = create_model("Item", __base__=_Item, **{name: (str, ...) for name in self.written_fields})
        item.min_words = {name: f["min_words"] for name, f in self.fields.items() if f and f.get("min_words")}
        item.optional = {name for name, f in self.fields.items() if f and f.get("required") is False}
        item.single_line = {name for name, f in self.fields.items() if f and f.get("single_line")}
        return create_model("Batch", __config__=ConfigDict(extra="forbid"), items=(list[item], ...))

    def read_cases(self, path=None, graded=True):
        """Validated cases from a JSONL file (the training data by default). `graded=False` accepts null answers."""
        path = Path(path or self.train_path)
        if not path.exists():
            return []
        cases = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                cases.append(self._check(json.loads(line), f"{path.name}:{n}", graded))
        return cases

    def _check(self, case, where, graded):
        fields, answers = case.get("fields") or {}, case.get("answers") or {}
        if set(fields) != set(self.fields):
            raise ValueError(f"{where}: fields {sorted(fields)}, the task expects {sorted(self.fields)}")
        for qid, q in self.questions.items():
            value = answers.get(qid)
            if value is None and not graded:
                continue
            if (type(value), value) not in {(type(k), k) for k in q.keys}:  # A noul takes true/false, not 1/0.
                raise ValueError(f"{where}: {qid}={value!r}, expected one of {q.keys}")
        return {**case, "id": case.get("id") or self.case_id(fields)}

    def test_cases(self):
        """The hand-written test set: never trained on, only measured."""
        return self.read_cases(self.test_path, graded=False)

    def contexts(self):
        path = self.spec.get("generation", {}).get("contexts")
        if not path:
            return [self.default_context]
        return [line.strip() for line in (self.root / path).read_text(encoding="utf-8").splitlines() if line.strip()]

    def fingerprint(self, cases):
        """Two evaluations with the same fingerprint measured the same questions on the same cases."""
        questions = [self.laya_for(c) for c in cases] if self.templated else self.laya
        data = json.dumps([cases, questions], ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:12]


def _matches(combo, rule):
    return all(combo[qid] in (value if isinstance(value, list) else [value]) for qid, value in rule.items())
