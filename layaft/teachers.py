"""Teachers: a paid decision model answers the same typed questions as Laya on every generated case.

Its distribution softens the training target (Laya also learns how much doubt is reasonable) and the cases where it
disagrees with the constructed label are dropped. Answers are cached per task in JSONL: only new cases are paid.

    create_teacher("jev")    # TypeSafe directly with TYPESAFE_API_KEY, otherwise through OpenRouter
    create_teacher("none")   # no teacher: the label is smoothed
"""
import json
from concurrent.futures import ThreadPoolExecutor

from layaft import config
from layaft.backends import OPENROUTER
from layaft.http import post

JEV_PRICE = 0.042  # US$ per million input tokens (output is free); TypeSafe's API does not return the cost.


class NoTeacher:
    """Null object: no answers, so every case trains on the smoothed label."""
    name = "none"

    def label(self, cases, task, path):
        return {}


class JevTeacher:
    """Jev speaks System One, the same contract as Laya: state + typed questions → typed answers."""
    name = "jev"

    def __init__(self):
        typesafe, openrouter = config.secret("TYPESAFE_API_KEY", "typesafe"), config.secret("OPENROUTER_API_KEY", "openrouter")
        if typesafe:
            self.url, self.key, self.model = "https://api.typesafe.ai/v1/systemone", typesafe, "jev-latest"
        else:
            self.url, self.key, self.model = f"{OPENROUTER}/systemone", openrouter, "typesafe/jev-1.13"
        self.cost = 0.0

    def ask(self, state, task):
        body = post(self.url, {"model": self.model, "state": state, "questions": task.laya}, self.key)
        usage = body.get("usage") or {}
        self.cost += float(usage.get("cost") or (usage.get("input_tokens") or 0) * JEV_PRICE / 1e6)
        answers = body.get("answers") or {}
        return {qid: q.normalize(answers[qid]) for qid, q in task.questions.items()}

    def label(self, cases, task, path):
        """Answers by case id: read from the cache, and only the missing ones are asked (8 at a time)."""
        cached = {r["id"]: r["answers"] for r in map(json.loads, path.open(encoding="utf-8"))} if path.exists() else {}
        missing = [c for c in cases if c["id"] not in cached]
        if not missing:
            print(f"Teacher: {sum(c['id'] in cached for c in cases)}/{len(cases)} cases answered by Jev, from the cache")
            return cached
        if not self.key:
            print(f"Teacher: no TYPESAFE_API_KEY or OPENROUTER_API_KEY; {len(missing)} uncached cases use the smoothed label")
            return cached

        def one(case):
            try:
                return case["id"], self.ask(task.state(case), task)
            except Exception as exc:
                print(case["id"], "failed:", exc, flush=True)
                return case["id"], None

        with ThreadPoolExecutor(8) as pool:
            new = {k: v for k, v in pool.map(one, missing) if v}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps({"id": k, "answers": v}, ensure_ascii=False) + "\n" for k, v in new.items())
        print(f"Teacher: Jev answered {len(new)} new cases for US${self.cost:.3f}")
        return {**cached, **new}


TEACHERS = {"jev": JevTeacher, "none": NoTeacher}


def create_teacher(name="jev"):
    if name not in TEACHERS:
        raise ValueError(f"Unknown teacher {name!r}: use {' or '.join(TEACHERS)}")
    return TEACHERS[name]()
