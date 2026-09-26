"""A second LLM answers every generated case blind; the cases where it does not reach the constructed label are set apart.

A label by construction is only as good as the generator's obedience: asked for a message that needs the email, it
sometimes writes one that does not. A judge of another family, reading only what Laya will read (the state) and
answering the same questions, catches those. Next to the input file:

    <name>_verified.jsonl   the cases where the judge agrees on every question: train on these
    <name>_rejected.jsonl   the rest, with the judge's answers under "judge", to audit

Cases already judged (in either file) are skipped, so it can run again after each generation round.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import ConfigDict, create_model

from layaft.data import io

PROMPT = '''You label data for a decision model. Read the state and answer every question from what it says, nothing else.

State:
{state}

Questions (each id with its instructions and, for a choice, the criteria of each option):
{questions}

Answer every question id with one of its option keys; a yes/no question with true or false.'''


def judge_schema(task):
    """Every question id, restricted to its keys: the reply can only be a valid answer."""
    fields = {qid: (Literal[tuple(q.keys)], ...) for qid, q in task.questions.items()}
    return create_model("Judgement", __config__=ConfigDict(extra="forbid"), **fields)


def outputs(path):
    return path.with_name(f"{path.stem}_verified.jsonl"), path.with_name(f"{path.stem}_rejected.jsonl")


def verify(task, llm, path=None):
    """Judges the cases of `path` (the task's training data by default) not judged yet. Returns (kept, rejected)."""
    path = path or task.train_path
    kept_path, rejected_path = outputs(path)
    done = {c["id"] for c in io.read(kept_path) + io.read(rejected_path)}
    cases = [c for c in task.read_cases(path) if c["id"] not in done]
    model, questions = judge_schema(task), json.dumps(task.laya, ensure_ascii=False, indent=1)
    schema = model.model_json_schema()
    llm.temperature = 0.0  # ponytail: only Ollama reads it; an OpenAI-compatible judge keeps its provider's default.

    def one(case):
        state = json.dumps(task.state(case), ensure_ascii=False, indent=1)
        try:
            content, _ = llm(PROMPT.format(state=state, questions=questions), schema)
            return case, model.model_validate_json(content).model_dump()
        except Exception as exc:
            print(f"{case['id']}: {type(exc).__name__}: {exc}", flush=True)
            return case, None  # Not written: judged again on the next run.

    kept, rejected = [], []
    with ThreadPoolExecutor(llm.parallel) as executor:
        for i, (case, judge) in enumerate(executor.map(one, cases), 1):
            if judge is None:
                continue
            if judge == case["answers"]:
                kept.append(case)
                io.append(kept_path, [case])
            else:
                rejected.append({**case, "judge": judge})
                io.append(rejected_path, [rejected[-1]])
            if i % 100 == 0:
                print(f"{i}/{len(cases)} judged · {len(kept)} kept · {len(rejected)} rejected", flush=True)
    print(f"Judge {llm.model}: {len(kept)} kept, {len(rejected)} rejected of {len(cases)} new cases → {kept_path}")
    return kept, rejected
