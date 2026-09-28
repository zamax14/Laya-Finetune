"""A second LLM answers every generated case blind; the cases where it does not reach the constructed label are set apart.

A label by construction is only as good as the generator's obedience: asked for a message that needs the email, it
sometimes writes one that does not. A judge of another family, reading only what Laya will read (the state) and
answering the same questions, catches those. Next to the input file:

    <name>_verified.jsonl     the cases where the judge agrees on every question: train on these
    <name>_rejected.jsonl     the rest, with the judge's answers under "judge", to audit
    <name>_relabelled.jsonl   only when judging a rejected file again: the cases where this judge answers exactly
                              like the first one, relabelled with that answer (the generated one kept as "generated")

Two judges of different families that agree blind are a better label than a generator that did not follow its
own brief, and those messages are the hard ones: written for one answer, they read as another. A relabelled case
must still be a valid combination of the task. Cases already judged (in any output file) are skipped, so it can run
again after each generation round.
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
    return tuple(path.with_name(f"{path.stem}_{name}.jsonl") for name in ("verified", "relabelled", "rejected"))


def verify(task, llm, path=None):
    """Judges the cases of `path` (the task's training data by default) not judged yet.
    Returns (kept, relabelled, rejected)."""
    path = path or task.train_path
    kept_path, relabelled_path, rejected_path = outputs(path)
    for output in outputs(path):  # Even if empty: the next step concatenates them.
        io.append(output, [])
    done = {c["id"] for c in io.read(kept_path) + io.read(relabelled_path) + io.read(rejected_path)}
    valid = [json.dumps(c, sort_keys=True) for c in task.combos()]
    cases = [c for c in task.read_cases(path) if c["id"] not in done]
    model = judge_schema(task)
    schema = model.model_json_schema()
    llm.temperature = 0.0  # ponytail: only Ollama reads it; an OpenAI-compatible judge keeps its provider's default.

    def one(case):
        state = json.dumps(task.state(case), ensure_ascii=False, indent=1)
        questions = json.dumps(task.laya_for(case), ensure_ascii=False, indent=1)
        try:
            content, _ = llm(PROMPT.format(state=state, questions=questions), schema)
            return case, model.model_validate_json(content).model_dump()
        except Exception as exc:
            print(f"{case['id']}: {type(exc).__name__}: {exc}", flush=True)
            return case, None  # Not written: judged again on the next run.

    kept, relabelled, rejected = [], [], []
    with ThreadPoolExecutor(llm.parallel) as executor:
        for i, (case, judge) in enumerate(executor.map(one, cases), 1):
            if judge is None:
                continue
            if judge == case["answers"]:
                kept.append(case)
                io.append(kept_path, [case])
            elif judge == case.get("judge") and json.dumps(judge, sort_keys=True) in valid:
                relabelled.append({**case, "answers": judge, "generated": case["answers"]})
                io.append(relabelled_path, [relabelled[-1]])
            else:
                rejected.append({**case, "judge": judge})
                io.append(rejected_path, [rejected[-1]])
            if i % 100 == 0:
                print(f"{i}/{len(cases)} judged · {len(kept)} kept · {len(relabelled)} relabelled · "
                      f"{len(rejected)} rejected", flush=True)
    print(f"Judge {llm.model}: {len(kept)} kept, {len(relabelled)} relabelled, {len(rejected)} rejected of "
          f"{len(cases)} new cases → {kept_path}")
    return kept, relabelled, rejected
