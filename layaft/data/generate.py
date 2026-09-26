"""Synthetic cases labelled by construction: the answer is decided first, then an LLM writes a text that has it.

Every call is for one fixed combination of answers, spread evenly, so each text's label is known without reading
it. The context only changes the setting: sector, country, who writes, format. Texts that give the answer away and
repeated titles (in the data or in the test set) are dropped. `generate_filler` writes the neutral documents that
`compose.LongStateBuilder` wraps around short cases to train and measure long context.
"""
import random
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from layaft.data import io

PROMPT = '''Write {n} distinct texts. Context: {context}

A careful reader must reach these answers from the facts in each text alone:
{answers}

Rules:
- Write them as the sender would. Vary the length, the tone and the level of detail.
- Give concrete facts: since when, which system or thing, literal messages, what was tried, who is affected, deadlines.
- Never state the answers or name them, and add no comment that gives them away.
- Fields of each text: {fields}.{choices}
- Situations different from these, which already exist: {seen}.'''

FILLER_PROMPT = '''Write {n} documents that could surround a text in this setting: {context}.
They need no action now: already-resolved past threads, automatic notifications, signatures and legal disclaimers,
routine logs, policy excerpts, meeting notes. Each one of 300 to 600 words, in the language of the setting.
None of them may describe an open problem, request or incident.'''


def norm(text):
    text = unicodedata.normalize("NFKD", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text.encode("ascii", "ignore").decode()).strip()


def leaks(task, text, combo):
    """Phrases that give the answer away, or the name of one of the answers."""
    text = text.lower()
    return any(p in text for p in task.leak_phrases) or any(
        term in text for qid, value in combo.items() for term in task.questions[qid].leak_terms(value))


def prompt_for(task, combo, context, seen, rng, count):
    values = {qid: task.questions[qid].describe(value) for qid, value in combo.items()}
    samples = {name: ", ".join(rng.sample(f["choices"], f.get("sample", len(f["choices"]))))
               for name, f in task.fields.items() if f and f.get("choices")}
    answers = "\n".join(f"- {task.questions[qid].instructions} → {d.label}" + (f" ({d.criteria})" if d.criteria else "")
                        + (f". The facts must show {d.signals}" if d.signals else "") for qid, d in values.items())
    choices = "".join(f" For {name}, one of: {s}." for name, s in samples.items())
    return (task.prompt or PROMPT).format(n=count, context=context, seen="; ".join(seen[-30:]) or "—", answers=answers,
                                          fields=", ".join(task.fields), choices=choices, **values, **samples)


def parse(content, model):
    """Validates the LLM's reply; a batch that is not well formed is dropped whole."""
    return [item.model_dump() for item in model.model_validate_json(content).items]


def generate(task, llm, n, context=None, path=None, seed=None, only=None):
    """About n new cases spread over the answer combinations, appended to the task's JSONL. Returns (rows, cost).

    `only` limits the combinations, e.g. {"categoria": ["seguridad"]} to reinforce one with few valid cases. Each
    combination is written as soon as it is done: if the run stops, what was generated stays.
    """
    context, path = context or task.default_context, path or io.cases_path(task)
    taken = {norm(task.title(c)) for c in task.test_cases() + io.read(path)}
    seed = seed if seed is not None else time.time_ns()
    pool = [c for c in task.combos() if not only or all(c[q] in v for q, v in only.items())]
    combos = random.Random(seed).sample(pool, min(n, len(pool)))  # With few rows, random combinations.
    per_combo = -(-n // len(combos))
    schema, written = task.schema.model_json_schema(), []

    def one(combo):
        # A combination's calls run in series so each sees the titles already used.
        rng, rows, seen, cost = random.Random(f"{seed}-{combo}"), [], [], 0.0
        for _ in range(3 * -(-per_combo // task.per_call)):  # Room for the dropped ones.
            if len(rows) >= per_combo:
                break
            try:
                content, spent = llm(prompt_for(task, combo, context, seen, rng, min(task.per_call, per_combo - len(rows))), schema)
                cost += spent
                batch = parse(content, task.schema)
            except ValidationError as exc:
                print(f"{combo}: reply dropped, {exc.error_count()} format errors", flush=True)
                continue
            except Exception as exc:
                print(f"{combo}: {type(exc).__name__}: {exc}", flush=True)
                continue
            for fields in batch:
                title = norm(fields.get(task.title_field) or fields[task.text_field][:60])
                if title in taken or title in map(norm, seen) or leaks(task, fields[task.text_field], combo):
                    continue
                seen.append(fields.get(task.title_field) or fields[task.text_field][:60])
                rows.append({"id": task.case_id(fields), "fields": fields, "answers": combo, "context": context,
                             "model": llm.model, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        io.append(path, rows[:per_combo])
        written.extend(rows[:per_combo])
        print(f"{len(written)}/{len(combos) * per_combo} rows", flush=True)
        return cost

    with ThreadPoolExecutor(llm.parallel) as executor:
        cost = sum(executor.map(one, combos))
    llm.unload()
    return written, cost


class _Filler(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str


FILLER = create_model("Fillers", __config__=ConfigDict(extra="forbid"), items=(list[_Filler], ...))


def generate_filler(task, llm, n, context=None, path=None, per_call=3):
    """About n neutral documents for long states, cached per task. Returns (rows, cost)."""
    context, path = context or task.default_context, path or io.filler_path(task)
    prompt = task.spec.get("generation", {}).get("filler_prompt") or FILLER_PROMPT
    schema, calls = FILLER.model_json_schema(), -(-n // per_call)

    def one(_):
        try:
            content, cost = llm(prompt.format(n=per_call, context=context), schema)
            docs = [d["text"] for d in parse(content, FILLER) if len(d["text"].split()) >= 120]
        except Exception as exc:
            print(f"filler: {type(exc).__name__}: {exc}", flush=True)
            return [], 0.0
        return [{"text": d, "context": context, "model": llm.model} for d in docs], cost

    with ThreadPoolExecutor(llm.parallel) as executor:
        results = list(executor.map(one, range(calls)))
    rows = [r for docs, _ in results for r in docs]
    io.append(path, rows)
    llm.unload()
    return rows, sum(c for _, c in results)
