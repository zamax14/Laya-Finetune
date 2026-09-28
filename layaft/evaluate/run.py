"""Measure a loaded agent on a task's cases: per-question metrics, and the same cases at longer lengths."""
import random
import statistics
import time

from layaft.data import io
from layaft.data.compose import POSITIONS, LongStateBuilder, state_budget, token_counter

LENGTHS = (8192, 16384, 32768, 65536)


def evaluate(agent, cases, task):
    """(summary, rows): each question's metrics over the cases, and one row per case."""
    agent.model.eval()
    rows = []
    for case in cases:
        started = time.perf_counter()
        answers = agent.predict(task.state(case), task.laya_for(case))["answers"]
        rows.append({"id": case["id"], "title": task.title(case), "latency_ms": round(1000 * (time.perf_counter() - started), 1),
                     "answers": {qid: q.record(answers[qid], case["answers"].get(qid)) for qid, q in task.questions.items()}})
    return summarize(task, rows), rows


def summarize(task, rows):
    latencies = sorted(r["latency_ms"] for r in rows)
    # The whole decision right: every graded question of the case at once (for routing, the exact tool set).
    whole = [all(a["predicted"] == a["expected"] for a in r["answers"].values() if a["expected"] is not None) for r in rows]
    return {"questions": {qid: q.metrics([r["answers"][qid] for r in rows]) for qid, q in task.questions.items()},
            "all_correct": sum(whole), "total": len(rows),
            "p50_latency_ms": round(statistics.median(latencies), 1),
            "p95_latency_ms": latencies[min(len(latencies) - 1, round(.95 * (len(latencies) - 1)))]}


def by_length(agent, cases, task, builder, lengths, seed=0):
    """The same cases wrapped in filler at each length, positions rotating start/middle/end: {length: summary}."""
    out = {"short": evaluate(agent, cases, task)[0]}
    for length in lengths:
        rng = random.Random(seed)
        long = [builder.build(c, length, POSITIONS[i % 3], rng) for i, c in enumerate(cases)]
        out[str(length)] = evaluate(agent, long, task)[0]
    return out


def by_ctx(agent, cases, task, seed=0):
    """`by_length` at 8k/16k/32k/64k up to the agent's context, with the held-out filler."""
    budget = state_budget(agent)
    builder = LongStateBuilder(task, io.fillers(task, held_out=True), token_counter(agent.tok))
    return by_length(agent, cases, task, builder, [n for n in LENGTHS if n < budget] + [budget], seed)
