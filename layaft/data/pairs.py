"""Negatives for one-question-per-candidate tasks: a request paired with catalog items it does not need.

A task like context prefiltering asks one templated question per candidate ("is «{name}» useful for this request?"),
with the candidate drawn from `generation.pool`. Every case generated for item X where the answer is yes is paired
with other items of the pool, labelled no by construction:

    near     the items most similar to X (embeddings): the hard negatives, two database MCPs or two docs skills
    random   any other items: the easy negatives, most of what a real catalog holds for a request

Near pairs go to <data>_near.jsonl and through `verify`, since a similar item may be needed too and the judges'
consensus relabels it; random pairs go to <data>_random.jsonl and train as they are. Pairs already written are
skipped, so it can run again after each generation round.
"""
import random

import numpy as np

from layaft.data import io


def item_text(row, fields):
    return ": ".join(str(row[f]) for f in fields)


def neighbours(rows, fields, embed):
    """Cosine similarity between every pair of pool rows, from `embed(texts) -> vectors`."""
    vectors = np.asarray(embed([item_text(r, fields) for r in rows]), dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-9)
    return vectors @ vectors.T


def outputs(path):
    return path.with_name(f"{path.stem}_near.jsonl"), path.with_name(f"{path.stem}_random.jsonl")


def pair(task, embed, near=3, random_=4, path=None, seed=0):
    """Writes the near and random pairs of every positive case in `path` (the training data by default).
    Returns (near, random) with the new pairs."""
    (qid,) = task.templated or {None}
    if qid is None or not task.pool_fields:
        raise ValueError(f"task {task.name}: pairing needs one templated question and generation.pool")
    path = path or task.train_path
    rows, fields = task.pool_rows, task.pool_fields
    texts = [item_text(r, fields) for r in rows]
    index = {text: i for i, text in enumerate(texts)}
    similarity = neighbours(rows, fields, embed)
    near_path, random_path = outputs(path)
    done = {c["id"] for c in io.read(near_path) + io.read(random_path)}
    rng, out = random.Random(seed), ([], [])
    for case in task.read_cases(path):
        own = index.get(item_text(case["fields"], fields))
        if case["answers"][qid] is not True or own is None:
            continue
        ranked = [int(j) for j in np.argsort(-similarity[own]) if texts[j] != texts[own]]  # Not a duplicate of X.
        chosen = ranked[:near]
        others = [j for j in range(len(rows)) if texts[j] != texts[own] and j not in chosen]
        for kind, picks in ((0, chosen), (1, rng.sample(others, min(random_, len(others))))):
            for j in picks:
                pair_fields = {**case["fields"], **{f: rows[j][f] for f in fields}}
                new = {"id": task.case_id(pair_fields), "fields": pair_fields, "answers": {**case["answers"], qid: False},
                       "source": case["id"], "pairing": ("near", "random")[kind]}
                if new["id"] not in done:
                    done.add(new["id"])
                    out[kind].append(new)
    io.append(near_path, out[0])
    io.append(random_path, out[1])
    print(f"Pairs: {len(out[0])} near → {near_path} · {len(out[1])} random → {random_path}")
    return out
