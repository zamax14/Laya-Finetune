"""Pairing with a fake embedding: near items are the most similar, random ones the rest, all labelled no."""
import tempfile
import unittest
from pathlib import Path

from layaft.data import io
from layaft.data.pairs import outputs, pair
from layaft.task import Task

ITEMS = [{"name": "postgres", "description": "query postgres"}, {"name": "mysql", "description": "query mysql"},
         {"name": "calendar", "description": "events"}, {"name": "email", "description": "mail"}]
VECTORS = {"postgres": [1, 0, 0], "mysql": [.9, .1, 0], "calendar": [0, 1, 0], "email": [0, 0, 1]}


def embed(texts):
    return [VECTORS[t.split(":")[0]] for t in texts]


def make_task(tmp):
    io.append(Path(tmp) / "catalog.jsonl", ITEMS)
    (Path(tmp) / "t.jsonl").write_text("")
    return Task({"name": "t", "fields": {"request": {}, "name": {}, "description": {}}, "state": {"request": "{request}"},
                 "questions": {"needed": {"type": "noul", "instructions": "Is «{name}»: {description} useful?"}},
                 "generation": {"text_field": "request", "id_fields": ["request", "name"],
                                "pool": {"file": "catalog.jsonl", "fields": ["name", "description"]}},
                 "data": {"test": "t.jsonl", "train": "cases.jsonl"}}, tmp)


class PairChecks(unittest.TestCase):
    def test_near_and_random_negatives_of_each_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(tmp)
            yes = {"fields": {"request": "cuántos pedidos hay en postgres", **ITEMS[0]}, "answers": {"needed": True}}
            no = {"fields": {"request": "hola", **ITEMS[2]}, "answers": {"needed": False}}
            io.append(task.train_path, [yes, no])
            near, rand = pair(task, embed, near=1, random_=2, seed=0)
            self.assertEqual([p["fields"]["name"] for p in near], ["mysql"])  # The most similar item.
            self.assertEqual({p["fields"]["name"] for p in rand}, {"calendar", "email"})
            for p in near + rand:
                self.assertFalse(p["answers"]["needed"])
                self.assertEqual(p["fields"]["request"], yes["fields"]["request"])
                self.assertEqual(p["id"], task.case_id(p["fields"]))
            self.assertEqual(len(io.read(outputs(task.train_path)[0])), 1)
            self.assertEqual(pair(task, embed, near=1, random_=2, seed=0), ([], []))  # Already paired.

    def test_pool_can_be_swapped_for_the_held_out_catalog(self):
        from layaft import _task
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(tmp)
            self.assertEqual(len(task.pool_rows), 4)
            io.append(Path(tmp) / "heldout.jsonl", ITEMS[:1])
            self.assertEqual(_task(task, Path(tmp) / "heldout.jsonl").pool_rows, ITEMS[:1])


if __name__ == "__main__":
    unittest.main()
