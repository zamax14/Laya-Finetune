"""Question types: everything that changes between choice, score and noul lives here, once.

Each type is a strategy with the same interface, created from a task's YAML with `Question.create`:

- `laya()`         the question in Laya's own format (what the model reads)
- `keys`           the possible answers, in the order of Laya's options
- `target()`       the training distribution: the constructed label, blended with the teacher or smoothed
- `normalize()`    any model's answer (Jev, an LLM) → Laya's answer shape
- `record()`       one prediction against its reference, for evaluation
- `metrics()`      the summary of many records, and `display()` / `headline()` to report it
- `describe()`     an answer as text for the data generator's prompt
"""
import math
import statistics
from types import SimpleNamespace

TEACHER_WEIGHT = 0.3  # Target = 70 % the constructed label + 30 % the teacher's distribution.
SMOOTHING = 0.1  # Without a teacher: 90 % on the label, the rest spread over the other options.


def probability(value, what):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{what} outside 0–1: {value}")
    return value


def ece(confidences, hits, bins=10):
    """Mean gap between confidence (0–1) and actual accuracy, by confidence band (0 is perfect)."""
    pairs = list(zip(confidences, hits))
    total = 0.0
    for b in range(bins):
        band = [(c, h) for c, h in pairs if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if band:
            accuracy = statistics.fmean(h for _, h in band)
            confidence = statistics.fmean(c for c, _ in band)
            total += len(band) / len(pairs) * abs(accuracy - confidence)
    return round(total, 3)


def _options(raw):
    """`{key: "criteria"}` or `{key: {criteria, label, signals}}` → `{key: SimpleNamespace}`."""
    out = {}
    for key, value in raw.items():
        value = value if isinstance(value, dict) else {"criteria": value}
        out[key] = SimpleNamespace(value=key, criteria=value["criteria"], label=value.get("label", key),
                                   signals=value.get("signals", ""))
    return out


class Question:
    type = None
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Question.registry[cls.type] = cls

    @classmethod
    def create(cls, qid, spec):
        """Factory: the class registered for `spec["type"]`."""
        if spec.get("type") not in cls.registry:
            raise ValueError(f"question {qid!r}: unknown type {spec.get('type')!r}; use one of {sorted(cls.registry)}")
        if not spec.get("instructions"):
            raise ValueError(f"question {qid!r}: missing 'instructions'")
        return cls.registry[spec["type"]](qid, spec)

    def __init__(self, qid, spec):
        self.id, self.spec, self.name = qid, spec, spec.get("name", qid)
        self.instructions = spec["instructions"]

    # ------------------------------------------------------------------ shared behaviour

    def index(self, value):
        return self.keys.index(value)

    def target(self, value, soft=None):
        """Distribution in Laya's option order: label blended with the teacher's, or smoothed without one."""
        i, n = self.index(value), len(self.keys)
        if soft is None:
            return [1 - SMOOTHING if k == i else SMOOTHING / (n - 1) for k in range(n)]
        return [(1 - TEACHER_WEIGHT) * float(k == i) + TEACHER_WEIGHT * s for k, s in enumerate(soft)]

    def record(self, answer, expected):
        return {"predicted": self.predicted(answer), "expected": expected,
                "confidence": round(100 * answer["confidence"], 1)}

    def leak_terms(self, value):
        """Words that give the answer away if a generated text contains them."""
        return []

    def headline(self, summary):
        """(hits, total) for the chart."""
        return summary["correct"], summary["total"]


class Choice(Question):
    type = "choice"

    def __init__(self, qid, spec):
        super().__init__(qid, spec)
        self.options = _options(spec["options"])
        self.keys = list(self.options)
        self.review = spec.get("review")  # {"green": 80, "yellow": 60}: the traffic light on the confidence.

    def laya(self):
        return {"type": "choice", "instructions": self.instructions,
                "criteria": {k: o.criteria for k, o in self.options.items()}}

    def describe(self, value):
        return self.options[value]

    def leak_terms(self, value):
        return [self.options[value].label.lower()]

    def teacher_dist(self, answer):
        return [answer["probabilities"][k] for k in self.keys]

    def predicted(self, answer):
        return answer["choice"]

    def normalize(self, answer):
        probs = _distribution(answer, self.keys, str(answer.get("choice")))
        pick = str(answer.get("choice")) if str(answer.get("choice")) in self.keys else max(probs, key=probs.get)
        return {"type": "choice", "choice": pick, "probabilities": probs,
                "confidence": round(float(answer.get("confidence", probs[pick])), 4)}

    def light(self, confidence):
        """Green above `green`, yellow from `yellow` to `green` (both included), red below (in %)."""
        return "green" if confidence > self.review["green"] else "yellow" if confidence >= self.review["yellow"] else "red"

    def record(self, answer, expected):
        # Laya's `confidence` for a choice is 1 − normalized entropy, not a probability (0.64 for 0.9/0.05/0.05):
        # the ECE and the traffic light need the probability of the chosen option, which calibration fits.
        p = (answer.get("probabilities") or {}).get(answer["choice"], answer["confidence"])
        out = {**super().record(answer, expected), "confidence": round(100 * p, 1)}
        if self.review:
            out["light"] = self.light(out["confidence"])
        return out

    def metrics(self, records):
        graded = [r for r in records if r["expected"] is not None]  # A null reference is ambiguous on purpose.
        hits = [r["predicted"] == r["expected"] for r in graded]
        out = {"correct": sum(hits), "total": len(graded), "ece": ece([r["confidence"] / 100 for r in graded], hits)}
        if self.review:
            out["lights"] = {color: {"correct": sum(r["predicted"] == r["expected"] for r in graded if r["light"] == color),
                                     "total": sum(r["light"] == color for r in graded)}
                             for color in ("green", "yellow", "red")}
        return out

    def display(self, s):
        rows = [(self.name, f"{s['correct']}/{s['total']}"), (f"{self.name} ECE", s["ece"])]
        if "lights" in s:
            rows.append((f"{self.name} hits in green", f"{s['lights']['green']['correct']}/{s['lights']['green']['total']}"))
        return rows


class Score(Question):
    type = "score"

    def __init__(self, qid, spec):
        super().__init__(qid, spec)
        self.levels = _options(spec["levels"])  # Lowest first: Laya scores it as a scale.
        self.keys = list(self.levels)

    def laya(self):
        return {"type": "score", "instructions": self.instructions, "criteria": [o.criteria for o in self.levels.values()]}

    def describe(self, value):
        return self.levels[value]

    def teacher_dist(self, answer):
        return [answer["probabilities"][str(i)] for i in range(len(self.keys))]

    def predicted(self, answer):
        return self.keys[min(len(self.keys) - 1, max(0, round(answer["score"])))]

    def normalize(self, answer):
        keys = [str(i) for i in range(len(self.keys))]
        probs = _distribution(answer, keys, str(answer.get("level")))
        expected = sum(i * p for i, p in enumerate(probs.values()))
        score = float(answer["score"]) if isinstance(answer.get("score"), (int, float)) else expected
        return {"type": "score", "score": round(score, 4), "probabilities": probs,
                "legend": {str(i): o.criteria for i, o in enumerate(self.levels.values())},
                "confidence": round(float(answer.get("confidence", max(probs.values()))), 4)}

    def record(self, answer, expected):
        return {**super().record(answer, expected), "score": round(answer["score"], 2)}

    def metrics(self, records):
        distance = [abs(self.index(r["predicted"]) - self.index(r["expected"])) for r in records]
        return {"correct": distance.count(0), "near": sum(d <= 1 for d in distance), "total": len(records)}

    def display(self, s):
        return [(f"{self.name} exact", f"{s['correct']}/{s['total']}"), (f"{self.name} ±1", f"{s['near']}/{s['total']}")]


class Noul(Question):
    type = "noul"
    keys = [False, True]  # Laya's noul options are always [false, true].

    def laya(self):
        return {k: v for k, v in self.spec.items() if k in ("type", "instructions", "criteria", "labels")}

    def describe(self, value):
        words = self.spec.get("words", {True: "yes", False: "no"})
        return SimpleNamespace(value=value, label=words[value], criteria="", signals="")

    def teacher_dist(self, answer):
        return [1 - answer["noul"], answer["noul"]]

    def predicted(self, answer):
        return answer["noul"] >= .5

    def normalize(self, answer):
        p = probability(answer["noul"] if "noul" in answer else answer["probability"], self.id)
        return {"type": "noul", "noul": p, "confidence": round(max(p, 1 - p), 4)}

    def record(self, answer, expected):
        return {**super().record(answer, expected), "p": round(answer["noul"], 3)}

    def metrics(self, records):
        return {"correct": sum(r["predicted"] == r["expected"] for r in records), "total": len(records),
                "brier": round(statistics.fmean((r["p"] - r["expected"]) ** 2 for r in records), 3)}

    def display(self, s):
        return [(self.name, f"{s['correct']}/{s['total']}"), (f"{self.name} Brier", s["brier"])]


def _distribution(answer, keys, stated):
    """Normalized probabilities over `keys`; without any, all the mass goes to what the model stated."""
    given = {str(k): v for k, v in (answer.get("probabilities") or {}).items()}
    probs = {k: max(0.0, float(given.get(k, 0))) for k in keys}
    if sum(probs.values()) <= 0:
        if stated not in keys:
            raise ValueError("answer without probabilities or a valid value")
        probs = {k: float(k == stated) for k in keys}
    total = sum(probs.values())
    return {k: round(v / total, 4) for k, v in probs.items()}
