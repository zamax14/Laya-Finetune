"""Evalúa checkpoints de Laya con los 20 tickets de prueba y dibuja la comparación.

    .venv/bin/python evaluar.py                                              # base frente a la última ajustada
    .venv/bin/python evaluar.py base v1=.model-cache/laya-mesa-de-ayuda-v1 v4=.model-cache/laya-mesa-de-ayuda

Las métricas son las del benchmark de Pondera: acierto de categoría (choice), prioridad exacta y a ±1 nivel (score),
bloqueo con umbral 0,5 y su Brier (noul), aciertos por color del semáforo y ECE de la confianza de la categoría.
Guarda resultados/evaluacion.json y docs/evaluacion.svg.
"""
import argparse
import hashlib
import json
import statistics
import time
from html import escape
from pathlib import Path

import modelo  # Antes que torch.
from tarea import LEVELS, QUESTIONS, TICKETS, light, ticket_state

RESULTS = modelo.ROOT / "resultados" / "evaluacion.json"
CHART = modelo.ROOT / "docs" / "evaluacion.svg"
FINETUNED = modelo.ROOT / ".model-cache" / "laya-mesa-de-ayuda"


def fingerprint():
    """Hash de tickets, preguntas y referencias: dos evaluaciones con el mismo hash midieron lo mismo."""
    data = json.dumps([TICKETS, QUESTIONS], ensure_ascii=False, sort_keys=True, default=list)
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def row(ticket, answers, elapsed_ms):
    category, blocking = answers["categoria"], answers["bloqueo"]["noul"]
    score = answers["prioridad"]["score"]
    confidence = round(100 * category["confidence"], 1)
    expected_category, expected_priority, _ = ticket["referencia"]
    return {"id": ticket["id"], "title": ticket["titulo"],
            "category": category["choice"], "category_confidence": confidence, "light": light(confidence)[0],
            "priority": LEVELS[min(len(LEVELS) - 1, max(0, round(score)))], "priority_score": round(score, 2),
            "blocking": round(blocking, 3),
            "expected_category": expected_category, "expected_priority": expected_priority,
            "expected_blocking": ticket["bloquea"], "latency_ms": round(elapsed_ms, 1)}


def ece(rows, bins=10):
    """Distancia media entre la confianza de la categoría y su acierto real, por tramos de confianza (0 es perfecto)."""
    graded = [r for r in rows if r["expected_category"]]
    total = 0.0
    for b in range(bins):
        band = [r for r in graded if b / bins < r["category_confidence"] / 100 <= (b + 1) / bins or (b == 0 and r["category_confidence"] == 0)]
        if band:
            accuracy = statistics.fmean(r["category"] == r["expected_category"] for r in band)
            confidence = statistics.fmean(r["category_confidence"] / 100 for r in band)
            total += len(band) / len(graded) * abs(accuracy - confidence)
    return round(total, 3)


def summarize(rows):
    graded = [r for r in rows if r["expected_category"]]
    distance = [abs(LEVELS.index(r["priority"]) - LEVELS.index(r["expected_priority"])) for r in rows]
    latencies = sorted(r["latency_ms"] for r in rows)
    lights = {}
    for color in ("verde", "amarillo", "rojo"):
        band = [r for r in graded if r["light"] == color]
        lights[color] = {"correct": sum(r["category"] == r["expected_category"] for r in band), "total": len(band)}
    return {"category_correct": sum(r["category"] == r["expected_category"] for r in graded),
            "category_total": len(graded),
            "priority_correct": distance.count(0), "priority_near": sum(d <= 1 for d in distance), "priority_total": len(rows),
            "blocking_correct": sum((r["blocking"] >= .5) == r["expected_blocking"] for r in rows), "blocking_total": len(rows),
            "blocking_brier": round(statistics.fmean((r["blocking"] - r["expected_blocking"]) ** 2 for r in rows), 3),
            "category_ece": ece(rows), "lights": lights,
            "p50_latency_ms": round(statistics.median(latencies), 1),
            "p95_latency_ms": latencies[min(len(latencies) - 1, round(.95 * (len(latencies) - 1)))]}


def evaluate(agent, dataset):
    """Evalúa un agente cargado sobre una lista de tickets con la forma de tarea.TICKETS."""
    agent.model.eval()
    rows = []
    for case in dataset:
        started = time.perf_counter()
        answers = agent.predict(ticket_state(case), QUESTIONS)["answers"]
        rows.append(row(case, answers, 1000 * (time.perf_counter() - started)))
    return summarize(rows), rows


METRICS = [("Categoría", lambda s: f"{s['category_correct']}/{s['category_total']}"),
           ("Prioridad exacta", lambda s: f"{s['priority_correct']}/{s['priority_total']}"),
           ("Prioridad a ±1", lambda s: f"{s['priority_near']}/{s['priority_total']}"),
           ("Bloqueo", lambda s: f"{s['blocking_correct']}/{s['blocking_total']}"),
           ("Brier del bloqueo", lambda s: s["blocking_brier"]),
           ("ECE de la categoría", lambda s: s["category_ece"]),
           ("Aciertos en verde", lambda s: f"{s['lights']['verde']['correct']}/{s['lights']['verde']['total']}"),
           ("Latencia p50", lambda s: f"{s['p50_latency_ms']:.0f} ms")]


def table(summaries):
    """Tabla en Markdown con una columna por modelo."""
    lines = ["| | " + " | ".join(summaries) + " |", "|---|" + "---|" * len(summaries)]
    lines += [f"| {name} | " + " | ".join(str(f(s)) for s in summaries.values()) + " |" for name, f in METRICS]
    return "\n".join(lines)


# ---------------------------------------------------------------- Gráfica, con el estilo del benchmark de Pondera

INK, GRID, MUTED = "#0d0d0d", "#d4d5d9", "#696969"
PALETTE = ["#4fa8f0", "#ffc53d", "#ff8e3c", "#d9376e", "#2fbf94", "#7757e4"]


def text(x, y, label, *, size=12, color=INK, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Nunito,system-ui,sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">{escape(str(label))}</text>')


def chart(summaries, path=CHART, footer=""):
    """Barras agrupadas: porcentaje de acierto por pregunta y por modelo."""
    width, height, left, right, top, bottom = 880, 440, 72, 840, 126, 354
    groups = [("Categoría", "category_correct", "category_total"), ("Prioridad exacta", "priority_correct", "priority_total"),
              ("Bloqueo", "blocking_correct", "blocking_total")]
    y = lambda v: bottom - v / 100 * (bottom - top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'role="img" aria-label="Acierto por pregunta">',
             f'<rect x="4" y="4" width="875" height="435" rx="12" fill="{INK}"/>',
             f'<rect x="1" y="1" width="875" height="435" rx="12" fill="white" stroke="{INK}" stroke-width="2"/>',
             text(24, 36, "Acierto por pregunta", size=20, weight=700),
             text(24, 57, "Porcentaje correcto en los 20 tickets de prueba, escritos a mano y nunca vistos al entrenar",
                  size=12, color=MUTED)]
    names = list(summaries)
    for i, name in enumerate(names):
        x = 72 + i * (760 // max(len(names), 1))
        parts += [f'<rect x="{x}" y="74" width="15" height="15" rx="3" fill="{PALETTE[i % len(PALETTE)]}" '
                  f'stroke="{INK}" stroke-width="2"/>', text(x + 23, 87, name, size=12, weight=700)]
    for tick in range(0, 101, 20):
        dash = "" if tick == 0 else ' stroke-dasharray="3 4"'
        parts += [f'<line x1="{left}" x2="{right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="{GRID}"{dash}/>',
                  text(left - 10, y(tick) + 4, f"{tick} %", color=MUTED, anchor="end")]
    group_width = (right - left) / len(groups)
    bar = min(group_width * .72 / len(names), 80)
    for g, (label, correct, total) in enumerate(groups):
        center = left + group_width * (g + .5)
        parts.append(text(center, bottom + 28, label, size=13, weight=700, anchor="middle"))
        for i, s in enumerate(summaries.values()):
            value = 100 * s[correct] / s[total]
            x = center - bar * len(names) / 2 + i * bar
            parts += [f'<rect x="{x + 2:.1f}" y="{y(value):.1f}" width="{bar - 4:.1f}" height="{bottom - y(value):.1f}" '
                      f'rx="4" fill="{PALETTE[i % len(PALETTE)]}" stroke="{INK}" stroke-width="2"/>',
                      text(x + bar / 2, y(value) - 7, f"{value:.0f} %", size=11, weight=700, anchor="middle")]
    parts += [text(24, 418, footer, size=11, color=MUTED), "</svg>"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evalúa checkpoints de Laya con los 20 tickets de prueba")
    parser.add_argument("modelos", nargs="*", help="«base» o nombre=ruta de un checkpoint local")
    args = parser.parse_args()
    specs = args.modelos or ["base"] + ([f"ajustada={FINETUNED}"] if FINETUNED.exists() else [])
    summaries, results = {}, {"fingerprint": fingerprint(), "modelos": {}}
    for spec in specs:
        name, _, path = spec.partition("=")
        agent = modelo.load(path or None)
        agent.predict(ticket_state(TICKETS[0]), QUESTIONS)  # Calentamiento fuera de la medición.
        summary, rows = evaluate(agent, TICKETS)
        label = "Laya base" if name == "base" else name
        summaries[label] = summary
        results["modelos"][label] = {"checkpoint": path or modelo.CHECKPOINT, "summary": summary, "rows": rows}
        del agent
    print(table(summaries))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    chart(summaries, footer=f"20 tickets de prueba · huella {results['fingerprint']}")
    print(f"\nGuardado en {RESULTS.relative_to(modelo.ROOT)} y {CHART.relative_to(modelo.ROOT)}")


if __name__ == "__main__":
    main()
