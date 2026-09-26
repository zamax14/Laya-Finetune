"""A Markdown table and an SVG bar chart comparing several evaluations (each a `run.summarize`)."""
from html import escape

INK, GRID, MUTED = "#0d0d0d", "#d4d5d9", "#696969"
PALETTE = ["#4fa8f0", "#ffc53d", "#ff8e3c", "#d9376e", "#2fbf94", "#7757e4"]


def table(task, summaries):
    """One column per evaluation; each question type decides its own rows."""
    first = next(iter(summaries.values()))
    names = [name for qid, q in task.questions.items() for name, _ in q.display(first["questions"][qid])]
    values = {label: [v for qid, q in task.questions.items() for _, v in q.display(s["questions"][qid])]
              + [f"{s['p50_latency_ms']:.0f} ms"] for label, s in summaries.items()}
    lines = ["| | " + " | ".join(summaries) + " |", "|---|" + "---|" * len(summaries)]
    lines += [f"| {name} | " + " | ".join(str(v[i]) for v in values.values()) + " |" for i, name in enumerate(names + ["Latency p50"])]
    return "\n".join(lines)


def _text(x, y, label, *, size=12, color=INK, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Nunito,system-ui,sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">{escape(str(label))}</text>')


def chart(task, summaries, path, subtitle="", footer=""):
    """Grouped bars: % correct per question and per evaluation."""
    width, height, left, right, top, bottom = 880, 440, 72, 840, 126, 354
    y = lambda v: bottom - v / 100 * (bottom - top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'role="img" aria-label="Accuracy per question">',
             f'<rect x="4" y="4" width="875" height="435" rx="12" fill="{INK}"/>',
             f'<rect x="1" y="1" width="875" height="435" rx="12" fill="white" stroke="{INK}" stroke-width="2"/>',
             _text(24, 36, "Accuracy per question", size=20, weight=700), _text(24, 57, subtitle, size=12, color=MUTED)]
    names = list(summaries)
    for i, name in enumerate(names):
        x = 72 + i * (760 // max(len(names), 1))
        parts += [f'<rect x="{x}" y="74" width="15" height="15" rx="3" fill="{PALETTE[i % len(PALETTE)]}" '
                  f'stroke="{INK}" stroke-width="2"/>', _text(x + 23, 87, name, size=12, weight=700)]
    for tick in range(0, 101, 20):
        dash = "" if tick == 0 else ' stroke-dasharray="3 4"'
        parts += [f'<line x1="{left}" x2="{right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="{GRID}"{dash}/>',
                  _text(left - 10, y(tick) + 4, f"{tick} %", color=MUTED, anchor="end")]
    group_width = (right - left) / len(task.questions)
    bar = min(group_width * .72 / len(names), 80)
    for g, (qid, q) in enumerate(task.questions.items()):
        center = left + group_width * (g + .5)
        parts.append(_text(center, bottom + 28, q.name, size=13, weight=700, anchor="middle"))
        for i, s in enumerate(summaries.values()):
            hits, total = q.headline(s["questions"][qid])
            value = 100 * hits / max(1, total)
            x = center - bar * len(names) / 2 + i * bar
            parts += [f'<rect x="{x + 2:.1f}" y="{y(value):.1f}" width="{bar - 4:.1f}" height="{bottom - y(value):.1f}" '
                      f'rx="4" fill="{PALETTE[i % len(PALETTE)]}" stroke="{INK}" stroke-width="2"/>',
                      _text(x + bar / 2, y(value) - 7, f"{value:.0f} %", size=11, weight=700, anchor="middle")]
    parts += [_text(24, 418, footer, size=11, color=MUTED), "</svg>"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
