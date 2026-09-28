"""Fills what public sources lack, with an LLM, and builds the pools of the context_prefilter task.

    python tasks/prefilter/synthesize.py qwen3.6:35b      # on a node with Ollama (slurm/prefilter/synthesize.sh)

- MCP tools: the Docker registry describes 242 servers without listing their tools; the LLM writes the 3–6 tools each
  would expose, grounded on that description.
- Rules: nobody publishes ~/.claude/rules; the LLM writes rules a team would install, by theme.
Held-out servers and themes stay out of training like the held-out real sources. Writes pool_train.jsonl and
pool_heldout.jsonl: the real catalog plus these items.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, ConfigDict, create_model

from layaft.backends import create_backend
from layaft.data.io import read

HERE = Path(__file__).parent
MAX_DESCRIPTION = 400
THEMES = ["git commits and branches", "pull request reviews", "Python code style", "TypeScript and React conventions",
          "testing and coverage", "security and secrets", "database migrations", "REST API design", "logging and errors",
          "dependency updates", "performance budgets", "accessibility", "internationalization", "documentation",
          "Terraform and infrastructure", "Kubernetes deployments", "CI pipelines", "data privacy and PII",
          "mobile apps", "machine learning experiments", "SQL and analytics", "Go services", "Rust crates",
          "frontend design systems", "incident response", "release notes and changelogs", "monorepo tooling",
          "Java and Spring", "data pipelines", "prompt and LLM usage"]
HELD_OUT_THEMES = {"Go services", "Rust crates", "incident response", "Java and Spring"}

TOOLS_PROMPT = """An MCP server for Claude Code is described as: «{title}: {description}» (category: {category}).
Write the {n} tools it most plausibly exposes, as its tools/list would return them: a snake_case name and a one or
two sentence description of what the tool does and its main inputs. Distinct tools, no invented features beyond
the description."""
RULES_PROMPT = """Write {n} different rules that a development team would install for Claude Code (files in
~/.claude/rules or sections of CLAUDE.md) about: {theme}. For each one a kebab-case file name and its first
paragraph: when it applies and what Claude must do, in one to three imperative sentences."""


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str
    description: str


BATCH = create_model("Entries", __config__=ConfigDict(extra="forbid"), items=(list[_Entry], ...))


def ask(llm, prompt):
    try:
        content, _ = llm(prompt, BATCH.model_json_schema())
        return BATCH.model_validate_json(content).items
    except Exception as exc:
        print(f"dropped: {type(exc).__name__}: {exc}", flush=True)
        return []


def item(kind, name, description, source, held_out):
    return {"type": kind, "name": name, "description": " ".join(description.split())[:MAX_DESCRIPTION],
            "source": source, "license": "synthetic", "held_out": held_out}


def main(model):
    llm = create_backend("ollama", model, parallel=8)
    rng = random.Random(0)
    servers = read(HERE / "mcp_servers.jsonl")

    def tools(server):
        entries = ask(llm, TOOLS_PROMPT.format(n=rng.randint(3, 6), **server))
        return [item("mcp", f"mcp__{server['server']}__{e.name}", e.description, f"synthetic:{server['server']}",
                     server["held_out"]) for e in entries]

    def rules(theme):
        entries = ask(llm, RULES_PROMPT.format(n=10, theme=theme))
        return [item("rule", e.name, e.description, f"synthetic:rules:{theme}", theme in HELD_OUT_THEMES) for e in entries]

    with ThreadPoolExecutor(llm.parallel) as pool:
        synthetic = [i for group in pool.map(tools, servers) for i in group]
        synthetic += [i for group in pool.map(rules, THEMES) for i in group]
    for name, held in (("train", False), ("heldout", True)):
        rows = read(HERE / f"catalog_{name}.jsonl") + [
            {k: v for k, v in i.items() if k != "held_out"} for i in synthetic if i["held_out"] == held]
        (HERE / f"pool_{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                 encoding="utf-8")
        kinds = {k: sum(r["type"] == k for r in rows) for k in ("skill", "agent", "mcp", "rule")}
        print(f"pool_{name}.jsonl: {len(rows)} items {kinds}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen3.6:35b")
