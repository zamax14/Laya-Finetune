"""Builds the catalog of Claude Code items for the context_prefilter task, from public MIT/Apache sources.

    python tasks/prefilter/collect.py     # → catalog_train.jsonl, catalog_heldout.jsonl, mcp_servers.jsonl here

Each item is what claude-decide shows Laya: {type, name, description (≤400 characters)}, plus its source and license.
Skills are SKILL.md files, agents are agents/*.md, MCP tools come from the Docker MCP registry (servers/*/tools.json),
named mcp__<server>__<tool> as Claude Code names them. Whole sources are held out (VoltAgent's agents and 15 % of the
MCP servers): the model never trains on them, so the held-out test measures catalogs it has not seen.
Needs `gh` (logged in) for the repository trees; files are read from raw.githubusercontent.com.
"""
import json
import random
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MAX_DESCRIPTION = 400  # As claude-decide's catalog.py: Laya reads at most 256 tokens of the question.
REPOS = {  # repo → license; only repositories that declare a permissive one.
    "anthropics/claude-plugins-official": "Apache-2.0",
    "obra/superpowers": "MIT",
    "wshobson/agents": "MIT",
    "VoltAgent/awesome-claude-code-subagents": "MIT",
    "davila7/claude-code-templates": "MIT",
}
MCP_REGISTRY = "docker/mcp-registry"  # MIT
HELD_OUT_REPOS = {"VoltAgent/awesome-claude-code-subagents"}
HELD_OUT_SERVERS = 0.15
# No source may dominate: one MCP server lists 300 tools and one repository aggregates 900 skills.
MAX_PER_SERVER, MAX_PER_REPO = 20, 400


def tree(repo):
    out = subprocess.run(["gh", "api", f"repos/{repo}/git/trees/main?recursive=1", "--jq", ".tree[].path"],
                         capture_output=True, text=True, check=True).stdout
    return out.split()


def raw(repo, path):
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/main/{path}", timeout=30) as reply:
        return reply.read().decode("utf-8", errors="replace")


def frontmatter(text):
    if not text.startswith("---"):
        return {}
    try:
        data = yaml.safe_load(text[3:text.find("\n---", 3)])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def item(kind, name, description, source, license_):
    return {"type": kind, "name": str(name), "description": " ".join(str(description).split())[:MAX_DESCRIPTION],
            "source": source, "license": license_}


def repo_items(repo, license_):
    paths = [p for p in tree(repo) if p.endswith(".md") and not p.lower().endswith("readme.md")]
    skills = [p for p in paths if p.endswith("/SKILL.md")]
    agents = [p for p in paths if re.search(r"(^|/)agents/[^/]+\.md$", p)
              or (repo in HELD_OUT_REPOS and p.startswith("categories/"))]

    def one(kind_path):
        kind, path = kind_path
        fields = frontmatter(raw(repo, path))
        if not fields.get("description") or fields.get("disable-model-invocation") is True:
            return None
        name = fields.get("name") or (Path(path).parent.name if kind == "skill" else Path(path).stem)
        return item(kind, name, fields["description"], repo, license_)

    with ThreadPoolExecutor(16) as pool:
        return [i for i in pool.map(one, [("skill", p) for p in skills] + [("agent", p) for p in agents]) if i]


def mcp_items():
    servers = sorted({p.split("/")[1] for p in tree(MCP_REGISTRY) if p.endswith("/tools.json")})

    def one(server):
        tools = json.loads(raw(MCP_REGISTRY, f"servers/{server}/tools.json") or "[]")
        return [item("mcp", f"mcp__{server}__{t['name']}", t.get("description") or t["name"], f"docker:{server}", "MIT")
                for t in tools if t.get("name")]

    with ThreadPoolExecutor(16) as pool:
        return [i for tools in pool.map(one, servers) for i in tools]


def mcp_servers(with_tools):
    """The registry's servers that list no tools: their description, for synthesize.py to write plausible tools."""
    servers = sorted({p.split("/")[1] for p in tree(MCP_REGISTRY) if p.endswith("/server.yaml")} - with_tools)

    def one(server):
        data = yaml.safe_load(raw(MCP_REGISTRY, f"servers/{server}/server.yaml")) or {}
        about = data.get("about") or {}
        return {"server": server, "title": about.get("title", server), "description": " ".join(str(about.get("description", "")).split()),
                "category": (data.get("meta") or {}).get("category", ""), "license": "MIT"}

    with ThreadPoolExecutor(16) as pool:
        return [s for s in pool.map(one, servers) if s["description"]]


def main():
    items = [i for repo, license_ in REPOS.items() for i in repo_items(repo, license_)] + mcp_items()
    unique = list({(i["type"], i["name"], i["description"]): i for i in items}.values())
    rng, capped = random.Random(0), []
    for source in sorted({i["source"] for i in unique}):
        rows = [i for i in unique if i["source"] == source]
        capped += rng.sample(rows, min(len(rows), MAX_PER_SERVER if source.startswith("docker:") else MAX_PER_REPO))
    unique = capped
    servers = sorted({i["source"] for i in unique if i["source"].startswith("docker:")})
    held_out = HELD_OUT_REPOS | set(random.Random(0).sample(servers, round(HELD_OUT_SERVERS * len(servers))))
    described = mcp_servers({s.removeprefix("docker:") for s in servers})
    held = set(random.Random(1).sample([s["server"] for s in described], round(HELD_OUT_SERVERS * len(described))))
    (HERE / "mcp_servers.jsonl").write_text("".join(json.dumps({**s, "held_out": s["server"] in held}) + "\n"
                                                    for s in described), encoding="utf-8")
    print(f"mcp_servers.jsonl: {len(described)} servers without a tools list ({len(held)} held out)")
    for name, rows in (("train", [i for i in unique if i["source"] not in held_out]),
                       ("heldout", [i for i in unique if i["source"] in held_out])):
        path = HERE / f"catalog_{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        kinds = {k: sum(r["type"] == k for r in rows) for k in ("skill", "agent", "mcp")}
        print(f"{path.name}: {len(rows)} items {kinds}")


if __name__ == "__main__":
    main()
