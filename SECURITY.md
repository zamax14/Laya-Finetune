# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately through
[GitHub's private vulnerability reporting](https://github.com/zamax14/Laya-Finetune/security/advisories/new)
with the affected version, the steps to reproduce it and its impact.

You will get an answer within a week. Once a fix is released, the advisory is published with credit to the reporter,
unless you prefer to stay anonymous.

## Supported versions

Only the latest release on [PyPI](https://pypi.org/project/laya-finetune/) receives security fixes.

## Scope

Laya Finetune sends task data to the LLM backend you configure and reads API keys from environment variables or local
files that are git-ignored (`openrouter`, `OPENAI`, `HF_TOKEN`, `typesafe`). Issues in that handling are in scope.
Problems in Laya itself, PyTorch, Transformers or the LLM providers belong to those projects.
