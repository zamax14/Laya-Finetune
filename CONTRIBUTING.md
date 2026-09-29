# Contributing to Laya Finetune

Thanks for helping. Bug reports, new LLM backends, question types and results on other GPUs are all welcome.

## Set up

```bash
git clone https://github.com/zamax14/Laya-Finetune.git && cd Laya-Finetune
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -t .    # no network, no GPU
```

The tests run on the task in [`tests/fixtures/helpdesk.yaml`](tests/fixtures/helpdesk.yaml) and never download the
model or call an LLM, so they pass on any machine. CI runs them on every push and pull request.

## Reporting a bug

Open an issue with:
- the command or Python call, and the full error;
- your GPU, driver and `torch` version (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`);
- the task YAML if the problem depends on it (a minimal one is best).

For a result that looks wrong, include the `runs/val/.../` table and how the test set was written.

## Pull requests

- One change per pull request, with a test that fails without it.
- Keep the code close to what is already there: small functions, no new dependency for what a few lines can do.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) on a single line:
  `feat(backends): add a vLLM backend`, `fix(verify): keep empty output files`.
- Never commit keys or data: `openrouter`, `OPENAI`, `HF_TOKEN`, `typesafe`, `data/*.jsonl` and `runs/` are
  git-ignored for that reason.

## Adding an LLM backend

Backends live in [`layaft/backends/`](layaft/backends). Anything that speaks the OpenAI chat API already works with
`backend=custom base_url=...`, so a new backend is only needed for a different protocol. Test it without the network,
as [`tests/test_backends.py`](tests/test_backends.py) does.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
