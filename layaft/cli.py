"""`layaft <mode> key=value ...`, like `yolo train data=coco.yaml`.

    layaft generate task=invoices backend=openrouter n=216 context=all
    layaft generate task=invoices backend=ollama llm=gemma3:12b filler=200
    layaft verify task=invoices llm=gemma4:31b
    layaft pair task=catalog near=3 random=4
    layaft train task=invoices ctx=16k profile=full
    layaft val task=invoices model=runs/invoices-16k ctx=16k
    layaft predict task=invoices model=runs/invoices-16k state="Iberia: flight MAD-MEX on 12 May, seat 23C"
    layaft extend model=multilingual ctx=32k

`model=` picks the checkpoint (multilingual by default); every other key goes to the mode.
"""
import json
import sys

MODES = ("generate", "verify", "pair", "train", "val", "predict", "extend")


def parse_value(text):
    """true/false/none, numbers, JSON lists and dicts; anything else stays a string (so `8k` reaches parse_ctx)."""
    lowered = text.lower()
    if lowered in ("true", "false", "none", "null"):
        return {"true": True, "false": False}.get(lowered)
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse(argv):
    """(mode, model, kwargs) from `mode key=value ...`."""
    if not argv or argv[0] not in MODES:
        raise SystemExit(__doc__ + f"\nMode must be one of: {', '.join(MODES)}")
    kwargs = {}
    for arg in argv[1:]:
        key, sep, value = arg.partition("=")
        if not sep:
            raise SystemExit(f"Arguments are key=value, got {arg!r}")
        kwargs[key.replace("-", "_")] = parse_value(value)
    return argv[0], kwargs.pop("model", None), kwargs


def main(argv=None):
    from layaft import LayaFT, config
    mode, model, kwargs = parse(sys.argv[1:] if argv is None else argv)
    result = getattr(LayaFT(model or config.DEFAULT_MODEL), mode)(**kwargs)
    if mode == "predict":
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
