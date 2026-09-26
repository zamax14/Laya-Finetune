"""Temperature scaling: one temperature per question type, fitted on held-out cases (laya-multilingual ships uncalibrated)."""
import torch


def fit_temperature(pairs):
    """Temperature minimizing the soft cross-entropy of (logits, target) pairs; 1.0 with fewer than 10."""
    from laya.common import clamp_temperature
    if len(pairs) < 10:
        return 1.0
    with torch.enable_grad():
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = -sum((t * torch.log_softmax(z / log_t.exp(), -1)).sum() for z, t in pairs) / len(pairs)
            loss.backward()
            return loss

        opt.step(closure)
    return clamp_temperature(float(log_t.exp()))


@torch.no_grad()
def calibrate(trainer, items):
    """Fits and applies the temperatures to the trainer's agent; returns them in Laya's order."""
    pairs = {qt: [] for qt in range(3)}
    for logits, _, chunk in trainer.logits(items):
        for r, it in enumerate(chunk):
            pairs[it["qtype"]].append((logits[r, :len(it["markers"])].cpu(), torch.tensor(it["target"])))
    fitted = [fit_temperature(pairs[qt]) for qt in range(3)]
    trainer.agent.temperature, trainer.agent.temperature_by_options = fitted, {}
    print("Temperatures:", {name: round(fitted[i], 3) for name, i in trainer.qtypes.items()},
          "(1.0 = fewer than 10 cases of that type)" if min(len(p) for p in pairs.values()) < 10 else "")
    return fitted
