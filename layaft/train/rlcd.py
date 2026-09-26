"""RLCD fine-tune of a loaded Laya agent: the recipe of Laya's official notebook, on one GPU.
(https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)

Noisy versions of each predicted distribution are rewarded with proper scoring rules (log, spherical and RPS for
`score`), plus the soft cross-entropy against the target. Sequences are built with the same code Laya predicts with.
"""
import random
import time

import numpy as np
import torch

LR_ENCODER, LR_HEAD = 2.5e-5, 1e-4
GROUP_SIZE = 4  # Noisy samples per sequence in the reinforcement part.
SIGMA_START, SIGMA_END = 0.4, 0.1


def targets(task, case, teacher):
    """Target distribution per question, in Laya's option order: the label blended with the teacher's answer."""
    t = teacher.get(case["id"])
    return {qid: q.target(case["answers"][qid], t and q.teacher_dist(t[qid])) for qid, q in task.questions.items()}


def batches(items, tokens_per_batch, rng):
    """Micro-batches of similar length within a token budget (at least one sequence), in random order."""
    ordered = sorted(items, key=lambda it: len(it["ids"]))
    out, current = [], []
    for it in ordered:
        if current and (len(current) + 1) * len(it["ids"]) > tokens_per_batch:
            out.append(current)
            current = []
        current.append(it)
    out += [current] if current else []
    rng.shuffle(out)
    return out


class Trainer:
    def __init__(self, agent, cfg, seed):
        from laya.common import QTYPES
        self.agent, self.cfg, self.seed, self.qtypes = agent, cfg, seed, QTYPES
        self.model, self.device = agent.model, agent.device
        self.pad = agent.tok.pad_token_id
        self.amp = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16  # The T4 has no bf16.

    def items(self, cases, task, teacher):
        """One sequence per case and question."""
        from laya.common import build_sequence
        out = []
        for case in cases:
            goal = targets(task, case, teacher)
            for qid, question in task.laya.items():
                internal = self.agent._to_internal(question)
                ids, markers = build_sequence(self.agent.tok, task.state(case), internal,
                                              self.agent.cfg["max_len"], self.agent.cfg["head_max_len"])
                assert len(markers) == len(goal[qid]), (case["id"], qid)
                out.append({"ids": ids, "markers": markers, "qtype": self.qtypes[internal["t"]], "target": goal[qid],
                            "label": int(np.argmax(goal[qid]))})
        return out

    def forward(self, batch):
        with torch.autocast("cuda", dtype=self.amp):
            logits, _ = self.model(*(batch[k].to(self.device)
                                     for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
        return logits.float()

    @staticmethod
    def soft_ce(logits, target, mask):
        return -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)

    @torch.no_grad()
    def logits(self, items):
        """(logits, item) for every item, in micro-batches: shared by validation and calibration."""
        from laya.common import collate_items
        self.model.eval()
        for chunk in batches(items, self.cfg["tokens_per_batch"], random.Random(0)):
            batch = collate_items([chunk], self.pad)
            yield self.forward(batch), batch, chunk

    def val_loss(self, items):
        total = sum(self.soft_ce(logits, batch["target"].to(self.device), batch["marker_mask"].to(self.device)).sum().item()
                    for logits, batch, _ in self.logits(items))
        return total / len(items)

    def snapshot(self):
        return {k: (v.half() if v.is_floating_point() else v).detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def train(self, train_items, val_items):
        """Keeps the epoch with the lowest validation CE. Returns (best loss, whether any epoch improved)."""
        from laya.common import collate_items, proper_reward
        cfg, model, device = self.cfg, self.model, self.device
        encoder = model.encoder
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = self.agent.cfg["max_len"] > 8192  # Laya's head attends over the whole sequence too.
        for module in ([encoder.embeddings, *encoder.layers[:cfg["freeze_layers"]]] if cfg["freeze_layers"] else []):
            module.requires_grad_(False)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([{"params": [p for n, p in trainable if n.startswith("encoder.")], "lr": LR_ENCODER},
                                       {"params": [p for n, p in trainable if not n.startswith("encoder.")], "lr": LR_HEAD}],
                                      weight_decay=0.01)
        per_epoch = batches(train_items, cfg["tokens_per_batch"], random.Random(self.seed))
        updates = cfg["epochs"] * -(-len(per_epoch) // cfg["grad_accum"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp == torch.float16)
        print(f"{sum(p.numel() for _, p in trainable) / 1e6:.0f} M trainable parameters · {updates} updates · {self.amp}")

        best_loss, best_state = self.val_loss(val_items), None
        print(f"Validation CE before training: {best_loss:.4f}", flush=True)
        for epoch in range(cfg["epochs"]):
            model.train()
            sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, cfg["epochs"] - 1)
            epoch_batches = batches(train_items, cfg["tokens_per_batch"], random.Random(self.seed + epoch))
            started, running = time.time(), 0.0
            for step, chunk in enumerate(epoch_batches, 1):
                batch = collate_items([chunk], self.pad)
                logits = self.forward(batch)
                mask, target, qtype = (batch[k].to(device) for k in ("marker_mask", "target", "qtype"))
                k = mask.sum(-1, keepdim=True).float()
                eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
                eps = (eps - eps.sum(-1, keepdim=True) / k) * mask  # Zero-mean noise over the logits.
                z = logits.detach().unsqueeze(0) + eps
                with torch.no_grad():
                    reward = proper_reward(torch.softmax(z.masked_fill(~mask, -1e4), -1), target.unsqueeze(0), qtype, mask,
                                           w_sph=0.75, w_rps=1.0)
                    advantage = reward - reward.mean(0, keepdim=True)
                    advantage = advantage / (advantage.std() + 1e-6)
                logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
                loss_ce = self.soft_ce(logits, target, mask).mean()
                scaler.scale((-(advantage * logp).mean() + loss_ce) / cfg["grad_accum"]).backward()
                running += loss_ce.item()
                if step % cfg["grad_accum"] == 0 or step == len(epoch_batches):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            loss = self.val_loss(val_items)
            print(f"epoch {epoch + 1}/{cfg['epochs']}: train CE {running / len(epoch_batches):.4f} · validation CE {loss:.4f} · "
                  f"{time.time() - started:.0f} s · max VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
            if loss < best_loss:
                best_loss, best_state = loss, self.snapshot()
        del optimizer, scaler
        torch.cuda.empty_cache()
        if best_state is not None:
            model.load_state_dict(best_state)
        return best_loss, best_state is not None
