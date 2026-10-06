import hashlib
import sys
from dataclasses import dataclass

import numpy as np

from .base import prepare
from .data import Run
from .linear import LinearPredictor


@dataclass
class Config:
    lookaheads: tuple[int, ...] = (1, 2, 3)
    rank: int = 64
    norm: bool = True
    loss: str = "softmax"
    epochs: int = 30
    batch: int = 512
    lr: float = 3e-3
    weight_decay: float = 1e-4
    max_tokens: int = 40000
    decode_weight: float = 1.0
    val_fraction: float = 0.15
    patience: int = 5
    eval_k: int = 16
    seed: int = 0
    threads: int = 4


def split_val(runs: list[Run], fraction: float) -> tuple[list[Run], list[Run]]:
    prompts = sorted({r.prompt_id for r in runs}, key=lambda p: hashlib.sha256(p.encode()).digest())
    n_val = int(round(len(prompts) * fraction)) if len(prompts) > 2 else 0
    val = set(prompts[:n_val])
    return [r for r in runs if r.prompt_id not in val], [r for r in runs if r.prompt_id in val]


def sample_rows(runs: list[Run], max_tokens: int, rng: np.random.Generator) -> list[np.ndarray]:
    total = sum(r.tokens for r in runs)
    keep = min(1.0, max_tokens / max(total, 1))
    rows = []
    for run in runs:
        n = max(1, int(round(run.tokens * keep)))
        rows.append(np.sort(rng.choice(run.tokens, size=min(n, run.tokens), replace=False)))
    return rows


class Batch:
    def __init__(self, runs: list[Run], rows: list[np.ndarray]):
        self.runs = runs
        self.rows = rows
        self.routes = np.concatenate([r.routes[i] for r, i in zip(runs, rows, strict=True)])
        self.phase = np.concatenate([r.phase[i] for r, i in zip(runs, rows, strict=True)])

    def hidden(self, layer: int, norm: bool) -> np.ndarray:
        parts = [r.hidden(layer, i) for r, i in zip(self.runs, self.rows, strict=True)]
        return prepare(np.concatenate(parts), norm)


def _multi_hot(torch, routes, experts: int):
    y = torch.zeros(routes.shape[0], experts)
    y.scatter_(1, torch.from_numpy(routes), 1.0)
    return y


def _recall(torch, logits, y, k: int) -> float:
    top = logits.topk(k, dim=-1).indices
    return float((y.gather(1, top).sum(-1) / y.sum(-1)).mean())


def _fit_head(torch, cfg: Config, x, base, y, w, vx, vbase, vy, gen):
    n, dim = x.shape
    experts = base.shape[1]
    scale = torch.ones(experts, requires_grad=True)
    bias = torch.zeros(experts, requires_grad=True)
    params = [scale, bias]
    down = up = None
    if cfg.rank:
        down = (torch.randn(dim, cfg.rank, generator=gen) / dim**0.5).requires_grad_()
        up = torch.zeros(cfg.rank, experts, requires_grad=True)
        params += [down, up]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def forward(xb, bb):
        out = bb * scale + bias
        if cfg.rank:
            out = out + (xb @ down) @ up
        return out

    def loss_fn(logits, yb, wb):
        if cfg.loss == "bce":
            per = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            ).sum(-1)
        else:
            per = -(yb * torch.log_softmax(logits, -1)).sum(-1) / yb.sum(-1)
        return (per * wb).sum() / wb.sum()

    def snapshot():
        return [p.detach().clone() for p in params]

    def score():
        with torch.no_grad():
            if vx is None:
                return -float(loss_fn(forward(x, base), y, w))
            return _recall(torch, forward(vx, vbase), vy, cfg.eval_k)

    initial = best = score()
    best_state, stale = snapshot(), 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, cfg.batch):
            idx = perm[start : start + cfg.batch]
            opt.zero_grad()
            loss_fn(forward(x[idx], base[idx]), y[idx], w[idx]).backward()
            opt.step()
        current = score()
        if current > best + 1e-5:
            best, best_state, stale = current, snapshot(), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    return [p.numpy() for p in best_state], initial, best


def fit_linear(runs: list[Run], gates: np.ndarray, cfg: Config, log=sys.stderr) -> LinearPredictor:
    import torch

    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    layers, experts, dim = gates.shape
    n_look = len(cfg.lookaheads)
    train_runs, val_runs = split_val(runs, cfg.val_fraction)
    train = Batch(train_runs, sample_rows(train_runs, cfg.max_tokens, rng))
    val = Batch(val_runs, sample_rows(val_runs, cfg.max_tokens // 4, rng)) if val_runs else None
    print(
        f"train {len(train_runs)} runs {len(train.phase)} tokens, "
        f"val {len(val_runs)} runs {0 if val is None else len(val.phase)} tokens",
        file=log,
    )
    w = torch.from_numpy(np.where(train.phase == 1, cfg.decode_weight, 1.0).astype(np.float32))
    weight_t = torch.from_numpy(np.ascontiguousarray(np.swapaxes(gates, 1, 2)))
    scale = np.ones((layers, n_look, experts), np.float32)
    bias = np.zeros((layers, n_look, experts), np.float32)
    down = np.zeros((layers, n_look, dim, cfg.rank), np.float32) if cfg.rank else None
    up = np.zeros((layers, n_look, cfg.rank, experts), np.float32) if cfg.rank else None
    for layer in range(layers):
        heads = [(i, layer + d) for i, d in enumerate(cfg.lookaheads) if layer + d < layers]
        if not heads:
            continue
        x = torch.from_numpy(train.hidden(layer, cfg.norm))
        vx = torch.from_numpy(val.hidden(layer, cfg.norm)) if val else None
        scores = []
        for i, target in heads:
            y = _multi_hot(torch, train.routes[:, target], experts)
            vy = _multi_hot(torch, val.routes[:, target], experts) if val else None
            base = x @ weight_t[target]
            vbase = vx @ weight_t[target] if val else None
            params, initial, best = _fit_head(torch, cfg, x, base, y, w, vx, vbase, vy, gen)
            scale[layer, i], bias[layer, i] = params[0], params[1]
            if cfg.rank:
                down[layer, i], up[layer, i] = params[2], params[3]
            scores.append(f"d{cfg.lookaheads[i]} {initial:.3f}->{best:.3f}")
        print(f"layer {layer:2d} " + " ".join(scores), file=log, flush=True)
    return LinearPredictor(gates, scale, bias, down, up, cfg.lookaheads, cfg.norm)
