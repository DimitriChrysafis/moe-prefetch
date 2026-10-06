import time

import numpy as np

from .data import Run
from .metrics import recall_at_k

PHASES = ("prefill", "decode")


class Tally:
    def __init__(self, keys, ks, layers: int):
        self.ks = tuple(ks)
        self.layers = layers
        self.sums = {(p, d): np.zeros((len(self.ks), layers)) for p in PHASES for d in keys}
        self.counts = {(p, d): np.zeros(layers) for p in PHASES for d in keys}

    def add(self, key, target: int, scores: np.ndarray, truth: np.ndarray, phase: np.ndarray):
        for i, k in enumerate(self.ks):
            r = recall_at_k(scores, truth, k)
            for code, name in enumerate(PHASES):
                sel = phase == code
                if sel.any():
                    self.sums[(name, key)][i, target] += r[sel].sum()
        for code, name in enumerate(PHASES):
            self.counts[(name, key)][target] += int((phase == code).sum())

    def report(self) -> tuple[dict, dict]:
        summary, curves = {}, {}
        for (phase, key), sums in self.sums.items():
            counts = self.counts[(phase, key)]
            valid = counts > 0
            curve = np.full(sums.shape, np.nan)
            curve[:, valid] = sums[:, valid] / counts[valid]
            if not valid.any():
                continue
            summary.setdefault(phase, {})[str(key)] = {
                str(k): float(np.nanmean(curve[i])) for i, k in enumerate(self.ks)
            }
            curves.setdefault(phase, {})[str(key)] = {
                str(k): [None if np.isnan(v) else round(float(v), 5) for v in curve[i]]
                for i, k in enumerate(self.ks)
            }
        return summary, curves


def _stateless(predictor, run: Run, tally: Tally, lookaheads):
    routes, phase = run.routes, run.phase
    layers = routes.shape[1]
    for layer in range(layers):
        out = predictor.scores(layer, run.hidden(layer))
        for target, scores in out.items():
            d = target - layer
            if d in lookaheads:
                tally.add(d, target, scores, routes[:, target], phase)


def _stateful(predictor, run: Run, tally: Tally, lookaheads):
    routes, phase = run.routes, run.phase
    hidden = run.array("hidden")
    tokens, layers = routes.shape[:2]
    predictor.reset()
    next_token = hasattr(predictor, "next_token")
    pending = [None] * layers
    for tok in range(tokens):
        x = np.asarray(hidden[tok], dtype=np.float32)
        ph = phase[tok : tok + 1]
        for layer in range(layers):
            if pending[layer] is not None:
                tally.add("next", layer, pending[layer], routes[tok : tok + 1, layer], ph)
            for target, scores in predictor.scores(layer, x[layer]).items():
                d = target - layer
                if d in lookaheads:
                    tally.add(d, target, scores, routes[tok : tok + 1, target], ph)
            predictor.update(layer, routes[tok, layer])
            if next_token:
                pending[layer] = predictor.next_token(layer, np.unique(routes[tok, layer]))[None]


def evaluate(predictor, runs: list[Run], ks=(10, 16, 24, 32), lookaheads=(1, 2, 3)) -> dict:
    if not runs:
        raise ValueError("no traces to evaluate")
    layers = runs[0].routes.shape[1]
    lookaheads = tuple(d for d in lookaheads if d in predictor.lookaheads)
    stateful = getattr(predictor, "stateful", False)
    keys = lookaheads + (("next",) if stateful and hasattr(predictor, "next_token") else ())
    tally = Tally(keys, ks, layers)
    for run in runs:
        (_stateful if stateful else _stateless)(predictor, run, tally, lookaheads)
    summary, curves = tally.report()
    phases = np.concatenate([r.phase for r in runs])
    return {
        "runs": [r.run_id for r in runs],
        "tokens": {p: int((phases == i).sum()) for i, p in enumerate(PHASES)},
        "summary": summary,
        "per_layer": curves,
    }


def latency(predictor, hidden_dim: int, layers: int, repeats: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, hidden_dim)).astype(np.float32)
    active = range(max(1, layers - min(predictor.lookaheads)))
    for layer in active:
        predictor.scores(layer, x)
    times = []
    for _ in range(repeats):
        for layer in active:
            t0 = time.perf_counter()
            predictor.scores(layer, x)
            times.append(time.perf_counter() - t0)
    us = np.array(times) * 1e6
    return {"median_us": float(np.median(us)), "p90_us": float(np.percentile(us, 90))}
