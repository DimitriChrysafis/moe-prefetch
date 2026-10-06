import numpy as np

from .base import save_artifact, targets


class HeuristicPredictor:
    kind = "heuristic"
    stateful = True

    def __init__(self, lookaheads=(1, 2, 3), layers: int = 48, experts: int = 512):
        self.lookaheads = tuple(int(d) for d in lookaheads)
        self.layers = layers
        self.experts = experts
        self.reset()

    def reset(self) -> None:
        self.freq = np.zeros((self.layers, self.experts), dtype=np.float32)
        self.transitions = np.zeros((self.layers, self.experts, self.experts), dtype=np.float32)
        self.previous: list[np.ndarray | None] = [None] * self.layers

    def update(self, layer: int, routes: np.ndarray) -> None:
        routes = np.asarray(routes).reshape(-1)
        selected = np.unique(routes)
        previous = self.previous[layer]
        if previous is not None:
            self.transitions[layer][np.ix_(previous, selected)] += 1
        np.add.at(self.freq[layer], routes, 1)
        self.previous[layer] = selected

    def scores(self, layer: int, hidden: np.ndarray) -> dict[int, np.ndarray]:
        n = 1 if np.ndim(hidden) == 1 else len(hidden)
        return {
            t: np.broadcast_to(self.freq[t], (n, self.experts))
            for t in targets(layer, self.lookaheads, self.layers)
        }

    def next_token(self, layer: int, selected: np.ndarray) -> np.ndarray:
        candidates = self.transitions[layer][np.asarray(selected)].sum(0)
        if not candidates.any():
            candidates = self.freq[layer].copy()
        if layer + 1 < self.layers:
            candidates = candidates + 2 * self.freq[layer + 1]
        return candidates

    def meta(self) -> dict:
        return {
            "kind": self.kind,
            "lookaheads": list(self.lookaheads),
            "layers": self.layers,
            "experts": self.experts,
        }

    def save(self, path):
        return save_artifact(path, self.meta(), {})

    @classmethod
    def from_artifact(cls, meta: dict, tensors: dict):
        return cls(meta["lookaheads"], meta["layers"], meta["experts"])
