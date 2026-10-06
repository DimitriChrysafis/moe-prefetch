import numpy as np

from .base import prepare, save_artifact, targets


class FutureGatePredictor:
    kind = "future_gate"
    stateful = False

    def __init__(self, gates: np.ndarray, lookaheads=(1, 2, 3), norm: bool = True):
        self.lookaheads = tuple(int(d) for d in lookaheads)
        self.norm = bool(norm)
        self.layers, self.experts, self.hidden_dim = gates.shape
        self.weight = np.ascontiguousarray(np.swapaxes(gates, 1, 2), dtype=np.float32)

    def scores(self, layer: int, hidden: np.ndarray) -> dict[int, np.ndarray]:
        x = prepare(hidden, self.norm)
        return {t: x @ self.weight[t] for t in targets(layer, self.lookaheads, self.layers)}

    def meta(self) -> dict:
        return {
            "kind": self.kind,
            "lookaheads": list(self.lookaheads),
            "norm": self.norm,
            "layers": self.layers,
            "experts": self.experts,
            "hidden_dim": self.hidden_dim,
        }

    def save(self, path):
        gates = np.swapaxes(self.weight, 1, 2).astype(np.float16)
        return save_artifact(path, self.meta(), {"gates": gates})

    @classmethod
    def from_artifact(cls, meta: dict, tensors: dict):
        return cls(tensors["gates"].astype(np.float32), meta["lookaheads"], meta["norm"])
