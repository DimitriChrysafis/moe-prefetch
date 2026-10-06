import numpy as np

from .base import prepare, save_artifact


class LinearPredictor:
    kind = "linear"
    stateful = False

    def __init__(
        self,
        gates: np.ndarray,
        scale: np.ndarray,
        bias: np.ndarray,
        down: np.ndarray | None = None,
        up: np.ndarray | None = None,
        lookaheads=(1, 2, 3),
        norm: bool = True,
    ):
        self.lookaheads = tuple(int(d) for d in lookaheads)
        self.norm = bool(norm)
        self.layers, self.experts, self.hidden_dim = gates.shape
        n = len(self.lookaheads)
        if scale.shape != (self.layers, n, self.experts) or bias.shape != scale.shape:
            raise ValueError("scale and bias must be [layers, lookaheads, experts]")
        self.weight = np.ascontiguousarray(np.swapaxes(gates, 1, 2), dtype=np.float32)
        self.scale = scale.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self.rank = 0 if down is None else down.shape[-1]
        if self.rank:
            if down.shape != (self.layers, n, self.hidden_dim, self.rank):
                raise ValueError("down must be [layers, lookaheads, hidden, rank]")
            if up.shape != (self.layers, n, self.rank, self.experts):
                raise ValueError("up must be [layers, lookaheads, rank, experts]")
            self.down = np.ascontiguousarray(
                np.moveaxis(down, 1, 2).reshape(self.layers, self.hidden_dim, n * self.rank),
                dtype=np.float32,
            )
            self.up = up.astype(np.float32)

    def scores(self, layer: int, hidden: np.ndarray) -> dict[int, np.ndarray]:
        x = prepare(hidden, self.norm)
        low = x @ self.down[layer] if self.rank else None
        out = {}
        for i, d in enumerate(self.lookaheads):
            t = layer + d
            if not 0 <= t < self.layers:
                continue
            s = (x @ self.weight[t]) * self.scale[layer, i] + self.bias[layer, i]
            if self.rank:
                s += low[:, i * self.rank : (i + 1) * self.rank] @ self.up[layer, i]
            out[t] = s
        return out

    def meta(self) -> dict:
        return {
            "kind": self.kind,
            "lookaheads": list(self.lookaheads),
            "norm": self.norm,
            "rank": self.rank,
            "layers": self.layers,
            "experts": self.experts,
            "hidden_dim": self.hidden_dim,
        }

    def tensors(self) -> dict[str, np.ndarray]:
        n = len(self.lookaheads)
        out = {
            "gates": np.swapaxes(self.weight, 1, 2).astype(np.float16),
            "scale": self.scale,
            "bias": self.bias,
        }
        if self.rank:
            down = self.down.reshape(self.layers, self.hidden_dim, n, self.rank)
            out["down"] = np.moveaxis(down, 2, 1).astype(np.float16)
            out["up"] = self.up.astype(np.float16)
        return out

    def save(self, path, extra: dict | None = None):
        return save_artifact(path, {**self.meta(), **(extra or {})}, self.tensors())

    @classmethod
    def from_artifact(cls, meta: dict, tensors: dict):
        f32 = {k: v.astype(np.float32) for k, v in tensors.items()}
        return cls(
            f32["gates"],
            f32["scale"],
            f32["bias"],
            f32.get("down"),
            f32.get("up"),
            meta["lookaheads"],
            meta["norm"],
        )
