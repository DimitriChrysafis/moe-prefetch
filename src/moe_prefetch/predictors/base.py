import json
from pathlib import Path
from typing import Protocol

import numpy as np
from safetensors.numpy import load_file, save_file

EPS = 1e-6


class Predictor(Protocol):
    lookaheads: tuple[int, ...]

    def scores(self, layer: int, hidden: np.ndarray) -> dict[int, np.ndarray]: ...


def targets(layer: int, lookaheads: tuple[int, ...], layers: int) -> list[int]:
    return [layer + d for d in lookaheads if 0 <= layer + d < layers]


def prepare(hidden: np.ndarray, norm: bool) -> np.ndarray:
    x = np.asarray(hidden, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if norm:
        x = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return x


def save_artifact(path: Path, meta: dict, tensors: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    tensors = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
    save_file(tensors, str(path / "weights.safetensors"))
    return path


def read_artifact(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no predictor at {path}")
    meta = json.loads(meta_path.read_text())
    weights = path / "weights.safetensors"
    return meta, load_file(str(weights)) if weights.exists() else {}
