from pathlib import Path

from .base import Predictor, read_artifact
from .future_gate import FutureGatePredictor
from .heuristic import HeuristicPredictor
from .linear import LinearPredictor

KINDS = {cls.kind: cls for cls in (HeuristicPredictor, FutureGatePredictor, LinearPredictor)}


def load_predictor(path: str | Path) -> Predictor:
    meta, tensors = read_artifact(Path(path))
    kind = meta.get("kind")
    if kind not in KINDS:
        raise ValueError(f"unknown predictor kind: {kind}")
    return KINDS[kind].from_artifact(meta, tensors)


__all__ = [
    "KINDS",
    "FutureGatePredictor",
    "HeuristicPredictor",
    "LinearPredictor",
    "Predictor",
    "load_predictor",
]
