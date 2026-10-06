import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

ARRAYS = ("routes", "weights", "hidden", "phase")


@dataclass(frozen=True)
class Run:
    path: Path
    meta: dict

    @property
    def run_id(self) -> str:
        return self.meta.get("run_id", self.path.name)

    @property
    def prompt_id(self) -> str:
        return self.meta.get("prompt_id", self.run_id)

    def array(self, name: str) -> np.ndarray:
        return np.load(self.path / f"{name}.npy", mmap_mode="r")

    @cached_property
    def routes(self) -> np.ndarray:
        return np.asarray(self.array("routes"), dtype=np.int64)

    @cached_property
    def phase(self) -> np.ndarray:
        return np.asarray(self.array("phase"))

    @property
    def tokens(self) -> int:
        return len(self.phase)

    def hidden(self, layer: int, rows: np.ndarray | None = None) -> np.ndarray:
        h = self.array("hidden")
        h = h[:, layer] if rows is None else h[rows, layer]
        return np.asarray(h, dtype=np.float32)


def open_run(path: Path) -> Run | None:
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.exists() or not all((path / f"{n}.npy").exists() for n in ARRAYS):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        shapes = {n: np.load(path / f"{n}.npy", mmap_mode="r").shape for n in ARRAYS}
    except (OSError, ValueError):
        return None
    t = shapes["phase"][0]
    if t == 0 or any(s[0] != t for s in shapes.values()):
        return None
    if shapes["routes"] != shapes["weights"] or shapes["hidden"][:2] != shapes["routes"][:2]:
        return None
    return Run(path, meta)


def list_runs(root: Path) -> list[Run]:
    root = Path(root)
    if not root.is_dir():
        return []
    runs = (open_run(p) for p in sorted(root.iterdir()) if p.is_dir())
    return [r for r in runs if r is not None]


def split_of(prompt_id: str) -> str:
    from moe_prefetch import prompts

    if any(p.id == prompt_id for p in prompts.PROMPTS):
        return prompts.split_of(prompt_id)
    rank = int(hashlib.sha256(prompt_id.encode()).hexdigest(), 16)
    return "eval" if rank % 4 == 0 else "train"


def split_runs(runs: list[Run]) -> dict[str, list[Run]]:
    out = {"train": [], "eval": []}
    for run in runs:
        out[split_of(run.prompt_id)].append(run)
    return out
