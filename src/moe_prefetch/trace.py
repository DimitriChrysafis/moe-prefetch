from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from colibri_runtime.generation import generate_tokens
from colibri_runtime.model import StreamedSparseMoeBlock

FORMAT = 1
PREFILL, DECODE = 0, 1
F16_MAX = float(np.finfo(np.float16).max)
ARRAYS = ("routes", "weights", "hidden", "phase", "tokens")

_active: Capture | None = None


class Capture:
    def __init__(self, model) -> None:
        text = model.args.text
        self.layers = text.num_hidden_layers
        self.experts = text.num_experts
        self.top_k = text.num_experts_per_tok
        self.hidden_dim = text.hidden_size
        self.blocks = {id(layer.mlp) for layer in model.layers}
        self.phase = PREFILL
        self.absmax = 0.0
        self._pending: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._chunks: list[tuple[np.ndarray, ...]] = []

    @property
    def tokens(self) -> int:
        return sum(chunk[3].shape[0] for chunk in self._chunks)

    def observe(self, block: StreamedSparseMoeBlock, x: mx.array) -> None:
        logits = block.gate(x.astype(mx.float32))
        k = block.top_k
        idx = mx.argpartition(-logits, k - 1, -1)[..., :k]
        top = mx.take_along_axis(logits, idx, -1)
        order = mx.argsort(-top, -1)
        idx = mx.take_along_axis(idx, order, -1)
        weights = mx.softmax(mx.take_along_axis(top, order, -1), -1, precise=True)
        absmax = mx.abs(x).max()
        hidden = mx.clip(x, -F16_MAX, F16_MAX).astype(mx.float16)
        mx.eval(idx, weights, hidden, absmax)
        n = idx.size // k
        self.absmax = max(self.absmax, float(absmax.item()))
        self._pending[block.layer_idx] = (
            np.array(idx, copy=True).reshape(n, k).astype(np.int16),
            np.array(weights, copy=True).reshape(n, k).astype(np.float16),
            np.array(hidden, copy=True).reshape(n, x.shape[-1]),
        )
        if block.layer_idx == self.layers - 1:
            self._flush()

    def _flush(self) -> None:
        if len(self._pending) != self.layers:
            missing = sorted(set(range(self.layers)) - set(self._pending))
            raise RuntimeError(f"forward pass skipped layers {missing}")
        parts = [self._pending[layer] for layer in range(self.layers)]
        self._pending.clear()
        routes, weights, hidden = (np.stack([p[i] for p in parts], 1) for i in range(3))
        phase = np.full(routes.shape[0], self.phase, dtype=np.uint8)
        self._chunks.append((routes, weights, hidden, phase))

    def arrays(self) -> dict[str, np.ndarray]:
        if self._pending:
            raise RuntimeError("capture ended in the middle of a forward pass")
        if not self._chunks:
            raise RuntimeError("nothing was captured")
        names = ("routes", "weights", "hidden", "phase")
        return {
            name: np.concatenate([chunk[i] for chunk in self._chunks])
            for i, name in enumerate(names)
        }


@contextmanager
def capture(model) -> Iterator[Capture]:
    global _active
    if _active is not None:
        raise RuntimeError("a capture is already running")
    cap = Capture(model)
    original = StreamedSparseMoeBlock.__call__

    def traced(block, x):
        if id(block) in cap.blocks:
            cap.observe(block, x)
        return original(block, x)

    StreamedSparseMoeBlock.__call__ = traced
    _active = cap
    try:
        yield cap
    finally:
        StreamedSparseMoeBlock.__call__ = original
        _active = None


def record(
    model,
    input_ids: list[int],
    max_tokens: int,
    eos_token_id: int | Iterable[int] | None = None,
    temperature: float = 0.0,
    seed: int = 0,
) -> tuple[list[int], dict[str, np.ndarray], dict[str, Any]]:
    with capture(model) as cap:

        def on_token(_token: int) -> None:
            cap.phase = DECODE

        generated, timing = generate_tokens(
            model, input_ids, max_tokens, temperature, 1.0, eos_token_id, seed, on_token
        )
    arrays = cap.arrays()
    arrays["tokens"] = np.asarray(list(input_ids) + generated[:-1], dtype=np.int32)
    if arrays["tokens"].shape[0] != arrays["phase"].shape[0]:
        raise RuntimeError("token count does not match captured positions")
    info = {
        "timing": timing,
        "hidden_absmax": cap.absmax,
        "layers": cap.layers,
        "experts": cap.experts,
        "top_k": cap.top_k,
        "hidden_dim": cap.hidden_dim,
    }
    return generated, arrays, info


def engine_git_sha() -> str | None:
    import colibri_runtime

    root = Path(colibri_runtime.__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def build_meta(run_id: str, arrays: dict[str, np.ndarray], info: dict[str, Any], **extra) -> dict:
    phase = arrays["phase"]
    meta = {
        "format": FORMAT,
        "run_id": run_id,
        "n_prefill": int((phase == PREFILL).sum()),
        "n_decode": int((phase == DECODE).sum()),
        "layers": info["layers"],
        "experts": info["experts"],
        "top_k": info["top_k"],
        "hidden_dim": info["hidden_dim"],
        "hidden_dtype": "float16",
        "hidden_absmax": info["hidden_absmax"],
        "engine_git_sha": engine_git_sha(),
    }
    meta.update(extra)
    return meta


def write_trace(path: str | Path, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"trace already exists: {path}")
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name in ARRAYS:
        if name in arrays:
            np.save(tmp / f"{name}.npy", arrays[name])
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp.rename(path)
    return path


def load_trace(path: str | Path, mmap: bool = True) -> dict[str, Any]:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"unsupported trace format in {path}: {meta.get('format')}")
    out: dict[str, Any] = {"meta": meta}
    for name in ARRAYS:
        file = path / f"{name}.npy"
        if file.exists():
            out[name] = np.load(file, mmap_mode="r" if mmap else None)
    return out


def list_traces(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p.parent for p in root.glob("*/meta.json") if not p.parent.name.endswith(".partial")
    )


def trace_bytes(root: str | Path) -> int:
    return sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())
