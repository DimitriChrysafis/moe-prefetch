import json
from pathlib import Path

import numpy as np


def random_gates(layers: int, experts: int, hidden_dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((layers, experts, hidden_dim)) / np.sqrt(hidden_dim)).astype(
        np.float32
    )


def write_trace(
    root: Path,
    run_id: str,
    gates: np.ndarray,
    n_prefill: int = 24,
    n_decode: int = 8,
    top_k: int = 4,
    prompt_id: str | None = None,
    seed: int = 0,
    drift: float = 0.85,
) -> Path:
    layers, experts, hidden_dim = gates.shape
    rng = np.random.default_rng(seed)
    t = n_prefill + n_decode
    hidden = np.empty((t, layers, hidden_dim), dtype=np.float32)
    h = rng.standard_normal((t, hidden_dim))
    for layer in range(layers):
        hidden[:, layer] = h * rng.uniform(0.5, 2.0, (t, 1))
        h = drift * h + np.sqrt(1 - drift**2) * rng.standard_normal((t, hidden_dim))
    hidden = hidden.astype(np.float16)
    logits = np.einsum("tlh,leh->tle", hidden.astype(np.float32), gates)
    logits += 0.05 * rng.standard_normal(logits.shape) + 0.3 * np.linspace(1, -1, experts)
    routes = np.argsort(-logits, axis=-1)[..., :top_k]
    top = np.take_along_axis(logits, routes, -1)
    weights = np.exp(top - top.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "routes.npy", routes.astype(np.int16))
    np.save(path / "weights.npy", weights.astype(np.float16))
    np.save(path / "hidden.npy", hidden)
    np.save(path / "phase.npy", np.r_[np.zeros(n_prefill), np.ones(n_decode)].astype(np.uint8))
    meta = {
        "format": 1,
        "run_id": run_id,
        "prompt_id": prompt_id or run_id,
        "prompt_category": "synthetic",
        "n_prefill": n_prefill,
        "n_decode": n_decode,
        "layers": layers,
        "experts": experts,
        "top_k": top_k,
        "hidden_dim": hidden_dim,
        "hidden_dtype": "float16",
        "engine_git_sha": "synthetic",
        "model_path": "synthetic",
        "max_tokens": n_decode,
        "temperature": 0.0,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path
