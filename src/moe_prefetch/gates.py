import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path.home() / "qwen3.8next/models/pipenetwork-Qwen3.8-Flash-Next-MLX-4bit"
CACHE_PATH = ROOT / "artifacts" / "gates.safetensors"
GATE_NAME = "language_model.model.layers.{}.mlp.gate"


def _num_layers(model_path: Path) -> int:
    config = json.loads((model_path / "config.json").read_text())
    return int(config.get("text_config", config)["num_hidden_layers"])


def _read(model_path: Path, weight_map: dict, name: str):
    import torch

    with safe_open(model_path / weight_map[name], framework="pt") as f:
        tensor = f.get_tensor(name)
    return (tensor.to(torch.float32) if tensor.is_floating_point() else tensor).numpy()


def _dequantize(weight, scales, biases, bits: int, group_size: int) -> np.ndarray:
    import mlx.core as mx

    out = mx.dequantize(
        mx.array(weight.astype(np.uint32)),
        mx.array(scales),
        mx.array(biases),
        group_size=group_size,
        bits=bits,
    )
    return np.asarray(out.astype(mx.float32))


def read_gates(model_path: Path = MODEL_PATH) -> np.ndarray:
    model_path = Path(model_path)
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    config = json.loads((model_path / "config.json").read_text())
    quant = config.get("quantization") or {}
    gates = []
    for layer in range(_num_layers(model_path)):
        base = GATE_NAME.format(layer)
        if f"{base}.weight" not in weight_map:
            raise KeyError(f"gate weight missing for layer {layer}")
        weight = _read(model_path, weight_map, f"{base}.weight")
        if f"{base}.scales" in weight_map:
            spec = quant.get(base.removeprefix("language_model."), quant.get(base, {}))
            weight = _dequantize(
                weight,
                _read(model_path, weight_map, f"{base}.scales"),
                _read(model_path, weight_map, f"{base}.biases"),
                int(spec.get("bits", quant.get("bits", 4))),
                int(spec.get("group_size", quant.get("group_size", 64))),
            )
        gates.append(weight.astype(np.float32))
    return np.stack(gates)


def load_gates(model_path: Path = MODEL_PATH, cache: Path | None = CACHE_PATH) -> np.ndarray:
    if cache is not None and Path(cache).exists():
        return load_file(str(cache))["gates"]
    gates = read_gates(model_path)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        save_file({"gates": gates}, str(cache))
    return gates
