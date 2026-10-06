from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest
from colibri_runtime.generation import generate_tokens
from colibri_runtime.load import load_model
from colibri_runtime.model import StreamedSparseMoeBlock
from safetensors.numpy import save_file

from moe_prefetch import trace
from moe_prefetch.prompts import EVAL_IDS, get_prompt, get_prompts, split_of

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

CFG = dict(
    vocab_size=64,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    layer_types=["linear_attention", "full_attention"],
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    hc_count=4,
    hc_lowrank=32,
    ple_layer_ids=[1],
    ple_embed_dim=64,
    ple_conv_kernel_size=2,
    ngram_size=3,
    heads_per_ngram=1,
    ngram_vocab_size_base=31,
    make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=2,
    seed=1234,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=8,
    indexer_budget=4,
    indexer_compress_ratio=2,
    output_gate_type="sigmoid",
    eos_token_id=1,
    bos_token_id=1,
    rope_parameters={
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
        "rope_type": "default",
    },
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)
CENTERED = (
    "hc_norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "indexer.q_layernorm.weight",
    "indexer.k_layernorm.weight",
    "ple.norm_key.weight",
    "ple.norm_query.weight",
    "ple.norm_conv.weight",
)
PROMPT = [2, 3, 4, 5, 6, 7, 8]
STEPS = 5


def build_checkpoint(path):
    torch.manual_seed(11)
    config = transformers.Qwen4ExpTextConfig(**CFG)
    reference = transformers.Qwen4ExpForCausalLM(config).eval()
    tensors = {}
    inter = CFG["moe_intermediate_size"]
    for key, value in reference.state_dict().items():
        array = value.detach().numpy().copy()
        if key.endswith(".mlp.experts.gate_up_proj"):
            base = "language_model." + key[: -len("experts.gate_up_proj")]
            tensors[base + "switch_mlp.gate_proj.weight"] = array[:, :inter].copy()
            tensors[base + "switch_mlp.up_proj.weight"] = array[:, inter:].copy()
            continue
        if key.endswith(".mlp.experts.down_proj"):
            base = "language_model." + key[: -len("experts.down_proj")]
            tensors[base + "switch_mlp.down_proj.weight"] = array
            continue
        if key.endswith("ple_embedding.ngram_embedding.weight"):
            rows = array.shape[0] // CFG["split_ngram_parts"]
            for shard in range(CFG["split_ngram_parts"]):
                name = key.replace(
                    "ngram_embedding.weight", f"ngram_embedding.shard_{shard}.weight"
                )
                tensors["language_model." + name] = array[shard * rows : (shard + 1) * rows]
            continue
        if "conv1d.weight" in key and array.ndim == 3:
            array = array.transpose(0, 2, 1)
        if key.endswith(CENTERED):
            array += 1.0
        tensors["language_model." + key] = array
    save_file(tensors, path / "model.safetensors")
    config = {"model_type": "qwen4_exp", "text_config": CFG, "vision_config": {}}
    (path / "config.json").write_text(json.dumps(config))


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny")
    build_checkpoint(path)
    streamed, _ = load_model(
        path,
        expert_budget_bytes=1 << 20,
        ple_budget_bytes=1 << 20,
        io_workers=2,
        prefetch_experts=1,
        prefetch_ple=1,
    )
    yield streamed
    streamed.close()


def logits_of(model, ids):
    out = model(mx.array([ids]))
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def test_capture_leaves_outputs_unchanged(model):
    original = StreamedSparseMoeBlock.__call__
    expected = logits_of(model, PROMPT)
    base, _ = generate_tokens(model, PROMPT, STEPS)
    with trace.capture(model) as cap:
        actual = logits_of(model, PROMPT)
    assert cap.tokens == len(PROMPT)
    np.testing.assert_array_equal(actual, expected)
    traced, _, _ = trace.record(model, PROMPT, STEPS)
    assert traced == base
    assert StreamedSparseMoeBlock.__call__ is original
    assert trace._active is None


def test_record_shapes_and_routes(model):
    seen = []
    store = model.expert_store
    original = store.record_route

    def spy(layer, experts):
        seen.append((layer, list(experts)))
        return original(layer, experts)

    store.record_route = spy
    try:
        generated, arrays, info = trace.record(model, PROMPT, STEPS, eos_token_id=None)
    finally:
        del store.record_route

    k, layers, experts = CFG["num_experts_per_tok"], CFG["num_hidden_layers"], CFG["num_experts"]
    total = len(PROMPT) + STEPS - 1
    assert len(generated) == STEPS
    assert arrays["routes"].shape == (total, layers, k)
    assert arrays["routes"].dtype == np.int16
    assert arrays["weights"].shape == (total, layers, k)
    assert arrays["weights"].dtype == np.float16
    assert arrays["hidden"].shape == (total, layers, CFG["hidden_size"])
    assert arrays["hidden"].dtype == np.float16
    assert arrays["phase"].dtype == np.uint8
    assert arrays["phase"].tolist() == [0] * len(PROMPT) + [1] * (STEPS - 1)
    assert arrays["tokens"].tolist() == PROMPT + generated[:-1]
    assert np.isfinite(arrays["hidden"]).all()
    assert info["hidden_absmax"] > 0

    routes = arrays["routes"]
    assert routes.min() >= 0 and routes.max() < experts
    w = arrays["weights"].astype(np.float32)
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=2e-3)
    assert (np.diff(w, axis=-1) <= 1e-3).all()

    engine = {}
    for layer, flat in seen:
        rows = np.asarray(flat).reshape(-1, k)
        engine.setdefault(layer, []).append(rows)
    for layer in range(layers):
        rows = np.concatenate(engine[layer])
        assert rows.shape[0] == total
        np.testing.assert_array_equal(np.sort(rows, -1), np.sort(routes[:, layer], -1))


def test_hidden_matches_gate_input(model):
    _, arrays, _ = trace.record(model, PROMPT, 2)
    for layer, block in enumerate(model.layers):
        h = mx.array(arrays["hidden"][:, layer].astype(np.float32))
        logits = np.array(block.mlp.gate(h))
        top = np.argsort(-logits, -1)[:, : CFG["num_experts_per_tok"]]
        agree = (np.sort(top, -1) == np.sort(arrays["routes"][:, layer], -1)).all(-1).mean()
        assert agree >= 0.9


def test_write_and_load_roundtrip(model, tmp_path):
    _, arrays, info = trace.record(model, PROMPT, 3)
    meta = trace.build_meta("tiny-run", arrays, info, prompt_id="tiny", prompt_category="test")
    out = trace.write_trace(tmp_path / "tiny-run", meta, arrays)
    loaded = trace.load_trace(out)
    assert loaded["meta"]["format"] == 1
    assert loaded["meta"]["n_prefill"] == len(PROMPT)
    assert loaded["meta"]["n_decode"] == 2
    assert loaded["meta"]["hidden_dtype"] == "float16"
    for name in trace.ARRAYS:
        np.testing.assert_array_equal(loaded[name], arrays[name])
    assert trace.list_traces(tmp_path) == [out]
    with pytest.raises(FileExistsError):
        trace.write_trace(out, meta, arrays)


def test_nested_capture_rejected(model):
    with trace.capture(model), pytest.raises(RuntimeError), trace.capture(model):
        pass


def test_prompt_split_is_stable():
    prompts = get_prompts()
    ids = [p.id for p in prompts]
    assert len(ids) == len(set(ids)) >= 28
    evals = get_prompts("eval")
    assert {p.id for p in evals} == EVAL_IDS
    assert 0.2 <= len(evals) / len(prompts) <= 0.34
    assert {p.category for p in evals} == {p.category for p in prompts}
    assert len(get_prompts("train")) + len(evals) == len(prompts)
    assert split_of(evals[0].id) == "eval"
    assert get_prompt(ids[0]).id == ids[0]
    with pytest.raises(ValueError):
        get_prompts("test")
