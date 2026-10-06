import json

import numpy as np
import pytest

from moe_prefetch.gates import load_gates
from moe_prefetch.predictors import (
    FutureGatePredictor,
    HeuristicPredictor,
    LinearPredictor,
    load_predictor,
)
from moe_prefetch.predictors.data import list_runs, split_runs
from moe_prefetch.predictors.evaluate import evaluate, latency
from moe_prefetch.predictors.metrics import recall_at_k
from moe_prefetch.predictors.synthetic import random_gates, write_trace
from moe_prefetch.predictors.train import Config, fit_linear

LAYERS, EXPERTS, DIM, TOP_K = 6, 32, 16, 4


@pytest.fixture
def gates():
    return random_gates(LAYERS, EXPERTS, DIM, seed=1)


@pytest.fixture
def traces(tmp_path, gates):
    root = tmp_path / "traces"
    for i in range(8):
        write_trace(root, f"run{i}", gates, n_prefill=20, n_decode=10, top_k=TOP_K, seed=i)
    return root


def linear(gates, rank=2, seed=0):
    rng = np.random.default_rng(seed)
    n = 3
    return LinearPredictor(
        gates,
        rng.uniform(0.5, 1.5, (LAYERS, n, EXPERTS)),
        rng.standard_normal((LAYERS, n, EXPERTS)),
        rng.standard_normal((LAYERS, n, DIM, rank)),
        rng.standard_normal((LAYERS, n, rank, EXPERTS)),
    )


def test_recall_hand_made():
    scores = np.array([[5.0, 4, 3, 2, 1, 0]])
    assert recall_at_k(scores, [[0, 2]], 1)[0] == 0.5
    assert recall_at_k(scores, [[0, 2]], 2)[0] == 0.5
    assert recall_at_k(scores, [[0, 2]], 3)[0] == 1.0
    assert recall_at_k(scores, [[5, 4]], 4)[0] == 0.0
    assert recall_at_k(scores, [[0, 0, 5]], 1)[0] == 0.5
    assert recall_at_k(scores, [[5, 4]], 99)[0] == 1.0


def test_recall_rows_independent():
    scores = np.array([[0.0, 1, 2, 3], [3, 2, 1, 0]])
    np.testing.assert_allclose(recall_at_k(scores, [[3, 2], [3, 2]], 2), [1.0, 0.0])


def test_trace_listing_and_split(traces):
    (traces / "partial").mkdir()
    np.save(traces / "partial" / "routes.npy", np.zeros((3, LAYERS, TOP_K), np.int16))
    runs = list_runs(traces)
    assert len(runs) == 8
    split = split_runs(runs)
    assert {r.run_id for r in split["train"]} | {r.run_id for r in split["eval"]} == {
        r.run_id for r in runs
    }
    assert not {r.prompt_id for r in split["train"]} & {r.prompt_id for r in split["eval"]}
    assert runs[0].routes.shape == (30, LAYERS, TOP_K)
    assert runs[0].hidden(2).shape == (30, DIM)


@pytest.mark.parametrize("kind", ["heuristic", "future_gate", "linear"])
def test_scores_shapes(kind, gates):
    predictor = {
        "heuristic": lambda: HeuristicPredictor((1, 2, 3), LAYERS, EXPERTS),
        "future_gate": lambda: FutureGatePredictor(gates),
        "linear": lambda: linear(gates),
    }[kind]()
    for layer in range(LAYERS):
        out = predictor.scores(layer, np.ones((5, DIM), np.float32))
        assert sorted(out) == [layer + d for d in (1, 2, 3) if layer + d < LAYERS]
        for s in out.values():
            assert s.shape == (5, EXPERTS)
            assert s.dtype == np.float32
    assert predictor.scores(0, np.ones(DIM))[1].shape == (1, EXPERTS)


@pytest.mark.parametrize("kind", ["heuristic", "future_gate", "linear"])
def test_save_load_roundtrip(kind, gates, tmp_path):
    predictor = {
        "heuristic": lambda: HeuristicPredictor((1, 2), LAYERS, EXPERTS),
        "future_gate": lambda: FutureGatePredictor(gates, norm=False),
        "linear": lambda: linear(gates),
    }[kind]()
    predictor.save(tmp_path / kind)
    assert json.loads((tmp_path / kind / "meta.json").read_text())["kind"] == kind
    loaded = load_predictor(tmp_path / kind)
    assert type(loaded) is type(predictor)
    assert loaded.lookaheads == predictor.lookaheads
    x = np.random.default_rng(0).standard_normal((3, DIM)).astype(np.float32)
    a, b = predictor.scores(1, x), loaded.scores(1, x)
    assert a.keys() == b.keys()
    for t in a:
        np.testing.assert_allclose(a[t], b[t], rtol=1e-2, atol=1e-2)


def test_load_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_predictor(tmp_path / "nope")


def test_future_gate_matches_router(gates):
    x = np.random.default_rng(0).standard_normal((4, DIM)).astype(np.float32)
    out = FutureGatePredictor(gates, norm=False).scores(0, x)
    np.testing.assert_allclose(out[2], x @ gates[2].T, rtol=1e-5, atol=1e-5)
    normed = FutureGatePredictor(gates, norm=True).scores(0, x)
    assert (np.argsort(out[2]) == np.argsort(normed[2])).all()


def test_linear_identity_is_future_gate(gates):
    n = 3
    ones, zeros = np.ones((LAYERS, n, EXPERTS)), np.zeros((LAYERS, n, EXPERTS))
    lp = LinearPredictor(gates, ones, zeros)
    fg = FutureGatePredictor(gates)
    x = np.random.default_rng(1).standard_normal((2, DIM))
    for t, s in lp.scores(1, x).items():
        np.testing.assert_allclose(s, fg.scores(1, x)[t], rtol=1e-5, atol=1e-5)


def test_heuristic_tracks_frequency():
    h = HeuristicPredictor((1,), 3, 8)
    h.update(1, np.array([3, 5]))
    h.update(1, np.array([5, 6]))
    s = h.scores(0, np.zeros(4))[1][0]
    assert s.argmax() == 5
    assert s[3] == s[6] == 1
    nxt = h.next_token(1, np.array([3, 5]))
    assert set(np.flatnonzero(nxt)) == {5, 6}
    h.reset()
    assert not h.scores(0, np.zeros(4))[1].any()


def test_evaluate_synthetic(traces, gates):
    runs = list_runs(traces)
    fg = evaluate(FutureGatePredictor(gates), runs, ks=(4, 8))
    heur = evaluate(HeuristicPredictor((1, 2, 3), LAYERS, EXPERTS), runs, ks=(4, 8))
    assert fg["tokens"] == {"prefill": 160, "decode": 80}
    assert set(heur["summary"]["decode"]) == {"1", "2", "3", "next"}
    for phase in ("prefill", "decode"):
        row = fg["summary"][phase]
        assert row["1"]["8"] >= row["1"]["4"]
        assert row["1"]["4"] > heur["summary"][phase]["1"]["4"]
        curve = fg["per_layer"][phase]["1"]["4"]
        assert len(curve) == LAYERS and curve[0] is None and curve[1] is not None
    assert latency(FutureGatePredictor(gates), DIM, LAYERS, repeats=2)["median_us"] > 0


def test_fit_linear_deterministic(traces, gates, tmp_path):
    runs = split_runs(list_runs(traces))["train"]
    cfg = Config(rank=2, epochs=3, batch=64, threads=1)
    a = fit_linear(runs, gates, cfg)
    b = fit_linear(runs, gates, cfg)
    for t, s in a.scores(0, np.ones(DIM)).items():
        np.testing.assert_array_equal(s, b.scores(0, np.ones(DIM))[t])
    a.save(tmp_path / "lin")
    assert load_predictor(tmp_path / "lin").rank == 2


def test_load_gates_from_shards(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 2}}))
    weights = {
        f"language_model.model.layers.{i}.mlp.gate.weight": torch.randn(4, 8).to(torch.bfloat16)
        for i in range(2)
    }
    save_file(weights, str(model / "model-00001.safetensors"))
    index = {"weight_map": dict.fromkeys(weights, "model-00001.safetensors")}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    cache = tmp_path / "gates.safetensors"
    gates = load_gates(model, cache)
    assert gates.shape == (2, 4, 8) and gates.dtype == np.float32
    expected = weights["language_model.model.layers.1.mlp.gate.weight"].float().numpy()
    np.testing.assert_array_equal(gates[1], expected)
    assert cache.exists()
    np.testing.assert_array_equal(load_gates(tmp_path / "missing", cache), gates)
