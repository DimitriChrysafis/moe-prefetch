from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from colibri_runtime.generation import generate_tokens
from colibri_runtime.load import load_model
from colibri_runtime.model import StreamedSparseMoeBlock

from moe_prefetch import runtime
from moe_prefetch.predictors import FutureGatePredictor

pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_trace import PROMPT, build_checkpoint  # noqa: E402

STEPS = 6


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-runtime")
    build_checkpoint(path)
    return path


def fresh(path):
    model, _ = load_model(
        path,
        expert_budget_bytes=1 << 20,
        ple_budget_bytes=1 << 20,
        io_workers=2,
        prefetch_experts=1,
        prefetch_ple=1,
    )
    return model


def gates_of(model):
    return np.stack([np.array(layer.mlp.gate.weight.astype(mx.float32)) for layer in model.layers])


def run(model):
    generated, _ = generate_tokens(model, PROMPT, STEPS)
    logits = model(mx.array([PROMPT]))
    mx.eval(logits)
    return generated, np.array(logits.astype(mx.float32))


def test_prefetch_keeps_outputs_identical(checkpoint):
    base_model = fresh(checkpoint)
    try:
        expected_tokens, expected_logits = run(base_model)
    finally:
        base_model.close()
    model = fresh(checkpoint)
    original = StreamedSparseMoeBlock.__call__
    try:
        predictor = FutureGatePredictor(gates_of(model), lookaheads=(1,))
        prefetcher = runtime.enable_predictive_prefetch(model, predictor, k=2, heuristic=False)
        tokens, logits = run(model)
        assert prefetcher.calls > 0
        assert prefetcher.requested > 0
        assert model.expert_store.stats.prefetch_submitted >= prefetcher.requested
    finally:
        runtime.disable_predictive_prefetch(model)
        model.close()
    assert tokens == expected_tokens
    np.testing.assert_array_equal(logits, expected_logits)
    assert StreamedSparseMoeBlock.__call__ is original
    assert "prepare" not in model.expert_store.__dict__
    assert "prefetch_predictions" not in model.expert_store.__dict__


def test_protect_and_parallel_keep_outputs_identical(checkpoint):
    base_model = fresh(checkpoint)
    try:
        expected_tokens, expected_logits = run(base_model)
    finally:
        base_model.close()
    model = fresh(checkpoint)
    try:
        predictor = FutureGatePredictor(gates_of(model), lookaheads=(1,))
        runtime.enable_predictive_prefetch(
            model, predictor, k=2, protect=True, parallel_demand=True
        )
        tokens, logits = run(model)
    finally:
        runtime.disable_predictive_prefetch(model)
        model.close()
    assert tokens == expected_tokens
    np.testing.assert_array_equal(logits, expected_logits)


def test_double_enable_rejected(checkpoint):
    model = fresh(checkpoint)
    try:
        predictor = FutureGatePredictor(gates_of(model), lookaheads=(1,))
        runtime.enable_predictive_prefetch(model, predictor)
        with pytest.raises(RuntimeError):
            runtime.enable_predictive_prefetch(model, predictor)
    finally:
        runtime.disable_predictive_prefetch(model)
        model.close()


def test_top_experts_votes_across_tokens():
    scores = np.array([[0.0, 5.0, 4.0, 0.0], [0.0, 5.0, 0.0, 4.0], [3.0, 5.0, 0.0, 0.0]])
    assert runtime.top_experts(scores, 1) == [1]
    assert runtime.top_experts(scores[:1], 2) == [1, 2]
    assert runtime.top_experts(np.array([1.0, 3.0, 2.0]), 5) == [1, 2, 0]
