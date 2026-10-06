from __future__ import annotations

from collections import Counter
from typing import Any

import mlx.core as mx
import numpy as np
from colibri_runtime.model import StreamedSparseMoeBlock

from .predictors.base import Predictor

_ACTIVE: dict[int, _Handle] = {}


def top_experts(scores: np.ndarray, k: int) -> list[int]:
    rows = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    take = min(max(int(k), 0), rows.shape[1])
    if take == 0:
        return []
    votes: Counter[int] = Counter()
    totals: Counter[int] = Counter()
    for row in rows:
        for expert in np.argpartition(-row, take - 1)[:take]:
            votes[int(expert)] += 1
            totals[int(expert)] += row[expert]
    ranked = sorted(votes, key=lambda e: (-votes[e], -totals[e], e))
    return ranked[:take]


def enable_predictive_prefetch(
    model,
    predictor: Predictor,
    k: int = 8,
    lookaheads: tuple[int, ...] | None = None,
    heuristic: bool = False,
    protect: bool = False,
    chunk: int = 1,
    parallel_demand: bool = False,
) -> _Handle:
    if id(model) in _ACTIVE:
        raise RuntimeError("predictive prefetch is already enabled for this model")
    handle = _Handle(model, predictor)
    _ACTIVE[id(model)] = handle
    try:
        handle.install(k, lookaheads, heuristic, protect, chunk, parallel_demand)
    except BaseException:
        _ACTIVE.pop(id(model), None)
        handle.disable()
        raise
    return handle


def disable_predictive_prefetch(model) -> None:
    handle = _ACTIVE.pop(id(model), None)
    if handle is not None:
        handle.disable()


class _Handle:
    def __init__(self, model, predictor: Predictor) -> None:
        self.model = model
        self.predictor = predictor
        self.calls = 0
        self.requested = 0
        self.pinned: set[tuple[int, int]] = set()
        self._restore: list[tuple[Any, str, Any, bool]] = []

    def _patch(self, obj, name: str, value) -> None:
        had = name in getattr(obj, "__dict__", {})
        self._restore.append((obj, name, getattr(obj, name), had))
        setattr(obj, name, value)

    def install(
        self,
        k: int,
        lookaheads: tuple[int, ...] | None,
        heuristic: bool,
        protect: bool,
        chunk: int,
        parallel_demand: bool,
    ) -> None:
        store = self.model.expert_store
        predictor = self.predictor
        want = tuple(lookaheads) if lookaheads else tuple(predictor.lookaheads)
        chunk = max(1, int(chunk))
        stateful = getattr(predictor, "stateful", False)
        state: dict[str, Any] = {}

        original_call = StreamedSparseMoeBlock.__call__
        original_predictions = store.prefetch_predictions
        original_route = store.record_route
        original_put = store.cache.put
        original_get = store.get

        def missing(layer: int, expert: int) -> bool:
            key = (layer, expert)
            return store.cache.peek(key) is None and key not in store._futures

        def release(key: tuple[int, int]) -> None:
            if key in self.pinned:
                self.pinned.discard(key)
                store.cache.unpin(key)

        def call(block, x):
            if stateful and block.layer_idx == 0 and x.shape[1] > 1:
                predictor.reset()
            state["x"] = x
            out = original_call(block, x)
            for key in [key for key in self.pinned if key[0] == block.layer_idx]:
                release(key)
            return out

        def record_route(layer: int, experts) -> None:
            state["routes"] = np.asarray(list(experts), dtype=np.int64).reshape(-1)
            original_route(layer, experts)

        def prefetch_predictions(layer: int, selected) -> None:
            if heuristic:
                original_predictions(layer, selected)
            x = state.get("x")
            if x is None:
                return
            self.calls += 1
            hidden = np.asarray(x.astype(mx.float32)).reshape(-1, x.shape[-1])
            scores = predictor.scores(layer, hidden)
            if stateful:
                routes = state.get("routes")
                if routes is not None:
                    predictor.update(layer, routes)
            for target, score in sorted(scores.items()):
                if target - layer not in want or not 0 <= target < store.num_layers:
                    continue
                picks = top_experts(score, k)
                for start in range(0, len(picks), chunk):
                    experts = [
                        expert for expert in picks[start : start + chunk] if missing(target, expert)
                    ]
                    if not experts:
                        continue
                    submitted = store.request_many(target, experts, prefetch=True)
                    self.requested += len(submitted)
                    if protect:
                        self.pinned.update(submitted)

        def put(key, value, nbytes=None, pin: bool = False, hot: bool = False) -> bool:
            return original_put(key, value, nbytes, pin or key in self.pinned, hot)

        def get(layer: int, expert: int):
            value = original_get(layer, expert)
            release((layer, expert))
            return value

        def prepare(layer: int, experts) -> None:
            store._ensure_open()
            store._harvest_prefetches()
            layer = store._validate_key(layer, 0)[0]
            experts = list(dict.fromkeys(store._validate_key(layer, e)[1] for e in experts))
            for start in range(0, len(experts), 32):
                batch = [
                    e
                    for e in experts[start : start + 32]
                    if store.cache.peek((layer, e)) is None and (layer, e) not in store._futures
                ]
                for expert in batch:
                    store.request_many(layer, [expert])

        self._patch(StreamedSparseMoeBlock, "__call__", call)
        if stateful:
            self._patch(store, "record_route", record_route)
        self._patch(store, "prefetch_predictions", prefetch_predictions)
        if protect:
            self._patch(store.cache, "put", put)
            self._patch(store, "get", get)
        if parallel_demand:
            self._patch(store, "prepare", prepare)

    def disable(self) -> None:
        for obj, name, value, had in reversed(self._restore):
            if had:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        self._restore.clear()
        cache = self.model.expert_store.cache
        for key in self.pinned:
            cache.unpin(key)
        self.pinned.clear()

    def __enter__(self) -> _Handle:
        return self

    def __exit__(self, *_: object) -> None:
        disable_predictive_prefetch(self.model)
