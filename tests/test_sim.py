import itertools

import numpy as np
import pytest
from colibri_runtime import expert_cache
from colibri_runtime.expert_cache import ExpertCache

from moe_prefetch.predictors.synthetic import random_gates, write_trace
from moe_prefetch.sim import (
    GIB,
    IOQueue,
    Ranks,
    SimCache,
    SimConfig,
    SimStore,
    Timing,
    Trace,
    rank_experts,
    simulate,
)

LAYERS, EXPERTS, DIM, TOP_K = 8, 48, 16, 4
EXPERT_BYTES = 4096


@pytest.fixture
def clock(monkeypatch):
    ticks = itertools.count()
    monkeypatch.setattr(expert_cache, "monotonic", lambda: float(next(ticks)) * 1e-6)
    return ticks


@pytest.fixture
def traces(tmp_path):
    gates = random_gates(LAYERS, EXPERTS, DIM, seed=3)
    root = tmp_path / "traces"
    for i in range(3):
        write_trace(root, f"run{i}", gates, n_prefill=16, n_decode=8, top_k=TOP_K, seed=i)
    return root


def gib_for(slots: int, nbytes: int = EXPERT_BYTES) -> float:
    return slots * nbytes / GIB


def test_rank_experts_orders_and_caps():
    scores = np.array([[1.0, 5, 2], [3, 5, 0]])
    top = rank_experts(scores, 2)
    assert top.tolist() == [1, 0]
    assert rank_experts(scores, 99).shape == (3,)
    assert rank_experts(np.zeros((0, 4)), 2).size == 0


def test_sim_cache_parity_with_engine(clock):
    rng = np.random.default_rng(7)
    for policy in ("lru", "lfu", "adaptive"):
        real = ExpertCache(max_bytes=24, policy=policy, partitions=4)
        sim = SimCache(max_bytes=24, policy=policy, partitions=4)
        for step in range(400):
            action = rng.integers(0, 10)
            key = (int(rng.integers(0, 4)), int(rng.integers(0, 9)))
            if action < 5:
                size = int(rng.integers(1, 6))
                pin = bool(rng.integers(0, 8) == 0)
                hot = bool(rng.integers(0, 3) == 0)
                assert real.put(key, 1, size, pin=pin, hot=hot) == sim.put(
                    key, 1, size, pin=pin, hot=hot
                )
            elif action < 8:
                got_real = real.get(key) is not None
                got_sim = sim.get(key) is not None
                assert got_real == got_sim
            elif action == 8:
                assert real.pin(key) == sim.pin(key)
                if real.peek(key) is not None and rng.integers(0, 2):
                    assert real.unpin(key) == sim.unpin(key)
            else:
                hot = bool(rng.integers(0, 2))
                assert real.mark_hot(key, hot) == sim.mark_hot(key, hot)
            assert set(real._items) == set(sim._items), f"{policy} step {step}"
        for key in real._items:
            want, have = real._items[key], sim._items[key]
            assert (want.nbytes, want.pinned, want.hot, want.frequency) == (
                have.nbytes,
                have.pinned,
                have.hot,
                have.frequency,
            )
        for field in ("hits", "misses", "evictions", "bytes", "rejected"):
            assert getattr(real.stats, field) == getattr(sim.stats, field)


def test_sim_cache_put_rolls_back(clock):
    sim = SimCache(max_bytes=10, policy="lru", partitions=2)
    real = ExpertCache(max_bytes=10, policy="lru", partitions=2)
    for cache in (sim, real):
        cache.put((0, 0), 1, 6, pin=True)
        cache.put((0, 1), 1, 4)
        assert cache.put((1, 0), 1, 4) is True
        assert cache.put((1, 1), 1, 6) is False
    assert set(sim._items) == set(real._items) == {(0, 0), (1, 0)}
    assert sim.stats.rejected == real.stats.rejected == 1
    assert sim.stats.bytes == real.stats.bytes == 10
    assert (1, 0) in sim._items and not sim._items[(1, 0)].pinned


def load_traces(root):
    return [Trace.load(p) for p in sorted(root.iterdir()) if (p / "meta.json").exists()]


def base_config(**over):
    out = dict(cache_gib=gib_for(24), prefetch_count=2, k=4)
    out.update(over)
    return SimConfig(**out)


def test_simulate_none_policy(traces):
    for trace in load_traces(traces):
        report = simulate(trace, base_config(policy="none"), expert_bytes=EXPERT_BYTES)
        dec, pre = report["decode"], report["prefill"]
        assert pre["tokens"] == 16 and dec["tokens"] == 8
        assert dec["gets"] >= dec["hits"]
        assert dec["loads"] <= dec["gets"]
        assert dec["prefetch_submitted"] == 0
        assert dec["prefetch_hits"] == 0
        assert dec["hit_rate"] < 1.0
        assert dec["tok_s"] > 0


def test_simulate_oracle_covers_misses(traces):
    trace = load_traces(traces)[0]
    config = SimConfig(
        policy="oracle", lookaheads=(1, 2), k=TOP_K, cache_gib=gib_for(96), prefetch_count=2
    )
    report = simulate(trace, config, expert_bytes=EXPERT_BYTES)
    dec = report["decode"]
    assert dec["prefetch_hits"] > 0
    assert dec["prefetch_hit_rate"] > 0.3
    assert (
        dec["stall_seconds"]
        < simulate(trace, base_config(policy="none"), expert_bytes=EXPERT_BYTES)["decode"][
            "stall_seconds"
        ]
    )


def test_simulate_prefetch_accounting_balances(traces):
    for policy in ("heuristic", "oracle"):
        for trace in load_traces(traces):
            config = base_config(policy=policy, lookaheads=(1, 2), k=8)
            report = simulate(trace, config, expert_bytes=EXPERT_BYTES)
            submitted = sum(report[p]["prefetch_submitted"] for p in ("prefill", "decode"))
            resolved = sum(
                report[p]["prefetch_hits"] + report[p]["prefetch_wasted"]
                for p in ("prefill", "decode")
            )
            hits = sum(report[p]["prefetch_hits"] for p in ("prefill", "decode"))
            wasted = sum(report[p]["prefetch_wasted"] for p in ("prefill", "decode"))
            assert resolved <= submitted, (policy, report)
            assert submitted - resolved <= submitted * 0.15, (policy, report)
            assert hits <= submitted and wasted <= submitted
            for phase in ("prefill", "decode"):
                s = report[phase]
                assert s["hits"] <= s["gets"]
                assert s["seconds"] >= s["stall_seconds"]


def test_parallel_demand_lowers_stall(traces):
    real_bytes = 2_764_800
    trace = load_traces(traces)[0]
    serial = simulate(
        trace,
        SimConfig(policy="none", cache_gib=gib_for(600, real_bytes)),
        expert_bytes=real_bytes,
        timing=Timing(),
    )
    parallel = simulate(
        trace,
        SimConfig(policy="none", cache_gib=gib_for(600, real_bytes), parallel_demand=True),
        expert_bytes=real_bytes,
        timing=Timing(),
    )
    assert parallel["prefill"]["stall_seconds"] < serial["prefill"]["stall_seconds"]


def test_io_queue_shares_bandwidth():
    queue = IOQueue(Timing(workers=2, thread_bandwidth=1e9, ssd_bandwidth=2e9, task_latency=0))
    assert queue.submit(0.0, 1000) == pytest.approx(1000 / 1e9)
    assert queue.submit(0.0, 1000) == pytest.approx(1000 / 1e9)
    queue = IOQueue(Timing(workers=3, thread_bandwidth=1e9, ssd_bandwidth=2e9, task_latency=0))
    queue.submit(0.0, 1000)
    queue.submit(0.0, 1000)
    assert queue.submit(0.0, 3000) == pytest.approx(3000 / (2e9 / 3))
    queue = IOQueue(Timing(workers=1, thread_bandwidth=1e9, ssd_bandwidth=2e9, task_latency=0))
    queue.submit(0.0, 1000)
    assert queue.submit(0.0, 1000) == pytest.approx(2 * 1000 / 1e9)


def test_store_protects_prefetch_until_used(traces):
    trace = load_traces(traces)[0]
    config = base_config(policy="heuristic", protect="prefetched")
    store = SimStore(trace.layers, trace.experts, EXPERT_BYTES, config, Timing())
    store.prefetch(0.0, 2, [5, 6, 7])
    assert store.pending == {(2, 5), (2, 6), (2, 7)}
    store.harvest(1.0)
    assert all(store.cache.peek((2, e)) == 1 for e in (5, 6, 7))
    assert all(store.cache._items[(2, e)].pinned for e in (5, 6, 7))
    store.get(1.0, (2, 5))
    assert not store.cache._items[(2, 5)].pinned
    store.expire(2)
    assert not store.protected
    assert not store.cache._items[(2, 6)].pinned


def test_ranks_oracle_predicts_future(traces):
    trace = load_traces(traces)[0]
    ranks = Ranks.oracle(trace, (1, 2), kmax=8)
    steps = trace.steps()
    routes = np.asarray(trace.routes)
    for i, step in enumerate(steps):
        for layer in range(trace.layers - 2):
            for d in (1, 2):
                picks = set(ranks.get(i, layer)[layer + d].tolist())
                if len(step) == 1:
                    actual = set(routes[step[0], layer + d].tolist())
                    assert actual <= picks


def test_trace_steps_split_phase(traces):
    trace = load_traces(traces)[0]
    steps = trace.steps()
    assert len(steps) == 9
    assert len(steps[0]) == 16
    assert all(len(s) == 1 for s in steps[1:])
    assert trace.n_prefill == 16
