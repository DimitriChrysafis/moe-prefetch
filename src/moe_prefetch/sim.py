from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from colibri_runtime import expert_cache as engine_cache
from colibri_runtime.expert_cache import ExpertCache, _Entry

GIB = 1 << 30
EXPERT_BYTES = 2_764_800
POLICIES = ("none", "heuristic", "predictor", "hybrid", "oracle")
PROTECT = ("none", "prefetched", "predicted")


class Predictor(Protocol):
    lookaheads: tuple[int, ...]

    def scores(self, layer: int, hidden: np.ndarray) -> dict[int, np.ndarray]: ...


def checkpoint_expert_bytes(model_dir: str | Path) -> int:
    from colibri_runtime.storage import SafeTensorIndex

    with SafeTensorIndex(model_dir, workers=1) as index:
        layers = sorted(
            {
                int(name.split(".")[2])
                for name in index.tensors
                if name.startswith("model.layers.") and ".mlp.switch_mlp." in name
            }
        )
        if not layers:
            raise ValueError(f"no streamed experts in {model_dir}")
        return max(
            sum(
                index.tensors[name].row_bytes
                for proj in ("gate", "up", "down")
                for suffix in ("weight", "scales", "biases")
                if (name := f"model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{suffix}")
                in index.tensors
            )
            for layer in layers
        )


@dataclass
class Trace:
    run_id: str
    meta: dict
    routes: np.ndarray
    phase: np.ndarray
    path: Path | None = None
    _hidden: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        if meta.get("format") != 1:
            raise ValueError(f"unsupported trace format in {path}")
        routes = np.load(path / "routes.npy", mmap_mode="r").astype(np.int16)
        phase = np.load(path / "phase.npy").astype(np.uint8)
        if len(phase) != len(routes):
            raise ValueError(f"routes and phase length differ in {path}")
        return cls(meta.get("run_id", path.name), meta, routes, phase, path)

    @property
    def hidden(self) -> np.ndarray:
        if self._hidden is None:
            if self.path is None:
                raise ValueError("trace has no hidden states")
            self._hidden = np.load(self.path / "hidden.npy", mmap_mode="r")
        return self._hidden

    @property
    def layers(self) -> int:
        return self.routes.shape[1]

    @property
    def experts(self) -> int:
        return int(self.meta.get("experts", 512))

    @property
    def n_prefill(self) -> int:
        return int(np.count_nonzero(self.phase == 0))

    def steps(self) -> list[np.ndarray]:
        prefill = np.flatnonzero(self.phase == 0)
        decode = np.flatnonzero(self.phase == 1)
        out = [prefill] if len(prefill) else []
        return out + [decode[i : i + 1] for i in range(len(decode))]


def rank_experts(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    flat = scores.reshape(-1, scores.shape[-1])
    if not flat.size:
        return np.empty(0, dtype=np.int64)
    row = flat.mean(0)
    k = min(k, row.shape[-1])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    top = np.argpartition(-row, k - 1)[:k]
    return top[np.argsort(-row[top], kind="stable")]


class Ranks:
    def __init__(self, kmax: int, steps: int, layers: int) -> None:
        self.kmax = kmax
        self.table: list[list[dict[int, np.ndarray]]] = [
            [{} for _ in range(layers)] for _ in range(steps)
        ]

    def get(self, step: int, layer: int) -> dict[int, np.ndarray]:
        return self.table[step][layer]

    @classmethod
    def from_predictor(cls, trace: Trace, predictor: Predictor, kmax: int = 32) -> Ranks:
        steps = trace.steps()
        out = cls(kmax, len(steps), trace.layers)
        stateful = hasattr(predictor, "update")
        if stateful and hasattr(predictor, "reset"):
            predictor.reset()
        routes = np.asarray(trace.routes)
        for layer in range(trace.layers):
            if stateful:
                break
            hidden = np.asarray(trace.hidden[:, layer], dtype=np.float32)
            scores = predictor.scores(layer, hidden)
            for i, step in enumerate(steps):
                out.table[i][layer] = {
                    t: rank_experts(s[step], kmax).astype(np.int16)
                    for t, s in scores.items()
                    if layer < t < trace.layers
                }
        if stateful:
            for i, step in enumerate(steps):
                for layer in range(trace.layers):
                    hidden = np.asarray(trace.hidden[step, layer], dtype=np.float32)
                    out.table[i][layer] = {
                        t: rank_experts(s, kmax).astype(np.int16)
                        for t, s in predictor.scores(layer, hidden).items()
                        if layer < t < trace.layers
                    }
                    predictor.update(layer, routes[step, layer])
        return out

    @classmethod
    def oracle(cls, trace: Trace, lookaheads: tuple[int, ...], kmax: int = 32) -> Ranks:
        steps = trace.steps()
        out = cls(kmax, len(steps), trace.layers)
        routes = np.asarray(trace.routes)
        for i, step in enumerate(steps):
            for layer in range(trace.layers):
                for d in lookaheads:
                    target = layer + d
                    if not layer < target < trace.layers:
                        continue
                    if len(step) == 1:
                        picks = routes[step[0], target, :kmax]
                    else:
                        counts = np.bincount(routes[step, target].ravel(), minlength=trace.experts)
                        picks = rank_experts(counts[None], kmax)
                        picks = picks[counts[picks] > 0]
                    out.table[i][layer][target] = picks.astype(np.int16)
        return out


class SimCache(ExpertCache):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._heaps: defaultdict[tuple, list] = defaultdict(list)

    def _push(self, key) -> None:
        entry = self._items.get(key)
        if entry is None or entry.pinned:
            return
        if self.policy == "lru":
            rank = (entry.touched,)
        elif self.policy == "lfu":
            rank = (entry.frequency, entry.touched)
        else:
            rank = (
                (entry.frequency + (4 if entry.hot else 0)) / max(entry.nbytes, 1),
                entry.touched,
            )
        heap = self._heaps[(self._partition(key), entry.hot)]
        heapq.heappush(heap, (rank, entry.touched, entry.frequency, key))

    def _top(self, bucket):
        heap = self._heaps.get(bucket)
        while heap:
            _, touched, frequency, key = heap[0]
            entry = self._items.get(key)
            if (
                entry is not None
                and not entry.pinned
                and entry.touched == touched
                and entry.frequency == frequency
                and entry.hot == bucket[1]
            ):
                return heap[0]
            heapq.heappop(heap)
        return None

    def _best(self, partitions, hot: bool):
        best = None
        for partition in partitions:
            top = self._top((partition, hot))
            if top is not None and (best is None or top[0] < best[0]):
                best = top
        return best

    def _victim(self):
        partitions = {bucket[0] for bucket in self._heaps}
        groups = [partitions]
        if self.partitions is not None and self.max_bytes is not None:
            fair = self.max_bytes / self.partitions
            usage = self._partition_usage
            over = {p for p in partitions if usage.get(p, 0) > fair}
            groups = [over, partitions - over]
        for group in groups:
            cold, hot = self._best(group, False), self._best(group, True)
            if cold is not None or hot is not None:
                return (cold or hot)[3]
        return None

    def _update_pinned(self) -> None:
        pass

    def get(self, key, count_miss: bool = True):
        value = super().get(key, count_miss)
        if value is not None:
            self._push(key)
        return value

    def put(self, key, value, nbytes=None, pin: bool = False, hot: bool = False) -> bool:
        size = int(nbytes if nbytes is not None else getattr(value, "nbytes", 1))
        if size < 0:
            raise ValueError("nbytes must be nonnegative")
        if self.max_bytes is not None and size > self.max_bytes:
            self.stats.rejected += 1
            return False
        old = self._items.pop(key, None)
        if old is not None:
            self._remove_bytes(key, old.nbytes)
            pin = pin or old.pinned
            hot = hot or old.hot
        evicted = []
        while (self.max_items is not None and len(self._items) + 1 > self.max_items) or (
            self.max_bytes is not None and self.stats.bytes + size > self.max_bytes
        ):
            victim = self._victim()
            if victim is None:
                for k, entry in reversed(evicted):
                    self._items[k] = entry
                    self.stats.bytes += entry.nbytes
                    self._adjust_partition(k, entry.nbytes)
                    self._push(k)
                if old is not None:
                    self._items[key] = old
                    self.stats.bytes += old.nbytes
                    self._adjust_partition(key, old.nbytes)
                    self._push(key)
                self.stats.rejected += 1
                return False
            entry = self._items.pop(victim)
            heapq.heappop(self._heaps[(self._partition(victim), entry.hot)])
            self._remove_bytes(victim, entry.nbytes)
            evicted.append((victim, entry))
        self._items[key] = _Entry(value, size, pin, hot, 1, engine_cache.monotonic())
        self.stats.bytes += size
        self._adjust_partition(key, size)
        self.stats.evictions += len(evicted)
        self.stats.peak_bytes = max(self.stats.peak_bytes, self.stats.bytes)
        self._push(key)
        return True

    def unpin(self, key) -> bool:
        if key not in self._items:
            return False
        self._items[key].pinned = False
        self._update_pinned()
        self._push(key)
        self._evict_if_needed()
        return True

    def mark_hot(self, key, hot: bool = True) -> bool:
        out = super().mark_hot(key, hot)
        if out:
            self._push(key)
        return out


@dataclass(frozen=True)
class Timing:
    layer_compute: float = 1.38e-3
    token_compute: float = 0.33e-3
    thread_bandwidth: float = 0.75e9
    ssd_bandwidth: float = 3.0e9
    task_latency: float = 0.2e-3
    workers: int = 6
    predictor_seconds: float = 0.0


@dataclass(frozen=True)
class SimConfig:
    policy: str = "heuristic"
    k: int = 8
    lookaheads: tuple[int, ...] = (1,)
    cache_gib: float = 8.0
    cache_policy: str = "adaptive"
    prefetch_count: int = 2
    protect: str = "none"
    chunk: int = 1
    fill: bool = False
    parallel_demand: bool = False
    max_speculative: int | None = None

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}")
        if self.protect not in PROTECT:
            raise ValueError(f"unknown protect mode {self.protect!r}")
        if self.chunk < 1:
            raise ValueError("chunk must be positive")

    @property
    def uses_ranks(self) -> bool:
        return self.policy in {"predictor", "hybrid", "oracle"}

    @property
    def uses_heuristic(self) -> bool:
        return self.policy in {"heuristic", "hybrid"}


@dataclass
class Stats:
    tokens: int = 0
    seconds: float = 0.0
    stall_seconds: float = 0.0
    gets: int = 0
    hits: int = 0
    loads: int = 0
    loaded_bytes: int = 0
    prefetch_submitted: int = 0
    prefetch_hits: int = 0
    prefetch_inflight_hits: int = 0
    prefetch_wasted: int = 0
    evictions: int = 0

    def report(self) -> dict:
        out = asdict(self)
        out["hit_rate"] = self.hits / self.gets if self.gets else 0.0
        out["prefetch_hit_rate"] = (
            self.prefetch_hits / self.prefetch_submitted if self.prefetch_submitted else 0.0
        )
        exposed = self.gets - self.hits + self.prefetch_hits - self.prefetch_inflight_hits
        out["miss_coverage"] = self.prefetch_hits / exposed if exposed else 0.0
        out["tok_s"] = self.tokens / self.seconds if self.seconds else 0.0
        out["gib_per_token"] = self.loaded_bytes / GIB / self.tokens if self.tokens else 0.0
        return out


class IOQueue:
    def __init__(self, timing: Timing) -> None:
        self.timing = timing
        self.free = [0.0] * timing.workers
        heapq.heapify(self.free)

    def submit(self, now: float, nbytes: int) -> float:
        start = max(now, self.free[0])
        busy = sum(f > start for f in self.free) + 1
        bandwidth = min(self.timing.thread_bandwidth, self.timing.ssd_bandwidth / busy)
        done = start + self.timing.task_latency + nbytes / bandwidth
        heapq.heapreplace(self.free, done)
        return done


@dataclass
class SimStore:
    layers: int
    experts: int
    expert_bytes: int
    config: SimConfig
    timing: Timing
    cache: ExpertCache = field(init=False)
    io: IOQueue = field(init=False)

    def __post_init__(self) -> None:
        budget = int(self.config.cache_gib * GIB)
        if budget < self.expert_bytes:
            raise ValueError("cache must hold at least one expert")
        self.cache = SimCache(
            max_bytes=budget, policy=self.config.cache_policy, partitions=self.layers
        )
        self.io = IOQueue(self.timing)
        slots = budget // self.expert_bytes
        count = self.config.prefetch_count if self.config.uses_heuristic else 0
        default = min(slots, max(self.timing.workers * 2, count * self.layers))
        self.max_speculative = self.config.max_speculative or default
        self.futures: dict[tuple[int, int], tuple[float, bool]] = {}
        self.pending: set[tuple[int, int]] = set()
        self.ready: set[tuple[int, int]] = set()
        self.protected: set[tuple[int, int]] = set()
        self.freq: defaultdict[int, Counter] = defaultdict(Counter)
        self.recent: defaultdict[int, deque] = defaultdict(lambda: deque(maxlen=8))
        self.transitions: defaultdict[tuple[int, int], Counter] = defaultdict(Counter)
        self.stats = Stats()

    def known(self, key: tuple[int, int]) -> bool:
        return self.cache.peek(key) is not None or key in self.futures

    def submit(self, now: float, keys: list[tuple[int, int]], prefetch: bool) -> None:
        if not keys:
            return
        done = self.io.submit(now, len(keys) * self.expert_bytes)
        for key in keys:
            self.futures[key] = (done, prefetch)
        self.stats.loads += len(keys)
        self.stats.loaded_bytes += len(keys) * self.expert_bytes
        if prefetch:
            self.pending.update(keys)
            self.stats.prefetch_submitted += len(keys)

    def prefetch(self, now: float, layer: int, experts, chunk: int = 1) -> None:
        room = self.max_speculative - len(self.pending)
        keys = []
        for expert in experts:
            key = (layer, int(expert))
            if self.known(key):
                if self.config.protect == "predicted":
                    self.protect(key)
                continue
            if len(keys) >= room:
                break
            keys.append(key)
        for start in range(0, len(keys), chunk):
            self.submit(now, keys[start : start + chunk], True)
        if self.config.protect != "none":
            self.protected.update(keys)

    def harvest(self, now: float) -> None:
        done = sorted(
            (self.futures[key][0], key) for key in self.pending if self.futures[key][0] <= now
        )
        for _, key in done:
            hot = self.freq[key[0]][key[1]] >= 4
            evictions = self.cache.stats.evictions
            inserted = self.cache.put(key, 1, self.expert_bytes, pin=key in self.protected, hot=hot)
            self.stats.evictions += self.cache.stats.evictions - evictions
            del self.futures[key]
            self.pending.discard(key)
            if inserted:
                self.ready.add(key)
            else:
                self.stats.prefetch_wasted += 1
        stale = [key for key in self.ready if self.cache.peek(key) is None]
        self.ready.difference_update(stale)
        self.stats.prefetch_wasted += len(stale)

    def prepare(self, now: float, layer: int, experts: list[int]) -> None:
        self.harvest(now)
        for start in range(0, len(experts), 32):
            missing = [
                (layer, e) for e in experts[start : start + 32] if not self.known((layer, e))
            ]
            if self.config.parallel_demand:
                for key in missing:
                    self.submit(now, [key], False)
            else:
                self.submit(now, missing, False)

    def release(self, key: tuple[int, int]) -> None:
        if key in self.protected:
            self.protected.discard(key)
            self.cache.unpin(key)

    def get(self, now: float, key: tuple[int, int]) -> float:
        self.stats.gets += 1
        if self.cache.get(key) is not None:
            self.stats.hits += 1
            if key in self.ready:
                self.ready.discard(key)
                self.stats.prefetch_hits += 1
            self.release(key)
            return now
        if key not in self.futures:
            self.submit(now, [key], False)
        done, prefetch = self.futures.pop(key)
        if done > now:
            self.stats.stall_seconds += done - now
            now = done
        if prefetch:
            self.pending.discard(key)
            self.stats.prefetch_hits += 1
            self.stats.prefetch_inflight_hits += 1
        hot = self.freq[key[0]][key[1]] >= 4
        evictions = self.cache.stats.evictions
        self.cache.put(key, 1, self.expert_bytes, hot=hot)
        self.stats.evictions += self.cache.stats.evictions - evictions
        self.release(key)
        return now

    def record_route(self, layer: int, routed: list[int]) -> None:
        selected = tuple(sorted(set(routed)))
        previous = self.recent[layer][-1] if self.recent[layer] else ()
        for old in previous:
            self.transitions[(layer, old)].update(selected)
        self.freq[layer].update(routed)
        self.recent[layer].append(selected)
        for expert in selected:
            self.cache.mark_hot((layer, expert), self.freq[layer][expert] >= 4)

    def heuristic(self, now: float, layer: int, selected: list[int]) -> None:
        count = self.config.prefetch_count
        if not count:
            return
        candidates = Counter()
        for expert in selected:
            candidates.update(self.transitions[(layer, expert)])
        if not candidates:
            candidates.update(self.freq[layer])
        if layer + 1 < self.layers:
            candidates.update({e: n * 2 for e, n in self.freq[layer + 1].items()})
        for expert, _ in candidates.most_common(count):
            self.prefetch(now, layer, [expert])
        if layer + 1 < self.layers:
            for expert, _ in self.freq[layer + 1].most_common(count):
                self.prefetch(now, layer + 1, [expert])

    def protect(self, key: tuple[int, int]) -> None:
        if key in self.protected:
            return
        if key in self.pending or (self.cache.peek(key) is not None and self.cache.pin(key)):
            self.protected.add(key)

    def expire(self, layer: int) -> None:
        for key in [key for key in self.protected if key[0] == layer]:
            self.release(key)

    def finish(self) -> None:
        self.stats.prefetch_wasted += len(self.pending) + len(self.ready)


def simulate(
    trace: Trace,
    config: SimConfig,
    timing: Timing | None = None,
    expert_bytes: int = EXPERT_BYTES,
    ranks: Ranks | None = None,
) -> dict:
    timing = timing or Timing()
    if config.uses_ranks and ranks is None:
        if config.policy != "oracle":
            raise ValueError(f"policy {config.policy} needs predictor ranks")
        ranks = Ranks.oracle(trace, config.lookaheads, max(config.k, 10))
    store = SimStore(trace.layers, trace.experts, expert_bytes, config, timing)
    routes = np.asarray(trace.routes)
    prefill_stats = None
    now = 0.0
    for index, step in enumerate(trace.steps()):
        decode = len(step) == 1 and trace.phase[step[0]] == 1
        if decode and prefill_stats is None:
            prefill_stats = store.stats
            store.stats = Stats()
        started = now
        for layer in range(trace.layers):
            now += timing.layer_compute + timing.token_compute * (len(step) - 1)
            routed = routes[step, layer].ravel()
            unique = np.unique(routed).tolist()
            store.prepare(now, layer, unique)
            if config.uses_ranks:
                now += timing.predictor_seconds
                for target, picks in sorted(ranks.get(index, layer).items()):
                    if target - layer not in config.lookaheads:
                        continue
                    if config.fill:
                        picks = [e for e in picks if not store.known((target, int(e)))]
                    store.prefetch(now, target, picks[: config.k], config.chunk)
            for expert in unique:
                now = store.get(now, (layer, expert))
            store.record_route(layer, routed.tolist())
            store.expire(layer)
            if config.uses_heuristic:
                store.heuristic(now, layer, unique)
        store.stats.seconds += now - started
        store.stats.tokens += len(step)
    store.finish()
    if prefill_stats is None:
        prefill_stats, store.stats = store.stats, Stats()
    return {
        "run_id": trace.run_id,
        "config": asdict(config),
        "decode": store.stats.report(),
        "prefill": prefill_stats.report(),
    }
