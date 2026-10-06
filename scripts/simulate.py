import argparse
import json
import sys
from pathlib import Path

from moe_prefetch import sim
from moe_prefetch.predictors import load_predictor
from moe_prefetch.predictors.data import list_runs, split_runs
from moe_prefetch.sim import SimConfig, Timing, Trace, simulate

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / "qwen3.8next/models/pipenetwork-Qwen3.8-Flash-Next-MLX-4bit"
PHASES = ("prefill", "decode")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(ROOT / "traces"))
    ap.add_argument("--split", default="eval", choices=["train", "eval", "all"])
    ap.add_argument("--policy", nargs="+", choices=sim.POLICIES, default=["heuristic"])
    ap.add_argument("--predictor")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lookaheads", type=int, nargs="+", default=[1])
    ap.add_argument("--cache-gib", type=float, default=8.0)
    ap.add_argument("--cache-policy", choices=["lru", "lfu", "adaptive"], default="adaptive")
    ap.add_argument("--prefetch-count", type=int, default=2)
    ap.add_argument("--protect", choices=sim.PROTECT, default="none")
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--parallel-demand", action="store_true")
    ap.add_argument("--max-speculative", type=int)
    ap.add_argument("--expert-bytes", type=int)
    ap.add_argument("--kmax", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thread-bandwidth", type=float, default=0.75e9)
    ap.add_argument("--ssd-bandwidth", type=float, default=3.0e9)
    ap.add_argument("--layer-compute", type=float, default=1.38e-3)
    ap.add_argument("--token-compute", type=float, default=0.33e-3)
    ap.add_argument("--task-latency", type=float, default=0.2e-3)
    ap.add_argument("--predictor-seconds", type=float, default=0.0)
    ap.add_argument("--max-runs", type=int)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "sim"))
    return ap.parse_args(argv)


def aggregate(reports: list[dict]) -> dict:
    out = {}
    for phase in PHASES:
        total = {}
        for key in (
            "tokens",
            "seconds",
            "stall_seconds",
            "gets",
            "hits",
            "loads",
            "loaded_bytes",
            "prefetch_submitted",
            "prefetch_hits",
            "prefetch_inflight_hits",
            "prefetch_wasted",
            "evictions",
        ):
            total[key] = sum(r[phase][key] for r in reports)
        out[phase] = {
            **total,
            "hit_rate": total["hits"] / total["gets"] if total["gets"] else 0.0,
            "prefetch_hit_rate": (
                total["prefetch_hits"] / total["prefetch_submitted"]
                if total["prefetch_submitted"]
                else 0.0
            ),
            "tok_s": total["tokens"] / total["seconds"] if total["seconds"] else 0.0,
            "gib_per_token": (
                total["loaded_bytes"] / sim.GIB / total["tokens"] if total["tokens"] else 0.0
            ),
        }
    return out


def table(results: dict[str, dict]) -> str:
    head = (
        "| policy | phase | tok/s | stall s | hit rate | prefetch hit | pf hits/submitted "
        "| wasted | GiB/tok |"
    )
    lines = [head, "|" + "---|" * 9]
    for name, res in results.items():
        for phase in PHASES:
            s = res[phase]
            lines.append(
                f"| {name} | {phase} | {s['tok_s']:.2f} | {s['stall_seconds']:.1f} "
                f"| {s['hit_rate']:.1%} | {s['prefetch_hit_rate']:.1%} "
                f"| {s['prefetch_hits']}/{s['prefetch_submitted']} "
                f"| {s['prefetch_wasted']} | {s['gib_per_token']:.3f} |"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    runs = list_runs(Path(args.traces))
    if args.split != "all":
        runs = split_runs(runs)[args.split]
    runs = runs[: args.max_runs]
    if not runs:
        print(f"no {args.split} traces in {args.traces}", file=sys.stderr)
        return 1
    traces = [Trace.load(r.path) for r in runs]
    tokens = sum(len(t.routes) for t in traces)
    print(f"{len(traces)} {args.split} runs, {tokens} tokens", file=sys.stderr)

    expert_bytes = args.expert_bytes or sim.checkpoint_expert_bytes(MODEL)
    timing = Timing(
        layer_compute=args.layer_compute,
        token_compute=args.token_compute,
        thread_bandwidth=args.thread_bandwidth,
        ssd_bandwidth=args.ssd_bandwidth,
        task_latency=args.task_latency,
        workers=args.workers,
        predictor_seconds=args.predictor_seconds,
    )
    predictor = load_predictor(args.predictor) if args.predictor else None
    ranks = {}
    need_ranks = set(args.policy) & {"predictor", "hybrid"}
    if need_ranks and predictor is None:
        print("predictor/hybrid policy needs --predictor <path>", file=sys.stderr)
        return 2

    results, raw = {}, []
    for policy in args.policy:
        config = SimConfig(
            policy=policy,
            k=args.k,
            lookaheads=tuple(args.lookaheads),
            cache_gib=args.cache_gib,
            cache_policy=args.cache_policy,
            prefetch_count=args.prefetch_count,
            protect=args.protect,
            chunk=args.chunk,
            fill=args.fill,
            parallel_demand=args.parallel_demand,
            max_speculative=args.max_speculative,
        )
        reports = []
        for trace in traces:
            if config.uses_ranks and policy != "oracle":
                key = (id(trace), id(predictor))
                if key not in ranks:
                    ranks[key] = sim.Ranks.from_predictor(trace, predictor, args.kmax)
                report = simulate(trace, config, timing, expert_bytes, ranks[key])
            else:
                report = simulate(trace, config, timing, expert_bytes)
            reports.append(report)
            raw.append(report)
        results[policy] = aggregate(reports)
        print(f"{policy} done", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {k: v for k, v in vars(args).items() if k != "out"},
        "aggregate": results,
        "runs": raw,
    }
    path = out / "latest.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(table(results))
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
