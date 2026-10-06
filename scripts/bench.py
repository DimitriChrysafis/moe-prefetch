#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from moe_prefetch.prompts import get_prompt, get_prompts

MODEL = Path.home() / "qwen3.8next/models/pipenetwork-Qwen3.8-Flash-Next-MLX-4bit"
GIB = 1 << 30
MEDIAN_KEYS = (
    "ttft_s",
    "prefill_tok_s",
    "decode_tok_s",
    "prefetch_hit_rate",
    "prefetch_hits",
    "prefetch_submitted",
    "prefetch_wasted",
    "decode_prefetch_hit_rate",
    "decode_prefetch_hits",
    "decode_prefetch_submitted",
    "decode_prefetch_wasted",
    "wait_s",
    "decode_wait_s",
    "loaded_bytes",
    "disk_bytes_read",
    "decode_disk_bytes_read",
    "cache_hits",
    "cache_misses",
    "peak_mlx_bytes",
    "peak_rss_bytes",
    "load_s",
)


def parse_opt(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", nargs="+", choices=["baseline", "predictor"], default=["baseline"])
    p.add_argument("--predictor", type=Path)
    p.add_argument("--opt", type=parse_opt, action="append", default=[])
    p.add_argument("--prompts", nargs="*")
    p.add_argument("--split", choices=["all", "train", "eval"], default="eval")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--expert-budget-gib", type=float, default=8.0)
    p.add_argument("--ple-budget-gib", type=float, default=1.0)
    p.add_argument("--mlx-cache-gib", type=float, default=1.0)
    p.add_argument("--mlx-memory-gib", type=float, default=28.0)
    p.add_argument("--cache-policy", choices=["lru", "lfu", "adaptive"], default="adaptive")
    p.add_argument("--io-workers", type=int, default=6)
    p.add_argument("--expert-prefetch", type=int, default=2)
    p.add_argument("--ple-prefetch", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("artifacts/bench"))
    p.add_argument("--compare", action="store_true")
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--prompt-id", help=argparse.SUPPRESS)
    p.add_argument("--worker-mode", help=argparse.SUPPRESS)
    return p


def counters(model) -> dict[str, Any]:
    expert = model.expert_store.snapshot()
    return {
        "expert": {k: v for k, v in expert.items() if isinstance(v, int | float)},
        "cache": {k: expert["cache"].get(k, 0) for k in ("hits", "misses", "evictions")},
        "disk": dict(model.expert_store.index.snapshot()),
    }


def delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    return {
        group: {k: v - before[group].get(k, 0) for k, v in values.items()}
        for group, values in after.items()
    }


def ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def worker(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from colibri_runtime.generation import MemorySampler, generate_tokens
    from colibri_runtime.load import load

    mx.set_cache_limit(int(args.mlx_cache_gib * GIB))
    mx.set_memory_limit(int(args.mlx_memory_gib * GIB))
    mx.reset_peak_memory()
    sampler = MemorySampler()
    t0 = time.perf_counter()
    model, tokenizer = load(
        args.model,
        expert_budget_bytes=int(args.expert_budget_gib * GIB),
        ple_budget_bytes=int(args.ple_budget_gib * GIB),
        cache_policy=args.cache_policy,
        io_workers=args.io_workers,
        prefetch_experts=args.expert_prefetch,
        prefetch_ple=args.ple_prefetch,
    )
    try:
        load_s = time.perf_counter() - t0
        if args.worker_mode == "predictor":
            from moe_prefetch import predictors, runtime

            predictor = predictors.load_predictor(args.predictor)
            runtime.enable_predictive_prefetch(model, predictor, **dict(args.opt))
        prompt = get_prompt(args.prompt_id)
        ids = prompt.encode(tokenizer)
        eos = model.args.text.eos_token_id
        sampler.reset()
        start = counters(model)
        marks: dict[str, Any] = {}

        def on_token(_token: int) -> None:
            if not marks:
                marks["prefill"] = counters(model)

        generated, timing = generate_tokens(
            model, ids, args.max_tokens, 0.0, 1.0, eos, 0, on_token=on_token
        )
        end = counters(model)
        memory = sampler.snapshot()
        total = delta(end, start)
        decode = delta(end, marks.get("prefill", end))
        prefill_s = timing["prefill_done"] - timing["started"]
        decode_s = timing["done"] - timing["first_token"]
        decode_tokens = max(0, len(generated) - 1)
        e, d = total["expert"], decode["expert"]
        result = {
            "prompt_id": prompt.id,
            "category": prompt.category,
            "mode": args.worker_mode,
            "prompt_tokens": len(ids),
            "generated_tokens": len(generated),
            "generated_ids": generated,
            "generated_text": tokenizer.decode(generated),
            "load_s": load_s,
            "ttft_s": timing["first_token"] - timing["started"],
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "prefill_tok_s": ratio(len(ids), prefill_s),
            "decode_tok_s": ratio(decode_tokens, decode_s),
            "prefetch_submitted": e["prefetch_submitted"],
            "prefetch_hits": e["prefetch_hits"],
            "prefetch_wasted": e["prefetch_wasted"],
            "prefetch_hit_rate": ratio(e["prefetch_hits"], e["prefetch_submitted"]),
            "decode_prefetch_submitted": d["prefetch_submitted"],
            "decode_prefetch_hits": d["prefetch_hits"],
            "decode_prefetch_wasted": d["prefetch_wasted"],
            "decode_prefetch_hit_rate": ratio(d["prefetch_hits"], d["prefetch_submitted"]),
            "wait_s": e["wait_seconds"],
            "decode_wait_s": d["wait_seconds"],
            "loads": e["loads"],
            "loaded_bytes": e["loaded_bytes"],
            "disk_bytes_read": total["disk"]["bytes_read"],
            "decode_disk_bytes_read": decode["disk"]["bytes_read"],
            "cache_hits": total["cache"]["hits"],
            "cache_misses": total["cache"]["misses"],
            "peak_mlx_bytes": mx.get_peak_memory(),
            "peak_rss_bytes": memory["peak_rss_bytes"],
            "counters": total,
            "decode_counters": decode,
        }
    finally:
        sampler.close()
        model.close()
    args.worker.write_text(json.dumps(result), encoding="utf-8")
    return 0


def worker_command(args: argparse.Namespace, mode: str, prompt_id: str, out: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(out),
        "--worker-mode",
        mode,
        "--prompt-id",
        prompt_id,
        "--model",
        str(args.model),
        "--max-tokens",
        str(args.max_tokens),
        "--expert-budget-gib",
        str(args.expert_budget_gib),
        "--ple-budget-gib",
        str(args.ple_budget_gib),
        "--mlx-cache-gib",
        str(args.mlx_cache_gib),
        "--mlx-memory-gib",
        str(args.mlx_memory_gib),
        "--cache-policy",
        args.cache_policy,
        "--io-workers",
        str(args.io_workers),
        "--expert-prefetch",
        str(args.expert_prefetch),
        "--ple-prefetch",
        str(args.ple_prefetch),
    ]
    if mode == "predictor":
        cmd += ["--predictor", str(args.predictor)]
        for key, value in args.opt:
            cmd += ["--opt", f"{key}={json.dumps(value)}"]
    return cmd


def run_one(args: argparse.Namespace, mode: str, prompt_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="moe-bench-") as tmp:
        out = Path(tmp) / "result.json"
        proc = subprocess.run(
            worker_command(args, mode, prompt_id, out), capture_output=True, text=True
        )
        if proc.returncode or not out.exists():
            return {
                "prompt_id": prompt_id,
                "mode": mode,
                "error": f"worker exited with {proc.returncode}",
                "stderr": proc.stderr[-4000:],
            }
        return json.loads(out.read_text(encoding="utf-8"))


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return {}
    return {key: statistics.median(r[key] for r in ok) for key in MEDIAN_KEYS}


def compare(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_prompt: dict[str, dict[str, list[list[int]]]] = {}
    for entry in results:
        for run in entry["runs"]:
            if "error" in run:
                continue
            by_prompt.setdefault(entry["prompt_id"], {}).setdefault(entry["mode"], []).append(
                run["generated_ids"]
            )
    report = {}
    for prompt_id, modes in by_prompt.items():
        seqs = [tuple(s) for runs in modes.values() for s in runs]
        first = seqs[0]
        diverge = None
        for seq in seqs[1:]:
            if seq != first:
                at = next(
                    (i for i, (a, b) in enumerate(zip(first, seq, strict=False)) if a != b),
                    min(len(first), len(seq)),
                )
                diverge = at if diverge is None else min(diverge, at)
        report[prompt_id] = {
            "modes": sorted(modes),
            "identical": diverge is None,
            "first_divergence": diverge,
        }
    return report


def fmt_gib(value: float) -> str:
    return f"{value / GIB:.2f}"


def table(results: list[dict[str, Any]]) -> str:
    head = (
        "| mode | prompt | prompt tok | gen tok | ttft s | prefill tok/s | decode tok/s "
        "| hit rate | decode hit rate | hits/submitted | wasted | wait s | GiB read "
        "| peak MLX GiB |"
    )
    rows = [head, "|" + "---|" * 14]
    for entry in results:
        m = entry["median"]
        ok = [r for r in entry["runs"] if "error" not in r]
        if not m or not ok:
            rows.append(f"| {entry['mode']} | {entry['prompt_id']} | failed |" + " |" * 11)
            continue
        rows.append(
            f"| {entry['mode']} | {entry['prompt_id']} | {ok[0]['prompt_tokens']} "
            f"| {ok[0]['generated_tokens']} | {m['ttft_s']:.2f} | {m['prefill_tok_s']:.2f} "
            f"| {m['decode_tok_s']:.3f} | {m['prefetch_hit_rate']:.1%} "
            f"| {m['decode_prefetch_hit_rate']:.1%} "
            f"| {m['prefetch_hits']:.0f}/{m['prefetch_submitted']:.0f} "
            f"| {m['prefetch_wasted']:.0f} | {m['wait_s']:.1f} | {fmt_gib(m['disk_bytes_read'])} "
            f"| {fmt_gib(m['peak_mlx_bytes'])} |"
        )
    return "\n".join(rows)


def system_info() -> dict[str, Any]:
    def sysctl(name: str) -> str:
        try:
            return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    import mlx.core as mx

    return {
        "platform": platform.platform(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(sysctl("hw.memsize") or 0),
        "mlx": mx.__version__,
        "python": platform.python_version(),
    }


def select_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        return [get_prompt(p).id for p in args.prompts]
    ids = [p.id for p in get_prompts(args.split)]
    return ids[: args.limit] if args.limit else ids


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.worker:
        return worker(args)
    if args.inputs:
        results = []
        for path in args.inputs:
            results += json.loads(path.read_text(encoding="utf-8"))["results"]
        print(table(results))
        report = compare(results)
        print(json.dumps(report, indent=2))
        return 0 if all(r["identical"] for r in report.values()) else 1
    if "predictor" in args.mode and args.predictor is None:
        print("--mode predictor needs --predictor <path>", file=sys.stderr)
        return 2
    prompt_ids = select_prompts(args)
    results = []
    for prompt_id in prompt_ids:
        for mode in args.mode:
            runs = []
            for run in range(args.runs):
                t0 = time.perf_counter()
                r = run_one(args, mode, prompt_id)
                runs.append(r)
                status = r.get("error") or (
                    f"ttft {r['ttft_s']:.1f}s, decode {r['decode_tok_s']:.3f} tok/s, "
                    f"hit {r['prefetch_hit_rate']:.1%}"
                )
                print(
                    f"{mode} {prompt_id} run {run + 1}/{args.runs}: {status} "
                    f"({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
                if "error" in r:
                    print(r["stderr"], file=sys.stderr)
            results.append(
                {"mode": mode, "prompt_id": prompt_id, "runs": runs, "median": summarize(runs)}
            )
    config = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
        if k not in {"worker", "prompt_id", "worker_mode", "inputs"}
    }
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "system": system_info(),
        "config": config,
        "results": results,
    }
    if args.compare or len(args.mode) > 1:
        payload["compare"] = compare(results)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(table(results))
    print(f"wrote {path}")
    failed = any("error" in r for e in results for r in e["runs"])
    if "compare" in payload:
        print(json.dumps(payload["compare"], indent=2))
        if args.compare and not all(r["identical"] for r in payload["compare"].values()):
            return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
