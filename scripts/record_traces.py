#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx

from moe_prefetch import trace
from moe_prefetch.prompts import get_prompts, split_of

MODEL = Path.home() / "qwen3.8next/models/pipenetwork-Qwen3.8-Flash-Next-MLX-4bit"
GIB = 1 << 30


def bytes_per_token(layers: int, top_k: int, hidden: int) -> int:
    return layers * (hidden * 2 + top_k * 4) + 5


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--out", type=Path, default=Path("traces"))
    parser.add_argument("--split", choices=["all", "train", "eval"], default="all")
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-total-gb", type=float, default=8.0)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--expert-budget-gib", type=float, default=8.0)
    parser.add_argument("--ple-budget-gib", type=float, default=1.0)
    parser.add_argument("--mlx-cache-gib", type=float, default=1.0)
    parser.add_argument("--mlx-memory-gib", type=float, default=28.0)
    parser.add_argument("--io-workers", type=int, default=6)
    parser.add_argument("--expert-prefetch", type=int, default=2)
    parser.add_argument("--ple-prefetch", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def log(message: str) -> None:
    print(message, flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if free_gb(args.out) < args.min_free_gb:
        log(f"only {free_gb(args.out):.1f} GB free, need {args.min_free_gb:.0f} GB, not starting")
        return 1
    prompts = get_prompts(args.split)
    if args.prompts:
        wanted = set(args.prompts)
        prompts = [p for p in prompts if p.id in wanted]
        unknown = wanted - {p.id for p in prompts}
        if unknown:
            log(f"unknown prompt ids: {sorted(unknown)}")
            return 1
    budget = int(args.max_total_gb * 1e9)

    from colibri_runtime.load import load

    mx.set_cache_limit(int(args.mlx_cache_gib * GIB))
    mx.set_memory_limit(int(args.mlx_memory_gib * GIB))
    t0 = time.perf_counter()
    model, tokenizer = load(
        args.model,
        expert_budget_bytes=int(args.expert_budget_gib * GIB),
        ple_budget_bytes=int(args.ple_budget_gib * GIB),
        io_workers=args.io_workers,
        prefetch_experts=args.expert_prefetch,
        prefetch_ple=args.ple_prefetch,
    )
    log(f"loaded model in {time.perf_counter() - t0:.1f}s")
    text = model.args.text
    per_token = bytes_per_token(text.num_hidden_layers, text.num_experts_per_tok, text.hidden_size)
    eos = text.eos_token_id
    done = 0
    started = time.perf_counter()
    try:
        for i, prompt in enumerate(prompts, 1):
            out = args.out / prompt.id
            if out.exists():
                if not args.overwrite:
                    log(f"[{i}/{len(prompts)}] {prompt.id}: exists, skipping")
                    continue
                shutil.rmtree(out)
            if free_gb(args.out) < args.min_free_gb:
                log(f"free disk dropped below {args.min_free_gb:.0f} GB, stopping")
                return 1
            used = trace.trace_bytes(args.out)
            room = (budget - used) // per_token - args.max_tokens
            limit = min(args.max_prompt_tokens, room)
            if limit < 16:
                log(f"trace budget of {args.max_total_gb} GB reached, stopping")
                break
            full = prompt.encode(tokenizer)
            ids = prompt.encode(tokenizer, limit)
            t1 = time.perf_counter()
            generated, arrays, info = trace.record(model, ids, args.max_tokens, eos, 0.0)
            elapsed = time.perf_counter() - t1
            timing = info["timing"]
            meta = trace.build_meta(
                prompt.id,
                arrays,
                info,
                prompt_id=prompt.id,
                prompt_category=prompt.category,
                split=split_of(prompt.id),
                model_path=str(args.model),
                max_tokens=args.max_tokens,
                temperature=0.0,
                expert_budget_gib=args.expert_budget_gib,
                prompt_tokens_full=len(full),
                truncated=len(ids) < len(full),
                generated_ids=generated,
                generated_text=tokenizer.decode(generated),
                prefill_seconds=timing["prefill_done"] - timing["started"],
                decode_seconds=timing["done"] - timing["first_token"],
                record_seconds=elapsed,
            )
            trace.write_trace(out, meta, arrays)
            done += 1
            prefill_tps = meta["n_prefill"] / meta["prefill_seconds"]
            decode_tps = (len(generated) - 1) / meta["decode_seconds"] if len(generated) > 1 else 0
            log(
                f"[{i}/{len(prompts)}] {prompt.id}: prefill {meta['n_prefill']} "
                f"({prefill_tps:.1f} tok/s), decode {meta['n_decode']} ({decode_tps:.2f} tok/s), "
                f"{elapsed:.0f}s, absmax {info['hidden_absmax']:.1f}, "
                f"total {trace.trace_bytes(args.out) / 1e9:.2f} GB"
            )
    finally:
        model.close()
    log(
        f"recorded {done} traces in {time.perf_counter() - started:.0f}s, "
        f"{trace.trace_bytes(args.out) / 1e9:.2f} GB under {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
