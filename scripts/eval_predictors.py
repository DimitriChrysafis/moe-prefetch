import argparse
import json
import sys
from pathlib import Path

from safetensors.numpy import load_file

from moe_prefetch.gates import load_gates
from moe_prefetch.predictors import FutureGatePredictor, HeuristicPredictor, load_predictor
from moe_prefetch.predictors.data import list_runs, split_runs
from moe_prefetch.predictors.evaluate import evaluate, latency

ROOT = Path(__file__).resolve().parents[1]


def build(spec: str, gates_path: str | None, lookaheads, layers: int, experts: int):
    if spec == "heuristic":
        return spec, HeuristicPredictor(lookaheads, layers, experts)
    if spec in ("future_gate", "future_gate_norm"):
        gates = load_file(gates_path)["gates"] if gates_path else load_gates()
        return spec, FutureGatePredictor(gates, lookaheads, norm=spec.endswith("_norm"))
    path = Path(spec)
    if not path.is_dir():
        path = ROOT / "artifacts" / "predictors" / spec
    return path.name, load_predictor(path)


def table(results: dict, ks, phases) -> str:
    head = ["predictor", "phase", "d"] + [f"R@{k}" for k in ks] + ["us/call"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, res in results.items():
        lat = f"{res['latency']['median_us']:.0f}"
        for phase in phases:
            for key, row in res["summary"].get(phase, {}).items():
                cells = [name, phase, key] + [f"{row[str(k)]:.3f}" for k in ks] + [lat]
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictors", nargs="*", default=["heuristic", "future_gate"])
    ap.add_argument("--traces", default=str(ROOT / "traces"))
    ap.add_argument("--split", default="eval", choices=["train", "eval", "all"])
    ap.add_argument("--gates")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 16, 24, 32])
    ap.add_argument("--lookaheads", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-runs", type=int)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "eval"))
    args = ap.parse_args()

    runs = list_runs(Path(args.traces))
    if args.split != "all":
        runs = split_runs(runs)[args.split]
    runs = runs[: args.max_runs]
    if not runs:
        print(f"no {args.split} traces in {args.traces}", file=sys.stderr)
        return 1
    meta = runs[0].meta
    layers, experts = int(meta["layers"]), int(meta["experts"])
    hidden_dim = int(meta["hidden_dim"])
    tokens = sum(r.tokens for r in runs)
    print(f"{len(runs)} {args.split} runs, {tokens} tokens", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in args.predictors:
        name, predictor = build(spec, args.gates, tuple(args.lookaheads), layers, experts)
        res = evaluate(predictor, runs, args.ks, tuple(args.lookaheads))
        res["latency"] = latency(predictor, hidden_dim, layers)
        res["split"] = args.split
        res["predictor"] = spec
        results[name] = res
        (out / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n")
        print(f"{name} done", file=sys.stderr)
        del predictor
    phases = ("prefill", "decode")
    print(table(results, args.ks, phases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
