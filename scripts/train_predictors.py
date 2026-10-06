import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

from safetensors.numpy import load_file

from moe_prefetch.gates import load_gates
from moe_prefetch.predictors.data import list_runs, split_runs
from moe_prefetch.predictors.train import Config, fit_linear

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    defaults = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="linear")
    ap.add_argument("--traces", default=str(ROOT / "traces"))
    ap.add_argument("--gates")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "predictors"))
    ap.add_argument("--lookaheads", type=int, nargs="+", default=list(defaults.lookaheads))
    ap.add_argument("--rank", type=int, default=defaults.rank)
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--loss", choices=["softmax", "bce"], default=defaults.loss)
    ap.add_argument("--epochs", type=int, default=defaults.epochs)
    ap.add_argument("--batch", type=int, default=defaults.batch)
    ap.add_argument("--lr", type=float, default=defaults.lr)
    ap.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    ap.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    ap.add_argument("--decode-weight", type=float, default=defaults.decode_weight)
    ap.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    ap.add_argument("--patience", type=int, default=defaults.patience)
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--threads", type=int, default=defaults.threads)
    args = ap.parse_args()

    cfg = Config(
        lookaheads=tuple(args.lookaheads),
        rank=args.rank,
        norm=not args.no_norm,
        loss=args.loss,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_tokens=args.max_tokens,
        decode_weight=args.decode_weight,
        val_fraction=args.val_fraction,
        patience=args.patience,
        seed=args.seed,
        threads=args.threads,
    )
    runs = split_runs(list_runs(Path(args.traces)))["train"]
    if not runs:
        print(f"no train traces in {args.traces}", file=sys.stderr)
        return 1
    gates = load_file(args.gates)["gates"] if args.gates else load_gates()
    start = time.time()
    predictor = fit_linear(runs, gates, cfg)
    extra = {
        "config": asdict(cfg),
        "train_runs": [r.run_id for r in runs],
        "train_tokens": sum(r.tokens for r in runs),
        "train_seconds": round(time.time() - start, 1),
    }
    path = predictor.save(Path(args.out) / args.name, extra)
    print(f"saved {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
