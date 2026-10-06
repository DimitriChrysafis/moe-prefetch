# moe-prefetch

learned expert prefetching for [qwen3.8-flash-next](https://github.com/DimitriChrysafis/qwen3.8-flash-next), my runtime that runs a 180b mixture-of-experts model on a 36 gb m3 max by reading experts off the ssd as the router asks for them.

every token passes through 48 moe layers, and each layer picks 10 of 512 experts. the engine can't start reading an expert until that layer's router has run, so the read sits right on the critical path. but the hidden state going into layer L already says a lot about what layers L+1 to L+3 will pick. a small model reading it can start those reads while layer L is still computing.

```
hidden state at layer L ──► router L ──► experts for L      (blocking read)
          │
          └──────────────► predictor ──► top-k for L+1..L+3 (async read, lands early)
```

the engine repo stays untouched. everything here patches it at runtime.

## pipeline

| step | script | what it does |
|---|---|---|
| trace | `scripts/record_traces.py` | runs the real model over 30 prompts (code, prose, math, agent tool use) and saves the moe input and the 10 routed experts for every token at every layer. 7,610 tokens, 1.8 gb |
| train | `scripts/train_predictors.py` | one head per (layer, lookahead). it starts from the target layer's own router applied to the current hidden state, then learns a per-expert scale and bias plus a rank-64 correction. a quarter of each category's prompts is held out for eval and never trained on |
| eval | `scripts/eval_predictors.py` | recall@k on the held-out prompts |
| simulate | `scripts/simulate.py` | replays traces through a copy of the engine's cache with a modeled ssd queue, so policies can be compared without loading the 97 gb model |
| run | `moe_prefetch.runtime` | after each moe layer, scores the hidden state and submits the top-k experts for upcoming layers as async reads. optionally pins prefetched experts until they're used and splits demand reads across io threads |
| bench | `scripts/bench.py` | fresh process per run, baseline vs predictor, and fails if the generated tokens differ |

## offline recall

share of the experts a future layer actually routes to that show up in the predictor's top-k. 8 held-out prompts, 248 decode tokens. d is how many layers ahead.

| predictor | d=1 R@10 | d=2 R@10 | d=3 R@10 | d=1 R@16 | µs per layer |
|---|---:|---:|---:|---:|---:|
| engine frequency heuristic | 0.222 | 0.224 | 0.226 | 0.294 | 5 |
| target layer's router, untrained | 0.634 | 0.541 | 0.496 | 0.765 | 82 |
| trained rank-64 heads | **0.672** | **0.615** | **0.587** | **0.798** | 158 |

the untrained baseline is already strong, likely because the residual stream changes slowly from layer to layer. training helps most further out: about 4 points at d=1, but 7 to 9 at d=2 and d=3.

## setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ../qwen3.8-flash-next
pip install -e '.[dev]'
pytest
```

the scripts default to `~/qwen3.8next/models/pipenetwork-Qwen3.8-Flash-Next-MLX-4bit`. pass `--model` to point elsewhere. only run one model process at a time, since two of them fight over the ssd and the numbers are meaningless.

```sh
python scripts/record_traces.py
python scripts/train_predictors.py --name linear
python scripts/eval_predictors.py heuristic future_gate linear
python scripts/bench.py --mode baseline predictor --predictor artifacts/predictors/linear \
  --opt k=8 --opt 'lookaheads=[1,2]' --opt protect=true --opt parallel_demand=true \
  --split eval --max-tokens 128 --compare
```

## tests

37 tests. the tracer is checked against a tiny transformers checkpoint, the predictors against synthetic traces with known structure, the simulator's cache against the engine's cache operation by operation, and the runtime hook for identical tokens and logits with prefetch on and off.
