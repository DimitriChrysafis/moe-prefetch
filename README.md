# moe-prefetch

learned expert prefetching for [qwen3.8-flash-next](https://github.com/DimitriChrysafis/qwen3.8-flash-next), my runtime that runs a 180b mixture-of-experts model on a 36 gb m3 max by reading experts off the ssd as the router asks for them.

every token passes through 48 moe layers, and each layer picks 10 of 512 experts. the engine can't start reading an expert until that layer's router has run, so the read sits right on the critical path. but the hidden state going into layer L already says a lot about what layers L+1 to L+3 will pick. a small model reading it can start those reads while layer L is still computing.

```
hidden state at layer L ──► router L ──► experts for L      (blocking read)
          │
          └──────────────► predictor ──► top-k for L+1..L+3 (async read, lands early)
```

the engine repo stays untouched. everything here patches it at runtime.

## results

real model, m3 max 36 gb, 8 gib expert cache, 8 held-out prompts, 128 greedy tokens, fresh process per run.

| | baseline | predictor |
|---|---:|---:|
| prefetch hit rate (median) | 0.4% | **54.8%** |
| time waiting on expert reads (median) | 37.9 s | **13.3 s** |
| decode (median) | 1.334 tok/s | 1.370 tok/s |
| generated tokens identical | | 8 / 8 |

the predictor works: more than half of what it prefetches gets used, and time spent blocked on reads drops by about 65%. decode speed mostly doesn't follow. the [why](#what-i-learned) is the interesting part.

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

## end to end, per prompt

predictor run: `k=8`, lookaheads 1 and 2, pinning on, parallel demand reads on. raw runs in `artifacts/bench/20261006-125327.json`.

| prompt | hit rate | read wait (s) | decode (tok/s) | ttft (s) |
|---|---:|---:|---:|---:|
| code-py-explain-gen | 0.5% → 62.4% | 33.4 → 13.1 | 1.369 → 1.418 | 16.7 → 18.2 |
| code-ts-explain-types | 0.3% → 57.0% | 30.4 → 11.8 | 1.396 → 1.427 | 14.3 → 15.9 |
| prose-ironman | 0.2% → 52.1% | 36.1 → 14.0 | 1.338 → 1.323 | 15.5 → 13.1 |
| prose-story | 0.4% → 44.6% | 37.8 → 12.5 | 1.330 → 1.260 | 12.8 → 12.1 |
| math-series | 0.2% → 59.8% | 38.5 → 13.4 | 1.423 → 1.552 | 14.3 → 11.8 |
| math-linalg | 0.2% → 49.1% | 38.0 → 12.2 | 1.277 → 1.517 | 14.9 → 14.2 |
| agent-fix-test | 1.1% → 54.5% | 46.9 → 22.1 | 0.958 → 0.687 | 47.4 → 80.4 |
| agent-refactor | 1.0% → 55.0% | 63.9 → 22.8 | 0.496 → 0.679 | 72.1 → 76.6 |

## ablations

same 6 non-agent eval prompts, medians.

| config | hit rate | read wait (s) | decode (tok/s) |
|---|---:|---:|---:|
| baseline (engine heuristic) | 0.3% | 37.0 | 1.354 |
| predictor, no pinning, serial demand reads | 8.1% | 22.0 | 1.063 |
| predictor, pinning, serial demand reads | 54.6% | 18.9 | 1.328 |
| predictor, pinning, parallel demand reads | 54.5% | 12.8 | 1.422 |

pinning is what makes it work at all. without it, the cache's adaptive policy gives a freshly prefetched expert the lowest possible use count and evicts it before the layer that needs it runs. hit rate falls to 8%, the wasted reads compete with real ones, and decode ends up slower than doing nothing.

## what i learned

**the simulator was too optimistic.** in `scripts/simulate.py` (engine heuristic, pinning on), splitting each layer's demand reads across the io threads took decode from 1.54 to 3.68 tok/s. on the real model with the predictor, the same change gave about 7% (1.328 → 1.422 above). the simulator's cost model is just ssd bandwidth plus a fixed compute time per layer. it charges nothing for turning rows into mlx arrays or for python overhead, so it treats almost every second not spent computing as a second spent waiting on the disk. my guess is that missing cost is most of the gap.

**less waiting didn't mean faster decode.** a 128-token run lasts about 95 s, and read wait fell by about 24 s per run, yet decode moved about 3%. waiting on the ssd wasn't the main bottleneck. the freed time went into other work on the main thread, most likely turning prefetched rows into mlx arrays, plus the extra reads from wrong guesses: bytes read went up on 7 of 8 prompts, prose-story from 83 to 111 gib.

**long prefills got worse.** agent-fix-test has an 850-token prompt, and its time to first token went from 47 s to 80 s. it isn't bad guesses: 96 to 100% of prefill prefetches get used on every prompt. my suspect is parallel demand reads. a long prefill touches hundreds of experts per layer, and that mode submits them one at a time instead of in batches of 32. next step is to limit parallel demand reads to decode and measure again.

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
