# moe-prefetch

learned expert prefetching for [qwen3.8-flash-next](https://github.com/DimitriChrysafis/qwen3.8-flash-next). it guesses which experts the next few layers will need and starts reading them off the ssd early. it patches the engine at runtime, so the engine repo stays unchanged.

## the idea

the model has 48 moe layers, and each layer's router picks 10 of 512 experts for every token. the engine only knows which experts a layer needs after that layer's router runs, so the read always blocks.

```
layer L ──► router L ──► read experts for L          blocks the layer
   │
   └──► predictor ──► read experts for L+1 to L+3    runs while L computes
```

the hidden state going into layer L already says a lot about which experts the later layers will pick. a small model can read it and start those reads early.

## results

![results](images/results.png)

8 held-out prompts, 128 tokens each, medians. m3 max, 36 gb, 8 gib expert cache, fresh process per run. raw runs are in `artifacts/bench/20261006-125327.json`.

- about 55% of the experts it prefetches get used. the baseline is 0.4%.
- time blocked on expert reads drops from 37.9 s to 13.3 s per run.
- decode speed barely changes, 1.33 to 1.37 tok/s. the output matches token for token on all 8 prompts.

the gap between the second and third bullets is the important part. see below.

## offline recall

how often a future layer's real experts show up in the predictor's top 10. 8 held-out prompts, 248 decode tokens.

| predictor | 1 layer ahead | 2 ahead | 3 ahead |
|---|---:|---:|---:|
| engine frequency heuristic | 0.222 | 0.224 | 0.226 |
| target layer's router, untrained | 0.634 | 0.541 | 0.496 |
| trained rank-64 heads | 0.672 | 0.615 | 0.587 |

the untrained router is already strong, probably because the hidden state changes slowly from layer to layer. training helps more the further ahead it looks.

## ablations

same 6 non-agent prompts, medians.

| config | hit rate | read wait (s) | decode (tok/s) |
|---|---:|---:|---:|
| engine heuristic (baseline) | 0.3% | 37.0 | 1.354 |
| predictor, no pinning, serial reads | 8.1% | 22.0 | 1.063 |
| predictor, pinning, serial reads | 54.6% | 18.9 | 1.328 |
| predictor, pinning, parallel reads | 54.5% | 12.8 | 1.422 |

## what didn't work

pinning is required. without it, the cache gives a freshly prefetched expert the lowest use count and evicts it before the layer that needs it runs. the hit rate falls to 8%, and decode ends up slower than with no prefetch at all.

less waiting didn't mean faster decode. a 128-token run waits about 24 s less, but decode only gains about 3%. my guess is the freed time goes to turning rows into mlx arrays on the main thread. wrong guesses also add reads: bytes read went up on 7 of 8 prompts, and prose-story went from 83 to 111 GiB.

the simulator was too optimistic. in `scripts/simulate.py`, splitting demand reads across io threads took decode from 1.54 to 3.68 tok/s. on the real model the same change gave about 7%. the simulator charges nothing for python overhead or for building mlx arrays, so it counts almost all non-compute time as disk wait.

long prompts got slower. on agent-fix-test, an 850-token prompt, time to first token went from 47 s to 80 s. my suspect is parallel demand reads on long prefills, which submit hundreds of reads one at a time instead of in batches of 32. next step is to limit parallel reads to decode and measure again.

## how it's built

| script | what it does |
|---|---|
| `scripts/record_traces.py` | runs the model over 30 prompts and saves the routed experts for every token |
| `scripts/train_predictors.py` | trains one head per layer and lookahead, starting from that layer's router |
| `scripts/eval_predictors.py` | recall@k on the held-out prompts |
| `scripts/simulate.py` | replays traces through a copy of the cache without loading the model |
| `scripts/bench.py` | baseline vs predictor in fresh processes, fails if the tokens differ |
| `moe_prefetch.runtime` | the hook that submits predictions as async reads |

## run it

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ../qwen3.8-flash-next
pip install -e '.[dev]'
pytest
```

the scripts default to `~/qwen3.8next/models/pipenetwork-Qwen3.8-Flash-Next-MLX-4bit`. pass `--model` to change it. run one model process at a time. two of them compete for the ssd and the timings mean nothing.

```sh
python scripts/record_traces.py
python scripts/train_predictors.py --name linear
python scripts/eval_predictors.py heuristic future_gate linear
python scripts/bench.py --mode baseline predictor --predictor artifacts/predictors/linear \
  --opt k=8 --opt 'lookaheads=[1,2]' --opt protect=true --opt parallel_demand=true \
  --split eval --max-tokens 128 --compare
```

## tests

37 tests. the tracer is checked against a tiny transformers checkpoint, the simulator's cache against the engine's cache, and the runtime hook must give identical tokens and logits with prefetch on and off.
