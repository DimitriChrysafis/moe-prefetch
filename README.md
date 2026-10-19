# moe-prefetch

learned expert prefetching for [qwen3.8-flash-next](https://github.com/DimitriChrysafis/qwen3.8-flash-next). it guesses which experts the next few layers will need and starts reading them off the ssd early. it patches the engine at runtime, so the engine repo stays unchanged.

## the idea

the model has 48 moe layers, and each layer's router picks 10 of 512 experts for every token. the engine only learns which experts a layer needs once that layer's router has run, so every expert read waits on the ssd.

the hidden state going into layer L already says a lot about which experts the next layers will pick. a small predictor reads it and starts those reads while layer L is still computing.

## results

![per prompt results](images/results.png)

8 prompts the predictor never trained on, 128 tokens each, on an m3 max with 36 gb and an 8 gib expert cache. raw runs are in `artifacts/bench/20261006-125327.json`.

- read wait fell by 53 to 68% on every prompt. the median run went from 37.9 s of waiting to 13.3 s.
- 45 to 62% of the prefetched experts got used. without the predictor it's about 1% or less.
- decode went up on 5 prompts and down on 3. the median went from 1.33 to 1.37 tok/s.
- every prompt produced the same tokens with and without the predictor.

the predictor works, but saving wait time didn't make decoding faster. the biggest drop was agent-fix-test, from 0.96 to 0.69 tok/s. the next section covers why.

## offline recall

how often the predictor's top 10 contains the experts a future layer actually routes to, on the 8 held-out prompts.

- engine's frequency heuristic: 22% one layer ahead, 23% three layers ahead
- the target layer's own router, untrained: 63% and 50%
- trained rank-64 heads: 67% and 59%

the untrained router is already strong, probably because the hidden state changes slowly from layer to layer. training matters more the further ahead it looks.

## what didn't work

pinning is required. without it, the cache's adaptive policy evicts a freshly prefetched expert before the layer that needs it runs. the hit rate falls from 55% to 8%, and decode drops to 1.06 tok/s, below the 1.35 baseline on the six non-agent prompts.

less waiting didn't mean faster decode. a 128-token run waits about 24 s less, but decode only gains about 3%. my guess is the freed time goes to turning rows into mlx arrays on the main thread. wrong guesses also add reads: bytes read went up on 7 of 8 prompts, and prose-story went from 83 to 111 GiB.

the simulator was too optimistic. in `scripts/simulate.py`, splitting demand reads across io threads took decode from 1.54 to 3.68 tok/s. on the real model the same change gave about 7%. the simulator charges nothing for python overhead or for building mlx arrays, so it counts almost all non-compute time as disk wait.

long prompts got slower. on agent-fix-test, an 850-token prompt, time to first token went from 47 s to 80 s. my suspect is parallel demand reads on long prefills, which submit hundreds of reads one at a time instead of in batches of 32. next step is to limit parallel reads to decode and measure again.

## how it's built

- `scripts/record_traces.py`: runs the model over 30 prompts and saves the routed experts for every token
- `scripts/train_predictors.py`: one head per layer and lookahead, starting from that layer's router
- `scripts/eval_predictors.py`: recall@k on the held-out prompts
- `scripts/simulate.py`: replays traces through a copy of the cache without loading the model
- `scripts/bench.py`: baseline vs predictor in fresh processes, fails if the tokens differ
- `moe_prefetch.runtime`: the hook that submits predictions as async reads

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
