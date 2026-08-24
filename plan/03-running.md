# Running it

Everything here runs on the machine with the GPUs. Nothing needs Docker except
the final scoring step, which can run anywhere.

## 1. Install

```bash
cd ~/Projects/swe-agent
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# REQUIRED: the fork. Stock langgraph has no `_vllm_agent`, no `_kv_forecast`
# and no `_is_root` kwarg, and `agent/graph.py` will not import without them.
pip install -e ~/Projects/langgraph-dev/libs/langgraph

# dataset loader + the scorer
pip install -e ~/Projects/SWE-bench

# used by the model path and the prefix analyzer
pip install langchain-openai transformers

cp env.template .env    # then fill it in
```

Check the fork is the branch you think it is:

```bash
(cd ~/Projects/langgraph-dev && git rev-parse --abbrev-ref HEAD)   # Prediction-SWEbench
```

A dirty or stale checkout of the fork is the classic silent failure here: the
`/v1/agents` calls go quiet and the run looks like a control arm that was
labelled treatment.

## 2. Start vLLM

```bash
cd ~/Projects/vllm       # branch vllm-v2
vllm serve ./Qwen2.5-72B-Instruct-AWQ/ \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 131072 \
  --hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
```

Tool calling **must** be enabled: both react loops depend on it, and without a
tool parser the routers see no `tool_calls` and every loop exits on its first
turn.

Per arm, additionally:

- prefix prefetch → the build with `/v1/agents/*` mounted
- KV provenance (for `run_evaluate_prefix.py`'s reuse snapshot) → `VLLM_KV_PROVENANCE=1`
- node-aware eviction → the eviction policy enabled, `VLLM_REQUEST_STATS_DIR`
  pointed at the directory kv-trace-analyser tails, and a reachable Redis

## 3. Verify the chain before spending GPU hours

```bash
python diag_prediction.py
```

Reports, in order: fork importable → prompt composition node count → transition
rule count → prefetch enabled and reachable. On branch `Prediction-SWEbench`
expect **6 nodes / 8460 static chars** and **11 transition rules**. A zero on
either line means the fork is not the branch you think it is.

It also writes `prompt_compile_metadata.json`, which is worth committing when
it changes — it is the diffable record of what the compiler saw.

## 4. Warm the repo cache (optional but recommended)

The first instance of each repo clones it from GitHub into
`~/.cache/swe-agent-bench/repos`. Doing that inside a measured phase adds a
network clone to a latency number. Pre-clone by running the cold phase once:

```bash
python tests/run_evaluate.py --max-instances 6 --completions-per-instance 0 \
  --cold-instances-per-query 1
```

## 5. Run the arms

```bash
# control: no prefetch
LANGGRAPH_VLLM_AGENT_ENABLE=0 \
python tests/run_evaluate.py --max-instances 6 --completions-per-instance 3 \
  --ablation-mode baseline

# treatment: prefetch on
LANGGRAPH_VLLM_AGENT_ENABLE=1 \
LANGGRAPH_VLLM_AGENT_BASE_URL=http://localhost:8000/v1/agents \
python tests/run_evaluate.py --max-instances 6 --completions-per-instance 3 \
  --ablation-mode full

# how much prefix is there to warm at all
python tests/run_evaluate_prefix.py --max-instances 6 --completions-per-instance 3

# node-aware eviction (clears the prefetch env itself)
KV_FORECAST_REDIS_URL=redis://localhost:6379/0 \
python tests/run_evaluate_node_eviction.py --max-instances 6 --completions-per-instance 3
```

Both arms must use the same instances. The sampler is
`random.Random(62).sample(...)` over the dataset, so the same
`--dataset/--split/--max-instances` gives the same set; pin explicitly with
`--instance-ids` if you change anything else.

## 6. Score the patches

Each harness writes `predictions.jsonl` in its metrics directory. Scoring is a
separate step, on any machine with Docker:

```bash
swebench eval \
  -p tests/timing_logs/<RUN_ID>/predictions.jsonl \
  -d SWE-bench/SWE-bench_Verified \
  --run-id <RUN_ID> -j 8
```

First run builds the instance images, which is slow and large. `swebench images
build verified -j 8` up front if you want that out of the measured path.

Scoring is deliberately **not** part of the benchmark process: the GPU box needs
no Docker daemon, and a scoring failure cannot corrupt a latency measurement.

## 7. What to compare

| Question | Where to look |
|---|---|
| Did the arm get faster end to end? | `job_instance_e2e_latency.csv`, warm rows only |
| Which node absorbed the change? | `node_metrics/*.jsonl`, `totals_by_node_name_seconds` |
| Did the server's cache behave differently? | the `kv_hbm` lines after the epoch in `kv_metrics_reset.json` |
| How much prompt was re-sent? | `prefix_logs/<RUN_ID>/prefix_analysis/prefix_totals.json` |
| Reuse across instances vs within one? | `kv_reuse_snapshot.json` (`reuse_cross_tokens` vs `reuse_self_tokens`) |
| Did the agent actually solve anything? | the `swebench eval` report |

Never mix cold-phase rows into a warm aggregate. They are in a separate
`cold_phase/` tree for exactly that reason, and every row also carries its own
`phase` label so a concatenation can still be split apart.
