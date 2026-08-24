# The three harnesses

Same layout, same file names and same metric schemas as
`open_deep_research_Amith/tests/`, so the ODR analysis scripts read both with
at most a column rename (`example_id` → `swe_instance_id`).

Shared pieces live in two modules rather than being copy-pasted three times:

- `tests/swebench_instances.py` — load instances, materialise `./workspace_repo`
  at `base_commit`, collect the patch, append to `predictions.jsonl`
- `tests/prefix_analysis.py` — the longest-repeating-prefix analyzer (ODR keeps
  this inline in its prefix harness; nothing in it is workload-specific)
- `tests/kv_metrics_reset.py`, `tests/kv_reuse_snapshot.py` — copied verbatim
  from ODR; they talk to the server, not to the workload

## `tests/run_evaluate.py` — latency, cold vs warm

The main benchmark. Two phases with a signal between them:

```
cold phase → POST /v1/kv_metrics/reset (+ HBM flush) → warm phase (measured)
```

The cold phase fills the caches; the warm phase is what is reported. Cold rows
go to `cold_phase/` so nothing aggregating the report can pick them up, and
every row also carries a `phase` label for when files get concatenated.

```
tests/timing_logs/<RUN_ID>/
  instance_summary.jsonl          one line per run: durations, per-node totals, errors, patch size
  job_instance_e2e_latency.csv    one row per run
  node_metrics/job_<id>_*.jsonl   one line per node execution
  predictions.jsonl               → swebench eval
  kv_metrics_reset.json           the epoch boundary and the discarded cold numbers
  cold_phase/…                    the same tree, for the unmeasured pass
SWE_predictions/<RUN_ID>/         the fork's prediction / prefix / vllm-agent logs
```

Key flags: `--max-instances`, `--completions-per-instance`, `--instance-ids`,
`--instances-file`, `--ablation-mode` (`full` … `baseline`), `--pseudo-dynamic`,
`--cold-instances-per-query`, `--skip-cold-phase`, `--no-hbm-flush`,
`--no-kv-metrics-reset`.

## `tests/run_evaluate_prefix.py` — how much prompt is re-sent

Single phase, plus a callback attached to every job that records every prompt
the agent sends and finds the longest common prefix with the best-matching
earlier prompt **from the same node in the same job**.

This is the harness that works today regardless of the fork's compile-time gap:
it measures the prompts client-side. It is also the right thing to run *first* —
it answers "is there a prefix worth warming here at all" before any effort goes
into warming one.

```
tests/prefix_logs/<RUN_ID>/
  prefix_analysis/
    prefix_edge_counts.json    hash → {"from_node:to_node": count}
    prefix_texts.json          hash → the prefix text
    prefix_totals.json         hash → {count, direct_count, prefix_tokens, prefix_chars}
    prefix_containment.json    hash → {parent, delta_chars, delta_tokens}
    job_<id>/…                 the same four, per job, plus prefix_turns.jsonl
  instance_summary.jsonl, job_instance_e2e_latency.csv, node_metrics/, predictions.jsonl
  kv_reuse_snapshot.json       GET /v1/kv_metrics — cross-job vs self reuse (needs VLLM_KV_PROVENANCE=1)
```

Read `prefix_totals.json` as: `direct_count` is how many turns had *exactly*
this prefix as their longest match; `count` rolls in every longer prefix that
contains it. Prefixes nest, so sum `delta_tokens` from
`prefix_containment.json`, never `prefix_tokens`.

Expected shape for this agent: large prefixes on the two react loops
(`tools:conduct_research`, `research_tool_node:get_clear_implementation_plan_for_atomic_task`),
and a large per-instance-constant block everywhere, because the gitingest repo
tree is interpolated into three prompts.

## `tests/run_evaluate_node_eviction.py` — node-aware KV eviction

Same two-phase shape as `run_evaluate.py`, plus a `KVForecastSession` driven
from the LangGraph task stream. It publishes `PROB|<job_id>` to Redis — "will
this (node, call_type) run again, and when" — from the workflow's topology and
observed transitions. `HISTORY|<job_id>` — what a miss costs — is published
separately by kv-trace-analyser from the request stats vLLM writes to
`VLLM_REQUEST_STATS_DIR`.

**It turns the prefix prefetch off** (`KV_EVICTION_DISABLE_PREFETCH=1` by
default, clearing both `LANGGRAPH_VLLM_AGENT_ENABLE` and
`LANGGRAPH_VLLM_AGENT_BASE_URL`). Two mechanisms running at once means a
hit-rate change cannot be attributed to either. The startup banner prints the
*real* prefetch state, not the ablation label.

`SWE_GRAPH_SPEC` (top of the file) is the hand-written topology. It has to be
hand-written for the same reason ODR's is: `get_graph(xray=True)` returns
path-prefixed names (`swe_architect:conduct_research`) while the runtime — and
therefore every `extra_body` — uses the bare name (`conduct_research`), so a
reflected spec would mis-join every row silently. `_reconcile_graph_spec()`
checks the top-level names at startup and `KV_FORECAST.unknown_nodes` reports
subgraph drift at the end.

**Keep `SWE_GRAPH_SPEC` in sync when you change the graph.** A node missing
from it is published in no PROB row, so its blocks stay unscored on plain LRU —
silently removed from the experiment.

State that persists across runs: `KV_PREDICTION_TRANSITION_STORE`
(`./kv_forecast/transitions.json`) holds observed edge weights, so a fresh job
starts from real transition counts instead of uniform successors. Delete it to
reset the prior; keep it to let it converge.

```
tests/eviction_logs/<RUN_ID>/
  instance_summary.jsonl, job_instance_e2e_latency.csv, node_metrics/, predictions.jsonl
  kv_metrics_reset.json
  vllm_stats/          (if VLLM_REQUEST_STATS_DIR is left at its default)
  cold_phase/…
```

## What is deliberately not here

- **No pairwise / LLM-judge evaluators.** ODR needs them because a research
  report has no ground truth. SWE-bench has one: `swebench eval` runs
  FAIL_TO_PASS and PASS_TO_PASS in a container. Use it.
- **No Tavily-style response cache.** Every tool here reads a checkout pinned to
  `base_commit`, so the inputs are already reproducible.
- **No `--max-concurrency`.** One shared `./workspace_repo`; see
  [02-required-changes.md](02-required-changes.md#d-known-limits-of-the-harness-as-written).
