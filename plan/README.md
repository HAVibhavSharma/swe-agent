# Running this agent as a SWE-bench workload for the KV-cache experiments

This directory explains how to drive the LangGraph SWE agent in this repo as
the *workload* for the same serving-systems experiment that
`~/Projects/open_deep_research_Amith` runs: a modified LangGraph predicts which
node runs next and tells a modified vLLM to pre-warm that node's prompt prefix,
while a modified KV-eviction policy ranks cached blocks by predicted value
instead of LRU.

Read in order:

| File | What it answers |
|---|---|
| [01-architecture.md](01-architecture.md) | How the four repos connect; the graph; where every LLM call goes and what it carries |
| [02-required-changes.md](02-required-changes.md) | What was changed in this repo and why; what is still missing before the prefetch arm is real; bugs found in the agent |
| [03-running.md](03-running.md) | Step-by-step execution on the GPU box, including scoring the patches |
| [04-harnesses.md](04-harnesses.md) | The three `tests/run_evaluate*.py` scripts: what each measures, flags, outputs |

## The 60-second version

```bash
# on the machine that runs things
pip install -e ~/Projects/langgraph-dev/libs/langgraph   # the fork; REQUIRED
pip install -e ~/Projects/SWE-bench                      # dataset + scorer
cp env.template .env                                     # then fill it in

python diag_prediction.py                                # is the chain wired?

python -m tests.run_evaluate --max-instances 6 --completions-per-instance 3
python -m tests.run_evaluate_prefix --max-instances 6
python -m tests.run_evaluate_node_eviction --max-instances 6
```

## Status, honestly

| Arm | Harness | Works today? |
|---|---|---|
| Baseline latency / cold-warm | `tests/run_evaluate.py` with `--ablation-mode baseline` | **Yes** |
| Prompt-reuse measurement | `tests/run_evaluate_prefix.py` | **Yes** — measures prompts client-side, independent of the fork's analysis |
| Node-aware KV eviction | `tests/run_evaluate_node_eviction.py` | **Yes**, given Redis + kv-trace-analyser + the eviction vLLM |
| Prefix prefetch (treatment) | `tests/run_evaluate.py` with the prefetch env on | **Yes**, on `langgraph-dev` branch `Prediction-SWEbench` — the analyser now reads LCEL prompt templates. 6 nodes, 8460 static chars, 11 transition rules. See [02](02-required-changes.md#b-prompt-composition-from-lcel-prompt-templates-implemented). |

Run `python diag_prediction.py` first: it prints exactly which of those four
are live on your machine rather than leaving it to be inferred from a hit rate.
