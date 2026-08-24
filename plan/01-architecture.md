# Architecture: four repos, one request path

## The repos

| Repo | Branch | Role |
|---|---|---|
| `~/Projects/swe-agent` (this one) | `main` | the workload: a LangGraph agent that reads a repo and writes a patch |
| `~/Projects/SWE-bench` | `main` @ `7a21e05` (5.0.2) | the task set (`problem_statement`, `repo`, `base_commit`) and the Docker scorer |
| `~/Projects/langgraph-dev` | `Prediction-SWEbench` | compile-time prompt/transition analysis + the runtime prediction worker that fires prefetch |
| `~/Projects/vllm` | `vllm-v2` | `/v1/agents/*` routes, the agent-prefetch registry, node-aware KV eviction |

SWE-bench itself contributes **no agent**: it is a dataset plus a Docker
evaluation harness plus single-shot inference (`swebench/inference/run_api.py`).
That is why this repo is the workload and SWE-bench is only the task source and
the scorer.

## The graph

Compiled at import in `agent/graph.py`. Both subgraphs compile first (they are
module-level objects imported at the top), which is the order the fork's
analyser needs — it must see inside a subgraph before the root is compiled.

```
swe_agent (root)
  START → swe_architect → swe_developer → END

swe_architect  (agent/architect/graph.py)
  START → come_up_with_research_next_step → check_research_step
        → [should_conduct_research]
              plan_is_valid      → conduct_research
              plan_is_not_valid  → come_up_with_research_next_step
  conduct_research → [should_call_tool]
              should_call_tool   → tools → conduct_research      ← react loop
              implement_plan     → extract_implementation_plan → END

swe_developer  (agent/developer/graph.py)
  START → start_implementing → prepare_for_implementation
        → get_clear_implementation_plan_for_atomic_task
        → [should_continue_implementation_research]
              should_continue_research → research_tool_node → back   ← react loop
              implement_plan           → creating_diffs_for_task
        → proceed_to_next_atomic_task → [is_implementation_complete]
              continue → prepare_for_implementation
              END      → END
```

| Node | LLM calls | Kind | Notes |
|---|---|---|---|
| `come_up_with_research_next_step` | 1 (structured) | non-react | proposes a hypothesis |
| `check_research_step` | 1 (structured) | non-react | accept/reject the hypothesis |
| `conduct_research` | 1 (tool-bound) | **react** | the architect's growing prefix lives here |
| `tools` | 0 | — | `ToolNode`, search + codemap |
| `extract_implementation_plan` | 1 | non-react | JSON plan out |
| `start_implementing`, `prepare_for_implementation`, `proceed_to_next_atomic_task` | 0 | — | state shuffling |
| `get_clear_implementation_plan_for_atomic_task` | 1 (tool-bound) | **react** | the developer's growing prefix |
| `research_tool_node` | 0 | — | `ToolNode` |
| `creating_diffs_for_task` | 1 | non-react | two call sites (new file / diff), one node name |

Compared with ODR: two react loops (same), a compression-like summarisation
step (`extract_implementation_plan`), a long final generation
(`creating_diffs_for_task`) — but **no parallel fan-out**. Nothing here plays
the role of ODR's ten concurrent researchers, so `AGENT_UNIT_METADATA_KEY` is
never set and every `agent_id` is a plain `(job, graph path)` bucket.

## The tools

All local filesystem reads over `./workspace_repo`:

- `agent/tools/search.py` — `search_keyword_in_directory`
- `agent/tools/codemap.py` — `get_code_definitions`, `get_function_implementation`,
  `get_code_definitions_multi`, `get_raw_file_content` (tree-sitter)
- `agent/tools/write.py` — `get_files_structure` (gitingest tree), `create_file`,
  `write_to_file`; only `get_files_structure` is bound to the model, the writes
  happen in `creating_diffs_for_task` directly

This is a **determinism win over ODR**. ODR had to freeze Tavily responses into
a cache (`TAVILY_CACHE_DIR`) to make two runs comparable, and its own docs admit
the frozen cache "manufactures some of the path stability the report shows".
Here every tool reads a checkout pinned to `base_commit`, so repeated runs see
identical inputs with no cache and no caveat. The remaining non-determinism is
the server's — continuous batching, APC, quantised kernels — which is the thing
being measured.

Note the prompt-size hazard: `get_files_structure` ingests the **whole repo
tree** and is interpolated into three different prompts. On a large repo
(django, sympy) that is a very large, per-instance-constant block — which is
also exactly the kind of shared prefix the KV cache should exploit, so it is
worth measuring before trimming.

## The LLM call path

Every call goes through `agent/common/model.py::get_model_config(config, max_tokens)`,
which is the single place per-call kwargs are decided:

```
node fn(state, config)
  → configurable_model.{bind_tools|with_structured_output}(…)
      .with_config(get_model_config(config, N))
        → langchain-openai ChatOpenAI → POST {OPENAI_BASE_URL}/chat/completions
```

Stock `/v1/chat/completions`. Like ODR, this workload never calls the custom
`/v1/agents/*` routes itself — the LangGraph fork does that, out of band.

`extra_body` carries the hint channel, built by
`agent/common/llm_request_metadata.py` (copied verbatim from ODR):

| key | source | consumed by |
|---|---|---|
| `job_id` | `configurable["job_id"]` — the harness sets one per instance run | KV eviction index, provenance |
| `langgraph_node` | `metadata["langgraph_node"]` — the bare runtime node name | KV eviction index |
| `call_type` | `derive_call_type(node)` | KV eviction index |
| `agent_id` | `derive_agent_id(graph_path, job_id, unit)` — only when prefetch is on | prefetch registry |
| `record_in_registry` | `True` alongside `agent_id` | prefetch registry |

## What the fork does with that

1. **Compile time** (`langgraph/graph/state.py:3318`, reached because
   `agent/graph.py` compiles with `_is_root=True`): walks every node function's
   AST and builds `prompt_composition` (static / pseudo-dynamic / true-dynamic
   prompt segments) and `transition_prediction` (priority-ordered rules mapping
   a streamed signal to the next node). `diag_prediction.py` dumps both.
2. **Run time** (`langgraph/pregel/main.py:2606`): a
   `BackgroundTransitionPredictionWorker` reads the streamed tool-call name,
   matches the rules, resolves the predicted next node, and POSTs
   `warm_agent_prefixes` → `{base}/prefetch` (`_vllm_agent.py:328`).
3. **Server** (`vllm/entrypoints/openai/agent_chat/api_router.py`): `/v1/agents/prefetch`
   submits phantom prefills for the registered prefix;
   `vllm/v1/agent_prefetch/auto_register.py` records prefixes under the
   `agent_id` from `extra_body`; `vllm/v1/core/node_eviction/` ranks blocks
   using Redis `PROB|<job_id>` (published by `KVForecastSession`) and
   `HISTORY|<job_id>` (published by kv-trace-analyser from
   `VLLM_REQUEST_STATS_DIR`).

Steps 1–2 are where this agent currently falls short; see
[02-required-changes.md](02-required-changes.md).
