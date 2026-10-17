# How this repo uses the `langgraph-dev` fork

## Setup

- The fork is installed in place of stock langgraph: `pip install -e ~/Projects/langgraph-dev/libs/langgraph`.
- It must be on branch `Prediction-SWEbench`. Older branches can't read LCEL prompt templates, so the prefetch has nothing to warm.
- `pyproject.toml` only lists `langgraph>=0.2.69`. Nothing there pins the fork, so a plain `uv sync` installs stock langgraph.
- `diag_prediction.py` checks whether the fork is installed, whether compile produced metadata, and whether prefetch is on.

## The fork's job, in one line

At compile time it reads the graph's prompts and edges. At run time it predicts the next node and asks vLLM to pre-warm that node's prompt prefix. It also tags every LLM request so vLLM can make KV-cache decisions per node.

## 1. Compile time (`graph/state.py`)

- `swe_agent_builder.compile(_is_root=True)` in `agent/graph.py` sets off the analysis.
- The analysis attaches two things to the compiled graph:
  - **`prompt_composition`**: the prompt segments for each node (static, state-bound, or pseudo-dynamic), read out of `(prompt_template | model).invoke(...)`.
  - **`transition_prediction`**: rules of the form "condition → next node", built from conditional edges, tool calls, and loop limits.
- It also dumps debug metadata to `LANGGRAPH_COMPILE_DEBUG_PATH`.
- The agent needs no changes for any of this. It only has to keep prompts as LCEL templates so the fork can parse them.

## 2. Run time: prefetch (`pregel/main.py`, `_prediction.py`, `_vllm_agent.py`)

- `stream()` starts a `BackgroundTransitionPredictionWorker` whenever `transition_prediction` is present.
- The worker watches task and stream events, matches the rules, and predicts the next node.
- It builds that node's longest known prefix with `_prompt_prefix.build_longest_prefix`.
- It sends the prefix to vLLM with a POST to `/v1/agents/prefetch` (or `/chat/completions`). This is `warm_agent_prefixes` / `seed_agent_registry`.
- Env switches:
  - `LANGGRAPH_VLLM_AGENT_ENABLE=1` or `LANGGRAPH_VLLM_AGENT_BASE_URL` turns it on.
  - `LANGGRAPH_VLLM_AGENT_NAMESPACE` sets the namespace.
  - `LANGGRAPH_PROMPT_PSEUDO_DYNAMIC` controls whether pseudo-dynamic segments are filled in.
- Predictions and prefix logs are written to `LANGGRAPH_PREDICTION_LOG_DIR`. The harnesses set this for each run.

## 3. Per-request tagging (this repo → fork helpers)

- `agent/common/llm_request_metadata.py` imports helpers from `langgraph.pregel._vllm_agent`.
- On every model call it adds these fields to `extra_body`:
  - `job_id`, taken from `configurable`
  - `langgraph_node`, taken from LangGraph's run metadata
  - `call_type`, from `derive_call_type(node)`
  - `agent_id` + `record_in_registry`, only when prefetch is on. The id comes from `derive_agent_id(graph_path, job_id, unit)` and is the registry key that vLLM records prefixes under.
- `agent/common/model.py` attaches this through `get_model_config()`, so every node's `.with_config(...)` carries it.
- `model.py` also clears the prefetch env vars whenever `MODEL_PROVIDER != openai`, because the fork's `vllm_agent_enabled()` doesn't know which provider is in use.

## 4. Node-aware KV eviction (`_kv_forecast.py`)

- `tests/run_evaluate_node_eviction.py` uses `KVForecastSession(SWE_GRAPH_SPEC, ...)`.
- `SWE_GRAPH_SPEC` is a hand-written topology: node type, call sites, and next nodes. It has to be hand-written because the runtime node names don't match what `get_graph(xray=True)` reports.
- The harness calls `start_job`, `on_task_start`, `on_task_finish`, and `end_job` from stream events. This publishes `PROB|<job_id>` rows to Redis (`KV_FORECAST_REDIS_URL`), which the eviction vLLM ranks blocks with.
- It turns prefetch off by default (`KV_EVICTION_DISABLE_PREFETCH=1`) so the two mechanisms don't mix in one result.
- Without a Redis URL the session does nothing.

## 5. Other touch points

- `tests/prefix_analysis.py` imports `AGENT_UNIT_METADATA_KEY` and copies the fork's namespace format (`|`, `:`, `/`).
- `tests/system_prompt_population.py` sends prompts directly to `/v1/agents/prefetch` and copies `derive_agent_namespace`.
- The graph code (`StateGraph`, `ToolNode`, `add_messages`, `MemorySaver`) uses the normal langgraph API. The fork leaves it unchanged.

## 6. Prompt seeds (`agent/common/prompt_seeds.py`)

- A port of ODR's `prompt_seeds.py`. The node that runs before a target rebuilds the target's prompt from outputs it already has. It sends that prompt to `/v1/agents/prefetch` as `messages` with `prefill_on_miss`, so the prefill happens while the producer node is still generating or its tools are running.
- Each seed is cut right after the conversation placeholder, so it is always a prefix of the real prompt.

| Producer | Seeds |
|---|---|
| `come_up_with_research_next_step` | `check_research_step` (on the request) and `conduct_research` (on the request, then extended with the reply) |
| `conduct_research` | `extract_implementation_plan` (on the request, then extended with the reply when a tool phase follows) |
| `prepare_for_implementation` | `get_clear_implementation_plan_for_atomic_task` (little lead time) |
| `get_clear_implementation_plan_for_atomic_task` | `creating_diffs_for_task` (on the request, then extended with the reply when a tool phase follows) |

- `come_up_with_research_next_step` is not a target, because its restart prompt starts with its own previous request, which is already cached.
- The seed inputs copy each target's `invoke({...})` rather than sharing code with it, because the fork's analyser reads that dict. The fork's compile metadata is unchanged by the seeding code.
- Tool-bound seeds carry `tools`, posted by the seeding code itself because this branch's `warm_agent_prefixes` doesn't accept `tools`.
- `tests/check_prompt_seeds.py` runs the whole graph against a fake server and checks that each seed is an exact prefix of its target's real request, with the same agent id and tools.
- Settings: `SWE_PROMPT_SEEDS=0` turns all seeds off; `SWE_PROMPT_SEEDS_SKIP=<nodes>` skips specific targets. Each seed is logged to `<LANGGRAPH_PREDICTION_LOG_DIR>/job_<id>.prompt_seeds.jsonl`.

## Gotcha

- `env.template` sets `LANGGRAPH_VLLM_AGENT_TIMEOUT=180`, but the fork reads `LANGGRAPH_VLLM_AGENT_TIMEOUT_SECONDS`. The timeout override therefore does nothing, and the fork's default is used instead.
