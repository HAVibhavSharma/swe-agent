# How LangGraph fires a prefetch

This file traces one prefetch end to end: from the token that makes the next
node knowable, through the fork's prediction worker, to the
`POST /v1/agents/prefetch` that vLLM answers with a phantom request. It is the
mechanism behind the "prefix prefetch (treatment)" row in
[README.md](README.md) and the runtime half of what
[02-required-changes.md](02-required-changes.md) made possible in this repo.

Everything described here lives in `~/Projects/langgraph-dev` on branch
`Prediction-SWEbench`, principally `langgraph/pregel/_prediction.py` and
`langgraph/pregel/_vllm_agent.py`. This repo contributes only the labels on its
requests and the graph the analyser reads.

---

## 1. Why there is a prefetch at all

An agent's time is dominated by prefill, and an agent's prompts are mostly
repetition: every node re-sends a long system block, and each turn of a react
loop re-sends the whole conversation so far. If those tokens are already in the
KV cache when the request arrives, the request skips straight to decode.

The opportunity the fork exploits is that **the next node becomes knowable
before its request exists**. When the architect's research node emits a tool
call, the edge it will take is already determined, yet the tool still has to run
and the next prompt still has to be assembled — hundreds of milliseconds in
which the server is free to warm that node's prefix. The prediction worker
exists to spend that window.

## 2. The mental model: two halves that meet at an `agent_id`

The single most common misreading of this code is that the predictor sends the
prompt it wants warmed. It does not. The design is two independent halves:

**Writing the registry is retrospective.** Every ordinary chat completion this
repo issues carries `agent_id` and `record_in_registry: True` in `extra_body`
(`agent/common/llm_request_metadata.py:55-76`). vLLM's
`vllm/v1/agent_prefetch/auto_register.py` records that request's chunk-aligned
prompt prefix under the `agent_id` it carried. The registry is therefore a
memory of what each place in the graph has actually sent.

**Firing a prefetch is a read of that memory.** The predictor's POST names an
`agent_id` and a `prefetch_top_k`, and vLLM fans out one phantom request per
selected prefix to pull those blocks into HBM.

So the stream content decides **when** a prefetch fires and **which bucket** it
reads. It never decides **what** is warmed. That content comes from the
bucket's history. Two consequences follow immediately, and both matter when
reading a run:

- A node's *first* execution can never be helped. Nothing is registered yet, so
  the endpoint answers `200 OK` with `submitted=0`. This is the hole
  `tests/system_prompt_population.py` fills by POSTing the same endpoint with an
  explicit `text` seed — see [04-harnesses.md](04-harnesses.md).
- The registry only exists if vLLM is configured to keep it
  (`VLLM_NODE_EVICTION_PREFETCH_DRAIN=1`). Without it every prefetch is a
  well-formed no-op.

## 3. Compile time: the two metadata maps

Nothing at runtime can predict anything unless the graph was compiled with the
fork's root flag:

```python
swe_agent = swe_agent_builder.compile(_is_root=True)   # agent/graph.py:41
```

`_is_root=True` makes `CompiledStateGraph.compile` walk this graph *and its
subgraphs* and attach two maps (`langgraph/graph/state.py:3319-3322`):

| Map | What it holds | Used for |
|---|---|---|
| `transition_prediction` | rules of the form *(producer node, condition) → next executable node* | deciding **that** a prefetch should fire, and for **which** node |
| `prompt_composition` | per node, the static/dynamic segments of the prompt it will send | building the prefix that gets logged, and the react/non-react classification |

Conditions come in a small closed set — `tool_call_name`, `any_tool_calls`,
`no_tool_calls`, `state_over_config_limit`, `state_over_static_limit`,
`config_key_equals`, plus `any_of` / `all_of` / `not` / `always`
(`_prediction.py:53-101`). Which condition a rule carries is what determines how
early it can fire, and that is the subject of §5.

Both maps are dumped at compile time for inspection; `python diag_prediction.py`
reports whether they came out non-empty.

## 4. Runtime wiring: a stream the caller never asked for

The harness streams tasks:

```python
async for part in graph.astream(..., stream_mode="tasks", subgraphs=True):   # tests/run_evaluate.py:501
```

but that is not what the predictor runs on. When `transition_prediction` is
present, the fork widens the stream behind the caller's back:

```python
if background_prediction_worker.enabled:
    stream_modes.update(("messages", "tasks", "updates"))   # langgraph/pregel/main.py:2643
```

Every chunk the loop emits is then handed to the worker
(`main.py:3713`), which is a bounded queue drained by a daemon thread
(`_prediction.py:1399-1406`). This is deliberate isolation: prediction work
never runs on the request path, and a queue that fills simply increments
`dropped_events` rather than slowing the graph. The cost is that a prefetch can
be late but never blocking.

## 5. The firing decision

`TransitionPredictionTracker.on_stream_chunk` (`_prediction.py:1667`) routes by
stream mode. There are three paths, and they differ in how much lead time they
can produce.

### Mid-decode, from message content — the path worth having

`_on_message_chunk` (`_prediction.py:1840`) parses the tool calls out of the
streaming `AIMessage` chunk while the producer node's LLM is still generating,
and matches them against that node's rules. Only rules whose condition is
decidable from the stream are eligible, which `_condition_uses_message_stream`
(`:1643`) restricts to `tool_call_name` and `any_tool_calls`.

The concrete trigger is therefore: *a tool call name appears in a partial
assistant message and matches a rule*. Each
`(rule_id, namespace, message_id, call_token)` fires at most once, so a tool
call that accretes tokens across many chunks does not re-fire.

This is the case that pays: the tool has not run yet, the next prompt has not
been built yet, and the warm has that whole interval to land.

### At node end, from the committed result

Conditions that depend on state rather than on tokens — `no_tool_calls`, the
two loop-bound comparisons, `config_key_equals` — cannot be decided mid-stream.
They are evaluated in `_on_task_result_event` (`:1973`) against the task's
result dict. The lead shrinks to whatever gap exists between one node finishing
and the next node's request reaching the server.

### At the successor's start — a backfill, not a prefetch

`_backfill_prediction_for_actual_start` (`:1692`, called from `_on_task_event`
at `:2120`) records that a rule *would* have predicted this transition when
nothing fired earlier. Lead is zero and no useful warm is possible; it exists so
the fidelity logs can tell "the rule was wrong" apart from "the rule was right
but too late."

Every path stamps the server's own clock through `_mark_min_lead` (`:1507`),
which posts a `min_lead` echo marker carrying the `source`
(`message_stream` / `task_result` / `task_start`). The marker is emitted
*outside* every gate that can suppress the warm, because the question it answers
is "when did this transition become knowable", not "was a warm sent".

## 6. From a matched rule to the POST

Once a rule matches, `rule["prediction"]["next_executable_node"]` names the
target and `_log_prefix_for_prediction` (`:1557`) carries out the rest:

1. Look up `prompt_composition[target_node]`. A node with no entry — a ToolNode,
   a pure state-shuffling node — produces no prefetch, correctly, because it
   caches nothing.
2. Call `build_longest_prefix` (`langgraph/pregel/_prompt_prefix.py`) against the
   last state snapshot for this scope, stopping at the first segment it cannot
   fill honestly. The result goes to `job_<id>.prefix.jsonl` with
   `blocked_on_segment_ordinal` and `blocked_on_state_keys`, which is how you see
   *why* a prefix came out short.
3. Call `enqueue_prediction_prefix` (`:1098`), which builds the identity and
   hands it to the background HTTP worker.

The identity is built for the **predicted** node, not the producer:

```python
agent_id = derive_agent_id(predicted_node, job_id=context["job_id"], unit=unit)
node     = bare_node_name(predicted_node)
call_type = derive_call_type(node)
```

`_submit` (`:1214`) then makes the call:

```python
warm_agent_prefixes(
    base_url=..., agent_id=agent_id, agent_kind=..., wait=False,
    prefetch_top_k=1,
    job_id=..., langgraph_node=node, call_type=call_type,
    issuer="langgraph",
)
```

Note what is absent. The `prefix_messages` built in step 2 are placed on the
queue and written to the prefix log, but **no `text` and no `messages` are sent
on the wire** for a `prediction_prefetch`. The endpoint's seeding fields exist
(`_vllm_agent.py:456-473`) and are used by the workflow-level oracle and by
`tests/system_prompt_population.py`; the in-graph predictor does not use them.
The one branch that ever sent content was `startup_warmup` via
`seed_agent_registry`, and it is now unreachable — `_enqueue_graph_warmup`
(`:1148`) has no caller. It was removed because it issued demand traffic the
baseline arm never sent, which made the two arms serve different work.

## 7. The identity contract

Two different keys ride on the same POST, and confusing them is the usual cause
of a prefetch that succeeds and helps nothing.

**`agent_id` — the registry key.** `{namespace}:{job_id}:{graph path}`
(`derive_agent_id`, `_vllm_agent.py:194`). One bucket per *place in the graph,
in this run*. This works only because vLLM's `auto_register.py` honours an
explicit `agent_id`; an id constructed here that no writer ever creates is
invisible, and every prefetch for it returns `200 OK … available=0 submitted=0`
in under a millisecond. The `#unit` suffix that once separated concurrent
instances of a node was removed: two callers resolved the unit at different
moments and so named different buckets, and both still got `200 OK`.

**`(job_id, langgraph_node, call_type)` — the eviction index key.** Sent
alongside, and it must be the identity the *next real call* will present —
which is what `inject_langgraph_request_metadata` builds from
`configurable["job_id"]`, `metadata["langgraph_node"]` and
`derive_call_type(node)`. Send all three or vLLM caches the blocks without
indexing them, leaving them on plain LRU. The response echoes `scored`, and
`_submit` logs it, so an arm that is not actually measuring the policy on
prefetched prefixes says so in its own log rather than in a disappointing hit
rate.

`agent_kind` (`react` / `non-react`) is derived from the target's call
expression — `.bind_tools(` means react (`detect_agent_kind`,
`_vllm_agent.py:349`) — and tells the server it is looking at a growing
conversational prefix rather than a fixed one.

`issuer="langgraph"` names this caller on the server's access line. It exists
because the workflow-level oracle warms the same agents with a near-identical
payload, and without it "which caller produced this hit" is unanswerable after
a run.

## 8. The whole path, in order

```
architect's LLM is decoding
        │  chunk carries tool_call "grep_repo"
        ▼
StreamMessagesHandler ──► stream.put(("messages", chunk))
        │
        ▼  main.py:3713 (non-blocking; drops on a full queue)
BackgroundTransitionPredictionWorker.enqueue
        │
        ▼  daemon thread, _prediction.py:1406
TransitionPredictionTracker.on_stream_chunk
        │  rule (conduct_research, tool_call_name=grep_repo) → tools
        ├──► echo_marker("min_lead", agent_id=…, source="message_stream")   → POST /v1/echo
        ▼
_log_prefix_for_prediction
        │  build_longest_prefix(prompt_composition["tools"], state)  → job_*.prefix.jsonl
        ▼
enqueue_prediction_prefix     agent_id = langgraph:<job>:<path to tools>
        │
        ▼  second daemon thread, _prediction.py:1214
yield_to_claimed_prefetch? ──yes──► echo_marker("prefetch_skipped") and stop
        │ no
        ▼
POST /v1/agents/prefetch {agent_id, agent_kind, prefetch_top_k: 1,
                          job_id, langgraph_node, call_type, issuer,
                          prefill_on_miss, wait: false}
        │
        ▼
vLLM looks the bucket up, fans out ≤1 phantom request over the recorded prefix,
answers {available, submitted, completed, scored}            → job_*.vllm_agent.jsonl
        │
        ▼
tool runs, next prompt is assembled, the real request for `tools` arrives and
hits the blocks the phantom promoted — and is itself auto-registered, which is
what keeps the bucket current for the next iteration of the loop.
```

## 9. Gates, and what each one looks like when it closes

Every one of these fails silently in the sense that the run still completes and
still produces numbers. This table is the checklist for "the prefetch is on but
nothing is happening".

| Gate | Where | Symptom when closed |
|---|---|---|
| `_is_root=True` unsupported (stock langgraph) | `agent/graph.py:41` | `TypeError` at import — deliberately loud |
| `transition_prediction` empty | compile time | no prediction worker at all; `diag_prediction.py` reports it |
| `vllm_agent_enabled()` false | `_vllm_agent.py:35` | worker `enabled=False`, no POST ever; this is what `--no-prefetch` and `KV_EVICTION_DISABLE_PREFETCH` set |
| no base url or model | `_prediction.py:1091` | same, silently |
| **prefix logger disabled** | `_log_prefix_for_prediction:1563` | **no prefetch is enqueued at all** — needs `LANGGRAPH_PREDICTION_LOG_DIR` *and* a `job_id` in `configurable` (`_default_prefix_path:632`). `tests/run_evaluate.py:89` sets the first; `_job_config` supplies the second |
| target has no `prompt_composition` entry | `:1565` | that node is never warmed |
| claim held by another caller | `yield_to_claimed_prefetch`, TTL `LANGGRAPH_VLLM_PREFETCH_CLAIM_TTL_S` (60 s) | `prefetch_skipped` marker with `reason=claimed_by_other_caller` |
| registry empty for this `agent_id` | vLLM | `submitted=0`, `available=0` |
| index triple missing or unparsed | vLLM | `scored=false`; blocks warmed but on LRU |
| queue full | `:1386`, `:1195` | `dropped_events` rises; predictions silently skipped |

## 10. Reading a run afterwards

Per job, under `LANGGRAPH_PREDICTION_LOG_DIR`:

- `job_<id>.prefix.jsonl` — every prefix built for a prediction, with what
  blocked it from growing further.
- `job_<id>.vllm_agent.jsonl` — one line per POST: `agent_id`, `agent_name`,
  `agent_kind`, the index triple, and the server's `scored` / `submitted` /
  `completed`.
- `job_<id>.trace.jsonl`, `job_<id>.fidelity.jsonl` — predictions against what
  actually ran.

On the server's own timeline, `min_lead` markers bracket each target's warm
window from below and the target's own chat-completion line closes it; a
workflow oracle, where one is running, supplies `max_lead` as the other end.
Both come from the server's clock, which is the point of routing them through
`/v1/echo` rather than timestamping them client-side.

## 11. Where this sits in the harnesses

`tests/run_evaluate.py` is the only harness that exercises this path. Its
fixed-trace machinery (`agent/common/trace_store.py`) is orthogonal to it: in
`pinned` mode the live request still goes out carrying the full `extra_body`, so
registration, prediction and the phantom are all genuinely exercised and timed,
and only the *response* handed back to the agent is replaced by the recording.
That is what makes the prediction problem identical in both arms. In `offline`
mode nothing reaches the server and nothing is warmed; `record` runs should pass
`--no-prefetch` so the recorded trajectory is a stock-stack one.

`tests/run_evaluate_node_eviction.py` disables this path outright
(`KV_EVICTION_DISABLE_PREFETCH=1`, clearing both env vars after `load_dotenv`)
so that a hit-rate change can be attributed to the eviction policy alone. Its
only `/v1/agents/prefetch` traffic is the seeding phase described in §2, which
uses the endpoint's `text` field and never the predictor.
