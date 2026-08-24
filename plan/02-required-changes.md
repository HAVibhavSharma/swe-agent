# What changed here, and what is still missing

## A. Changes already applied to this repo

### 1. The model is built per call, from the node's config

**Was:** `ChatAnthropic(model="claude-sonnet-4-20250514")` constructed at import
time in eight places, wired into module-level LCEL chains.

**Now:** `agent/common/model.py` — one configurable model, re-configured per
call via `get_model_config(config, max_tokens)`. Node signatures gained
`config: RunnableConfig`.

Why it had to change: the config is what carries `metadata["langgraph_node"]`.
A model built at import time cannot see it, so every request would reach vLLM
without a node label and be indexed under the empty label — cached, but joined
to nothing. It is also what makes `OPENAI_BASE_URL` (the local vLLM) reachable
at all.

`MODEL_PROVIDER=anthropic` still works for debugging; the vLLM-only kwargs
(`extra_body`, `stream_usage`, `seed`) are omitted for it, and the prefetch env
vars are cleared so nothing warms a cache the traffic never reaches.

### 2. Request metadata

`agent/common/llm_request_metadata.py`, copied verbatim from ODR. It is already
workload-agnostic — no ODR node names in it.

### 3. Loop bounds, in the shape the analyser reads

`agent/common/configuration.py` plus counters in both subgraph states:

| Loop | Counter | Bound |
|---|---|---|
| architect research (outer) | `research_iterations` | `configurable.max_researcher_iterations` |
| architect react | `tool_call_iterations` | `configurable.max_react_tool_calls` |
| developer react | `tool_call_iterations` (reset per atomic task) | `configurable.max_react_tool_calls` |
| developer atomic tasks | `atomic_tasks_done` | `configurable.max_atomic_tasks` |

Two reasons. First, the upstream agent had **no bound on any loop** — it ended
when the model stopped calling tools, or at the root `recursion_limit=200`. On a
benchmark that is an unbounded per-instance cost.

Second, the names are deliberately ODR's. `langgraph/graph/state.py:1481
_has_iteration_limit_condition` matches `tool_call_iterations` /
`max_react_tool_calls` and `research_iterations` / `max_researcher_iterations`
**literally**, so reusing them buys working `state_over_config_limit`
prediction rules with no fork change. That coupling is a wart, not a design —
see §C.

### 4. `_is_root=True`

`agent/graph.py` now compiles with the fork-only kwarg, which is what triggers
the recursive prompt/transition analysis. On stock langgraph this raises
`TypeError`, deliberately: the workload has no meaning without the fork.

The harnesses compile `swe_agent_builder` themselves (with a checkpointer),
exactly as ODR's harnesses compile `deep_researcher_builder`.

### 5. Two pre-existing bugs, fixed while here

- **`architect/graph.py` had both a conditional edge and an unconditional edge
  from `check_research_step`.** `add_conditional_edges(check_research_step, …)`
  and `add_edge("check_research_step", "conduct_research")` were both declared,
  so `conduct_research` ran on *every* visit regardless of the router's verdict,
  and `plan_is_not_valid` fanned out to both successors. The unconditional edge
  is removed. This changes behaviour — it is the difference between a router
  that decides and a router that is ignored.
- **Dead `call_model` function** in `architect/graph.py`, referencing prompt
  variables the template does not declare and registered as no node. Removed.

## B. Prompt composition from LCEL prompt templates (implemented)

**Was the blocker.** `diag_prediction.py` used to report
`prompt_composition: 0 node(s)`, which meant `/v1/agents/prefetch` warmed
nothing: the predictor fired on time and sent an empty prefix.

The cause was a shape mismatch, not a misconfiguration.
`_extract_prompt_composition` (`langgraph/graph/state.py`) looks for
`.invoke` / `.ainvoke` whose first argument "looks like a prompt"
(`_looks_like_prompt_input_expr`), and a dict literal qualified **only** with a
`"content"` key. This agent passes a dict of *template variables*:

```python
runnable = plan_next_step_prompt | model      # LCEL chain
response = runnable.invoke({
    "implementation_research_scratchpad": ...,  # no "content" key
    "codebase_structure": ...,
})
```

and the prompt text is not in the source at all — it comes from
`agent/*/prompts/*.md` via `helpers/prompts.py::markdown_to_prompt_template`.

**Fixed on `langgraph-dev` branch `Prediction-SWEbench`** (~570 lines in
`langgraph/graph/state.py`). The template was always reachable: the chain is a
live object by compile time, and `inspect.getclosurevars` already gives the
analyser the names a node references. So:

1. `_resolve_prompt_template_object` resolves an invoke target to the
   `BasePromptTemplate` at the head of its chain — through a local assignment,
   through `|`, and through the chain-shaping methods (`with_config`,
   `bind_tools`, `with_structured_output`, …).
2. `_prompt_template_parts` flattens the template into ordered parts: message
   templates, fixed-content messages, `MessagesPlaceholder`s, and anything
   unreadable (multimodal), which is recorded rather than skipped.
3. `_analyze_template_text` splits each message's template on its `{variables}`
   with `string.Formatter`, exactly as `analyze_format_call` splits a
   `str.format` call. Literal text → `static`. A variable → whatever the invoke
   dict binds it to, analysed by the existing `_analyze_prompt_expr`. An unbound
   variable → `pseudo_dynamic` placeholder.
4. `_composition_from_prompt_template` emits the standard `PromptComposition`,
   including a `message_envelope` that keeps the roles — so the warm request has
   the same shape as the real one instead of one flattened system message. A
   `MessagesPlaceholder` bound to a state key becomes a `state_messages` part;
   one bound to a function call closes the envelope, because a prefix that is
   wrong after that point is worse than a prefix that is short.

Everything is wrapped so a template shape it cannot read degrades to the old
behaviour instead of failing the compile.

**Also fixed: state attribute access.** `_extract_state_key_from_attribute`
teaches the analyser that `state.foo` is the same read as `state["foo"]`.
Pydantic and dataclass state schemas are read as attributes, and without this
every segment in such a graph came out opaque with no state key — and a segment
with no state key cannot be filled from the state snapshot, so it blocked the
prefix for a value that was available all along. Wired into
`_analyze_prompt_expr`, `_looks_like_message_state_ref` and
`_extract_prompt_message_envelope`.

**Measured on this agent** (`python diag_prediction.py`):

| | before | after |
|---|---|---|
| nodes with prompt composition | 0 | 6 |
| static prompt chars found | 0 | 8460 |
| transition rules | 11 | 11 |
| warm prefix for `conduct_research` | — | 1229 chars, 2 messages (system + user), blocking correctly on the scratchpad |

**No regression on ODR.** The full compile metadata — every composition entry
and every transition rule — is byte-identical before and after the change. The
template path only engages when a `BasePromptTemplate` is actually found at the
head of the chain, and ODR has none.

Three new tests in `libs/langgraph/tests/test_utils.py` cover the chain read,
the envelope truncation, and attribute access.

## C. Smaller gaps in the same family

| Gap | Where | Effect | Fix |
|---|---|---|---|
| `_infer_message_state_key` requires a key ending in `_messages`, reached by `state["k"]` or `state.get("k")` | this agent's message lists are `implementation_research_scratchpad` / `atomic_implementation_research` | the branch rules compile, but with an empty `message_state_key` | accept any `add_messages`-reduced field rather than matching the name |
| `_has_no_tool_calls_condition` matches three literal source strings | our routers use `not most_recent_message.tool_calls`, which is one of them — by construction | works, accidentally | match the AST shape, not the text |
| `_extract_tool_call_name_literals` requires the loop variable to be named `tool_call` | this agent never branches on a tool *name* | no `tool_call_name` rules; only `any_tool_calls` / `no_tool_calls` | take the name from the enclosing `for` target |

Unrelated but worth knowing: three tests in `libs/langgraph/tests/test_utils.py`
(`…classifies_state_independent_function_calls`,
`…preserves_concat_order_for_prefix_planning`,
`…skips_subgraph_invocations`) fail on the branch, and failed identically
before this change. The first two assert that a prompt produced by a plain
function call (`prompt = build_prompt()`) is recognised;
`_looks_like_prompt_input_expr` rejects it.

What **does** work today, unchanged: `_extract_tool_call_branch_targets`
(`state.py:1382`) reads `if <expr>.tool_calls: return "<literal>"` out of a
conditional-edge router and maps the returned label through the path map
(`state.py:3476-3495`). Both of this agent's react loops are written that way,
so the `any_tool_calls` / `no_tool_calls` rules compile for `conduct_research`
and `get_clear_implementation_plan_for_atomic_task` with no fork change.

## D. Known limits of the harness as written

1. **One workspace, so jobs are strictly sequential.** The agent hard-codes
   `./workspace_repo` in four places, so `tests/swebench_instances.py` wipes and
   re-creates that one directory per job. ODR's `--max-concurrency` has no
   equivalent until the workspace is per job (which needs the agent's path
   references parameterised).
2. **No test execution.** The agent never runs the repo's tests, so there is no
   repair loop — it writes a patch and stops. Scoring happens afterwards, out of
   process, via `swebench eval`. That keeps Docker off the GPU box, and keeps a
   scoring failure from corrupting a latency measurement.
3. **`creating_diffs_for_task` applies edits by line number** parsed out of the
   model's own `original_code_snippet` block. It is a fragile edit mechanism and
   a likely source of empty or broken patches; `empty_patch` is reported per run
   in the summary so a low resolve rate can be attributed to it rather than to
   the serving stack.
4. **The `.md` prompts are unmodified.** Nothing in them mentions SWE-bench, a
   failing test, or a patch format; the issue text arrives as the architect's
   first scratchpad message. Expect a low resolve rate compared with purpose-built
   SWE agents. For this experiment that is acceptable — the workload needs to be
   *representative and stable*, not state of the art — but do not report the
   resolve rate as an agent result without saying so.
