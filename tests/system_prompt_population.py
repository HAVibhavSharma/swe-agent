"""Fill vLLM's ``AgentPrefixRegistry`` with this graph's system prompts.

The counterpart of ``open_deep_research/tests/system_prompt_population.py``,
carrying the same idea to the SWE agent: instead of a cold pass of real
benchmark instances whose only job is to leave the caches warm, a handful of
``POST /v1/agents/prefetch`` calls seed the registry directly.

Why the swap
------------
``/v1/agents/prefetch`` is normally a *reader*: it looks up whatever prefixes
an agent has already accumulated and fires one phantom request per prefix. The
registry is written retrospectively, by ordinary chat traffic
(``vllm/v1/agent_prefetch/auto_register.py``), so on a node's first execution
there is nothing to look up and the endpoint answers 200 OK having submitted
nothing at all.

The endpoint's ``text`` field is the way out. When it is set, the server wraps
the text as a message of ``text_role``, runs it through the served model's chat
template with ``add_generation_prompt=False`` -- so the tokens are a strict
prefix of any real chat opening with the same message -- chunk-aligns the
result and *records* it under ``agent_id`` before fanning out. One call per
node therefore puts every node's system prompt in the registry before the first
real request exists.

What this phase does, and what ``prefill_on_miss`` changes
----------------------------------------------------------
It always populates the **registry**. Whether it also fills the KV cache
depends on ``prefill_on_miss``.

By default a ``prefetch_only`` request whose connector reports zero external
tokens is aborted before prefill. That is the right contract for a
*prediction-driven* prefetch -- a phantom is a promotion from LMCache L1 into
HBM, and spending a full prompt's prefill on a guess is worse than letting the
real request pay it. But a seeding call is not a guess: the prefix it just
recorded is one nothing has ever computed, so the miss is certain and the abort
makes the seed a no-op.

``prefill_on_miss=True`` (the default here, ``--no-prefill-on-miss`` to turn it
off) opts those phantoms out of the abort. Each one prefills once, and the
LMCache store path writes the result as a side effect, so the prefix is in L1
before the first real request. Against a vLLM without the matching field the
value is dropped by pydantic and the phase degrades to registry-only.

Prefix construction
-------------------
Every prompt in this repo is a markdown chat template (``helpers/prompts.py``)
whose **first** section is ``# System``, and in all of them the ``{variables}``
live in the later ``# Human`` / ``# Placeholder`` sections. So the seed is that
leading system block, rendered whole: a byte-exact prefix of what the node
sends at runtime.

``_build_seed_text`` still stops at the first field it cannot fill honestly, so
a prompt that grows a variable in its system block degrades to a shorter but
still true prefix rather than a guess that matches nothing and fails silently.
``text_role`` follows the section's own role, because the chat template wraps a
system block in different control tokens than a user block and a role mismatch
records a prefix no real request can hit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from string import Formatter
from typing import Any

DEFAULT_TIMEOUT_S = 120.0

PREFETCH_PATH = "/v1/agents/prefetch"

# Name of this phase wherever it is printed or written to a report.
PHASE_LABEL = "system prompt population"


def agent_namespace() -> str:
    """The namespace half of the registry key.

    Must match what the *server* uses, which it reads from its node-eviction
    config (``NodeEvictionConfig.prefetch_agent_namespace``) rather than from
    anything sent on the wire. Both sides default to ``langgraph``; this env var
    is the client-side override that already drives
    ``langgraph.pregel._vllm_agent.derive_agent_namespace``, so setting it keeps
    the two in step.
    """
    return os.getenv("LANGGRAPH_VLLM_AGENT_NAMESPACE", "langgraph")


def resolve_prefetch_url() -> str | None:
    """Where to POST, or None if there is no local vLLM to seed.

    ``SYSTEM_PROMPT_POPULATION_URL`` wins if set. Otherwise it is derived from
    the same base URL the traffic uses, so the seed cannot drift onto a
    different host from the requests it is meant to warm.
    """
    explicit = os.getenv("SYSTEM_PROMPT_POPULATION_URL", "").strip()
    if explicit:
        return explicit

    base = (
        os.getenv("LANGGRAPH_VLLM_AGENT_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
    )
    if not base:
        return None

    # The route is mounted at the server root. Both conventional spellings of
    # the base URL have to be stripped back to that root, longest first:
    # `LANGGRAPH_VLLM_AGENT_BASE_URL` is normalized to end in `/v1/agents`
    # while `OPENAI_BASE_URL` ends in `/v1`.
    root = base.rstrip("/")
    for suffix in ("/v1/agents", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return f"{root}{PREFETCH_PATH}"


# ---------------------------------------------------------------------------
# Which prompt each node sends
# ---------------------------------------------------------------------------
# `node` is the bare runtime name -- the value LangGraph puts in
# `metadata["langgraph_node"]` and therefore the value the real request sends,
# not the graph path (`swe_architect:conduct_research`). The same names as
# SWE_GRAPH_SPEC in run_evaluate_node_eviction.py, for the same reason:
# `auto_register.agent_ids_for_extra_args` records every real prefix under
# `{namespace}:{langgraph_node}`, and engine core can only build that form, so
# seeding it is what makes the seed and the traffic land in the same bucket.
#
# `agent_kind` follows `detect_agent_kind`: "react" for a call whose model was
# `.bind_tools(...)`, since those accumulate several prefixes per agent;
# "non-react" for a single static preamble, which makes the server drop the
# agent's existing entries before recording (enforcing one prefix) and forces
# the fan-out to 1.
#
# `creating_diffs_for_task` is the one node with two call sites -- new-file
# creation and diff extraction, two different prompts under one node name,
# because the node name is the label vLLM sees. Both are marked "react" so the
# second seed does not evict the first; "non-react" would leave the node
# holding whichever prompt was posted last.
_NODE_PROMPTS: tuple[tuple[str, str, str, str], ...] = (
    # (node, "module:attribute", text_role, agent_kind)
    (
        "come_up_with_research_next_step",
        "agent.architect.graph:plan_next_step_prompt",
        "system",
        "non-react",
    ),
    (
        "check_research_step",
        "agent.architect.graph:check_research_prompt",
        "system",
        "non-react",
    ),
    (
        "conduct_research",
        "agent.architect.graph:conduct_research_prompt",
        "system",
        "react",
    ),
    (
        "extract_implementation_plan",
        "agent.architect.graph:extract_implementation_prompt",
        "system",
        "non-react",
    ),
    (
        "get_clear_implementation_plan_for_atomic_task",
        "agent.developer.graph:get_clear_implementation_plan_prompt",
        "system",
        "react",
    ),
    (
        "creating_diffs_for_task",
        "agent.developer.graph:implement_new_file_prompt",
        "system",
        "react",
    ),
    (
        "creating_diffs_for_task",
        "agent.developer.graph:extract_diffs_tasks_prompt",
        "system",
        "react",
    ),
)


def _fillable_values() -> dict[str, Any]:
    """Every template field this process can fill the way the run will.

    Empty today, and kept as the seam it is: every ``{variable}`` in this
    repo's prompts is runtime state (the scratchpad, the checked-out repo's
    file listing, the current task), none of which exists before the first
    instance is materialized. A field added here is filled into the seed; a
    field left out truncates it.
    """
    return {}


def _leading_message(prompt: Any) -> tuple[str | None, str | None, str | None]:
    """The first message of a prompt template: ``(template, role, error)``.

    A chat template's leading section is the only part of the prompt that is a
    prefix of the request in the wire sense -- everything after it is preceded
    by content this process does not have.
    """
    messages = getattr(prompt, "messages", None)
    if messages is None:
        # A plain PromptTemplate (no `_type: "chat"` header): the whole
        # template is one user turn.
        template = getattr(prompt, "template", None)
        if isinstance(template, str) and template:
            return template, "user", None
        return None, None, "prompt is neither a chat template nor a string template"
    if not messages:
        return None, None, "chat template has no messages"

    first = messages[0]
    inner = getattr(first, "prompt", None)
    template = getattr(inner, "template", None)
    if not isinstance(template, str) or not template:
        # MessagesPlaceholder or similar: nothing static to seed with.
        return (
            None,
            None,
            f"leading message is {type(first).__name__}, which carries no text",
        )

    class_name = type(first).__name__.lower()
    if "system" in class_name:
        role = "system"
    elif "human" in class_name:
        role = "user"
    else:
        return None, None, f"leading message role is unsupported ({type(first).__name__})"
    return template, role, None


def _build_seed_text(template: str, values: dict[str, Any]) -> tuple[str, str | None]:
    """Fill what is knowable, stop at the first field that is not.

    Returns ``(text, blocked_on_field)``; ``blocked_on_field`` is None when the
    whole template was fillable, which is the case for every system block in
    this repo today.
    """
    out: list[str] = []
    for literal, field, _spec, _conversion in Formatter().parse(template):
        if literal:
            out.append(literal)
        if field is None:
            continue
        if field not in values:
            return "".join(out), field
        out.append(str(values[field]))
    return "".join(out), None


def _load(target: str) -> Any:
    """Import ``module:attribute``. Raises with the name that failed."""
    import importlib

    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def build_seeds() -> list[dict[str, Any]]:
    """One seed descriptor per call site, longest-honest-prefix first.

    Ordered by descending seed length so the prompts that carry real prefill
    savings are registered before anything that might time out.
    """
    values = _fillable_values()
    seeds: list[dict[str, Any]] = []
    for node, target, text_role, agent_kind in _NODE_PROMPTS:
        try:
            prompt = _load(target)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            # A renamed or moved prompt is a silent hole in the registry, so
            # say so rather than skipping quietly.
            seeds.append(
                {
                    "node": node,
                    "prompt": target,
                    "text": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        template, role, error = _leading_message(prompt)
        if template is None:
            seeds.append(
                {"node": node, "prompt": target, "text": None, "error": error}
            )
            continue
        if role != text_role:
            # The declared role is what the payload sends; a mismatch would
            # record the prefix under the wrong control tokens.
            seeds.append(
                {
                    "node": node,
                    "prompt": target,
                    "text": None,
                    "error": (
                        f"leading message is a {role} block but the table "
                        f"declares {text_role}"
                    ),
                }
            )
            continue

        text, blocked_on = _build_seed_text(template, values)
        seeds.append(
            {
                "node": node,
                "prompt": target,
                "text": text,
                "text_role": text_role,
                "agent_kind": agent_kind,
                "blocked_on_field": blocked_on,
                "text_chars": len(text),
            }
        )
    seeds.sort(key=lambda seed: len(seed.get("text") or ""), reverse=True)
    return seeds


async def populate_system_prompts(
    *,
    label: str,
    top_k: int | None = None,
    prefill_on_miss: bool = True,
    record_to: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Seed the registry with every node's system prompt. Never raises.

    With ``prefill_on_miss`` (the default) each seed is also prefilled once so
    it lands in LMCache; see the module docstring for why that is the whole
    difference between warming the cache and only warming the lookup.

    A failed seed means the warm phase's first visit to that node prefills as it
    always did -- worth knowing, never worth throwing away a run for. The
    per-node outcome is returned and written next to the run's other metrics, so
    a warm phase that never hits cannot be discovered later by wondering why the
    hit rate looks low.
    """
    url = resolve_prefetch_url()
    namespace = agent_namespace()

    if url is None:
        result: dict[str, Any] = {
            "ok": False,
            "reason": "no_prefetch_url",
            "detail": (
                "None of SYSTEM_PROMPT_POPULATION_URL, "
                "LANGGRAPH_VLLM_AGENT_BASE_URL or OPENAI_BASE_URL is set, so "
                "there is no vLLM server to seed. The warm phase will prefill "
                "every node's system prompt on first visit."
            ),
            "label": label,
            "nodes": [],
        }
        _report(result, record_to)
        return result

    try:
        seeds = build_seeds()
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        result = {
            "ok": False,
            "reason": "seed_build_failed",
            "detail": f"{type(exc).__name__}: {exc}",
            "label": label,
            "url": url,
            "nodes": [],
        }
        _report(result, record_to)
        return result

    # Imported here so a run that skips this phase does not require httpx.
    import httpx

    nodes: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for seed in seeds:
            node = seed["node"]
            agent_id = f"{namespace}:{node}"
            row: dict[str, Any] = {
                "node": node,
                "agent_id": agent_id,
                "prompt": seed["prompt"],
                "text_role": seed.get("text_role"),
                "text_chars": seed.get("text_chars", 0),
                "blocked_on_field": seed.get("blocked_on_field"),
            }
            if not seed.get("text"):
                row.update(
                    {
                        "ok": False,
                        "reason": "empty_seed",
                        "detail": seed.get("error", "the template blocked before any text"),
                    }
                )
                nodes.append(row)
                continue

            payload: dict[str, Any] = {
                "agent_id": agent_id,
                "text": seed["text"],
                "text_role": seed["text_role"],
                "agent_kind": seed["agent_kind"],
                # Block until the server has recorded and fanned out, so the
                # phase boundary is a real boundary: nothing from here is still
                # in flight when the warm phase starts.
                "wait": True,
                # The node-eviction index key the *upcoming real request* will
                # present. job_id is deliberately absent -- no job exists yet,
                # and vLLM declines to index rather than index under a wrong
                # key. call_type is the bare node, matching
                # `derive_call_type(node)` on the client side.
                "langgraph_node": node,
                "call_type": node,
                # The seed is a prefix nothing has computed, so LMCache is
                # guaranteed to miss. Without this the phantom is aborted before
                # prefill and the cache stays empty; with it the prefill happens
                # once here and the store path puts the prefix in L1. Ignored by
                # a vLLM without the matching field.
                "prefill_on_miss": prefill_on_miss,
            }
            if top_k is not None:
                payload["prefetch_top_k"] = top_k

            try:
                response = await client.post(url, json=payload)
                body = response.json()
                if not isinstance(body, dict):
                    body = {"reason": "bad_payload", "body": body}
                row.update(body)
                row["status_code"] = response.status_code
                row["ok"] = response.status_code == 200
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                row.update(
                    {
                        "ok": False,
                        "reason": "request_failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
            nodes.append(row)

    seeded = [row for row in nodes if row.get("ok")]
    result = {
        "ok": bool(seeded) and len(seeded) == len(nodes),
        "label": label,
        "url": url,
        "namespace": namespace,
        "nodes_attempted": len(nodes),
        "nodes_seeded": len(seeded),
        "prefill_on_miss": prefill_on_miss,
        "nodes": nodes,
    }
    _report(result, record_to)
    return result


def _report(result: dict[str, Any], record_to: Path | None) -> None:
    if record_to is not None:
        record_to.parent.mkdir(parents=True, exist_ok=True)
        record_to.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if result.get("reason") in {"no_prefetch_url", "seed_build_failed"}:
        print(
            f"System prompt population FAILED ({result['reason']}: "
            f"{result.get('detail', '')}) -- the registry is empty and every "
            "node prefills on first visit"
        )
        return

    print(
        f"System prompt population: {result['nodes_seeded']}/"
        f"{result['nodes_attempted']} call sites seeded at {result['url']} "
        f"(namespace={result['namespace']}, "
        f"prefill_on_miss={result.get('prefill_on_miss')})"
    )
    for row in result["nodes"]:
        if not row.get("ok"):
            print(
                f"  {row['node']:<46} FAILED "
                f"({row.get('reason')}: {row.get('detail', '')})"
            )
            continue
        # `available_prefixes` is what the registry held *after* the seed was
        # recorded, so a 0 here means the seed itself did not stick -- the
        # single most useful number on the line, and the one that would
        # otherwise only show up as an unexplained miss much later.
        blocked = row.get("blocked_on_field")
        truncation = f" (truncated at {{{blocked}}})" if blocked else " (full)"
        print(
            f"  {row['node']:<46} {row['text_chars']:>6} chars{truncation}, "
            f"available={row.get('available_prefixes')} "
            f"submitted={row.get('submitted')} "
            f"completed={row.get('completed')}"
        )
        # The server echoes the field back only if it understood it. A None here
        # against a run that asked for prefill means the vLLM on the other end
        # predates the change, and the phase warmed the registry only -- which
        # looks identical in every other number.
        if result.get("prefill_on_miss") and row.get("prefill_on_miss") is None:
            print(
                "    WARNING: the server did not echo prefill_on_miss -- it "
                "predates that field, so this seed was NOT prefilled and "
                "LMCache is still empty for it"
            )
        if row.get("available_prefixes") == 0:
            print(
                "    WARNING: the server recorded nothing for this node -- the "
                "seed was shorter than one LMCache chunk, or the render failed"
            )
