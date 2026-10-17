"""Build the prompt a later node will send, and pay its prefill early.

The counterpart of ``open_deep_research/prompt_seeds.py``. Several prompts in
this graph share no prefix with anything the server has already computed,
because they copy an earlier node's conversation behind a different system
block:

``extract_implementation_plan``   its own system block, the repo tree, then the
                                  architect's whole research scratchpad,
                                  rewritten by ``convert_tools_messages_to_ai_and_human``
``creating_diffs_for_task``       its own system block, the target file
                                  (line-numbered) and task, then the atomic
                                  research loop, rewritten the same way
``check_research_step``           its own system block, then the scratchpad
``conduct_research``              its own system block (and tools), the repo
                                  tree, then the scratchpad
``get_clear_implementation_plan_for_atomic_task``
                                  its own system block (and tools), the repo
                                  tree, the target file and the task

Every one of them is constructible from requests that have already gone out
and replies already received. The node that runs before the target rebuilds
the target's prompt as far as it is known and sends it to
``/v1/agents/prefetch`` as ``messages``; with ``prefill_on_miss`` the server
records it under the target's ``agent_id`` and computes it in idle GPU time,
while the producer node is still generating or its tools are running.

A seed is always a *prefix* of the real prompt: it is cut right after the
placeholder that carries the conversation, because the conversation only ever
grows by appending and everything after the placeholder (closing
instructions, ``{output_format}``) would make it a prefix of nothing. So the
seed sent on any turn is safe, whichever turn turns out to be the last, and
each turn's seed extends the previous one -- the server prefills only what was
added.

When each seed fires
--------------------
``come_up_with_research_next_step``  request: seed ``check_research_step`` and
                                     ``conduct_research``; response: extend
                                     the ``conduct_research`` seed by the reply
``conduct_research``                 request: seed ``extract_implementation_plan``;
                                     response with tool calls: extend it
``prepare_for_implementation``       seed ``get_clear_implementation_plan_for_atomic_task``
                                     (little lead: the target starts next)
``get_clear_implementation_plan_for_atomic_task``
                                     request: seed ``creating_diffs_for_task``;
                                     response with tool calls: extend it

``come_up_with_research_next_step`` itself is not a target: when the research
loop restarts, its prompt opens with its own previous request, which is
already cached.

Nothing is read ahead: every value comes from state the producer node holds,
or the reply it has just received. Prompt construction mirrors the target
node's own ``invoke({...})`` without sharing code with it, because the fork's
compile-time analyser reads that dict literal; ``tests/check_prompt_seeds.py``
asserts each seed is a prefix of the request the target really sends.

Environment
-----------
``SWE_PROMPT_SEEDS``           ``0`` disables every seed
``SWE_PROMPT_SEEDS_SKIP``      comma-separated target nodes not to seed
``SWE_PROMPT_SEEDS_LOG_PATH``  explicit log path; otherwise derived from
                               ``LANGGRAPH_PREDICTION_LOG_DIR`` and the job id

Each also reads its ``ODR_`` spelling when the ``SWE_`` one is unset, which is
what pitlane sets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence
from urllib import error, request

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.pregel._vllm_agent import (
    AGENT_UNIT_METADATA_KEY,
    derive_agent_id,
    derive_call_type,
    graph_path_from_checkpoint_ns,
    vllm_agent_base_url,
    vllm_agent_enabled,
)

logger = logging.getLogger(__name__)

# One worker: a turn's extended seed must reach the server after the seed it
# extends, so the server prefills only the difference. Never on a node's own
# thread -- rendering a prompt and the POST stay off the critical path.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swe-seed")

_lock = threading.Lock()
# (job, agent_id) -> digest of the last seed sent, so an unchanged prompt is
# not prefilled twice.
_seeded: dict[tuple[str, str], str] = {}
_prefix_templates: dict[tuple[int, str], ChatPromptTemplate] = {}
_wire_model: Any = None

_log_lock = threading.Lock()


def enabled() -> bool:
    """True when a seed should be built and sent.

    Read per call, not at import: the harnesses change the prefetch env after
    the agent is imported (``--no-prefetch``, ``KV_EVICTION_DISABLE_PREFETCH``).
    """
    return (
        _env("PROMPT_SEEDS", "1") == "1"
        and vllm_agent_enabled()
        and bool(vllm_agent_base_url())
    )


def _env(name: str, default: str = "") -> str:
    """``SWE_<name>`` if set, else the ``ODR_<name>`` spelling.

    The fallback is what pitlane sets (``BENCH_PROMPT_SEEDS`` becomes
    ``ODR_PROMPT_SEEDS``; the ``ours`` arm points ``ODR_PROMPT_SEEDS_LOG_PATH``
    into its cell), the same convention as ``trace_store``.
    """
    value = os.getenv(f"SWE_{name}")
    if value is None or value == "":
        value = os.getenv(f"ODR_{name}")
    return default if value is None or value == "" else value


def _skipped(target: str) -> bool:
    skip = _env("PROMPT_SEEDS_SKIP")
    return target in {name.strip() for name in skip.split(",") if name.strip()}


# -- identity --------------------------------------------------------------
def _target_identity(
    config: RunnableConfig | None, target: str
) -> tuple[str | None, str | None]:
    """``(job_id, agent_id)`` the target's own request will carry.

    Built from the *producer's* namespace with the leaf swapped, because the
    target runs in the same subgraph: its request derives
    ``swe_architect:extract_implementation_plan`` from its own namespace
    exactly as this derives it from ``swe_architect:conduct_research``. Same
    path, same job, same unit -- the id ``inject_langgraph_request_metadata``
    will send, so the seed lands in the bucket the real request reads.
    """
    if not isinstance(config, dict):
        return None, None
    configurable = config.get("configurable") or {}
    metadata = config.get("metadata") or {}
    job_id = configurable.get("job_id")
    job_id = job_id if isinstance(job_id, str) and job_id else None
    path = graph_path_from_checkpoint_ns(
        metadata.get("langgraph_checkpoint_ns"), metadata.get("langgraph_node")
    )
    if not path:
        return job_id, None
    parent = path.rsplit(":", 1)[0] if ":" in path else ""
    target_path = f"{parent}:{target}" if parent else target
    agent_id = derive_agent_id(
        target_path, job_id=job_id, unit=metadata.get(AGENT_UNIT_METADATA_KEY)
    )
    return job_id, agent_id


# -- prompt construction ---------------------------------------------------
def _prefix_template(prompt: ChatPromptTemplate, history_key: str) -> ChatPromptTemplate:
    """``prompt`` cut right after the placeholder named ``history_key``.

    What follows the conversation is the node's closing instruction; leaving
    it in would make the seed a prefix of nothing once the conversation grows.
    """
    key = (id(prompt), history_key)
    with _lock:
        cached = _prefix_templates.get(key)
    if cached is not None:
        return cached
    cut = next(
        index
        for index, message in enumerate(prompt.messages)
        if isinstance(message, MessagesPlaceholder)
        and message.variable_name == history_key
    )
    template = ChatPromptTemplate.from_messages(prompt.messages[: cut + 1])
    with _lock:
        _prefix_templates[key] = template
    return template


def _to_wire(
    messages: Sequence[Any], tools: Sequence[Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Messages and tools as ChatOpenAI puts them on the wire.

    Uses ChatOpenAI's own payload builder, so a tool-call-only turn carries
    ``content: null`` and the tools are formatted the way ``bind_tools`` sends
    them -- a mismatch in either renders different tokens. Tools matter: the
    chat template renders them into the system block, so a seed of a
    tool-bound node without them matches nothing. The model is never called.
    """
    global _wire_model
    from langchain_openai import ChatOpenAI

    from agent.common.model import MODEL_NAME

    if _wire_model is None:
        _wire_model = ChatOpenAI(
            model=MODEL_NAME, api_key="prompt-seed", base_url="http://prompt-seed.invalid"
        )
    kwargs = _wire_model.bind_tools(list(tools)).kwargs if tools else {}
    payload = _wire_model._get_request_payload(list(messages), **kwargs)
    return payload["messages"], payload.get("tools")


# -- logging ---------------------------------------------------------------
def _log_path(job_id: str | None) -> Path | None:
    configured = _env("PROMPT_SEEDS_LOG_PATH")
    if configured:
        return Path(configured).expanduser()
    directory = os.getenv("LANGGRAPH_PREDICTION_LOG_DIR")
    if directory and job_id:
        return Path(directory).expanduser() / f"job_{job_id}.prompt_seeds.jsonl"
    return None


def _emit(job_id: str | None, record: dict[str, Any]) -> None:
    path = _log_path(job_id)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except Exception:  # noqa: BLE001
        logger.warning("Prompt seeds: log write failed", exc_info=True)


# -- sending ---------------------------------------------------------------
def _post(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to ``/prefetch``.

    Not ``warm_agent_prefixes``: this fork branch's version takes no
    ``tools``, and two of the targets are tool-bound. The payload is otherwise
    the one it builds. A server without the ``tools`` field ignores it, and
    the seed then records a prefix the real request cannot hit -- the log row
    is where that shows.
    """
    try:
        timeout = float(os.getenv("LANGGRAPH_VLLM_AGENT_TIMEOUT_SECONDS") or 30.0)
    except ValueError:
        timeout = 30.0
    req = request.Request(
        url=f"{base_url}/prefetch",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"/prefetch failed with HTTP {exc.code}: {detail}") from exc
    return json.loads(body) if body else {}


def _build_and_send(
    *,
    event: str,
    producer_node: str | None,
    target: str,
    job_id: str | None,
    agent_id: str,
    prompt: ChatPromptTemplate,
    inputs: dict[str, Any],
    history_key: str,
    tools: Sequence[Any] | None,
    issued_at_ns: int,
) -> None:
    """Render, dedupe, POST and log one seed. Runs on the executor; never raises."""
    try:
        base_url = vllm_agent_base_url()
        if not base_url:
            return
        resolved = {key: value() if callable(value) else value for key, value in inputs.items()}
        rendered = _prefix_template(prompt, history_key).invoke(resolved).to_messages()
        messages, wire_tools = _to_wire(rendered, tools)
        digest = hashlib.sha256(
            json.dumps([messages, wire_tools], sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        with _lock:
            key = (job_id or "_", agent_id)
            if _seeded.get(key) == digest:
                return
            _seeded[key] = digest
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            # One prompt per target, superseded rather than accumulated.
            "agent_kind": "non-react",
            "wait": False,
            "prefetch_top_k": 1,
            "langgraph_node": target,
            "call_type": derive_call_type(target),
            "messages": messages,
            # Explicit: the miss is certain by construction, so a server whose
            # default flips must not silently turn the seed into a no-op.
            "prefill_on_miss": True,
        }
        if job_id:
            payload["job_id"] = job_id
        if wire_tools:
            payload["tools"] = wire_tools
    except Exception:  # noqa: BLE001 - a seed must never fail the run
        logger.warning("Prompt seed %s: build failed for %s", event, target, exc_info=True)
        return

    started = time.perf_counter_ns()
    response: dict[str, Any] = {}
    failure: str | None = None
    try:
        response = _post(base_url, payload)
    except Exception as exc:  # noqa: BLE001
        failure = str(exc)[:500]
        logger.warning("Prompt seed %s failed for %s: %s", event, target, exc)
    _emit(job_id, {
        "event": event,
        "job_id": job_id,
        "producer_node": producer_node,
        "predicted_node": target,
        "agent_id": agent_id,
        "seed_messages": len(messages),
        "seed_chars": sum(len(str(m.get("content") or "")) for m in messages),
        "seed_tools": len(wire_tools or []),
        # None means a server that ignored `messages`: it warmed whatever the
        # registry already held instead, and this row is the only sign.
        "seeded_from_messages": response.get("seeded_from_messages"),
        "prefill_on_miss": response.get("prefill_on_miss"),
        "issued_at_ns": issued_at_ns,
        "post_elapsed_ns": time.perf_counter_ns() - started,
        "scored": response.get("scored"),
        "submitted": response.get("submitted"),
        "error": failure,
    })


def seed(
    config: RunnableConfig | None,
    *,
    target: str,
    prompt: ChatPromptTemplate,
    inputs: dict[str, Any],
    history_key: str,
    tools: Sequence[Any] | None = None,
    event: str,
) -> None:
    """Seed ``target``'s prompt, built from ``prompt`` and ``inputs``.

    ``config`` is the *producer* node's config; ``target`` a bare node in the
    same subgraph. ``inputs`` are the target's template variables as the
    target will pass them, with the conversation under ``history_key`` as far
    as it is known now. Variables after that placeholder may be omitted. A
    callable value is called on the worker -- for an input that is expensive
    to build (``get_files_structure`` walks the whole repo) and that the
    producer cannot hoist out of its own ``invoke({...})`` without changing
    what the fork's compile-time analyser reads there.
    ``tools`` is the target's ``bind_tools`` list, when it has one.

    Returns at once; the work runs on a background thread. Never raises.
    """
    try:
        if not enabled() or _skipped(target):
            return
        job_id, agent_id = _target_identity(config, target)
        if not agent_id:
            return
        metadata = (config or {}).get("metadata") or {}
        # Snapshot now: the conversation list must be the one this node saw,
        # not whatever the reducer has made of it by the time the worker runs.
        snapshot = {
            key: list(value) if isinstance(value, list) else value
            for key, value in inputs.items()
        }
        _EXECUTOR.submit(
            _build_and_send,
            event=event,
            producer_node=metadata.get("langgraph_node"),
            target=target,
            job_id=job_id,
            agent_id=agent_id,
            prompt=prompt,
            inputs=snapshot,
            history_key=history_key,
            tools=tools,
            issued_at_ns=time.perf_counter_ns(),
        )
    except Exception:  # noqa: BLE001 - a seed must never fail the run
        logger.warning("Prompt seed %s: not issued for %s", event, target, exc_info=True)


def drain(timeout: float | None = None) -> None:
    """Block until every seed issued so far has been sent. For tests."""
    _EXECUTOR.submit(lambda: None).result(timeout=timeout)


def reset() -> None:
    """Drop per-job dedupe state. For tests and between jobs in one process."""
    with _lock:
        _seeded.clear()
