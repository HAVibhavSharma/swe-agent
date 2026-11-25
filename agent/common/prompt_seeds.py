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
``SWE_PROMPT_SEEDS_PREFILL_URL``  base URL of a second vLLM engine, e.g.
                               ``http://127.0.0.1:8200``. Each seed is sent
                               there as a one-token chat completion instead of
                               to ``/prefetch``: that engine computes it on its
                               own GPU and its LMCache connector stores the KV,
                               which the serving engine then loads from the
                               shared LMCache when the real request arrives.
                               The serving GPU does no seed work at all.
                               Under vLLM data parallelism it is the
                               serving port itself. With question pinning
                               (the server's ``VLLM_AGENT_DP_PIN``) a seed
                               waits there until an engine has no agent call
                               in flight and runs on it; a seed whose target
                               call arrives first, or whose question ends, is
                               dropped (HTTP 409) and logged as skipped
                               ``dropped: <reason>``. Each seed carries
                               ``seed_issued_at``, the time it was issued, so
                               the server can tell.
``SWE_PROMPT_SEEDS_PREFILL_TIMEOUT``  seconds to wait for one such prefill
                               (default 300)
``SWE_PROMPT_SEEDS_PRIORITY``  vLLM ``priority`` of each seed sent to the
                               prefill engine (unset or ``0``: not sent). For
                               seeds that share an engine with real calls,
                               which are sent at 0: under vLLM's priority
                               policy a seed then never queues ahead of a real
                               call, and is the first preempted. Needs the
                               server's ``--scheduling-policy priority``.
``SWE_PROMPT_SEEDS_LOOKUP_URL``  the LMCache server's ``/lookup``, e.g.
                               ``http://127.0.0.1:9000/lookup``. With it, a
                               seed bound for the prefill engine is first
                               tokenized there (``/tokenize``, no GPU work) and
                               looked up in LMCache (nothing is loaded); if
                               LMCache already holds all of it but fewer than
                               ``SWE_PROMPT_SEEDS_MIN_NEW_TOKENS`` (default
                               1024) tokens, it is not sent. Such a seed would
                               compute almost nothing and read its whole
                               prefix from disk to do it, beside the real
                               calls reading theirs. A failed probe sends the
                               seed as before.
``SWE_PROMPT_SEEDS_AS_WARMS``  ``1`` names each seed sent to the prefill
                               engine as a warm, request id
                               ``prefetch::<agent_id>::seed-<hex>``, which
                               reaches LMCache as ``chatcmpl-prefetch::...``.
                               LMCache then treats it as it treats vLLM's
                               phantoms: its loads queue behind real requests',
                               the blocks it reads are retained, and DemandLRU
                               tags them for the job and node it is for.

Each also reads its ``ODR_`` spelling when the ``SWE_`` one is unset, which is
what pitlane sets.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence
from urllib import error, request

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langsmith import tracing_context
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
# With a prefill engine each POST waits for the prefill itself (seconds for a
# long prompt), so one worker for every question would queue one question's
# seed behind another's. One worker per job keeps each question's seeds in
# order -- the extended seed after the one it extends -- and lets questions
# run side by side. The /prefetch path returns at once and keeps the one.
_JOB_EXECUTORS: dict[str, ThreadPoolExecutor] = {}

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
        and bool(vllm_agent_base_url() or prefill_url())
    )


def prefill_url() -> str:
    """The second engine seeds are computed on, or "" to send them to
    ``/prefetch`` on the serving engine as before."""
    return _env("PROMPT_SEEDS_PREFILL_URL").rstrip("/")


def lookup_url() -> str:
    """LMCache's ``/lookup``, or "" to send every seed without asking."""
    return _env("PROMPT_SEEDS_LOOKUP_URL").rstrip("/")


_DEFAULT_MIN_NEW_TOKENS = 1024


def min_new_tokens() -> int:
    """A seed LMCache already holds all but fewer than this many tokens of is
    not sent."""
    try:
        return max(0, int(_env("PROMPT_SEEDS_MIN_NEW_TOKENS") or _DEFAULT_MIN_NEW_TOKENS))
    except ValueError:
        return _DEFAULT_MIN_NEW_TOKENS


def seed_priority() -> int:
    """The vLLM priority seeds go to the prefill engine at; 0 sends none."""
    try:
        return int(_env("PROMPT_SEEDS_PRIORITY") or 0)
    except ValueError:
        return 0


def as_warms() -> bool:
    """Whether seeds go to the prefill engine named as warms."""
    return _env("PROMPT_SEEDS_AS_WARMS", "0") == "1"


def _seed_request_id(agent_id: str) -> str:
    """A warm's id for LMCache (``prefetch::<agent>::...``), unique per seed:
    the agent id carries the job and the target node, which DemandLRU reads."""
    return f"prefetch::{agent_id}::seed-{uuid.uuid4().hex[:16]}"


def _executor(job_id: str | None) -> ThreadPoolExecutor:
    if not prefill_url():
        return _EXECUTOR
    key = job_id or "_"
    with _lock:
        executor = _JOB_EXECUTORS.get(key)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"swe-seed-{key}")
            _JOB_EXECUTORS[key] = executor
        return executor


def _submit(job_id: str | None, fn: Any, /, **kwargs: Any) -> Any:
    """Run ``fn`` on the job's seed worker in a copy of the caller's context.

    A thread pool does not carry context variables over, and the job's
    checkout is one (``workspace.set_job_root``): a callable input such as the
    architect's repo tree, evaluated on the worker without it, walks the shared
    ``./workspace_repo`` instead of this question's checkout, so with questions
    at once the seed is a prefix of no request the question sends.
    """
    return _executor(job_id).submit(contextvars.copy_context().run, fn, **kwargs)


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

    Not ``warm_agent_prefixes``: it predates that function's ``tools`` on
    this branch, and a seed must carry them, since two of the targets are
    tool-bound. The payload is otherwise the one it builds. A server without the ``tools`` field ignores it, and
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


def _post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} failed with HTTP {exc.code}: {detail[:300]}") from exc
    return json.loads(raw) if raw else {}


def _probe(helper: str, lookup: str, messages: list[dict[str, Any]],
           tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """How many tokens the seed has, and how many of them LMCache holds.

    The prefill engine tokenizes it exactly as its chat completion would
    (same template, ``add_generation_prompt`` off, the same tools); LMCache
    hashes those tokens as a vLLM lookup would and reports the prefix it
    holds, loading nothing. ``{"tokens", "cached", "error"}``; on any
    failure ``error`` is set and the seed is sent anyway.
    """
    from agent.common.model import MODEL_NAME

    body: dict[str, Any] = {"model": MODEL_NAME, "messages": messages,
                            "add_generation_prompt": False}
    if tools:
        body["tools"] = tools
    try:
        tokens = _post_json(f"{helper}/tokenize", body, timeout=30.0).get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("/tokenize returned no tokens")
        found = _post_json(lookup, {"token_ids": tokens}, timeout=10.0)
        return {"tokens": len(tokens), "cached": int(found["cached_prefix_tokens"]),
                "error": None}
    except Exception as exc:  # noqa: BLE001 - a probe must never stop a seed
        return {"tokens": None, "cached": None, "error": str(exc)[:300]}


def _post_prefill(url: str, messages: list[dict[str, Any]],
                  tools: list[dict[str, Any]] | None,
                  request_id: str | None = None,
                  issued_at_unix: float | None = None) -> dict[str, Any]:
    """Compute ``messages`` on the prefill engine: a one-token chat completion.

    ``add_generation_prompt`` off, so the rendered prompt ends where the seed
    ends and stays a prefix of the real request, which continues with the
    node's closing messages and only then the assistant header. A plain
    request, not a phantom: it takes vLLM's ordinary prefill path, whose
    LMCache connector stores the computed blocks -- a phantom that misses
    LMCache is finished without prefilling on this fork, and would store
    nothing. The one generated token is thrown away. ``request_id`` (with
    ``SWE_PROMPT_SEEDS_AS_WARMS``) names it as a warm for LMCache, and
    ``issued_at_unix`` tells a pinning server which calls came after it.
    """
    from agent.common.model import MODEL_NAME

    try:
        timeout = float(_env("PROMPT_SEEDS_PREFILL_TIMEOUT") or 300.0)
    except ValueError:
        timeout = 300.0
    body: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "add_generation_prompt": False,
    }
    if tools:
        body["tools"] = tools
    if request_id:
        body["request_id"] = request_id
    if issued_at_unix is not None:
        body["seed_issued_at"] = issued_at_unix
    if seed_priority():
        body["priority"] = seed_priority()
    try:
        return _post_json(f"{url}/v1/chat/completions", body, timeout=timeout)
    except RuntimeError as exc:
        raise RuntimeError(f"prefill engine: {exc}") from exc


def _question_ended(job_id: str | None) -> bool:
    """Under question pinning, whether ``job_id``'s question has ended (it had
    an engine and no longer has). Always False without pinning."""
    from agent.common import dp_routing

    return dp_routing.enabled() and job_id is not None and dp_routing.ended(job_id)


def _finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices") or []
    return choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None


def _dropped_reason(message: str) -> str | None:
    """Why a pinning server dropped a seed (HTTP 409 ``seed dropped: ...``),
    or None for any other failure."""
    if "HTTP 409" not in message:
        return None
    at = message.find("seed dropped: ")
    if at < 0:
        return None
    return message[at + len("seed dropped: "):].split('"', 1)[0].strip() or "unknown"


def _build_and_send(**kwargs: Any) -> None:
    """``_send_seed`` with LangSmith tracing off.

    The worker thread does not inherit the node's run context, so with
    LANGSMITH_TRACING on every prompt rendered and repo tree built for a seed
    was traced as a root run of its own -- dozens of ChatPromptTemplate and
    get_files_structure rows per question, outside the job's trace. A seed is
    a side channel to the server, not part of the agent's work.
    """
    with tracing_context(enabled=False):
        _send_seed(**kwargs)


def _send_seed(
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
    issued_at_unix: float | None = None,
) -> None:
    """Render, dedupe, POST and log one seed. Runs on the executor; never raises."""
    try:
        helper = prefill_url()
        base_url = vllm_agent_base_url()
        if not (helper or base_url):
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
    skipped: str | None = None
    lookup = lookup_url() if helper else ""
    ended = _question_ended(job_id)
    probe: dict[str, Any] = (_probe(helper, lookup, messages, wire_tools) if lookup and not ended
                             else {"tokens": None, "cached": None, "error": None})
    request_id = _seed_request_id(agent_id) if helper and as_warms() else None
    if ended or _question_ended(job_id):
        # Pinned questions only: it ended while this seed was being built or
        # queued behind the question's earlier seeds. Its call never comes.
        skipped = "dropped: its question ended"
        request_id = None
    elif (probe["error"] is None and probe["tokens"] is not None
            and probe["tokens"] - probe["cached"] < min_new_tokens()):
        # Nearly all of it is in LMCache already: the seed would compute a few
        # hundred tokens and read the rest from disk to do so.
        skipped = "cached"
        request_id = None
    else:
        try:
            if helper:
                response = _post_prefill(helper, messages, wire_tools, request_id, issued_at_unix)
            else:
                response = _post(base_url, payload)
        except Exception as exc:  # noqa: BLE001
            dropped = _dropped_reason(str(exc))
            if dropped is not None:
                # The server held it for an idle GPU and gave up: not a failure.
                skipped = f"dropped: {dropped}"
            else:
                failure = str(exc)[:500]
                logger.warning("Prompt seed %s failed for %s: %s", event, target, exc)
    if helper and not skipped and _finish_reason(response) == "abort":
        # The server aborted it when its question was released.
        skipped = "dropped: aborted, its question ended"
    usage = (response.get("usage") or {}) if helper else {}
    details = usage.get("prompt_tokens_details") or {}
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
        # Wall clock, so the harness can line a seed up with the request it
        # predicted (vLLM's stats are in Unix time) and judge it.
        "issued_at_unix": issued_at_unix,
        "post_elapsed_ns": time.perf_counter_ns() - started,
        "scored": response.get("scored"),
        "submitted": response.get("submitted"),
        # On a prefill engine: the prompt it computed and how much of it it
        # already held (its GPU cache or LMCache), i.e. what was new work.
        "prefill_url": helper or None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        # SWE_PROMPT_SEEDS_LOOKUP_URL: the seed's tokens and the prefix
        # LMCache held before it was sent; "cached" in `skipped` means it was
        # not sent, as all but fewer than `min_new_tokens` were there.
        # "dropped: <reason>" means a pinning server held it for an idle GPU
        # and dropped it (`post_elapsed_ns` is then how long it waited).
        "lookup_tokens": probe["tokens"],
        "lookup_cached_tokens": probe["cached"],
        "lookup_error": probe["error"],
        "min_new_tokens": min_new_tokens() if lookup else None,
        "skipped": skipped,
        # SWE_PROMPT_SEEDS_AS_WARMS: the id it was sent under.
        "request_id": request_id,
        # SWE_PROMPT_SEEDS_PRIORITY: the vLLM priority it was sent at.
        "priority": seed_priority() if helper and skipped != "cached" else None,
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
        _submit(
            job_id,
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
            issued_at_unix=time.time(),
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
