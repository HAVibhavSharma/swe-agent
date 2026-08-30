"""Record / replay of LLM wire traffic so two runs execute an identical workload.

Ported from ``open_deep_research/trace_store.py``; same modes, same trace
format, so a trace recorded by either repo is readable by the other's analysis
scripts.

Motivation
----------
The A/B study compares a baseline (stock vLLM + stock LangGraph) against the
modified stack (custom vLLM + prefix prefetch). Both arms run the same agent
code, so a latency difference should come from the serving layer alone. It does
not: the agent is a feedback loop, so one differing sampled token changes the
tool call, which changes the file read, which changes every later prompt. The
arms then do different amounts of work and the numbers are not comparable.

This module pins the trajectory. A baseline run in ``record`` mode dumps every
chat-completion request/response pair to a JSONL trace. Both arms then run in
``pinned`` mode: the request still goes to the real server (so prefill, decode
and prefetch are genuinely exercised and timed), but the sampled output is
discarded and the *recorded* response is handed back to the agent.

Modes (``SWE_TRACE_MODE``, or ``ODR_TRACE_MODE``)
-------------------------------------------------
``off``      Default. No interception.
``record``   Live calls, every request/response appended to the trace.
``pinned``   Live call issued for timing (output capped at the recorded length),
             recorded response returned.
``offline``  No server contact; recorded response returned immediately. Cheap
             way to check that a code change keeps the trajectory intact.

Environment (``SWE_`` prefix preferred, ``ODR_`` accepted for parity with ODR)
-----------------------------------------------------------------------------
``SWE_TRACE_MODE``     one of the modes above
``SWE_TRACE_PATH``     trace JSONL (read in pinned/offline, written in record)
``SWE_TRACE_ON_MISS``  ``strict`` (raise) or ``live`` (fall back, default)
``SWE_TRACE_REPORT``   divergence report JSONL; defaults next to the trace

Interception is at the HTTP layer, not the LangChain call sites, because the
agent invokes the model from several places with different wrappers
(``with_structured_output``, ``bind_tools``, ``with_retry``). The wire is the
one place every call passes through, and it captures the exact bytes the server
sees -- which is what a serving-layer study cares about.

Unlike ODR's copy, the environment is read **lazily**, on the first model call,
not at import: the harnesses call ``load_dotenv(override=True)`` after importing
the graph, so anything read at import time would miss ``.env``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

VALID_MODES = {"off", "record", "pinned", "offline"}

# Request fields that identify the *work* being asked for. Everything else
# (job_id, thread_id, agent id, streaming plumbing) varies run to run without
# changing what the model is asked to do, so it must stay out of the key.
_KEYED_FIELDS = (
    "model",
    "messages",
    "tools",
    "tool_choice",
    "response_format",
    "temperature",
    "top_p",
    "seed",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "stop",
)


def _env(name: str, default: str = "") -> str:
    """``SWE_TRACE_*`` if set, else the ``ODR_TRACE_*`` spelling."""
    for prefix in ("SWE_", "ODR_"):
        value = os.getenv(prefix + name)
        if value is not None and value.strip():
            return value.strip()
    return default


@dataclass(frozen=True)
class TraceConfig:
    mode: str
    path: str
    on_miss: str
    report_path: Path | None
    # Where the side-by-side dumps go when a pinned live call stops before the
    # baseline's token count. One file per such request.
    early_exit_dir: Path | None


_config: TraceConfig | None = None
_config_lock = threading.Lock()

# How many chat completions the transport has actually seen. A run that ends
# with this at 0 while the mode says `record` did not route its model traffic
# through here at all -- which is the failure that otherwise looks like "the
# trace file was never created".
_intercepted = 0
_intercepted_lock = threading.Lock()


def intercepted_count() -> int:
    with _intercepted_lock:
        return _intercepted


def config() -> TraceConfig:
    """Resolve (once) and validate the trace configuration."""
    global _config
    with _config_lock:
        if _config is None:
            _config = _resolve()
        return _config


def reset_config() -> None:
    """Forget the resolved config; for tests that flip the env var."""
    global _config, _client, _sync_client
    with _config_lock:
        _config = None
    with _client_lock:
        _client = None
        _sync_client = None
    global _intercepted
    with _intercepted_lock:
        _intercepted = 0


def _resolve() -> TraceConfig:
    mode = (_env("TRACE_MODE", "off") or "off").lower()
    if mode not in VALID_MODES:
        raise ValueError(f"TRACE_MODE={mode!r} is not one of {sorted(VALID_MODES)}")

    path = _env("TRACE_PATH")
    on_miss = (_env("TRACE_ON_MISS", "live") or "live").lower()
    if mode == "off":
        return TraceConfig(mode, path, on_miss, None, None)

    if not path:
        raise ValueError(f"TRACE_MODE={mode} requires SWE_TRACE_PATH to be set")

    # Fail before the run rather than mid-stream. An exception raised inside the
    # transport is caught by the OpenAI client and re-raised as a generic
    # APIConnectionError, so a bad path shows up as a fake network failure
    # hundreds of lines into a benchmark.
    trace_path = Path(path)
    if trace_path.is_dir():
        raise ValueError(
            f"TRACE_PATH={path!r} is a directory; it must be a file, "
            f"e.g. {path.rstrip('/')}/trace.jsonl"
        )
    if mode == "record":
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        # Appending to an existing trace would interleave two recordings.
        if trace_path.exists() and trace_path.stat().st_size > 0:
            raise ValueError(
                f"TRACE_PATH={path!r} already exists and is non-empty; recording "
                "would append to it. Move it aside or pick a new name."
            )
    elif not trace_path.exists():
        raise FileNotFoundError(f"TRACE_MODE={mode} but trace file not found: {path}")

    report = _env("TRACE_REPORT")
    report_path = (
        Path(report)
        if report
        else trace_path.with_name(trace_path.stem + ".divergence.jsonl")
    )
    early_exit = _env("TRACE_EARLY_EXIT_DIR")
    early_exit_dir = (
        Path(early_exit)
        if early_exit
        else trace_path.with_name(trace_path.stem + ".early_exits")
    )
    return TraceConfig(mode, path, on_miss, report_path, early_exit_dir)


def enabled() -> bool:
    """True when this module should intercept model traffic."""
    return config().mode in {"record", "pinned", "offline"}


def _job_id(body: dict[str, Any]) -> str:
    """Best-effort per-run identity, so repeats are scoped to one benchmark job.

    ``inject_langgraph_request_metadata`` puts the job id in ``extra_body``,
    which the OpenAI client flattens into the top level of the request body.
    """
    for key in ("job_id", "langgraph_job_id"):
        value = body.get(key)
        if isinstance(value, (str, int)):
            return str(value)
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("job_id")
        if value is not None:
            return str(value)
    return "_"


def request_key(body: dict[str, Any]) -> str:
    """Stable content hash of a chat-completion request."""
    canonical = {field: body[field] for field in _KEYED_FIELDS if field in body}
    encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class TraceStore:
    """Reads a recorded trace; hands out responses keyed by request content.

    An agent legitimately issues byte-identical requests more than once (retries,
    two atomic tasks with the same brief). Those are disambiguated by a
    per-(job, key) occurrence counter rather than by a global sequence number,
    because global ordering is exactly what shifts between the two arms.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._entries: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._cursor: dict[tuple[str, str], int] = defaultdict(int)
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Trace file not found: {self.path}")
        count = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._entries[(record["job_id"], record["key"])].append(record)
                count += 1
        logger.info("Loaded %d trace entries from %s", count, self.path)

    def take(self, job_id: str, key: str) -> Optional[dict[str, Any]]:
        """Return the next unused recording for this request, if any.

        Falls back to a job-agnostic lookup: job ids come from a counter in
        ``run_evaluate.py`` and only line up across arms when both runs execute
        the same instances in the same order.
        """
        with self._lock:
            for lookup in ((job_id, key), ("_", key)):
                bucket = self._entries.get(lookup)
                if not bucket:
                    continue
                index = self._cursor[lookup]
                if index < len(bucket):
                    self._cursor[lookup] = index + 1
                    return bucket[index]
                # Exhausted: reuse the last recording rather than diverging.
                return bucket[-1]
            # Content hash missed the job scope entirely; try any job.
            for (recorded_job, recorded_key), bucket in self._entries.items():
                if recorded_key == key and bucket:
                    index = self._cursor[(recorded_job, recorded_key)]
                    self._cursor[(recorded_job, recorded_key)] = index + 1
                    return bucket[min(index, len(bucket) - 1)]
        return None


class _TraceTransportBase:
    """Mode logic shared by the sync and async transports.

    Two transports are needed, not one: this repo's graph nodes are plain
    ``def`` functions calling ``runnable.invoke(...)``, so LangGraph runs them
    in a worker thread and ChatOpenAI uses its **sync** client. An async-only
    transport sees none of that traffic and records an empty trace.
    """

    def __init__(self) -> None:
        cfg = config()
        self._mode = cfg.mode
        self._on_miss = cfg.on_miss
        self._trace_path = Path(cfg.path)
        self._report_path = cfg.report_path or self._trace_path.with_name(
            self._trace_path.stem + ".divergence.jsonl"
        )
        self._early_exit_dir = cfg.early_exit_dir or self._trace_path.with_name(
            self._trace_path.stem + ".early_exits"
        )
        self._store = TraceStore(cfg.path) if self._mode in {"pinned", "offline"} else None
        self._write_lock = threading.Lock()

    # -- helpers ---------------------------------------------------------
    def _append(self, path: Path, record: dict[str, Any]) -> None:
        with self._write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _replayed_response(
        self, request: httpx.Request, entry: dict[str, Any]
    ) -> httpx.Response:
        body = entry["response_body"].encode("utf-8")
        content_type = entry.get("content_type") or "application/json"
        return httpx.Response(
            entry.get("status_code", 200),
            headers={"content-type": content_type},
            content=body,
            request=request,
        )

    def _completion_tokens(self, body_text: str, is_stream: bool) -> int | None:
        """Pull completion_tokens out of a recorded response for length pinning."""
        try:
            if not is_stream:
                return json.loads(body_text)["usage"]["completion_tokens"]
            for line in reversed(body_text.splitlines()):
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    return usage["completion_tokens"]
        except Exception:  # noqa: BLE001 - pinning is best-effort
            return None
        return None

    def _response_error(self, body_text: str, is_stream: bool) -> str | None:
        """Detect a failed generation that still returned HTTP 200.

        vLLM commits the status line before decoding starts, so an abort during
        generation (grammar rejection, engine error, cancelled request) arrives
        as an error payload or a truncated stream under a 200.
        """
        try:
            if not is_stream:
                error = json.loads(body_text).get("error")
                return str(error)[:500] if error else None
            saw_done = False
            for line in body_text.splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    saw_done = True
                    continue
                error = json.loads(payload).get("error")
                if error:
                    return str(error)[:500]
            if not saw_done:
                return "stream ended without [DONE] (generation aborted)"
        except Exception as exc:  # noqa: BLE001 - diagnosis is best-effort
            return f"unparseable response body: {exc}"[:500]
        return None

    def _message_text(self, body_text: str, is_stream: bool) -> str | None:
        """The assistant text a response carries, for eyeballing two of them.

        Best-effort and deliberately lossy: it is the field a human compares,
        while the raw bodies alongside it are what survives when the shape is
        something this does not know about (tool calls, reasoning content).
        """
        try:
            if not is_stream:
                message = json.loads(body_text)["choices"][0].get("message") or {}
                return message.get("content")
            parts: list[str] = []
            for line in body_text.splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    continue
                delta = (json.loads(payload).get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    parts.append(piece)
            return "".join(parts)
        except Exception:  # noqa: BLE001 - the raw bodies are the fallback
            return None

    def _dump_early_exit(
        self,
        *,
        key: str,
        job_id: str,
        body: dict[str, Any],
        entry: dict[str, Any],
        is_stream: bool,
        live_text: str,
        live_status: int | None,
        live_completion_tokens: int | None,
        pinned_limit: int | None,
        elapsed_ns: int,
    ) -> Path:
        """Both outputs for one request, side by side, in one JSON file.

        Written only when the live call stopped short of the baseline's token
        count. The agent is handed the recording and the live output is thrown
        away, so without this the two are never comparable: an early exit is
        exactly the case where the live call decoded something *different*, and
        the divergence report can only say that it was shorter.
        """
        node = body.get("langgraph_node")
        name = f"{key[:12]}_job{job_id}"
        if isinstance(node, str) and node:
            name = f"{name}_{node}"
        path = self._early_exit_dir / f"{name}.json"
        # A key can legitimately repeat (retries, the same brief twice); keep
        # every occurrence rather than overwriting the first.
        with self._write_lock:
            self._early_exit_dir.mkdir(parents=True, exist_ok=True)
            suffix = 1
            while path.exists():
                suffix += 1
                path = self._early_exit_dir / f"{name}.{suffix}.json"
            path.write_text(
                json.dumps(
                    {
                        "key": key,
                        "job_id": job_id,
                        "langgraph_node": node,
                        "call_type": body.get("call_type"),
                        "model": body.get("model"),
                        "stream": is_stream,
                        "reason": "early_exit",
                        "pinned_max_tokens": pinned_limit,
                        "live_completion_tokens": live_completion_tokens,
                        "live_status": live_status,
                        "live_latency_ns": elapsed_ns,
                        "baseline_latency_ns": entry.get("record_latency_ns"),
                        # What the agent was given, and what the live call
                        # produced instead. Text first because that is what a
                        # human reads; the raw bodies keep whatever the text
                        # extraction dropped.
                        "recorded_text": self._message_text(
                            entry["response_body"], bool(entry.get("stream", is_stream))
                        ),
                        "live_text": self._message_text(live_text, is_stream),
                        "recorded_response_body": entry["response_body"],
                        "live_response_body": live_text,
                        "request_body": body,
                    },
                    ensure_ascii=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return path

    # -- shared steps ----------------------------------------------------
    def _parse(self, raw: bytes) -> dict[str, Any] | None:
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return None

    def _note_interception(self, job_id: str) -> None:
        global _intercepted
        with _intercepted_lock:
            _intercepted += 1
            first = _intercepted == 1
        if first:
            # One line, on stdout, the first time traffic actually arrives: the
            # difference between "recording" and "wired up but never called" is
            # otherwise invisible until the run ends with an empty directory.
            print(
                f"[trace] {self._mode}: first chat completion intercepted "
                f"(job={job_id}) -> {self._trace_path}",
                flush=True,
            )

    def _write_record(
        self,
        *,
        key: str,
        job_id: str,
        body: dict[str, Any],
        is_stream: bool,
        status_code: int,
        content_type: str | None,
        text: str,
        elapsed_ns: int,
    ) -> None:
        self._append(
            self._trace_path,
            {
                "key": key,
                "job_id": job_id,
                "stream": is_stream,
                "status_code": status_code,
                "content_type": content_type,
                "request_body": body,
                "response_body": text,
                "record_latency_ns": elapsed_ns,
            },
        )

    def _miss(self, *, key: str, job_id: str, body: dict[str, Any]) -> None:
        """Record the divergence, and stop the run when asked to."""
        # The agent asked something the baseline never asked: the trajectory has
        # diverged, and from here the two arms are no longer comparable.
        self._append(
            self._report_path,
            {
                "key": key,
                "job_id": job_id,
                "reason": "miss",
                "mode": self._mode,
                "request_body": body,
            },
        )
        if self._on_miss == "strict":
            raise RuntimeError(
                f"Trace miss for job={job_id} key={key[:12]}; trajectory diverged "
                "from the recorded baseline (SWE_TRACE_ON_MISS=strict)."
            )
        logger.warning(
            "Trace miss (job=%s key=%s); falling back to live call", job_id, key[:12]
        )

    def _pinned_request(
        self,
        request: httpx.Request,
        body: dict[str, Any],
        entry: dict[str, Any],
        is_stream: bool,
    ) -> tuple[httpx.Request, int | None]:
        """The live call to issue for timing: same request, baseline's length."""
        pinned_body = dict(body)
        completion_tokens = self._completion_tokens(
            entry["response_body"], entry.get("stream", is_stream)
        )
        if completion_tokens:
            # Pin whichever length field the client actually sent: vLLM honours
            # max_completion_tokens over the deprecated max_tokens, so setting
            # the wrong one leaves the request uncapped.
            if "max_completion_tokens" in pinned_body:
                pinned_body["max_completion_tokens"] = completion_tokens
            else:
                pinned_body["max_tokens"] = completion_tokens
            # NOTE: do not set ignore_eos here. Under guided decoding (which
            # every structured-output / tool-calling request uses) the grammar
            # accepts only EOS once the JSON is closed. Suppressing EOS makes the
            # sampler emit the next-best token, the scheduler rejects it and
            # aborts the request mid-stream -- after the 200 status line is
            # already on the wire. Capping is enough: max_tokens bounds the live
            # call above, so decode cost can only come in at or under baseline.
        raw = json.dumps(pinned_body).encode("utf-8")
        headers = [
            (name, value)
            for name, value in request.headers.raw
            if name.lower() != b"content-length"
        ]
        headers.append((b"content-length", str(len(raw)).encode()))
        live_request = httpx.Request(
            request.method, request.url, headers=headers, content=raw
        )
        limit = pinned_body.get("max_completion_tokens", pinned_body.get("max_tokens"))
        return live_request, limit

    def _report_pinned(
        self,
        *,
        key: str,
        job_id: str,
        body: dict[str, Any],
        entry: dict[str, Any],
        is_stream: bool,
        live_text: str | None,
        live_status: int | None,
        live_error: str | None,
        elapsed_ns: int,
        pinned_limit: int | None,
    ) -> None:
        live_completion_tokens: int | None = None
        if live_text is not None:
            # The status line is sent before generation starts, so a stream that
            # dies mid-flight still reports 200. Recover the real outcome from
            # the body instead of trusting the status code.
            live_completion_tokens = self._completion_tokens(live_text, is_stream)
            live_error = live_error or self._response_error(live_text, is_stream)
            # A rejected request never decodes, so it costs almost nothing and
            # the recording still satisfies the agent -- the run looks healthy
            # and fast while the timing arm measured nothing. vLLM's flat error
            # shape ({"object": "error", ...}) has no "error" key, so the body
            # check above cannot catch it on its own.
            if (
                live_error is None
                and live_status is not None
                and not 200 <= live_status < 300
            ):
                live_error = f"HTTP {live_status}: {live_text[:400]}"

        # "Early exit": the live call hit EOS before the cap, so it decoded
        # fewer tokens than the baseline did -- the one case where the discarded
        # live output is known to differ from the recording the agent got.
        early_exit_path: Path | None = None
        if (
            live_text is not None
            and live_completion_tokens is not None
            and pinned_limit is not None
            and live_completion_tokens < pinned_limit
        ):
            try:
                early_exit_path = self._dump_early_exit(
                    key=key,
                    job_id=job_id,
                    body=body,
                    entry=entry,
                    is_stream=is_stream,
                    live_text=live_text,
                    live_status=live_status,
                    live_completion_tokens=live_completion_tokens,
                    pinned_limit=pinned_limit,
                    elapsed_ns=elapsed_ns,
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics, never fatal
                logger.warning("Could not write early-exit dump: %s", exc)

        if live_error:
            logger.warning(
                "Pinned live call reported an error (job=%s key=%s status=%s): %s",
                job_id,
                key[:12],
                live_status,
                live_error,
            )

        self._append(
            self._report_path,
            {
                "key": key,
                "job_id": job_id,
                "reason": "pinned",
                "live_status": live_status,
                "live_latency_ns": elapsed_ns,
                "pinned_max_tokens": pinned_limit,
                # Fidelity of the timing call: equal to pinned_max_tokens means
                # the live run decoded exactly as many tokens as the baseline.
                # Fewer means it hit EOS early; null usually means the stream was
                # aborted before it carried usage.
                "live_completion_tokens": live_completion_tokens,
                "live_error": live_error,
                "baseline_latency_ns": entry.get("record_latency_ns"),
                # Set when the live call stopped short: the file holding both
                # outputs for this request.
                "early_exit_json": str(early_exit_path) if early_exit_path else None,
            },
        )


class AsyncTraceTransport(_TraceTransportBase, httpx.AsyncBaseTransport):
    """The async half: used by ChatOpenAI's ``ainvoke`` / ``astream`` path."""

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        super().__init__()
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Only chat completions carry the workload; leave models/health alone.
        if not request.url.path.endswith("/chat/completions"):
            return await self._inner.handle_async_request(request)

        body = self._parse(await request.aread())
        if body is None:
            return await self._inner.handle_async_request(request)

        key = request_key(body)
        job_id = _job_id(body)
        is_stream = bool(body.get("stream"))
        self._note_interception(job_id)

        if self._mode == "record":
            started = time.perf_counter_ns()
            response = await self._inner.handle_async_request(request)
            text = (await response.aread()).decode("utf-8", errors="replace")
            await response.aclose()
            elapsed = time.perf_counter_ns() - started
            content_type = response.headers.get("content-type")
            self._write_record(
                key=key,
                job_id=job_id,
                body=body,
                is_stream=is_stream,
                status_code=response.status_code,
                content_type=content_type,
                text=text,
                elapsed_ns=elapsed,
            )
            return httpx.Response(
                response.status_code,
                headers={"content-type": content_type or "application/json"},
                content=text.encode("utf-8"),
                request=request,
            )

        entry = self._store.take(job_id, key) if self._store else None
        if entry is None:
            self._miss(key=key, job_id=job_id, body=body)
            return await self._inner.handle_async_request(request)

        if self._mode == "offline":
            return self._replayed_response(request, entry)

        # pinned: issue the real call so the serving stack is exercised and
        # timed, cap it at the baseline's output length, then discard what it
        # produced in favour of the recording.
        live_request, pinned_limit = self._pinned_request(request, body, entry, is_stream)
        started = time.perf_counter_ns()
        live_text: str | None = None
        live_status: int | None = None
        live_error: str | None = None
        try:
            live_response = await self._inner.handle_async_request(live_request)
            live_text = (await live_response.aread()).decode("utf-8", errors="replace")
            await live_response.aclose()
            live_status = live_response.status_code
        except Exception as exc:  # noqa: BLE001
            # A failed timing call must not change what the agent sees.
            logger.warning("Pinned live call failed (%s); serving recording only", exc)
            live_error = str(exc)
        elapsed = time.perf_counter_ns() - started

        self._report_pinned(
            key=key,
            job_id=job_id,
            body=body,
            entry=entry,
            is_stream=is_stream,
            live_text=live_text,
            live_status=live_status,
            live_error=live_error,
            elapsed_ns=elapsed,
            pinned_limit=pinned_limit,
        )
        return self._replayed_response(request, entry)


class SyncTraceTransport(_TraceTransportBase, httpx.BaseTransport):
    """The sync half: what this repo's ``runnable.invoke(...)`` nodes actually use."""

    def __init__(self, inner: httpx.BaseTransport | None = None) -> None:
        super().__init__()
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/chat/completions"):
            return self._inner.handle_request(request)

        body = self._parse(request.read())
        if body is None:
            return self._inner.handle_request(request)

        key = request_key(body)
        job_id = _job_id(body)
        is_stream = bool(body.get("stream"))
        self._note_interception(job_id)

        if self._mode == "record":
            started = time.perf_counter_ns()
            response = self._inner.handle_request(request)
            text = response.read().decode("utf-8", errors="replace")
            response.close()
            elapsed = time.perf_counter_ns() - started
            content_type = response.headers.get("content-type")
            self._write_record(
                key=key,
                job_id=job_id,
                body=body,
                is_stream=is_stream,
                status_code=response.status_code,
                content_type=content_type,
                text=text,
                elapsed_ns=elapsed,
            )
            return httpx.Response(
                response.status_code,
                headers={"content-type": content_type or "application/json"},
                content=text.encode("utf-8"),
                request=request,
            )

        entry = self._store.take(job_id, key) if self._store else None
        if entry is None:
            self._miss(key=key, job_id=job_id, body=body)
            return self._inner.handle_request(request)

        if self._mode == "offline":
            return self._replayed_response(request, entry)

        live_request, pinned_limit = self._pinned_request(request, body, entry, is_stream)
        started = time.perf_counter_ns()
        live_text: str | None = None
        live_status: int | None = None
        live_error: str | None = None
        try:
            live_response = self._inner.handle_request(live_request)
            live_text = live_response.read().decode("utf-8", errors="replace")
            live_response.close()
            live_status = live_response.status_code
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pinned live call failed (%s); serving recording only", exc)
            live_error = str(exc)
        elapsed = time.perf_counter_ns() - started

        self._report_pinned(
            key=key,
            job_id=job_id,
            body=body,
            entry=entry,
            is_stream=is_stream,
            live_text=live_text,
            live_status=live_status,
            live_error=live_error,
            elapsed_ns=elapsed,
            pinned_limit=pinned_limit,
        )
        return self._replayed_response(request, entry)


# The async transport under its original name.
TraceTransport = AsyncTraceTransport


_client: httpx.AsyncClient | None = None
# The one that carries this repo's traffic: every node is a plain `def` calling
# `runnable.invoke(...)`, so LangGraph runs it in a worker thread and ChatOpenAI
# uses its *sync* client. Wiring only the async half records nothing.
_sync_client: httpx.Client | None = None
_client_lock = threading.Lock()
_TIMEOUT = httpx.Timeout(600.0, connect=30.0)


def get_http_client() -> httpx.AsyncClient | None:
    """Shared httpx client wired to the trace transport, or None when off."""
    global _client
    if not enabled():
        return None
    with _client_lock:
        if _client is None:
            _client = httpx.AsyncClient(
                transport=AsyncTraceTransport(), timeout=_TIMEOUT
            )
            print(f"[trace] async transport attached ({describe()})", flush=True)
        return _client


def get_sync_http_client() -> httpx.Client | None:
    """Shared sync httpx client wired to the trace transport, None when off."""
    global _sync_client
    if not enabled():
        return None
    with _client_lock:
        if _sync_client is None:
            _sync_client = httpx.Client(transport=SyncTraceTransport(), timeout=_TIMEOUT)
            print(f"[trace] sync transport attached ({describe()})", flush=True)
        return _sync_client


def describe() -> str:
    """One-line summary for a harness banner."""
    cfg = config()
    if cfg.mode == "off":
        return "Trace mode: off (live calls, nothing recorded)"
    return (
        f"Trace mode: {cfg.mode} (path={cfg.path}, on_miss={cfg.on_miss}, "
        f"report={cfg.report_path})"
    )


def summary() -> dict[str, Any]:
    """What the harness prints at the end of a run."""
    cfg = config()
    written = 0
    path = Path(cfg.path) if cfg.path else None
    if cfg.mode == "record" and path and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            written = sum(1 for line in handle if line.strip())
    return {
        "trace_mode": cfg.mode,
        "trace_path": cfg.path or None,
        "trace_chat_completions_intercepted": intercepted_count(),
        "trace_records_written": written,
    }
