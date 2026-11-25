"""Keep each question on one vLLM data-parallel engine.

Under vLLM data parallelism the server places every request by itself. For a
dense model its balancer does not see the engines' load, and an arm's warms
count there like calls, so where a question's calls land depends on the arm.
With ``SWE_DP_PIN_ENGINES=<n>`` the agent places questions instead:

* when a question starts, it gets the engine with the fewest questions
  running on it (the lowest index on a tie). The batch starts a question only
  when one ends, so the engine that just lost a question takes the next one
  and the questions stay evenly split until the last few;
* every chat call of the question carries ``X-data-parallel-rank`` (vLLM's
  own header for this), so its prefix stays in one engine's GPU cache;
* the engine is named to the server before the question's first request
  (``POST /v1/agents/dp_rank``), so the question's warms go there too
  (the server needs ``VLLM_AGENT_DP_PIN=1``).

Prompt seeds carry no rank: the server holds each until an engine has no
agent call in flight and runs it there, and drops it if its target call
arrives first (``prompt_seeds``). The placement of questions is the same rule
in every arm and reacts only to questions starting and ending, never to warms
or seeds.

Environment
-----------
``SWE_DP_PIN_ENGINES``   number of engines; unset or below 2 turns this off
``SWE_DP_PIN_LOG_PATH``  JSONL of every assignment and release (optional)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any
from urllib import error, parse, request

logger = logging.getLogger(__name__)

HEADER = "X-data-parallel-rank"

_lock = threading.Lock()
_rank_of: dict[str, int] = {}


def engines() -> int:
    try:
        return int(os.getenv("SWE_DP_PIN_ENGINES") or 0)
    except ValueError:
        return 0


def enabled() -> bool:
    return engines() >= 2


def rank_for(job_id: str | None) -> int | None:
    if job_id is None:
        return None
    with _lock:
        return _rank_of.get(str(job_id))


def headers_for(job_id: str | None) -> dict[str, str]:
    """The header a chat call of ``job_id`` carries, or {} when unpinned."""
    rank = rank_for(job_id)
    return {} if rank is None else {HEADER: str(rank)}


def _server_url() -> str:
    base = (os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
    return f"{base}/agents/dp_rank" if base else ""


def _tell_server(job_id: str, rank: int) -> dict[str, Any]:
    url = _server_url()
    if not url:
        return {"error": "OPENAI_BASE_URL unset"}
    query = parse.urlencode({"job_id": job_id, "rank": rank})
    req = request.Request(f"{url}?{query}", data=b"", method="POST")
    try:
        with request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:200]}"}
    except Exception as exc:  # noqa: BLE001 - placement must never fail a question
        return {"error": str(exc)[:200]}


def _record(row: dict[str, Any]) -> None:
    path = os.getenv("SWE_DP_PIN_LOG_PATH")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError:
        pass


def assign(job_id: str) -> int | None:
    """Give ``job_id`` the engine with the fewest running questions, tell the
    server, and return the rank (None when pinning is off)."""
    n = engines()
    if n < 2:
        return None
    job_id = str(job_id)
    with _lock:
        running = [0] * n
        for rank in _rank_of.values():
            running[rank] += 1
        rank = running.index(min(running))
        _rank_of[job_id] = rank
    answer = _tell_server(job_id, rank)
    if answer.get("error") or not answer.get("enabled", False):
        logger.warning("dp pin: job %s -> engine %d, but the server said %s", job_id, rank, answer)
    _record({"event": "assign", "job_id": job_id, "rank": rank, "running_before": running,
             "at_unix": time.time(), "server": answer})
    return rank


def release(job_id: str) -> None:
    """The question ended: free its engine for the next one."""
    if not enabled():
        return
    job_id = str(job_id)
    with _lock:
        rank = _rank_of.pop(job_id, None)
    if rank is None:
        return
    answer = _tell_server(job_id, -1)
    _record({"event": "release", "job_id": job_id, "rank": rank, "at_unix": time.time(), "server": answer})
