"""Empty LMCache's L1 before a question, once everything in it is on L2.

A question's population phase leaves the prefixes it seeded in LMCache's L1
(CPU memory), and, since LMCache writes L2 through, on their way to the L2 disk
tier. Which of them the question then finds in L1 is whatever L1's LRU happened
to keep: in run swe_lite10_q1_20261102_010542 the floor found extract's whole
prompt in L1 and loaded it in 0.3 s, where an earlier run had to go to disk.
With this on, every question starts from the same place in every arm -- GPU
prefix cache flushed (`reset_kv_metrics`), L1 empty, every seed on L2 -- so a
question's first calls read their prefixes from disk, on demand or prefetched.

LMCache's multiprocess server serves an HTTP frontend (port 8080 by default):

* ``GET /status`` reports the L2 store queue as
  ``storage_manager.store_controller.pending_keys_count`` and
  ``in_flight_task_count``. Both 0: everything L1 holds is on L2 too.
* ``POST /clear-cache`` empties L1. It drops blocks something still holds as
  well, which is why this waits for the queue first and is only called between
  questions, when nothing is in flight.

On when ``LMCACHE_L1_FLUSH_URL`` names that frontend (``http://127.0.0.1:8080``).
Never raises: a flush that fails is returned, printed and recorded, and the
question runs anyway.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

URL_ENV_VAR = "LMCACHE_L1_FLUSH_URL"
# Writing a question's seeds (~5 x 9 GB) to a slow disk takes minutes.
DEFAULT_DRAIN_TIMEOUT_S = 900.0
DEFAULT_POLL_S = 0.5
REQUEST_TIMEOUT_S = 30.0


def resolve_url() -> str | None:
    """LMCache's HTTP frontend, or None when the flush is off."""
    url = os.getenv(URL_ENV_VAR, "").strip().rstrip("/")
    return url or None


def _store_queue(status: dict[str, Any]) -> tuple[int, int]:
    store = status["storage_manager"]["store_controller"]
    return int(store["pending_keys_count"]), int(store["in_flight_task_count"])


def _l1_objects(status: dict[str, Any]) -> int | None:
    try:
        return int(status["storage_manager"]["l1_manager"]["total_object_count"])
    except (KeyError, TypeError, ValueError):
        return None


async def flush_lmcache_l1(
    label: str,
    *,
    url: str | None = None,
    drain_timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
    record_to: Path | None = None,
) -> dict[str, Any] | None:
    """Wait for LMCache's L2 writes to finish, then clear L1. None when off.

    A queue that does not drain within ``drain_timeout_s`` is a failure and L1
    is left alone: clearing it then would drop blocks that never reached L2.
    """
    url = url or resolve_url()
    if url is None:
        return None
    # Imported here so a run with the flush off does not require httpx.
    import httpx

    started = time.monotonic()
    result: dict[str, Any] = {"label": label, "url": url, "ok": False}
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            while True:
                status = (await client.get(f"{url}/status")).json()
                pending, in_flight = _store_queue(status)
                if pending == 0 and in_flight == 0:
                    break
                if time.monotonic() - started > drain_timeout_s:
                    result.update(
                        reason="drain_timeout",
                        detail=(f"{pending} keys pending and {in_flight} store "
                                f"tasks in flight after {drain_timeout_s:.0f} s"),
                    )
                    break
                await asyncio.sleep(poll_s)
            result["drain_s"] = round(time.monotonic() - started, 3)
            if "reason" not in result:
                result["l1_objects_before"] = _l1_objects(status)
                response = await client.post(f"{url}/clear-cache")
                body = response.json()
                if response.status_code == 200 and body.get("status") == "ok":
                    after = (await client.get(f"{url}/status")).json()
                    result.update(ok=True, l1_objects_after=_l1_objects(after))
                else:
                    result.update(reason="clear_failed", status_code=response.status_code,
                                  body=body)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        result.update(reason="request_failed", detail=f"{type(exc).__name__}: {exc}")
    result["seconds"] = round(time.monotonic() - started, 3)

    if record_to is not None:
        record_to.parent.mkdir(parents=True, exist_ok=True)
        with record_to.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")

    if result["ok"]:
        print(
            f"LMCache L1 flush ({label}): L2 writes drained in {result['drain_s']:.1f} s, "
            f"L1 cleared ({result.get('l1_objects_before')} -> "
            f"{result.get('l1_objects_after')} objects)"
        )
    else:
        print(
            f"LMCache L1 flush ({label}) FAILED ({result.get('reason')}: "
            f"{result.get('detail', result.get('body', ''))}) -- the question starts "
            "with whatever L1 holds"
        )
    return result
