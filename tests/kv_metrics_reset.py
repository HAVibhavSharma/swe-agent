"""Tell the vLLM server when the cold phase is over.

The benchmark has to fill the caches before it measures anything — an arm
scored against an empty cache is scoring the cold start, not the policy. But
the cold phase then sits inside every number the server reports, and none of
them can be corrected after the fact:

* ``ttft_ms`` / ``ttft_p50_ms`` / ``ttft_p95_ms`` are computed over a ring of
  retained samples, and the cold phase's prefills *are* the tail. They keep
  being reported as the warm phase's percentiles until the ring turns over.
* ``hit_rate`` is cumulative for the life of the server, so a cold pass drags
  it down for the whole run, by an amount that depends on how long the run is.
* Anything rate-like read off the ``kv_hbm`` line is implicitly over "time
  since the server booted", which includes the cold phase.

``POST /v1/kv_metrics/reset`` (added to all three vLLM builds: ours,
vllm-continuum and baseline-vllm) zeroes those counters and opens a new
measurement epoch. Every subsequent ``kv_hbm`` line carries ``epoch=N`` and
``epoch_age_s=``, and a ``kv_hbm_reset`` marker line names the boundary in the
server log — so the cold phase's lines are still there, just no longer mixed
in with the warm ones.

By default it evicts nothing. With ``flush_hbm=True`` it also drops every
resident block from the GPU prefix cache at that same instant, while leaving
any KV connector's store alone -- so the warm phase starts with **cold HBM in
front of a warm LM cache**, and the prefixes the cold phase produced are pulled
back up from the CPU tier instead of recomputed. That transfer is the thing
worth measuring; leaving HBM full would measure a cache that never has to
fetch, and wiping the connector too would measure a cold start.

Note the dependency: with no KV connector configured, flushing HBM leaves
nothing underneath and the warm phase is simply cold. The server reports which
case it is in as ``kv_connector_configured`` in the response, and this module
prints a warning when the two disagree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_S = 60.0

# Path is the same on every build, so one harness talks to all three arms.
RESET_PATH = "/v1/kv_metrics/reset"


def resolve_reset_url() -> str | None:
    """Where to post the marker, or None if there is no local vLLM to tell.

    ``KV_METRICS_RESET_URL`` wins if set (point it anywhere, including a
    proxy). Otherwise it is derived from ``OPENAI_BASE_URL``, which is how
    ``open_deep_research.utils`` already reaches the local server — deriving
    it means the marker cannot drift onto a different host from the traffic.
    """
    explicit = os.getenv("KV_METRICS_RESET_URL", "").strip()
    if explicit:
        return explicit

    base = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base:
        return None
    # OPENAI_BASE_URL conventionally ends in `/v1`; the route is mounted at
    # the server root, so strip it rather than producing `/v1/v1/...`.
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}{RESET_PATH}"


async def reset_kv_metrics(
    label: str,
    *,
    flush_hbm: bool = True,
    record_to: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Mark the end of the cold phase on the server. Never raises.

    A failed marker means the reported numbers still contain the cold phase —
    bad, but not a reason to throw away a benchmark run that is otherwise
    fine. The failure is returned and written next to the run's other metrics
    so it cannot be discovered later by wondering why the hit rate looks low.
    """
    url = resolve_reset_url()
    if url is None:
        result: dict[str, Any] = {
            "ok": False,
            "reason": "no_reset_url",
            "detail": (
                "Neither KV_METRICS_RESET_URL nor OPENAI_BASE_URL is set, so "
                "there is no vLLM server to mark. Measurements will still "
                "include the cold phase."
            ),
            "label": label,
        }
    else:
        # Imported here so a run that never resets does not require httpx.
        import httpx

        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(
                    url,
                    params={
                        "label": label,
                        # Sent lower-case: FastAPI parses "True" too, but the
                        # access log and any proxy in between read better with
                        # the JSON spelling.
                        "flush_hbm": "true" if flush_hbm else "false",
                    },
                )
            payload = response.json()
            if not isinstance(payload, dict):
                payload = {"ok": False, "reason": "bad_payload", "body": payload}
            result = {**payload, "url": url, "status_code": response.status_code}
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            result = {
                "ok": False,
                "reason": "request_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "url": url,
                "label": label,
                "flush_hbm_requested": flush_hbm,
            }

    if record_to is not None:
        record_to.parent.mkdir(parents=True, exist_ok=True)
        record_to.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if result.get("ok"):
        flushed = result.get("hbm_flushed", "skipped")
        print(
            f"KV metrics reset: epoch {result.get('prev_epoch')} -> "
            f"{result.get('epoch')} (label={result.get('label')}, "
            f"hbm_flushed={flushed}); the cold phase is excluded from "
            f"everything measured after this point"
        )
        if flush_hbm and flushed == "false":
            # The engine refused: blocks were still referenced. The warm phase
            # is about to run against a partly-resident HBM cache, which is
            # neither of the two conditions being compared.
            print(
                "  WARNING: the HBM flush was refused (blocks still "
                "referenced by in-flight requests) -- the warm phase starts "
                "with cold-phase blocks resident"
            )
        if flush_hbm and result.get("kv_connector_configured") is False:
            print(
                "  WARNING: HBM was flushed but this server has no KV "
                "connector, so there is no CPU tier to serve the warm phase "
                "-- it will run fully cold"
            )
        marker = result.get("marker")
        if marker:
            print(f"  {marker}")
    else:
        print(
            "KV metrics reset FAILED "
            f"({result.get('reason')}: {result.get('detail', '')}) — reported "
            "numbers will still include the cold phase"
        )
    return result
