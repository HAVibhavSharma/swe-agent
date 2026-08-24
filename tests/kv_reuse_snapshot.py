"""Read the server's KV reuse rollup without ending the measurement epoch.

``kv_metrics_reset.py`` marks the boundary between the cold and warm phases,
and the response it gets back happens to carry the numbers it just discarded.
That is the wrong instrument for reading the *warm* phase: the only way to
see those numbers through the reset is to end the phase you are measuring.

``GET /v1/kv_metrics`` returns the same structure and changes nothing, so it
can be called at the end of a run — or repeatedly during one, which is the
only way to watch cross-question reuse accumulate as questions pile up rather
than seeing one number after the fact.

With ``VLLM_KV_PROVENANCE=1`` on the server, the payload's ``stats`` includes:

``reuse_cross_tokens``       tokens served from blocks another job computed
``reuse_self_tokens``        tokens a job re-read from its own earlier prefill
``reuse_unknown_tokens``     hits on content this server never saw claimed
``reuse_by_source_job``      job -> tokens it supplied to other jobs
``reuse_by_consumer_job``    job -> tokens it took from other jobs
``reuse_top_pairs``          the largest source -> consumer flows

Per-request detail behind those totals goes to the server's
``VLLM_KV_PROVENANCE_PATH`` JSONL; ``analyze_kv_reuse.py`` joins it to the
question text.

Standalone::

    python tests/kv_reuse_snapshot.py --out reuse_snapshot.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_S = 60.0

# Same path on every build, so one harness reads all three arms.
SNAPSHOT_PATH = "/v1/kv_metrics"


def resolve_snapshot_url() -> str | None:
    """Where to read from, or None if there is no local vLLM to ask.

    ``KV_METRICS_URL`` wins if set. Otherwise derived from
    ``OPENAI_BASE_URL`` — the same derivation ``kv_metrics_reset.py`` uses, so
    the snapshot cannot drift onto a different host from the traffic.
    """
    explicit = os.getenv("KV_METRICS_URL", "").strip()
    if explicit:
        return explicit

    base = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base:
        return None
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}{SNAPSHOT_PATH}"


async def snapshot_kv_metrics(
    *,
    record_to: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Fetch the rollup. Never raises — instrumentation must not fail a run."""
    url = resolve_snapshot_url()
    if url is None:
        result: dict[str, Any] = {
            "ok": False,
            "reason": "no_metrics_url",
            "detail": (
                "Neither KV_METRICS_URL nor OPENAI_BASE_URL is set, so there "
                "is no vLLM server to read."
            ),
        }
    else:
        # Imported here so a run that never snapshots does not require httpx.
        import httpx

        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.get(url)
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
            }

    if record_to is not None:
        record_to.parent.mkdir(parents=True, exist_ok=True)
        record_to.write_text(json.dumps(result, indent=2), encoding="utf-8")

    _print_summary(result)
    return result


def _print_summary(result: dict[str, Any]) -> None:
    if not result.get("ok"):
        print(
            "KV metrics snapshot FAILED "
            f"({result.get('reason')}: {result.get('detail', '')})"
        )
        return

    stats = result.get("stats") or {}
    if "reuse_cross_tokens" not in stats:
        # The server is up and instrumented, but provenance is off. Worth
        # saying plainly: the run will otherwise produce an empty analysis
        # with no explanation.
        print(
            "KV metrics snapshot: hit_rate="
            f"{stats.get('hit_rate', 0):.4f} epoch={stats.get('metrics_epoch')} "
            "-- no reuse attribution (start the server with "
            "VLLM_KV_PROVENANCE=1)"
        )
        return

    hits = stats.get("reuse_hit_tokens", 0) or 1
    print(
        f"KV reuse snapshot (epoch={stats.get('metrics_epoch')}): "
        f"{stats.get('reuse_requests', 0)} prefills, "
        f"{stats.get('reuse_hit_tokens', 0)} hit tokens -- "
        f"cross-question {stats.get('reuse_cross_tokens', 0)} "
        f"({100.0 * stats.get('reuse_cross_tokens', 0) / hits:.1f}%), "
        f"same-job {stats.get('reuse_self_tokens', 0)}, "
        f"unknown {stats.get('reuse_unknown_tokens', 0)}"
    )
    for pair in (stats.get("reuse_top_pairs") or [])[:5]:
        print(
            f"  job {pair['source_job']} -> job {pair['consumer_job']}: "
            f"{pair['tokens']} tokens"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the raw payload here as JSON.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args()
    asyncio.run(snapshot_kv_metrics(record_to=args.out, timeout_s=args.timeout))


if __name__ == "__main__":
    main()
