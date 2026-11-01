"""A refused HBM flush is asked again; one that succeeds, or no flush, is not.

    python tests/check_kv_metrics_reset.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["KV_METRICS_RESET_URL"] = "http://stub/v1/kv_metrics/reset"

import kv_metrics_reset as kmr  # noqa: E402


def _server(answers: list[str]):
    calls: list[bool] = []

    async def post(url, label, flush_hbm, timeout_s):
        calls.append(flush_hbm)
        flushed = answers[min(len(calls), len(answers)) - 1]
        return {"ok": True, "hbm_flushed": flushed, "epoch": len(calls) + 1,
                "prev_epoch": len(calls), "label": label}
    return post, calls


async def main() -> None:
    post, calls = _server(["false", "false", "true"])
    kmr._post_reset = post
    result = await kmr.reset_kv_metrics("x", flush_hbm=True, flush_attempts=20,
                                        retry_wait_s=0)
    assert result["hbm_flushed"] == "true" and result["flush_attempts"] == 3, result
    assert calls == [True, True, True]

    post, calls = _server(["false"])
    kmr._post_reset = post
    result = await kmr.reset_kv_metrics("x", flush_hbm=True, flush_attempts=4,
                                        retry_wait_s=0)
    assert result["hbm_flushed"] == "false" and result["flush_attempts"] == 4, result

    post, calls = _server(["skipped"])
    kmr._post_reset = post
    result = await kmr.reset_kv_metrics("x", flush_hbm=False, flush_attempts=20,
                                        retry_wait_s=0)
    assert result["flush_attempts"] == 1 and len(calls) == 1, result
    print("all kv-metrics-reset assertions passed")


if __name__ == "__main__":
    asyncio.run(main())
