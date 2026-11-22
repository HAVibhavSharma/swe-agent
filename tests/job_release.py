"""When a question ends, let go of what was kept for it -- and only that.

Two things are held for a running question that its end makes useless:

* the staged warms vLLM still has in flight for it (each holds its prefix
  read-locked in LMCache's L1): ``POST <vLLM>/v1/agents/release?job_id=<id>``,
  with ``SWE_RELEASE_STAGED_ON_END=1``;
* the demand its predictor reported to LMCache (``DemandLRU`` keeps blocks of
  the nodes running questions are heading to): ``DELETE <LMCache>/demand/<id>``,
  whenever ``LANGGRAPH_PREDICTION_DEMAND_URL`` is set.

Both name the question exactly, so the questions still running beside it keep
their warms and their demand, and a block another question holds keeps that
question's lock. A failure is logged and never fails the question.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import parse, request


def _call(method: str, url: str, timeout: float = 10.0) -> dict[str, Any]:
    req = request.Request(url, method=method, data=b"" if method == "POST" else None)
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def release_job(job_id: str, *, record_to: Path | None = None) -> dict[str, Any]:
    """Release ``job_id``'s staged warms and withdraw its demand, as configured."""
    row: dict[str, Any] = {"job_id": job_id, "at_unix": time.time(),
                           "released": None, "demand_cleared": None, "errors": []}
    base = os.getenv("LANGGRAPH_VLLM_AGENT_BASE_URL", "").rstrip("/")
    if os.getenv("SWE_RELEASE_STAGED_ON_END", "0") == "1" and base:
        try:
            out = _call("POST", f"{base}/release?{parse.urlencode({'job_id': job_id})}")
            row["released"] = len(out.get("released") or [])
        except Exception as exc:  # noqa: BLE001
            row["errors"].append(f"release: {exc}")
    demand = os.getenv("LANGGRAPH_PREDICTION_DEMAND_URL", "").rstrip("/")
    if demand:
        try:
            out = _call("DELETE", f"{demand}/{parse.quote(str(job_id), safe='')}")
            row["demand_cleared"] = bool(out.get("applied"))
        except Exception as exc:  # noqa: BLE001
            row["errors"].append(f"demand: {exc}")
    if record_to is not None and (row["released"] is not None or row["demand_cleared"] is not None
                                  or row["errors"]):
        try:
            record_to.parent.mkdir(parents=True, exist_ok=True)
            with record_to.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except OSError:
            pass
    return row
