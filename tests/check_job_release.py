"""Check tests/job_release.py against a fake vLLM and a fake LMCache.

A question's end releases its staged warms (``POST /v1/agents/release``) and
withdraws its demand (``DELETE /demand/<job>``), naming that question exactly;
with neither configured it calls nothing.

    python tests/check_job_release.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_release import release_job  # noqa: E402

CALLS: list[tuple[str, str]] = []
FAILURES: list[str] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _reply(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        CALLS.append(("POST", self.path))
        self._reply({"released": ["a::l1", "b::l1"]})

    def do_DELETE(self) -> None:
        CALLS.append(("DELETE", self.path))
        self._reply({"status": "ok", "applied": True})


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label}{f' -- {detail}' if detail and not ok else ''}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    for key in ("SWE_RELEASE_STAGED_ON_END", "LANGGRAPH_PREDICTION_DEMAND_URL"):
        os.environ.pop(key, None)
    os.environ["LANGGRAPH_VLLM_AGENT_BASE_URL"] = f"{base}/v1/agents"

    row = release_job("1")
    check("nothing is called unless asked", CALLS == [] and row["released"] is None, str(CALLS))

    os.environ["SWE_RELEASE_STAGED_ON_END"] = "1"
    os.environ["LANGGRAPH_PREDICTION_DEMAND_URL"] = f"{base}/demand"
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "job_release.jsonl"
        row = release_job("1", record_to=log)
        check("releases that question's staged warms and withdraws its demand, by its exact id",
              CALLS == [("POST", "/v1/agents/release?job_id=1"), ("DELETE", "/demand/1")], str(CALLS))
        check("reports what was released", row["released"] == 2 and row["demand_cleared"] is True
              and row["errors"] == [], str(row))
        check("one log row", len(log.read_text().splitlines()) == 1)

    CALLS.clear()
    os.environ["LANGGRAPH_VLLM_AGENT_BASE_URL"] = "http://127.0.0.1:9/v1/agents"  # nothing listens
    row = release_job("10")
    check("a failure is recorded, not raised", row["released"] is None
          and any(e.startswith("release:") for e in row["errors"]), str(row))
    check("the demand is still withdrawn", CALLS == [("DELETE", "/demand/10")], str(CALLS))

    server.shutdown()
    print(f"\n{'all checks passed' if not FAILURES else f'{len(FAILURES)} FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
