"""Check question pinning (agent/common/dp_routing.py) against a stand-in server.

Each question gets the engine with the fewest questions running when it
starts, the server hears it before the question's first request, and the
engine is freed when the question ends.

    python tests/check_dp_routing.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dp_routing", ROOT / "agent" / "common" / "dp_routing.py")
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)

POSTS: list[tuple[str, str, int]] = []
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(("   ok   " if ok else "   FAIL ") + label + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(label)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        url = urlsplit(self.path)
        q = parse_qs(url.query)
        POSTS.append((url.path, q["job_id"][0], int(q["rank"][0])))
        body = json.dumps({"job_id": q["job_id"][0], "rank": int(q["rank"][0]), "enabled": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log = Path(tempfile.mkdtemp()) / "dp_pin.jsonl"
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
    os.environ.pop("SWE_DP_PIN_ENGINES", None)
    check("off without SWE_DP_PIN_ENGINES", dp.assign("1") is None and dp.headers_for("1") == {} and not POSTS)

    os.environ.update({"SWE_DP_PIN_ENGINES": "2", "SWE_DP_PIN_LOG_PATH": str(log)})
    first = [dp.assign(j) for j in ("1", "2", "3", "5")]
    check("four questions alternate engines", first == [0, 1, 0, 1], str(first))
    check("each assignment reaches the server first", POSTS[:4] == [
        ("/v1/agents/dp_rank", "1", 0), ("/v1/agents/dp_rank", "2", 1),
        ("/v1/agents/dp_rank", "3", 0), ("/v1/agents/dp_rank", "5", 1)], str(POSTS[:4]))
    check("calls carry the header", dp.headers_for("3") == {"X-data-parallel-rank": "0"})
    dp.release("2")
    check("a release frees the engine at the server", POSTS[-1] == ("/v1/agents/dp_rank", "2", -1))
    check("the engine that lost a question takes the next", dp.assign("7") == 1)
    check("a released question carries no header", dp.headers_for("2") == {})
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    check("every assignment and release is logged",
          [r["event"] for r in rows] == ["assign"] * 4 + ["release", "assign"]
          and rows[-1]["running_before"] == [2, 1], str(rows[-1]))
    print(f"\n{'all checks passed' if not FAILURES else f'{len(FAILURES)} FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
