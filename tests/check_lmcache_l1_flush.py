"""LMCache's L1 is cleared only once its L2 writes have drained.

    python tests/check_lmcache_l1_flush.py

A stand-in for LMCache's HTTP frontend on a local port: `/status` reports a
store queue that drains after a few polls (or never), `/clear-cache` records
the call and empties a counter of L1 objects.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lmcache_l1_flush as flush  # noqa: E402


class _State:
    def __init__(self, pending_polls: int | None):
        self.pending_polls = pending_polls  # None: never drains
        self.polls = 0
        self.clears = 0
        self.l1_objects = 1700


def _serve(state: _State) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _reply(self, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.path == "/status", self.path
            state.polls += 1
            draining = (state.pending_polls is None
                        or state.polls <= state.pending_polls)
            self._reply({"storage_manager": {
                "store_controller": {"pending_keys_count": 512 if draining else 0,
                                     "in_flight_task_count": 1 if draining else 0},
                "l1_manager": {"total_object_count": state.l1_objects},
            }})

        def do_POST(self):
            assert self.path == "/clear-cache", self.path
            state.clears += 1
            state.l1_objects = 0
            self._reply({"status": "ok"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


async def main() -> None:
    os.environ.pop(flush.URL_ENV_VAR, None)
    assert await flush.flush_lmcache_l1("off") is None, "off without the URL"

    # Drains after three polls: then exactly one clear, and L1 is empty.
    state = _State(pending_polls=3)
    server, url = _serve(state)
    os.environ[flush.URL_ENV_VAR] = url + "/"
    with tempfile.TemporaryDirectory() as tmp:
        record = Path(tmp) / "flush.jsonl"
        result = await flush.flush_lmcache_l1("q1", poll_s=0.01, record_to=record)
        assert result["ok"] and state.clears == 1 and state.polls >= 4, (result, state.__dict__)
        assert result["l1_objects_before"] == 1700 and result["l1_objects_after"] == 0, result
        assert json.loads(record.read_text().splitlines()[0])["label"] == "q1"
    server.shutdown()

    # Never drains: give up and leave L1 alone, rather than drop unwritten blocks.
    state = _State(pending_polls=None)
    server, url = _serve(state)
    result = await flush.flush_lmcache_l1("q2", url=url, drain_timeout_s=0.1, poll_s=0.01)
    assert not result["ok"] and result["reason"] == "drain_timeout", result
    assert state.clears == 0, "cleared L1 with writes still pending"
    server.shutdown()

    # Nothing listening: a failure, not an exception.
    result = await flush.flush_lmcache_l1("q3", url="http://127.0.0.1:9", poll_s=0.01)
    assert not result["ok"] and result["reason"] == "request_failed", result
    print("all lmcache-l1-flush assertions passed")


if __name__ == "__main__":
    asyncio.run(main())
