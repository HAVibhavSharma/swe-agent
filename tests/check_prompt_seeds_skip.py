"""Check that seeds LMCache already holds are not sent, and seeds go as warms.

Runs the real graph against the fake serving engine of ``check_prompt_seeds.py``
plus a fake prefill engine and a fake LMCache server:

* the prefill engine answers ``/tokenize`` (one token per character of the
  rendered messages and tools, so prefixes stay prefixes) and one-token chat
  completions, and each completion "stores" its tokens in the fake LMCache;
* LMCache's ``/lookup`` reports the longest prefix of the given tokens that
  some stored prompt shares.

With ``SWE_PROMPT_SEEDS_LOOKUP_URL`` set, a seed whose tokens LMCache holds
all but fewer than ``SWE_PROMPT_SEEDS_MIN_NEW_TOKENS`` of must not reach the
prefill engine, every other seed must, and each log row must say which. With
``SWE_PROMPT_SEEDS_AS_WARMS`` every sent seed carries a warm's request id,
``prefetch::<agent_id>::seed-<hex>``. Then a lookup that fails must not stop
a seed.

    python tests/check_prompt_seeds_skip.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("check_prompt_seeds", ROOT / "tests" / "check_prompt_seeds.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

MIN_NEW = 200
PREFILLS: list[dict] = []
STORED: list[list[int]] = []
LOOKUPS: list[int] = []
_lock = threading.Lock()


def fake_tokens(messages, tools) -> list[int]:
    return [ord(c) for c in json.dumps([tools or [], messages], sort_keys=True)]


def _reply(handler: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    data = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class PrefillHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tokens = fake_tokens(body["messages"], body.get("tools"))
        if self.path == "/tokenize":
            if body.get("add_generation_prompt") is not False:
                return _reply(self, 400, {"error": "tokenize must match the seed's rendering"})
            return _reply(self, 200, {"count": len(tokens), "tokens": tokens})
        with _lock:
            PREFILLS.append(body)
            STORED.append(tokens)
        with base._events_lock:
            base.EVENTS.append(("prefill", len(PREFILLS) - 1))
        _reply(self, 200, {"id": "p", "object": "chat.completion", "model": body.get("model"),
                           "choices": [{"index": 0, "finish_reason": "length",
                                        "message": {"role": "assistant", "content": "x"}}],
                           "usage": {"prompt_tokens": len(tokens), "completion_tokens": 1}})


class LookupHandler(BaseHTTPRequestHandler):
    broken = False

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if LookupHandler.broken:
            return _reply(self, 503, {"status": "error"})
        tokens = body["token_ids"]
        with _lock:
            best = 0
            for stored in STORED:
                n = 0
                for a, b in zip(tokens, stored):
                    if a != b:
                        break
                    n += 1
                best = max(best, n)
            LOOKUPS.append(len(tokens))
        _reply(self, 200, {"status": "ok", "tokens": len(tokens), "cached_prefix_tokens": best})


def main() -> int:
    check = base.check
    servers = [ThreadingHTTPServer(("127.0.0.1", 0), h)
               for h in (base.Handler, PrefillHandler, LookupHandler)]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    serving, helper, lmcache = servers
    api = f"http://127.0.0.1:{serving.server_port}/v1"
    helper_url = f"http://127.0.0.1:{helper.server_port}"

    work = Path(tempfile.mkdtemp(prefix="swe-seeds-skip-"))
    (work / "workspace_repo").mkdir()
    (work / "workspace_repo" / "a.py").write_text("x = 1\nprint(x)\n")
    os.environ.update({
        "MODEL_PROVIDER": "openai", "MODEL_NAME": "fake-model", "OPENAI_MODEL": "fake-model",
        "OPENAI_BASE_URL": api, "OPENAI_API_KEY": "EMPTY",
        "LANGGRAPH_VLLM_AGENT_BASE_URL": f"{api}/agents",
        "LANGGRAPH_PREDICTION_LOG_DIR": str(work / "logs"),
        "LANGGRAPH_TIMING_LOG_DIR": str(work / "timing"),
        "LANGGRAPH_COMPILE_DEBUG_PATH": str(work / "compile_debug.json"),
        "SWE_TRACE_MODE": "off", "LANGSMITH_TRACING": "false",
        # The pitlane spellings, which the module falls back to.
        "ODR_PROMPT_SEEDS_PREFILL_URL": helper_url,
        "ODR_PROMPT_SEEDS_LOOKUP_URL": f"http://127.0.0.1:{lmcache.server_port}/lookup/",
        "ODR_PROMPT_SEEDS_MIN_NEW_TOKENS": str(MIN_NEW),
        "ODR_PROMPT_SEEDS_AS_WARMS": "1",
    })
    os.chdir(work)

    from agent.common import prompt_seeds
    from agent.graph import swe_agent

    check("lookup URL, threshold and warm naming are read",
          prompt_seeds.lookup_url().endswith("/lookup") and prompt_seeds.min_new_tokens() == MIN_NEW
          and prompt_seeds.as_warms())

    class Inline:
        def submit(self, fn, *args, **kwargs):
            from concurrent.futures import Future
            future: Future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    prompt_seeds._executor = lambda job_id: Inline()
    config = {"configurable": {"thread_id": "t1", "job_id": "job-1",
                               "max_researcher_iterations": 6, "max_react_tool_calls": 8,
                               "max_atomic_tasks": 20},
              "recursion_limit": 200}
    for _ in swe_agent.stream(
        {"implementation_research_scratchpad": [("user", "Rename x to y in a.py and add new_mod.py")]},
        config, stream_mode="updates", subgraphs=True,
    ):
        pass

    check("the run reached creating_diffs_for_task for both tasks",
          base._turns["creating_diffs_for_task"] == 2, str(dict(base._turns)))
    log = work / "logs" / "job_job-1.prompt_seeds.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    skipped = [r for r in rows if r["skipped"] == "cached"]
    sent = [r for r in rows if r["skipped"] is None]
    check("every seed was looked up", len(LOOKUPS) == len(rows) and all(
        r["lookup_error"] is None and r["lookup_tokens"] for r in rows), f"{len(LOOKUPS)} lookups")
    check("some seeds were skipped and some sent", skipped and sent,
          f"{len(skipped)} skipped, {len(sent)} sent")
    check("a skipped seed had fewer than the threshold of new tokens",
          all(r["lookup_tokens"] - r["lookup_cached_tokens"] < MIN_NEW for r in skipped))
    check("a sent seed had at least the threshold",
          all(r["lookup_tokens"] - r["lookup_cached_tokens"] >= MIN_NEW for r in sent))
    check("only the sent seeds reached the prefill engine", len(PREFILLS) == len(sent),
          f"{len(PREFILLS)} prefills, {len(sent)} sent")
    check("nothing went to the serving engine's /prefetch as a seed", base.SEEDS == [])
    check("skipped rows carry no request id and no prefill",
          all(r["request_id"] is None and r["prompt_tokens"] is None for r in skipped))
    ids = [p.get("request_id") for p in PREFILLS]
    check("every sent seed is named as a warm for its agent",
          all(r["request_id"] and r["request_id"].startswith(f"prefetch::{r['agent_id']}::seed-")
              for r in sent) and ids == [r["request_id"] for r in sent] and len(set(ids)) == len(ids))

    # A lookup that fails must not stop a seed: it is sent as before.
    LookupHandler.broken = True
    before = len(PREFILLS)
    prompt_seeds.reset()
    prompt_seeds._send_seed(
        event="probe_failure", producer_node="x", target="conduct_research", job_id="job-2",
        agent_id="langgraph:job-2:swe_architect:conduct_research",
        prompt=__import__("langchain_core.prompts", fromlist=["ChatPromptTemplate"])
        .ChatPromptTemplate.from_messages([("system", "s"), ("placeholder", "{h}")]),
        inputs={"h": [("user", "a question nobody has asked")]}, history_key="h", tools=None,
        issued_at_ns=0, issued_at_unix=0.0)
    row = json.loads((work / "logs" / "job_job-2.prompt_seeds.jsonl").read_text().splitlines()[-1])
    check("a failed lookup still sends the seed", len(PREFILLS) == before + 1
          and row["lookup_error"] and row["skipped"] is None and row["error"] is None, str(row))

    print(f"\n{len(rows)} seeds: {len(skipped)} skipped as cached, {len(sent)} sent; "
          f"{'all checks passed' if not base.FAILURES else f'{len(base.FAILURES)} FAILED'}")
    return 1 if base.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
