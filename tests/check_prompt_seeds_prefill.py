"""Check prompt seeds sent to a prefill engine (SWE_PROMPT_SEEDS_PREFILL_URL).

Runs the real graph against the fake server of ``check_prompt_seeds.py`` (the
serving engine) plus a second fake server standing in for the prefill engine
on the other GPU. With the URL set every seed must go to the prefill engine as
a one-token chat completion -- nothing to the serving engine's ``/prefetch`` --
and each must still be a prefix of the request its target really sends: the
same leading messages, the same tools, rendered without the generation prompt
so the real request extends it.

    python tests/check_prompt_seeds_prefill.py
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

PREFILLS: list[dict] = []


class PrefillHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with base._events_lock:
            PREFILLS.append({"path": self.path, **body})
            base.EVENTS.append(("prefill", len(PREFILLS) - 1))
        data = json.dumps({
            "id": "p", "object": "chat.completion", "created": 0, "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1234, "completion_tokens": 1, "total_tokens": 1235,
                      "prompt_tokens_details": {"cached_tokens": 1000}},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    check = base.check
    serving = ThreadingHTTPServer(("127.0.0.1", 0), base.Handler)
    helper = ThreadingHTTPServer(("127.0.0.1", 0), PrefillHandler)
    for server in (serving, helper):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    api = f"http://127.0.0.1:{serving.server_port}/v1"
    helper_url = f"http://127.0.0.1:{helper.server_port}"

    work = Path(tempfile.mkdtemp(prefix="swe-seeds-prefill-"))
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
        # The pitlane spelling, which the module falls back to.
        "ODR_PROMPT_SEEDS_PREFILL_URL": helper_url + "/",
    })
    os.chdir(work)

    from agent.common import prompt_seeds
    from agent.graph import swe_agent

    check("the prefill URL is read, trailing slash dropped", prompt_seeds.prefill_url() == helper_url)
    check("no seed priority unless asked", prompt_seeds.seed_priority() == 0)
    # As pitlane sets it under data parallelism, where seeds share the serving
    # engines with the real calls.
    os.environ["ODR_PROMPT_SEEDS_PRIORITY"] = "1"
    check("the seed priority is read", prompt_seeds.seed_priority() == 1)
    first, again, other = (prompt_seeds._executor("job-a"), prompt_seeds._executor("job-a"),
                           prompt_seeds._executor("job-b"))
    check("one seed worker per job with a prefill engine", first is again and first is not other
          and first is not prompt_seeds._EXECUTOR)

    class Inline:
        def submit(self, fn, *args, **kwargs):
            from concurrent.futures import Future
            future: Future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    # Inline, so each seed reaches its server before its target's request.
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
    check("nothing went to the serving engine's /prefetch as a seed", base.SEEDS == [],
          f"{len(base.SEEDS)} seeds there")
    check("the prefill engine got every seed", len(PREFILLS) >= 10, f"{len(PREFILLS)} prefills")

    for position, (kind, index) in enumerate(base.EVENTS):
        if kind != "prefill":
            continue
        seed = PREFILLS[index]
        problems = []
        if seed["path"] != "/v1/chat/completions":
            problems.append(f"posted to {seed['path']}")
        if seed.get("max_tokens") != 1:
            problems.append(f"max_tokens {seed.get('max_tokens')}")
        if seed.get("add_generation_prompt") is not False:
            problems.append("add_generation_prompt not off")
        if seed.get("model") != "fake-model":
            problems.append(f"model {seed.get('model')}")
        if seed.get("priority") != 1:
            problems.append(f"priority {seed.get('priority')}")
        later = [base.CHAT[i] for k, i in base.EVENTS[position + 1:] if k == "chat"
                 and base.CHAT[i]["messages"][: len(seed["messages"])] == seed["messages"]]
        if not later:
            problems.append("a prefix of no later request")
        elif later[0].get("tools") != seed.get("tools"):
            problems.append("tools differ from the request it is a prefix of")
        if len(seed["messages"]) < 2:
            problems.append("holds nothing beyond the system block")
        target = later[0].get("langgraph_node") if later else "?"
        check(f"prefill {index} -> {target} ({len(seed['messages'])} msgs)", not problems,
              "; ".join(problems))

    log = work / "logs" / "job_job-1.prompt_seeds.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    check("one log row per prefill", len(rows) == len(PREFILLS), f"{len(rows)} rows")
    check("no prefill errored", all(row["error"] is None for row in rows))
    check("rows name the prefill engine and its token counts",
          all(row["prefill_url"] == helper_url and row["prompt_tokens"] == 1234
              and row["cached_tokens"] == 1000 for row in rows))
    check("rows record the priority each seed was sent at",
          all(row["priority"] == 1 for row in rows))

    print(f"\n{len(PREFILLS)} prefills, {len(base.CHAT)} chat requests; "
          f"{'all checks passed' if not base.FAILURES else f'{len(base.FAILURES)} FAILED'}")
    return 1 if base.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
