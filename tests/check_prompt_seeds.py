"""Check the prompt seeds end to end, without a GPU.

Runs the real graph against one local fake server that plays both the OpenAI
chat endpoint and vLLM's ``/v1/agents/prefetch``, with scripted replies that
walk every seeded edge: a rejected research step, a tool turn in both react
loops, and one existing-file and one new-file atomic task.

The property that matters is that a seed is a prefix of the request its
target really sends -- same ``agent_id``, same leading messages byte for
byte, same tools. A seed that differs in one character warms a prefix nothing
hits and fails silently on the GPU box, so this asserts it on every seed.

    python tests/check_prompt_seeds.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHAT: list[dict] = []
SEEDS: list[dict] = []
EVENTS: list[tuple[str, int]] = []  # ("chat" | "seed", index), in arrival order
_events_lock = threading.Lock()
_turns: dict[str, int] = defaultdict(int)


def _reply(body: dict) -> dict:
    """The scripted assistant message for one chat request."""
    node = body.get("langgraph_node")
    _turns[node] += 1
    turn = _turns[node]
    system = body["messages"][0]["content"]

    def tool_call(name: str, args: dict) -> dict:
        return {"content": None, "tool_calls": [{
            "id": f"call_{node}_{turn}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}

    if node == "come_up_with_research_next_step":
        return {"content": json.dumps({"reasoning": f"r{turn}", "hypothesis": f"h{turn}"})}
    if node == "check_research_step":
        # Reject the first step, so the research loop restarts once.
        return {"content": json.dumps({"reasoning": f"c{turn}", "is_valid": turn > 1})}
    if node in ("conduct_research", "get_clear_implementation_plan_for_atomic_task"):
        if turn % 2:
            return tool_call("get_raw_file_content", {"file_path": "./workspace_repo/a.py"})
        return {"content": f"findings {turn}"}
    if node == "extract_implementation_plan":
        return {"content": json.dumps({"tasks": [
            {"file_path": "./workspace_repo/a.py", "logical_task": "edit",
             "atomic_tasks": [{"atomic_task": "rename x", "additional_context": "ctx"}]},
            {"file_path": "./workspace_repo/new_mod.py", "logical_task": "add",
             "atomic_tasks": [{"atomic_task": "create it", "additional_context": ""}]},
        ]})}
    if node == "creating_diffs_for_task":
        if "create a new file" in json.dumps(body["messages"]):
            return {"content": "VALUE = 1\n"}
        return {"content": "<code_change_request>\noriginal_code_snippet:\n1| x = 1\n"
                           "edit_code_snippet:\ny = 1\n</code_change_request>"}
    raise AssertionError(f"unscripted node {node!r}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _json(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/agents/prefetch"):
            if "messages" not in body:  # the fork's own predicted warm
                return self._json({"submitted": 0})
            with _events_lock:
                SEEDS.append(body)
                EVENTS.append(("seed", len(SEEDS) - 1))
            return self._json({"seeded_from_messages": True, "prefill_on_miss": True,
                               "submitted": 1, "scored": True})
        with _events_lock:
            CHAT.append(body)
            EVENTS.append(("chat", len(CHAT) - 1))
        message = {"role": "assistant", **_reply(body)}
        if not body.get("stream"):
            return self._json({"id": "x", "object": "chat.completion", "created": 0,
                               "model": body["model"], "choices": [{
                                   "index": 0, "message": message,
                                   "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        delta = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = [{"index": 0, **call} for call in message["tool_calls"]]
        chunks = [
            {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {},
                          "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            chunk.update({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body["model"]})
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label}{f' -- {detail}' if detail and not ok else ''}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/v1"

    work = Path(tempfile.mkdtemp(prefix="swe-seeds-"))
    (work / "workspace_repo").mkdir()
    (work / "workspace_repo" / "a.py").write_text("x = 1\nprint(x)\n")
    os.environ.update({
        "MODEL_PROVIDER": "openai", "MODEL_NAME": "fake-model", "OPENAI_MODEL": "fake-model",
        "OPENAI_BASE_URL": base, "OPENAI_API_KEY": "EMPTY",
        "LANGGRAPH_VLLM_AGENT_BASE_URL": f"{base}/agents",
        "LANGGRAPH_PREDICTION_LOG_DIR": str(work / "logs"),
        "LANGGRAPH_TIMING_LOG_DIR": str(work / "timing"),
        "LANGGRAPH_COMPILE_DEBUG_PATH": str(work / "compile_debug.json"),
        "SWE_TRACE_MODE": "off", "LANGSMITH_TRACING": "false",
    })
    os.chdir(work)  # the agent hard-codes ./workspace_repo

    from agent.common import prompt_seeds
    from agent.graph import swe_agent

    # Send each seed inline, so it reaches the server before its target's
    # request does -- the order the GPU box's real latencies give, and what
    # lets each seed be checked against the very request it was built for.
    class Inline:
        def submit(self, fn, *args, **kwargs):
            from concurrent.futures import Future
            future: Future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    prompt_seeds._EXECUTOR = Inline()

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
          _turns["creating_diffs_for_task"] == 2, str(dict(_turns)))
    check("the existing file was edited", (work / "workspace_repo" / "a.py").read_text().startswith("y = 1"))
    check("the new file was created", (work / "workspace_repo" / "new_mod.py").exists())

    expected_targets = {
        "check_research_step", "conduct_research", "extract_implementation_plan",
        "get_clear_implementation_plan_for_atomic_task", "creating_diffs_for_task",
    }
    seeded = {seed["langgraph_node"] for seed in SEEDS}
    check("every target was seeded", seeded == expected_targets,
          f"seeded={sorted(seeded)}")
    check("come_up_with_research_next_step is never a target",
          "come_up_with_research_next_step" not in seeded)

    for position, (kind, index) in enumerate(EVENTS):
        if kind != "seed":
            continue
        seed = SEEDS[index]
        target = seed["langgraph_node"]
        label = f"seed {index} -> {target} ({len(seed['messages'])} msgs)"
        following = [CHAT[i] for k, i in EVENTS[position + 1:]
                     if k == "chat" and CHAT[i]["langgraph_node"] == target]
        if not following:
            check(label, False, "no later request from the target")
            continue
        real = following[0]
        problems = []
        if real.get("agent_id") != seed["agent_id"]:
            problems.append(f"agent_id {seed['agent_id']} != {real.get('agent_id')}")
        if real["messages"][: len(seed["messages"])] != seed["messages"]:
            problems.append("messages are not a prefix of the real request")
        if len(seed["messages"]) < 2:
            problems.append("seed holds nothing beyond the system block")
        if real.get("tools") != seed.get("tools"):
            problems.append("tools differ")
        if seed.get("job_id") != real.get("job_id") or seed.get("call_type") != real.get("call_type"):
            problems.append("eviction index key differs")
        if seed.get("prefill_on_miss") is not True:
            problems.append("prefill_on_miss not explicit")
        check(label, not problems, "; ".join(problems))

    log = work / "logs" / "job_job-1.prompt_seeds.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    check("one log row per seed", len(rows) == len(SEEDS), f"{len(rows)} rows, {len(SEEDS)} seeds")
    check("no seed errored", all(row["error"] is None for row in rows))

    # The seed worker is untraced. It runs on a thread that does not inherit the
    # node's run, so with LangSmith on its prompt renders and repo-tree builds
    # would each become a stray root run.
    from langchain_core.callbacks.manager import CallbackManager

    seen: list[list[str]] = []
    send_seed = prompt_seeds._send_seed
    prompt_seeds._send_seed = lambda **kwargs: seen.append(
        [type(h).__name__ for h in CallbackManager.configure().handlers])
    os.environ.update({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "x",
                       "LANGSMITH_ENDPOINT": base})
    worker = threading.Thread(target=prompt_seeds._build_and_send)
    worker.start()
    worker.join()
    os.environ["LANGSMITH_TRACING"] = "false"
    prompt_seeds._send_seed = send_seed
    check("the seed worker runs with no LangSmith tracer", seen == [[]], str(seen))

    # Off means off: nothing is posted once the prefetch path is disabled.
    sent = len(SEEDS)
    os.environ["SWE_PROMPT_SEEDS"] = "0"
    prompt_seeds.reset()
    _turns.clear()
    for _ in swe_agent.stream(
        {"implementation_research_scratchpad": [("user", "again")]},
        {**config, "configurable": {**config["configurable"], "job_id": "job-2", "thread_id": "t2"}},
        stream_mode="updates",
    ):
        pass
    check("SWE_PROMPT_SEEDS=0 sends nothing", len(SEEDS) == sent)

    server.shutdown()
    print(f"\n{len(SEEDS)} seeds, {len(CHAT)} chat requests; "
          f"{'all checks passed' if not FAILURES else f'{len(FAILURES)} FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
