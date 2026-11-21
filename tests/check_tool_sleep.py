"""Check SWE_TOOL_SLEEP_S without a GPU.

A tiny graph with one tool node, run the way run_evaluate.py runs the agent
(`astream`, several instances on one event loop):

- unset, the graph gets langgraph's own ToolNode;
- set, each tool turn sleeps once and still returns the tool's result;
- instances at once sleep side by side, not one after another;
- a ranged sleep is the same for the same tool calls, and stays in range;
- langgraph's prompt analysis reads the same source for the node as for a
  plain ToolNode, so the predictor sees no new model call.

    python tests/check_tool_sleep.py
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

FAILURES = 0


def check(label: str, ok: bool) -> None:
    global FAILURES
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    FAILURES += 0 if ok else 1


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return f"contents of {path}"


class State(TypedDict):
    scratch: Annotated[list[AnyMessage], add_messages]


def graph_with(node):
    builder = StateGraph(State)
    builder.add_node("tools", node)
    builder.add_edge(START, "tools")
    return builder.compile()


def turn(path: str) -> dict:
    call = {"name": "read_file", "args": {"path": path}, "id": f"call-{path}"}
    return {"scratch": [AIMessage(content="", tool_calls=[call])]}


async def run(graph, state) -> list:
    out = []
    async for part in graph.astream(state, stream_mode="updates"):
        out.append(part)
    return out


def make(env: str | None):
    if env is None:
        os.environ.pop("SWE_TOOL_SLEEP_S", None)
    else:
        os.environ["SWE_TOOL_SLEEP_S"] = env
    from agent.tools import sleep
    importlib.reload(sleep)
    return sleep, sleep.make_tool_node([read_file], messages_key="scratch")


def main() -> int:
    print("== unset")
    sleep, node = make(None)
    check("unset gives langgraph's own ToolNode", type(node) is ToolNode)
    _, node = make("0")
    check("0 gives langgraph's own ToolNode", type(node) is ToolNode)

    print("== a fixed sleep")
    sleep, node = make("0.5")
    check("set gives a SleepingToolNode, still a ToolNode",
          isinstance(node, sleep.SleepingToolNode) and isinstance(node, ToolNode))
    graph = graph_with(node)
    t0 = time.monotonic()
    out = asyncio.run(run(graph, turn("a.py")))
    took = time.monotonic() - t0
    message = out[-1]["tools"]["scratch"][-1]
    check(f"one turn slept ~0.5 s ({took:.2f} s)", 0.45 <= took < 0.9)
    check("and returned the tool's result",
          message.content == "contents of a.py" and message.tool_call_id == "call-a.py")

    async def four():
        return await asyncio.gather(*(run(graph, turn(f"{i}.py")) for i in range(4)))
    t0 = time.monotonic()
    asyncio.run(four())
    took = time.monotonic() - t0
    check(f"four instances at once slept side by side ({took:.2f} s, not ~2 s)", took < 0.9)

    print("== a ranged sleep")
    sleep, _ = make("5-30")
    bounds = sleep.parse_tool_sleep("5-30")
    calls_a = [{"name": "read_file", "args": {"path": "a.py"}, "id": "x"}]
    calls_a_new_id = [{"name": "read_file", "args": {"path": "a.py"}, "id": "y"}]
    calls_b = [{"name": "read_file", "args": {"path": "b.py"}, "id": "x"}]
    a = sleep.tool_sleep_seconds(calls_a, bounds)
    check(f"the same calls sleep the same, whatever their id ({a:.2f} s)",
          a == sleep.tool_sleep_seconds(calls_a_new_id, bounds))
    spread = [sleep.tool_sleep_seconds([{"name": "t", "args": {"i": i}}], bounds)
              for i in range(200)]
    check(f"200 turns stay in 5-30 s (min {min(spread):.1f}, max {max(spread):.1f})",
          all(5 <= s <= 30 for s in spread) and max(spread) - min(spread) > 15)
    check("other calls sleep differently", a != sleep.tool_sleep_seconds(calls_b, bounds))
    for bad in ("x", "30-5", "-1"):
        try:
            sleep.parse_tool_sleep(bad)
            check(f"{bad!r} is rejected", False)
        except ValueError:
            check(f"{bad!r} is rejected", True)

    print("== langgraph's prompt analysis")
    import inspect
    sleep, node = make("1")
    plain = ToolNode([read_file], messages_key="scratch")
    check("the node's sync body reads as ToolNode's own source",
          inspect.getsource(node.func) == inspect.getsource(plain.func))
    check("its async body too",
          inspect.getsource(node.afunc) == inspect.getsource(plain.afunc))
    check("and RunnableCallable sees ToolNode's signature",
          inspect.signature(node.afunc) == inspect.signature(plain.afunc))

    print("\n" + ("all tool-sleep checks passed" if FAILURES == 0 else f"{FAILURES} FAILURES"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
