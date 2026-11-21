"""SWE_TOOL_SLEEP_S: make every tool turn take longer, as a real agent's would.

The research tools here read files and search a local checkout, so a tool turn
takes ~0.2 s. A real coding agent's tools run tests, install packages or call
the network, and take seconds to minutes. The KV prefetcher's lead time is
that gap: with ~0.2 s it can hide almost none of a multi-second LMCache read.
This stretches each tool turn so the two arms can be compared at real lengths.

    SWE_TOOL_SLEEP_S=10      every tool turn sleeps 10 s
    SWE_TOOL_SLEEP_S=5-30    each sleeps 5-30 s, fixed by the turn's tool calls

The ranged sleep is a hash of the turn's tool names and arguments, not a
random draw, so under a pinned trace replay every arm sleeps the same amount
at the same turn. One sleep per tool turn, before the tools run: the node has
already started, so a warm fired at its start gets the whole sleep as lead.
Unset, empty or 0: the graphs get langgraph's own ToolNode, unchanged.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import time
from typing import Any, Sequence

from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

ENV = "SWE_TOOL_SLEEP_S"


def parse_tool_sleep(value: str | None) -> tuple[float, float] | None:
    """`"10"` as (10, 10), `"5-30"` as (5, 30); unset, empty or 0 as None."""
    if value is None or not value.strip():
        return None
    lo, sep, hi = value.strip().partition("-")
    try:
        lo_s = float(lo)
        hi_s = float(hi) if sep else lo_s
    except ValueError:
        raise ValueError(f"{ENV}={value!r} is neither seconds nor LO-HI seconds") from None
    if lo_s < 0 or hi_s < lo_s:
        raise ValueError(f"{ENV}={value!r}: need 0 <= LO <= HI")
    return None if hi_s == 0 else (lo_s, hi_s)


def tool_sleep_seconds(tool_calls: Sequence[dict[str, Any]], bounds: tuple[float, float]) -> float:
    """The turn's sleep: LO when fixed, else LO..HI by a hash of its calls."""
    lo, hi = bounds
    if lo == hi:
        return lo
    key = json.dumps(
        [[call.get("name"), call.get("args")] for call in tool_calls],
        sort_keys=True,
        default=str,
    )
    frac = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return lo + (hi - lo) * frac


class SleepingToolNode(ToolNode):
    """A ToolNode that sleeps once per turn before running the turn's tools.

    A subclass, not a wrapper function, so the graph still holds a ToolNode:
    langgraph's prompt analysis reads a node's source for `.invoke(` /
    `.ainvoke(` model calls, and a wrapper calling the ToolNode would look
    like one. The overrides keep ToolNode's signatures (`functools.wraps`),
    which RunnableCallable reads to decide what to inject.
    """

    def __init__(self, tools: Sequence[Any], *, bounds: tuple[float, float],
                 messages_key: str = "messages", **kwargs: Any) -> None:
        super().__init__(tools, messages_key=messages_key, **kwargs)
        self._sleep_bounds = bounds
        self._sleep_messages_key = messages_key

    def _turn_sleep(self, input: Any) -> float:
        if isinstance(input, list):
            messages = input
        elif isinstance(input, dict):
            messages = input.get(self._sleep_messages_key) or []
        else:
            messages = getattr(input, self._sleep_messages_key, None) or []
        calls = list(getattr(messages[-1], "tool_calls", None) or []) if messages else []
        seconds = tool_sleep_seconds(calls, self._sleep_bounds)
        logger.info(
            "tool sleep %.2f s before %s",
            seconds,
            ",".join(str(c.get("name")) for c in calls) or "no tool call",
        )
        return seconds

    @functools.wraps(ToolNode._afunc)
    async def _afunc(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(self._turn_sleep(input))
        return await super()._afunc(input, *args, **kwargs)

    @functools.wraps(ToolNode._func)
    def _func(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        time.sleep(self._turn_sleep(input))
        return super()._func(input, *args, **kwargs)


def make_tool_node(tools: Sequence[Any], **kwargs: Any) -> ToolNode:
    """ToolNode, or SleepingToolNode when SWE_TOOL_SLEEP_S asks for a sleep."""
    bounds = parse_tool_sleep(os.getenv(ENV))
    if bounds is None:
        return ToolNode(tools, **kwargs)
    logger.warning("%s=%s: every tool turn sleeps %s s", ENV, os.getenv(ENV),
                   bounds[0] if bounds[0] == bounds[1] else f"{bounds[0]}-{bounds[1]}")
    return SleepingToolNode(tools, bounds=bounds, **kwargs)
