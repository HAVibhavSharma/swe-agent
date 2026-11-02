"""A pinned live call decodes exactly the recorded length, as ODR's does.

    python tests/check_trace_exact_tokens.py

Greedy decoding still stops wherever a run's arithmetic puts EOS: the same
prompt gave 525 tokens in one run and 787 in the next, and two arms' decode
times differed by that alone. With SWE_TRACE_PINNED_EXACT_TOKENS (default on)
the timing call is held to the recorded count with ignore_eos + min_tokens,
which first needs the decode constraints gone; the agent still gets the
recording.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx

# By path: agent/common has no __init__.py, and this needs nothing else of it.
_spec = importlib.util.spec_from_file_location(
    "trace_store",
    Path(__file__).resolve().parents[1] / "agent" / "common" / "trace_store.py",
)
ts = importlib.util.module_from_spec(_spec)
sys.modules["trace_store"] = ts
_spec.loader.exec_module(ts)

RECORDED = 787


def _transport(exact: bool, pin_length: bool):
    transport = object.__new__(ts.AsyncTraceTransport)
    transport._exact_tokens = exact
    transport._pin_length = pin_length
    return transport


def _call(transport, body: dict):
    request = httpx.Request("POST", "http://vllm/v1/chat/completions",
                            content=json.dumps(body).encode())
    entry = {"response_body": json.dumps({"usage": {"completion_tokens": RECORDED}}),
             "stream": False}
    live, limit, baseline, exact = transport._pinned_request(request, body, entry, False)
    return json.loads(live.content), limit, baseline, exact


def main() -> None:
    body = {
        "model": "m", "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "required", "max_tokens": 16384,
        "response_format": {"type": "json_object"}, "stop": ["</s>"],
    }
    original = json.loads(json.dumps(body))

    # Default: exactly the recorded length, constraints stripped, tools kept.
    sent, limit, baseline, exact = _call(_transport(exact=True, pin_length=False), body)
    assert exact and limit == baseline == RECORDED, (exact, limit, baseline)
    assert sent["ignore_eos"] is True and sent["min_tokens"] == RECORDED, sent
    assert sent["max_tokens"] == sent["max_completion_tokens"] == RECORDED, sent
    assert "response_format" not in sent and "stop" not in sent, sent
    assert sent["tool_choice"] == "auto" and sent["tools"] == body["tools"], sent
    assert sent["messages"] == body["messages"], "the prompt must not change"
    assert body == original, "the agent's own request body was modified"

    # Off, uncapped: the request goes out as the agent wrote it.
    sent, limit, _, exact = _call(_transport(exact=False, pin_length=False), body)
    assert not exact and limit is None and sent == body, sent

    # Off, capped: the old ceiling-only pin, EOS still allowed.
    sent, limit, _, exact = _call(_transport(exact=False, pin_length=True), body)
    assert not exact and limit == RECORDED and sent["max_tokens"] == RECORDED, sent
    assert "ignore_eos" not in sent and "min_tokens" not in sent, sent

    # The switch: on unless set to 0.
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SWE_TRACE_MODE"] = "pinned"
        trace = Path(tmp) / "trace.jsonl"
        trace.write_text("")
        os.environ["SWE_TRACE_PATH"] = str(trace)
        os.environ.pop("SWE_TRACE_PINNED_EXACT_TOKENS", None)
        assert ts._resolve().exact_tokens is True
        os.environ["SWE_TRACE_PINNED_EXACT_TOKENS"] = "0"
        assert ts._resolve().exact_tokens is False
    print("all trace-exact-tokens assertions passed")


if __name__ == "__main__":
    main()
