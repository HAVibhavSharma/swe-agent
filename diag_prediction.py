#!/usr/bin/env python3
"""Health check for the prefetch path. Run this before any benchmark.

Answers, in order, the four questions that decide whether a "treatment" run is
actually a treatment run:

1. Is the LangGraph fork installed at all? (`_vllm_agent` imports or not)
2. Did the compile produce prompt composition metadata? Zero segments means
   the prefix builder has nothing to warm, and the prefetch is inert no matter
   what the env says.
3. Did it produce transition prediction rules? Zero rules means nothing is ever
   predicted, so no prefetch is ever fired.
4. Is the prefetch actually enabled, and pointed at a reachable server?

Writes the two metadata blobs to prompt_compile_metadata.json so they can be
diffed across changes to the agent -- the same file ODR keeps.

    python diag_prediction.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

OUT_PATH = Path(__file__).resolve().parent / "prompt_compile_metadata.json"


def main() -> int:
    # 1 --------------------------------------------------------------------
    try:
        from langgraph.pregel._vllm_agent import (  # noqa: F401
            derive_agent_id,
            vllm_agent_enabled,
        )
    except ModuleNotFoundError:
        print("FAIL: langgraph.pregel._vllm_agent is missing.")
        print("      This checkout has stock langgraph. Install the fork:")
        print("        pip install -e ~/Projects/langgraph-dev/libs/langgraph")
        return 1
    print("ok: LangGraph fork is importable")

    from agent.graph import swe_agent_builder

    graph = swe_agent_builder.compile(_is_root=True)

    # 2 --------------------------------------------------------------------
    composition = getattr(graph, "prompt_composition", None) or {}
    print(f"prompt_composition: {len(composition)} node(s)")
    for node, entries in sorted(composition.items()):
        segments = sum(len(entry.get("segments", [])) for entry in entries)
        static = sum(
            len(segment.get("text") or "")
            for entry in entries
            for segment in entry.get("segments", [])
            if segment.get("classification") == "static"
        )
        print(f"  {node}: {len(entries)} call(s), {segments} segment(s), {static} static chars")
    if not composition:
        print(
            "  WARNING: no prompt composition was extracted, so the prefetch has\n"
            "  nothing to warm. Reading `(prompt_template | model).invoke({...})`\n"
            "  needs langgraph-dev branch Prediction-SWEbench; check which branch\n"
            "  is installed. See plan/02-required-changes.md."
        )

    # 3 --------------------------------------------------------------------
    prediction = getattr(graph, "transition_prediction", None) or {}
    rules = prediction.get("rules", []) if isinstance(prediction, dict) else []
    print(f"transition_prediction: {len(rules)} rule(s)")
    for rule in rules:
        condition = rule.get("condition", {}).get("kind")
        target = rule.get("prediction", {}).get("next_executable_node")
        print(f"  [{rule.get('priority')}] {rule.get('id')}: {condition} -> {target}")

    OUT_PATH.write_text(
        json.dumps(
            {"prompt_composition": composition, "transition_prediction": prediction},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT_PATH}")

    # 4 --------------------------------------------------------------------
    enabled = vllm_agent_enabled()
    print(f"prefetch enabled: {enabled}")
    print(f"  LANGGRAPH_VLLM_AGENT_ENABLE={os.getenv('LANGGRAPH_VLLM_AGENT_ENABLE')!r}")
    print(f"  LANGGRAPH_VLLM_AGENT_BASE_URL={os.getenv('LANGGRAPH_VLLM_AGENT_BASE_URL')!r}")
    print(f"  OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL')!r}")
    if enabled:
        import urllib.request

        base = (
            os.getenv("LANGGRAPH_VLLM_AGENT_BASE_URL")
            or f"{(os.getenv('OPENAI_BASE_URL') or '').rstrip('/')}/agents"
        )
        url = f"{base.rstrip('/')}/prefetch"
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(
                    {"agent_id": "diag:probe", "agent_kind": "non-react", "wait": True}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                print(f"  POST {url} -> {response.status}")
        except Exception as exc:  # noqa: BLE001
            print(f"  POST {url} FAILED: {type(exc).__name__}: {exc}")
            print("  (is this the modified vLLM, with the /v1/agents routes mounted?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
