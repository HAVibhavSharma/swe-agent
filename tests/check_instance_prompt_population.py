"""Check the per-instance seeds and ODR_REGISTRY_RECORD, without a GPU.

Seeds each node's prefix as far as an instance fixes it -- system block, repo
tree, issue -- the way the harness does after checking out an instance, then runs the real graph against the fake server of
check_prompt_seeds.py, as arm A does: the predictor's warms only load
(LANGGRAPH_PREDICTION_PROMOTE_ONLY=1), no prompt seeds, and no real request
recorded in the prefix registry (ODR_REGISTRY_RECORD=none).

The property that matters is that each seed is a prefix of the first request
its node really sends -- the same leading messages byte for byte, the last one
a prefix of the real one, and the same tools -- and that it reaches as far as
the instance allows: the tree where the node sends one, and the issue where
the node's history opens with it. A seed that
differs in one character is a prefix nothing hits, and that fails silently on
the GPU box.

    python tests/check_instance_prompt_population.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import check_prompt_seeds as fake  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label}{f' -- {detail}' if detail and not ok else ''}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/v1"

    work = Path(tempfile.mkdtemp(prefix="swe-tree-seeds-"))
    repo = work / "workspace_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "a.py").write_text("x = 1\nprint(x)\n")
    (repo / "pkg" / "__init__.py").write_text("")
    # Braces in a path: the tree is a value, never a template.
    (repo / "pkg" / "fmt_{name}.py").write_text("s = '{}'\n")
    os.environ.update({
        "MODEL_PROVIDER": "openai", "MODEL_NAME": "fake-model", "OPENAI_MODEL": "fake-model",
        "OPENAI_BASE_URL": base, "OPENAI_API_KEY": "EMPTY",
        "LANGGRAPH_VLLM_AGENT_BASE_URL": f"{base}/agents",
        "LANGGRAPH_PREDICTION_LOG_DIR": str(work / "logs"),
        "LANGGRAPH_TIMING_LOG_DIR": str(work / "timing"),
        "LANGGRAPH_COMPILE_DEBUG_PATH": str(work / "compile_debug.json"),
        "SWE_TRACE_MODE": "off", "LANGSMITH_TRACING": "false",
        # Arm A's workflow settings.
        "ODR_PROMPT_SEEDS": "0", "ODR_REGISTRY_RECORD": "none",
        "LANGGRAPH_PREDICTION_PROMOTE_ONLY": "1",
    })
    os.chdir(work)  # the agent hard-codes ./workspace_repo

    import system_prompt_population as population
    from agent.tools.write import get_files_structure

    issue = "Rename x to y in a.py and add new_mod.py"
    record = work / "instance_prompt_population.jsonl"
    result = asyncio.run(population.populate_instance_prompts(
        label="check", issue=issue, record_to=record))
    tree = get_files_structure.invoke({"directory": "./workspace_repo"})
    nodes = [node for node, _, _ in population._INSTANCE_PROMPTS]
    tree_nodes = {"come_up_with_research_next_step", "conduct_research",
                  "extract_implementation_plan", "get_clear_implementation_plan_for_atomic_task"}
    issue_nodes = {"come_up_with_research_next_step", "check_research_step",
                   "conduct_research", "extract_implementation_plan"}

    check("every node was seeded", result.get("ok") is True
          and result.get("nodes_seeded") == len(nodes), str(result.get("nodes")))
    check("one line per instance in the record", len(record.read_text().splitlines()) == 1)
    tree_seeds = {seed["langgraph_node"]: seed for seed in fake.SEEDS}
    check("one seed per node", sorted(tree_seeds) == sorted(nodes)
          and len(fake.SEEDS) == len(nodes), str([s["langgraph_node"] for s in fake.SEEDS]))
    for node, seed in tree_seeds.items():
        check(f"{node}: wildcard key, replaces, waits, prefills",
              seed["agent_id"] == f"langgraph:*:**:{node}"
              and seed["agent_kind"] == "non-react" and seed["wait"] is True
              and seed["prefill_on_miss"] is True and "job_id" not in seed, str(seed.keys()))

    seeded = len(fake.SEEDS)
    from agent.graph import swe_agent

    config = {"configurable": {"thread_id": "t1", "job_id": "job-1",
                               "max_researcher_iterations": 6, "max_react_tool_calls": 8,
                               "max_atomic_tasks": 20},
              "recursion_limit": 200}
    for _ in swe_agent.stream(
        {"implementation_research_scratchpad": [("user", issue)]},
        config, stream_mode="updates", subgraphs=True,
    ):
        pass
    check("arm A sends no prompt seeds", len(fake.SEEDS) == seeded)
    check("arm A's predictor warms carry no prompt",
          fake.WARMS and all("messages" not in warm for warm in fake.WARMS), str(len(fake.WARMS)))

    for node, seed in tree_seeds.items():
        real = next((body for body in fake.CHAT if body.get("langgraph_node") == node), None)
        if real is None:
            check(f"{node}: is a prefix of its first request", False, "no request from the node")
            continue
        want, got = seed["messages"], real["messages"]
        problems = []
        if got[: len(want) - 1] != want[:-1]:
            problems.append("leading messages differ")
        last, real_last = want[-1], got[len(want) - 1] if len(got) >= len(want) else {}
        if last["role"] != real_last.get("role") or not str(
                real_last.get("content") or "").startswith(last["content"]):
            problems.append("the tree message is not a prefix of the real one")
        contents = [m["content"] for m in want]
        if node in tree_nodes and not any(tree in c for c in contents):
            problems.append("the seed does not hold the tree")
        if node in issue_nodes and (last["role"], last["content"]) != ("user", issue):
            problems.append("the seed does not end with the issue")
        if real.get("tools") != seed.get("tools"):
            problems.append("tools differ")
        check(f"{node}: is a prefix of its first request ({len(want)} msgs, "
              f"{len(seed.get('tools') or [])} tools)", not problems, "; ".join(problems))

    for node in sorted(issue_nodes):
        seed = tree_seeds[node]
        real = next(body for body in fake.CHAT if body.get("langgraph_node") == node)
        check(f"{node}: every seed message is sent whole",
              real["messages"][: len(seed["messages"])] == seed["messages"])
    check("check_research_step: its system block and the issue",
          [m["role"] for m in tree_seeds["check_research_step"]["messages"]] == ["system", "user"])

    agent_calls = [body for body in fake.CHAT if body.get("agent_id")]
    check("ODR_REGISTRY_RECORD=none: no real request records its prompt",
          agent_calls and all(body.get("record_in_registry") is False for body in agent_calls),
          str({body.get("record_in_registry") for body in agent_calls}))

    from agent.common import llm_request_metadata as meta

    def records(mode: str, node: str) -> bool:
        os.environ["ODR_REGISTRY_RECORD"] = mode
        return meta._records_in_registry(node)

    check("react: only the tool-bound nodes record",
          records("react", "conduct_research")
          and records("react", "get_clear_implementation_plan_for_atomic_task")
          and not records("react", "come_up_with_research_next_step"))
    check("all, and unset: every node records",
          records("all", "check_research_step")
          and (os.environ.pop("ODR_REGISTRY_RECORD") and meta._records_in_registry("check_research_step")))
    os.environ["SWE_REGISTRY_RECORD"] = "none"
    os.environ["ODR_REGISTRY_RECORD"] = "all"
    check("the SWE_ spelling wins over ODR_", not meta._records_in_registry("conduct_research"))

    seeds_before = len(fake.SEEDS)
    os.environ["SWE_SEED_AGENT_ID_PATTERN"] = "0"
    import importlib
    importlib.reload(population)
    asyncio.run(population.populate_instance_prompts(label="exact", issue=issue))
    check("SWE_SEED_AGENT_ID_PATTERN=0: the old exact key",
          [s["agent_id"] for s in fake.SEEDS[seeds_before:]]
          == [f"langgraph:{node}" for node in nodes])

    server.shutdown()
    print(f"\n{'all checks passed' if not FAILURES else f'{len(FAILURES)} FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
