"""SWE-bench run for the node-aware KV eviction policy.

The counterpart of ``open_deep_research/tests/run_evaluate_node_eviction.py``.
Same two-phase shape as ``run_evaluate.py``, plus the forecast session that
publishes ``PROB|<job_id>`` to Redis from LangGraph task events -- one half of
what vLLM's node-aware eviction ranks blocks with. The other half,
``HISTORY|<job_id>``, is published by kv-trace-analyser from the request stats
vLLM writes to ``VLLM_REQUEST_STATS_DIR``.

The prefix prefetch is turned **off** here: it is an independent mechanism, and
running both at once means a hit-rate change cannot be attributed to either.
"""

import argparse
import csv
import itertools
import json
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
import asyncio
from langgraph.checkpoint.memory import MemorySaver

from langgraph.pregel._kv_forecast import KVForecastSession

from agent.graph import swe_agent_builder
from agent.common.model import MODEL_NAME, MODEL_PROVIDER

try:
    # `python tests/run_evaluate_node_eviction.py` puts tests/ on sys.path...
    from kv_metrics_reset import reset_kv_metrics
    from system_prompt_population import populate_system_prompts
    from swebench_instances import (
        append_prediction,
        collect_patch,
        load_instances,
        load_instances_file,
        materialize_workspace,
        patch_stats,
        problem_statement_message,
    )
except ImportError:  # ...`python -m tests.run_evaluate_node_eviction` does not.
    from tests.kv_metrics_reset import reset_kv_metrics
    from tests.system_prompt_population import populate_system_prompts
    from tests.swebench_instances import (
        append_prediction,
        collect_patch,
        load_instances,
        load_instances_file,
        materialize_workspace,
        patch_stats,
        problem_statement_message,
    )

load_dotenv(find_dotenv(), override=True)

RANDOM_SEED = 62
TARGET_COMPLETIONS_PER_INSTANCE = 3

seed = 0
max_researcher_iterations = int(os.getenv("MAX_RESEARCHER_ITERATIONS", "6"))
max_react_tool_calls = int(os.getenv("MAX_REACT_TOOL_CALLS", "8"))
max_atomic_tasks = int(os.getenv("MAX_ATOMIC_TASKS", "20"))
recursion_limit = int(os.getenv("GRAPH_RECURSION_LIMIT", "200"))

job_counter = itertools.count(1)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
METRICS_DIR = Path(__file__).resolve().parent / "eviction_logs" / RUN_ID
NODE_METRICS_DIR = METRICS_DIR / "node_metrics"
SUMMARY_PATH = METRICS_DIR / "instance_summary.jsonl"
E2E_LATENCY_CSV_PATH = METRICS_DIR / "job_instance_e2e_latency.csv"
PREDICTIONS_PATH = METRICS_DIR / "predictions.jsonl"
KV_METRICS_RESET_PATH = METRICS_DIR / "kv_metrics_reset.json"
# Per-node outcome of the seeding phase. Kept with the run it belongs to: a warm
# phase that never hits is explained by the node whose seed did not stick, and
# nothing else in the report says which one that was.
SYSTEM_PROMPT_POPULATION_PATH = METRICS_DIR / "system_prompt_population.json"
PREDICTION_DIR = Path(__file__).resolve().parents[1] / "SWE_predictions" / RUN_ID
os.environ["LANGGRAPH_PREDICTION_LOG_DIR"] = str(PREDICTION_DIR)

COLD_METRICS_DIR = METRICS_DIR / "cold_phase"
COLD_PREDICTION_DIR = PREDICTION_DIR / "cold_phase"
PHASE_COLD = "cold"
PHASE_WARM = "warm"


def _phase_metrics_dir(phase: str) -> Path:
    return METRICS_DIR if phase == PHASE_WARM else COLD_METRICS_DIR


def _phase_node_metrics_dir(phase: str) -> Path:
    return _phase_metrics_dir(phase) / "node_metrics"


def _phase_summary_path(phase: str) -> Path:
    return _phase_metrics_dir(phase) / "instance_summary.jsonl"


def _phase_e2e_csv_path(phase: str) -> Path:
    return _phase_metrics_dir(phase) / "job_instance_e2e_latency.csv"


def _phase_predictions_path(phase: str) -> Path:
    return _phase_metrics_dir(phase) / "predictions.jsonl"


# ---------------------------------------------------------------------------
# Node-aware KV eviction
# ---------------------------------------------------------------------------
# LANGGRAPH_ABLATION_MODE does NOT switch the prefetch off. Nothing in
# langgraph or this repo reads it -- it only names the experiment. The real
# switch is `vllm_agent_enabled()`, true if LANGGRAPH_VLLM_AGENT_ENABLE == "1"
# OR LANGGRAPH_VLLM_AGENT_BASE_URL is set, so both must be cleared -- and after
# load_dotenv, which may have set them from .env.
DISABLE_PREFETCH = os.getenv("KV_EVICTION_DISABLE_PREFETCH", "1") == "1"
if DISABLE_PREFETCH:
    os.environ["LANGGRAPH_VLLM_AGENT_ENABLE"] = "0"
    os.environ.pop("LANGGRAPH_VLLM_AGENT_BASE_URL", None)

ABLATION_MODE = (
    os.getenv("LANGGRAPH_ABLATION_MODE", "baseline").strip().lower() or "baseline"
)
os.environ["LANGGRAPH_ABLATION_MODE"] = ABLATION_MODE

# vLLM writes one JSONL per engine here; kv-trace-analyser tails the directory
# and publishes HISTORY. Point both at the same path -- there is no network
# transport for the trace, so the analyser must share this filesystem.
VLLM_STATS_DIR = os.getenv("VLLM_REQUEST_STATS_DIR") or str(METRICS_DIR / "vllm_stats")
os.environ["VLLM_REQUEST_STATS_DIR"] = VLLM_STATS_DIR

# Transition counts are a property of the workflow, not the run, so this path
# must survive across invocations: it is what lets a fresh job start from
# observed edge weights instead of uniform successors.
KV_TRANSITION_STORE = os.getenv("KV_PREDICTION_TRANSITION_STORE") or str(
    Path(__file__).resolve().parents[1] / "kv_forecast" / "transitions.json"
)
os.environ["KV_PREDICTION_TRANSITION_STORE"] = KV_TRANSITION_STORE

# The SWE agent topology, in the names the *runtime* uses (bare node names, as
# they appear in `metadata["langgraph_node"]` and therefore in the extra_body
# every request carries).
#
# Hand-written for the same reason ODR's is: `get_graph(xray=True)` prefixes
# subgraph nodes (`swe_architect:conduct_research`) while the runtime name is
# the bare `conduct_research`, so every row would mis-join silently.
#
# `calls` lists the node's LLM call sites -- empty for the two ToolNodes and
# for the pure state-shuffling nodes, which get no PROB row because they cache
# nothing. `type: react` marks the two loops that accumulate a growing prefix
# across turns; those are what the policy exists to protect.
SWE_GRAPH_SPEC: dict[str, dict[str, Any]] = {
    # --- root ---------------------------------------------------------------
    # Subgraph wrapper nodes. No LLM call of their own, but real states the
    # workflow passes through: leaving them out breaks the transition chain and
    # makes the task stream report them as drift.
    "swe_architect": {
        "type": "non-react",
        "calls": [],
        "next": ["come_up_with_research_next_step", "swe_developer"],
    },
    "swe_developer": {
        "type": "non-react",
        "calls": [],
        "next": ["start_implementing"],
    },
    # --- architect ----------------------------------------------------------
    "come_up_with_research_next_step": {
        "type": "non-react",
        "calls": ["come_up_with_research_next_step"],
        "next": ["check_research_step"],
    },
    "check_research_step": {
        "type": "non-react",
        "calls": ["check_research_step"],
        "next": ["conduct_research", "come_up_with_research_next_step"],
    },
    "conduct_research": {
        "type": "react",
        "calls": ["conduct_research"],
        "next": ["tools", "extract_implementation_plan"],
    },
    "tools": {
        "type": "non-react",
        "calls": [],
        "next": ["conduct_research"],
    },
    "extract_implementation_plan": {
        "type": "non-react",
        "calls": ["extract_implementation_plan"],
        "next": ["swe_developer"],
    },
    # --- developer ----------------------------------------------------------
    "start_implementing": {
        "type": "non-react",
        "calls": [],
        "next": ["prepare_for_implementation"],
    },
    "prepare_for_implementation": {
        "type": "non-react",
        "calls": [],
        "next": ["get_clear_implementation_plan_for_atomic_task"],
    },
    "get_clear_implementation_plan_for_atomic_task": {
        "type": "react",
        "calls": ["get_clear_implementation_plan_for_atomic_task"],
        "next": ["research_tool_node", "creating_diffs_for_task"],
    },
    "research_tool_node": {
        "type": "non-react",
        "calls": [],
        "next": ["get_clear_implementation_plan_for_atomic_task"],
    },
    # Two call sites live here (new-file creation and diff extraction); they
    # share the node name because that is the label vLLM sees.
    "creating_diffs_for_task": {
        "type": "non-react",
        "calls": ["creating_diffs_for_task"],
        "next": ["proceed_to_next_atomic_task"],
    },
    "proceed_to_next_atomic_task": {
        "type": "non-react",
        "calls": [],
        "next": ["prepare_for_implementation"],
    },
}

KV_FORECAST = KVForecastSession(
    SWE_GRAPH_SPEC,
    name="swe_agent",
    transition_store=KV_TRANSITION_STORE,
)


def _pseudo_dynamic_enabled() -> bool:
    return os.getenv("LANGGRAPH_PROMPT_PSEUDO_DYNAMIC", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _reconcile_graph_spec() -> None:
    """Fail loudly at startup if SWE_GRAPH_SPEC has drifted from the graph.

    A hand-written spec is the only way to name subgraph nodes the way the
    runtime does, and the price is drift when the workflow changes. An
    undeclared node is not an error -- it appears in no PROB row, so vLLM
    leaves its blocks unscored and in LRU order -- but it silently removes the
    node from the experiment, which is worse than a warning at import.
    """
    declared = set(SWE_GRAPH_SPEC)
    print(
        f"KV forecast: {'enabled' if KV_FORECAST.enabled else 'disabled'} "
        f"({len(declared)} nodes declared, label={ABLATION_MODE})"
    )
    # Print the real prefetch state, not the label. If this says enabled, the
    # /v1/agents warm calls are firing and phantom prefetch_only requests are
    # reaching vLLM alongside the eviction policy -- two mechanisms, one number.
    from langgraph.pregel._vllm_agent import vllm_agent_enabled

    prefetch_on = vllm_agent_enabled()
    print(f"Prefix prefetch (/v1/agents): {'ENABLED' if prefetch_on else 'disabled'}")
    print(
        "Pseudo-dynamic prefix fill: "
        f"{'enabled' if _pseudo_dynamic_enabled() else 'DISABLED'}"
        f"{'' if prefetch_on else ' (inert, prefetch off)'}"
    )
    if not KV_FORECAST.enabled:
        print(
            "  KV_FORECAST_REDIS_URL is unset, so no PROB/INFO is published "
            "and vLLM's policy will index blocks but rank nothing."
        )

    # Only the top-level names are reachable by reflection: the two subgraphs'
    # nodes live in their own builders. So this catches drift in the outer
    # graph at startup, and `KV_FORECAST.unknown_nodes` catches the rest from
    # the task stream as the run proceeds (reported at the end of main()).
    top_level = set(getattr(swe_agent_builder, "nodes", {}) or {})
    undeclared = top_level - declared
    if undeclared:
        print(
            "  WARNING: top-level graph nodes missing from SWE_GRAPH_SPEC: "
            + ", ".join(sorted(undeclared))
            + " -- their blocks will run unscored."
        )


def _slugify_model(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"


EXPERIMENT_NAME = os.getenv(
    "LANGSMITH_EXPERIMENT",
    f"swe-evict-{MODEL_PROVIDER}-{_slugify_model(MODEL_NAME)}-{ABLATION_MODE}-{RUN_ID}",
)
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_PROJECT"] = EXPERIMENT_NAME
os.environ["LANGCHAIN_PROJECT"] = EXPERIMENT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench against vLLM's node-aware KV eviction."
    )
    parser.add_argument("--dataset", type=str, default=os.getenv("SWE_DATASET", "SWE-bench/SWE-bench_Verified"))
    parser.add_argument("--split", type=str, default=os.getenv("SWE_SPLIT", "test"))
    parser.add_argument("--max-instances", type=int, default=6)
    parser.add_argument("--instance-ids", type=str, nargs="*", default=None)
    parser.add_argument("--instances-file", type=Path, default=None)
    parser.add_argument(
        "--completions-per-instance", type=int, default=TARGET_COMPLETIONS_PER_INSTANCE
    )
    parser.add_argument("--cold-instances-per-query", type=int, default=1)
    parser.add_argument("--skip-cold-phase", action="store_true")
    parser.add_argument(
        "--skip-system-prompt-population",
        action="store_true",
        help=(
            "Do not seed vLLM's AgentPrefixRegistry in this process. For when a "
            "separate invocation already seeded it -- the reset still fires "
            "before the warm phase, so the boundary is still drawn."
        ),
    )
    parser.add_argument(
        "--system-prompt-population-top-k",
        type=int,
        default=None,
        help=(
            "prefetch_top_k sent with each seed. Omitted by default, which tells "
            "the server to fan out over every prefix the agent has -- one, at "
            "this point, since the registry starts empty."
        ),
    )
    parser.add_argument(
        "--no-prefill-on-miss",
        dest="prefill_on_miss",
        action="store_false",
        help=(
            "Seed the registry without prefilling. The lookup is warm and the "
            "KV cache is not, so the warm phase's first visit to each node "
            "still pays a full prefill -- see tests/system_prompt_population.py."
        ),
    )
    parser.set_defaults(prefill_on_miss=True)
    parser.add_argument("--no-hbm-flush", action="store_true")
    parser.add_argument("--no-kv-metrics-reset", action="store_true")
    return parser.parse_args()


class TimingCollector:
    """Collect per-task timings from LangGraph task stream events."""

    def __init__(self) -> None:
        self._active: dict[str, dict[str, Any]] = {}
        self._invocation_counts: dict[str, int] = defaultdict(int)
        self.series: list[dict[str, Any]] = []
        self.totals_by_name_ns: dict[str, int] = defaultdict(int)

    def start(self, namespace: tuple[str, ...], payload: dict[str, Any]) -> None:
        node_name = payload["name"]
        self._invocation_counts[node_name] += 1
        self._active[payload["id"]] = {
            "node_name": node_name,
            "namespace": list(namespace),
            "invocation_index": self._invocation_counts[node_name],
            "started_at_ns": time.perf_counter_ns(),
        }

    def finish(self, payload: dict[str, Any]) -> None:
        started = self._active.pop(payload["id"], None)
        if started is None:
            return
        duration_ns = time.perf_counter_ns() - started["started_at_ns"]
        self.series.append(
            {
                "task_id": payload["id"],
                "node_name": started["node_name"],
                "namespace": started["namespace"],
                "invocation_index": started["invocation_index"],
                "duration_ns": duration_ns,
                "duration_seconds": duration_ns / 1_000_000_000,
                "error": _normalize_json_error(payload.get("error")),
                "interrupt_count": len(payload.get("interrupts", [])),
            }
        )
        self.totals_by_name_ns[started["node_name"]] += duration_ns

    def build_summary(self) -> dict[str, Any]:
        now_ns = time.perf_counter_ns()
        unfinished_tasks = [
            {
                "task_id": task_id,
                "node_name": started["node_name"],
                "namespace": started["namespace"],
                "invocation_index": started["invocation_index"],
                "elapsed_ns_so_far": now_ns - started["started_at_ns"],
                "elapsed_seconds_so_far": (now_ns - started["started_at_ns"]) / 1_000_000_000,
            }
            for task_id, started in self._active.items()
        ]
        totals_by_node_name_ns = dict(sorted(self.totals_by_name_ns.items()))
        return {
            "time_series": self.series,
            "totals_by_node_name_ns": totals_by_node_name_ns,
            "totals_by_node_name_seconds": {
                key: value / 1_000_000_000 for key, value in totals_by_node_name_ns.items()
            },
            "unfinished_tasks": unfinished_tasks,
        }


def _normalize_json_error(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _select_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.instances_file is not None:
        instances = load_instances_file(args.instances_file)
        if 0 < args.max_instances < len(instances):
            instances = instances[: args.max_instances]
        return instances
    return load_instances(
        args.dataset,
        args.split,
        instance_ids=args.instance_ids,
        max_instances=args.max_instances,
        seed=RANDOM_SEED,
    )


async def target(
    instance: dict[str, Any],
    *,
    logical_job_index: int,
    instance_index: int,
    phase: str = PHASE_WARM,
):
    graph = swe_agent_builder.compile(checkpointer=MemorySaver(), _is_root=True)
    job_id = str(next(job_counter))
    thread_id = str(uuid.uuid4())
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_at_ns = time.perf_counter_ns()

    config = {
        "configurable": {
            "thread_id": thread_id,
            "job_id": job_id,
            "seed": seed,
            "max_researcher_iterations": max_researcher_iterations,
            "max_react_tool_calls": max_react_tool_calls,
            "max_atomic_tasks": max_atomic_tasks,
        },
        "recursion_limit": recursion_limit,
        "run_name": f"swe-job{job_id}-{instance['instance_id']}-inst{instance_index}",
        "metadata": {
            "run_id": RUN_ID,
            "experiment": EXPERIMENT_NAME,
            "model": MODEL_NAME,
            "model_provider": MODEL_PROVIDER,
            "ablation_mode": ABLATION_MODE,
            "swe_instance_id": instance["instance_id"],
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "job_id": job_id,
            "logical_job_index": logical_job_index,
            "instance_index": instance_index,
            "phase": phase,
            "seed": seed,
        },
        "tags": [
            f"provider:{MODEL_PROVIDER}",
            f"model:{_slugify_model(MODEL_NAME)}",
            f"ablation:{ABLATION_MODE}",
            f"run:{RUN_ID}",
            f"phase:{phase}",
        ],
    }

    collector = TimingCollector()
    setup_error: str | None = None
    stream_error: str | None = None
    state_error: str | None = None
    final_state: dict[str, Any] | None = None
    patch = ""

    # `finish` events carry only the task id, so remember the node name from
    # the matching `start` in order to close the node out on the forecast.
    forecast_task_nodes: dict[str, str] = {}

    try:
        materialize_workspace(instance)
    except Exception as exc:  # noqa: BLE001
        setup_error = f"{type(exc).__name__}: {exc}"

    if setup_error is None:
        # INFO plus an initial PROB for every node in the graph -- including
        # nodes this job has not reached yet, so the policy can score a prefix
        # before it is first visited.
        KV_FORECAST.start_job(job_id)
        try:
            async for part in graph.astream(
                {"implementation_research_scratchpad": [problem_statement_message(instance)]},
                config,
                stream_mode="tasks",
                subgraphs=True,
                version="v2",
            ):
                if part["type"] != "tasks":
                    continue
                payload = part["data"]
                if "triggers" in payload:
                    collector.start(tuple(part["ns"]), payload)
                    node_name = payload.get("name")
                    if isinstance(node_name, str):
                        forecast_task_nodes[payload["id"]] = node_name
                        KV_FORECAST.on_task_start(node_name, job_id)
                else:
                    collector.finish(payload)
                    node_name = forecast_task_nodes.pop(payload.get("id"), None)
                    if node_name is not None:
                        KV_FORECAST.on_task_finish(node_name, job_id)
        except Exception as exc:
            stream_error = str(exc)
        finally:
            # Floor every key for this job and persist the transition counts.
            # In `finally` because a job that died still has to release its
            # blocks; otherwise they stay valuable until the age term decays
            # them out, which on back-to-back jobs is long enough to crowd out
            # the job that replaced it.
            KV_FORECAST.end_job(job_id)
            try:
                final_state_snapshot = await graph.aget_state(config)
                final_state = final_state_snapshot.values
            except Exception as exc:
                state_error = str(exc)
            try:
                patch = collect_patch()
            except Exception as exc:  # noqa: BLE001
                state_error = state_error or f"patch_collect: {exc}"

    finished_at_ns = time.perf_counter_ns()
    finished_at_utc = datetime.now(timezone.utc).isoformat()
    query_duration_ns = finished_at_ns - started_at_ns
    summary = collector.build_summary()
    node_metrics_path = (
        _phase_node_metrics_dir(phase) / f"job_{job_id}_thread_{thread_id}.jsonl"
    )

    append_prediction(
        _phase_predictions_path(phase),
        instance,
        patch,
        f"{EXPERIMENT_NAME}-inst{instance_index}",
    )

    base_record = {
        "run_id": RUN_ID,
        "swe_instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "logical_job_index": logical_job_index,
        "instance_index": instance_index,
        "phase": phase,
        "thread_id": thread_id,
        "job_id": job_id,
        "started_at": started_at_utc,
        "finished_at": finished_at_utc,
    }

    for event in summary["time_series"]:
        _append_jsonl(node_metrics_path, {**base_record, **event})

    _append_jsonl(
        _phase_summary_path(phase),
        {
            **base_record,
            **patch_stats(patch),
            "node_metrics_jsonl": str(node_metrics_path.resolve()),
            "query_duration_ns": query_duration_ns,
            "query_duration_seconds": query_duration_ns / 1_000_000_000,
            "node_execution_count": len(summary["time_series"]),
            "totals_by_node_name_ns": summary["totals_by_node_name_ns"],
            "totals_by_node_name_seconds": summary["totals_by_node_name_seconds"],
            "unfinished_tasks": summary["unfinished_tasks"],
            "setup_error": setup_error,
            "stream_error": stream_error,
            "state_error": state_error,
            "final_state_present": final_state is not None,
            "empty_patch": not patch.strip(),
        },
    )

    success = setup_error is None and stream_error is None and state_error is None
    return {
        "success": success,
        "run_id": RUN_ID,
        "swe_instance_id": instance["instance_id"],
        "logical_job_index": logical_job_index,
        "instance_index": instance_index,
        "phase": phase,
        "job_id": job_id,
        "thread_id": thread_id,
        "started_at": started_at_utc,
        "finished_at": finished_at_utc,
        "query_duration_ns": query_duration_ns,
        "query_duration_seconds": query_duration_ns / 1_000_000_000,
        "setup_error": setup_error,
        "stream_error": stream_error,
        "state_error": state_error,
        "empty_patch": not patch.strip(),
        **patch_stats(patch),
    }


def _e2e_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "swe_instance_id": result["swe_instance_id"],
        "logical_job_index": result["logical_job_index"],
        "instance_index": result["instance_index"],
        "phase": result["phase"],
        "job_id": result["job_id"],
        "thread_id": result["thread_id"],
        "started_at": result["started_at"],
        "finished_at": result["finished_at"],
        "query_duration_ns": result["query_duration_ns"],
        "query_duration_seconds": result["query_duration_seconds"],
        "success": result["success"],
        "empty_patch": result["empty_patch"],
        "patch_files": result["patch_files"],
        "setup_error": result["setup_error"],
        "stream_error": result["stream_error"],
        "state_error": result["state_error"],
    }


async def _run_phase(
    selected: list[dict[str, Any]],
    runs_per_instance: int,
    phase: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = runs_per_instance * len(selected)
    position = 0
    for instance_index in range(1, runs_per_instance + 1):
        for job_index, instance in enumerate(selected, start=1):
            position += 1
            label = "cold -- filling caches, not measured" if phase == PHASE_COLD else "warm"
            print(f"[{phase} {position}/{total}] {instance['instance_id']} ({label})")
            result = await target(
                instance,
                logical_job_index=job_index,
                instance_index=instance_index,
                phase=phase,
            )
            rows.append(result)
            _append_csv_row(_phase_e2e_csv_path(phase), _e2e_csv_row(result))
    return rows


async def _run_cold_phase_isolated(
    selected: list[dict[str, Any]],
    runs_per_instance: int,
) -> list[dict[str, Any]]:
    previous = os.environ.get("LANGGRAPH_PREDICTION_LOG_DIR")
    os.environ["LANGGRAPH_PREDICTION_LOG_DIR"] = str(COLD_PREDICTION_DIR)
    try:
        return await _run_phase(selected, runs_per_instance, PHASE_COLD)
    finally:
        if previous is None:
            os.environ.pop("LANGGRAPH_PREDICTION_LOG_DIR", None)
        else:
            os.environ["LANGGRAPH_PREDICTION_LOG_DIR"] = previous


async def main():
    args = parse_args()
    selected = _select_instances(args)

    _reconcile_graph_spec()
    print(f"Metrics directory: {METRICS_DIR}")
    print(f"System prompt population report: {SYSTEM_PROMPT_POPULATION_PATH}")
    print(f"Predictions JSONL: {PREDICTIONS_PATH}")
    print(f"Prediction logs directory: {PREDICTION_DIR}")
    print(f"vLLM request stats directory: {VLLM_STATS_DIR}")
    print(f"KV transition store: {KV_TRANSITION_STORE}")
    print(f"Model: {MODEL_PROVIDER}:{MODEL_NAME}")
    print(f"LangSmith project: {EXPERIMENT_NAME}")
    print(
        "Loop bounds: "
        f"max_researcher_iterations={max_researcher_iterations}, "
        f"max_react_tool_calls={max_react_tool_calls}, "
        f"max_atomic_tasks={max_atomic_tasks}"
    )
    print(
        f"Running {len(selected)} instances sequentially, "
        f"{args.completions_per_instance} run(s) each"
    )
    print(
        "Phases: cold "
        f"({0 if args.skip_cold_phase else args.cold_instances_per_query} "
        "run(s) per instance) -> system prompt population "
        f"({'SKIPPED' if args.skip_system_prompt_population else 'seeding'} "
        "the AgentPrefixRegistry"
        f"{'' if args.prefill_on_miss else ', registry only'}) "
        "-> POST /v1/kv_metrics/reset"
        f"{'' if args.no_hbm_flush else ' (+ HBM flush)'} "
        "-> warm (measured)"
    )

    try:
        # ---- cold phase ---------------------------------------------------
        cold_rows: list[dict[str, Any]] = []
        if args.skip_cold_phase:
            print("Cold phase: SKIPPED (--skip-cold-phase)")
        elif args.cold_instances_per_query > 0:
            cold_rows = await _run_cold_phase_isolated(selected, args.cold_instances_per_query)
            failed_cold = [row for row in cold_rows if not row["success"]]
            print(f"Cold phase complete: {len(cold_rows)} runs, {len(failed_cold)} failed")
            if failed_cold:
                print(
                    "  WARNING: some cold runs failed, so the caches are only "
                    "partly filled for the warm phase"
                )

        # ---- system prompt population -------------------------------------
        # One POST /v1/agents/prefetch per call site, each carrying that node's
        # system prompt as `text`, so vLLM's AgentPrefixRegistry holds every
        # prompt before the first real request exists. Fills the registry
        # always, the KV cache only with prefill_on_miss -- see
        # tests/system_prompt_population.py.
        system_prompt_population: dict[str, Any] | None = None
        if args.skip_system_prompt_population:
            print("System prompt population: SKIPPED (--skip-system-prompt-population)")
        else:
            system_prompt_population = await populate_system_prompts(
                label=f"system_prompt_population_{RUN_ID}",
                top_k=args.system_prompt_population_top_k,
                prefill_on_miss=args.prefill_on_miss,
                record_to=SYSTEM_PROMPT_POPULATION_PATH,
            )
            if not system_prompt_population.get("ok"):
                # A node whose seed did not stick prefills on every visit, not
                # just the first, and that shows up in the warm phase as a miss
                # with nothing to do with the policy under test.
                print(
                    "  WARNING: not every call site was seeded, so the registry "
                    "is only partly populated for the warm phase"
                )

        # ---- the boundary -------------------------------------------------
        # Everything the server counted up to here -- the population phase's
        # phantom requests, their LMCache misses, the epoch clock behind any
        # rate -- belongs to the phase before the boundary, and this is what
        # tells it so.
        kv_metrics_reset: dict[str, Any] | None = None
        if not args.no_kv_metrics_reset:
            kv_metrics_reset = await reset_kv_metrics(
                f"cold_done_{RUN_ID}",
                flush_hbm=not args.no_hbm_flush,
                record_to=KV_METRICS_RESET_PATH,
            )

        # ---- warm phase ---------------------------------------------------
        print(f"Warm phase: measuring, report goes to {METRICS_DIR}")
        warm_rows = await _run_phase(selected, args.completions_per_instance, PHASE_WARM)

        results = {
            "run_id": RUN_ID,
            "sampled_instances": len(selected),
            "completions_per_instance": args.completions_per_instance,
            "completed_runs": len(warm_rows),
            "failed_runs": sum(1 for row in warm_rows if not row["success"]),
            "empty_patch_runs": sum(1 for row in warm_rows if row["empty_patch"]),
            "metrics_dir": str(METRICS_DIR),
            "prediction_logs_dir": str(PREDICTION_DIR),
            "instance_summary_jsonl": str(SUMMARY_PATH),
            "e2e_latency_csv": str(E2E_LATENCY_CSV_PATH),
            "predictions_jsonl": str(PREDICTIONS_PATH),
            "vllm_request_stats_dir": VLLM_STATS_DIR,
            "kv_transition_store": KV_TRANSITION_STORE,
            "cold_runs": len(cold_rows),
            "cold_runs_failed": sum(1 for row in cold_rows if not row["success"]),
            "cold_phase_metrics_dir": str(COLD_METRICS_DIR),
            "system_prompt_population_ok": bool(
                system_prompt_population and system_prompt_population.get("ok")
            ),
            "system_prompt_population_seeded": (
                system_prompt_population.get("nodes_seeded")
                if system_prompt_population
                else None
            ),
            "system_prompt_population_attempted": (
                system_prompt_population.get("nodes_attempted")
                if system_prompt_population
                else None
            ),
            "system_prompt_population_json": (
                str(SYSTEM_PROMPT_POPULATION_PATH) if system_prompt_population else None
            ),
            "kv_metrics_reset_ok": bool(kv_metrics_reset and kv_metrics_reset.get("ok")),
            "kv_metrics_epoch": kv_metrics_reset.get("epoch") if kv_metrics_reset else None,
            "hbm_flushed": kv_metrics_reset.get("hbm_flushed") if kv_metrics_reset else None,
        }
        return results
    finally:
        # Flushes the publisher's queue and saves the transition counts that
        # seed the next run.
        KV_FORECAST.close()
        unknown = KV_FORECAST.unknown_nodes
        if unknown:
            print(
                "KV forecast: these runtime nodes were NOT in SWE_GRAPH_SPEC, "
                "so their blocks ran unscored: " + ", ".join(sorted(unknown))
            )


if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)
